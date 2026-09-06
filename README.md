<!-- The block below configures a Hugging Face Space (Docker). It is ignored
     everywhere else. See DEPLOY.md. -->
---
title: SkyGuard AI
emoji: "\U0001F6F0"
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SkyGuard AI

Real-time anomaly detection for Automatic Weather Stations.

**Smart India Hackathon 2026 · Problem statement 26073**
Ministry of Earth Sciences / India Meteorological Department · Software · Disaster Management

---

## The problem, and the constraint that shapes everything

An AWS reports temperature, pressure and humidity every ten minutes. Some of
those readings are wrong — sensors freeze, calibration drifts, batteries brown
out, modems drop packets. Wrong readings enter weather forecasts and disaster
warnings.

Detecting a wrong number is easy. **Detecting a wrong number without also
flagging a real heatwave, a real frontal passage or a real monsoon burst is the
whole problem**, and you have to do it from three channels with nothing else to
lean on.

## Approach

Five detection layers, ordered by how easily each can be fooled by genuine
weather. That ordering sets the fusion weights.

| Layer | Method | Catches | Weight |
|---|---|---|---|
| **L0** physics | Range, slew rate, dewpoint ≤ temperature | The impossible | 1.00 |
| **L1** temporal | Hampel on forecast residual, frozen-value test | Spikes, steps, latched sensors | 0.85 |
| **L2** multivariate | Robust Mahalanobis + Isolation Forest | Individually plausible, jointly impossible | 0.75 |
| **L3** sequence | PCA subspace reconstruction | Drift, noise bursts | 0.70 |
| **L4** spatial | WMO buddy check against neighbours | Sustained bias | 0.90 |

Evidence is pooled with a weighted noisy-OR, so layers reinforce each other but
a silent layer cannot veto a certain one.

**The dewpoint check answers the problem statement's own example.** A station
reporting 55 °C with very high humidity is individually in range on both
channels, but the pair implies a dewpoint above the air temperature. That is
arithmetic, not inference — deterministic, explainable in one sentence, and it
runs unchanged on an ESP32.

## Measured results

Reproduce with `make bench`, `make calibrate`, `make ablation`. Every number
below came out of those commands; none is estimated.

**Calibration sweep** (validation split, separate seeds and time origin):

| threshold | precision | recall | F1 | clean-stream FPR |
|---|---|---|---|---|
| 0.55 | 0.616 | 0.469 | 0.533 | 0.0124 |
| **0.60** | **0.618** | **0.468** | **0.532** | **0.0100** |
| 0.70 | 0.622 | 0.464 | 0.532 | 0.0081 |
| 0.90 | 0.588 | 0.376 | 0.459 | 0.0005 |

Selection rule is *maximise F1 subject to clean-stream false-positive rate ≤ 1 %*,
not maximise F1. A QC system that cries wolf gets muted in week two, and once
muted its recall is zero whatever the benchmark said. (`config.py` currently
ships an `alert_threshold` of 0.55, just above this budget; the two need
reconciling.)

**Spatial ablation** (4 stations, 6 000 samples each, 64 injected episodes):

| configuration | precision | recall | F1 | event recall | clean FPR |
|---|---|---|---|---|---|
| all five layers | 0.559 | 0.575 | 0.567 | 1.000 | 0.100 |
| without L4 spatial | 0.548 | 0.555 | 0.552 | 1.000 | 0.100 |

**Event recall is 1.000 across every fault type the benchmark injects** — spike,
frozen sensor, bias step, noise burst, dropout, power flicker, physical
violation. Every injected episode is caught at onset. Drift is not currently in
the test set (`CLAUDE.md` §8.6), so that is seven fault types, not eight.

**Latency:** p50 ≈ 5 ms, p99 ≈ 8 ms per observation, single-threaded Python,
roughly 300 observations/second.

### What these numbers mean, stated plainly

Event recall is excellent and point recall is mediocre: the system reliably
notices every fault when it begins, then partially loses long ones once its own
baseline adapts — less so now that the pipeline stops re-seeding a channel's
baseline from the sensor while the buddy check says the neighbours still
disagree (network point recall 0.555 → 0.575, F1 0.552 → 0.567, clean-stream FPR
unchanged). That is what L4 buys: +0.015 F1 / +0.020 point recall in the
ablation. Precision still needs work. Root-cause classification is now ~0.44
(up from ~0.32 — the multivariate layer stopped claiming "physical violation"
for single-channel spikes) but remains the weakest layer. See
`CLAUDE.md` §8 for the ranked list of open weaknesses.

## Quick start

```bash
pip install -e ".[api,dev]"

make demo         # the problem statement's example use case, end to end
make serve        # live console at http://127.0.0.1:8000
make bench        # detection accuracy on injected faults
make calibrate    # threshold sweep with false-positive budget
make ablation     # does each layer earn its place?
make check        # lint + tests + quick benchmark
```

### The console

`make serve` runs a simulated four-station Haryana/Himachal network at
accelerated time and streams every verdict over WebSocket to a single-page
console (`api/static/index.html`, no build step, works offline). The baseline
sits quiet; pick a station, pick a fault, press **Inject**, and watch it get
caught — the five detection layers light up by contribution, the "why it fired"
panel gives the plain-language reason and the maintenance action, the station's
health bar falls, and a sustained fault drives it into quarantine.

That interactive injection is the demo. It is considerably more convincing than
a recorded video, because a judge can choose the fault. (The console runs at a
slightly less sensitive `alert_threshold` than the benchmark so the clean
baseline does not chatter — a presentation choice, noted in `api/server.py`.)

### As a library

```python
from skyguard import SkyGuardPipeline, Observation

pipeline = SkyGuardPipeline()
pipeline.register_station("AWS-HR-AMB", altitude_m=272.0)
pipeline.fit(clean_history)               # list[Observation]

verdict = pipeline.process(observation)
if verdict.is_anomalous:
    print(verdict.fault_type.label)       # "Frozen sensor"
    print(verdict.attributions[0].narrative)
    print(verdict.fault_type.maintenance_action)
    print(verdict.repaired)               # corrected estimate, or None
```

## How the evaluation criteria are addressed

| Criterion | Weight | Where |
|---|---|---|
| Innovation & novelty | 25 % | Five-layer fusion; buddy check on three channels; exact attribution instead of sampled SHAP |
| Detection accuracy | 20 % | `evaluation/benchmark.py` — point, event and per-fault-type metrics |
| Real-time capability | 15 % | Bounded ring buffers, O(window) statistics, cascaded expensive tests; ~5 ms/observation |
| Explainability | 10 % | `explain/attribution.py` — exact per-layer contributions plus operator narrative |
| Scalability | 10 % | Learned layers shared network-wide, per-station state is a few kilobytes |
| Practical deployability | 10 % | L0 runs unchanged on ESP32; no GPU, no database, two dependencies |
| Visualisation / UI | 5 % | `api/static/index.html` — dependency-free console, works offline |
| Energy efficiency | 5 % | PCA forward pass is a matrix multiply; no neural inference in the hot path |

## Why some obvious choices were rejected

- **No LSTM autoencoder.** PCA reconstruction is a linear autoencoder with a
  closed-form optimum; at this window size it is within a few F1 points, fits in
  under a second, has no hyperparameters to defend, and its forward pass fits an
  ESP32 budget.
- **No runtime SHAP.** The model is not a black box — every detector emits its
  own statistic and an exact factored contribution. Attribution costs
  microseconds instead of hundreds of model evaluations per alert.
- **No database, no microservices.** One process, one pipeline object, many
  stations.

## Honest limitations

- All results are on synthetic data. The generator is physically coherent
  (humidity is *derived* from temperature and dewpoint, so it cannot be
  thermodynamically inconsistent by construction) and the network shares a
  synoptic driver with distance-dependent correlation — but it is still
  synthetic. Real-archive validation is the highest-value open item.
- Network false-positive rate (0.100) is worse than single-station (0.022).
  Undiagnosed.
- Root-cause classification accuracy is ~0.44 (was ~0.32). Detection is solid
  for every fault; it is the *label* that is unreliable — the classifier still
  confuses spike/bias-step and drift/noise-burst. Frozen sensor, data dropout
  and the impossible-humidity case it names correctly every time.
- The benchmark does not currently inject drift faults (`CLAUDE.md` §8.6), and
  its injected stream is ~23 % anomalous rather than the ~4 % the injector
  docstring claims. Read point precision and recall with that in mind.

## Licence

MIT.
