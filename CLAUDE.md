# CLAUDE.md — SkyGuard AI

Working agreement for this repository. Read this fully before touching code.

**Project:** Smart India Hackathon 2026, problem statement **26073** —
AI/ML-based intelligent anomaly detection for Automatic Weather Stations.
**Client:** Ministry of Earth Sciences / India Meteorological Department.
**Deliverable:** fully executable code, plus a document explaining use cases.

---

## 1. What this system is

An online quality-control system for AWS telemetry. It reads a stream of
temperature (°C), atmospheric pressure (hPa) and relative humidity (%) and, for
every single observation, decides whether the reading is trustworthy — then
says *why*, in language a duty forecaster can act on.

The constraint that shapes everything: **only three channels are available.**
No wind, no radiation, no rain gauge. Three numbers, and the requirement to
distinguish a broken sensor from real weather. The whole architecture is a
response to that constraint.

## 2. The one thing that must not be lost

**Distinguishing a sensor fault from a genuine meteorological event is the
entire problem.** Anything can flag a 55 °C reading. The value is in *not*
flagging a real heatwave, a real frontal passage, or a real monsoon burst — and
in being able to explain the difference.

Every design decision here traces back to that. When you are unsure whether a
change is right, ask: does it make the system better at telling weather from
faults, or does it just make the anomaly score bigger?

## 3. Mistakes already made — do not repeat these

These are real failures found by measurement during the initial build. Each cost
hours. They are recorded because the naive design is *seductive* and you will be
tempted back toward it.

### 3.1 Never baseline against a trailing median

Comparing a reading to the median of a trailing window produced a **90 %
false-positive rate on clean data.** Temperature has an ~8 °C diurnal swing, so
during the morning ramp a 4-hour trailing median sits several degrees below the
current reading and every clean sunrise scores as a multi-sigma anomaly.

The baseline is a **one-step-ahead local linear forecast**
(`StationState.predict`). The diurnal ramp is absorbed into the slope; you test
the residual. Do not "simplify" this back to a median.

### 3.2 Scale estimates must be floored at sensor resolution

Barometers report to 0.1 hPa and pressure moves slowly, so consecutive reports
are frequently bit-identical. The median absolute forecast residual then
evaluates to **exactly zero**, every standardised score divides by ~0, and the
detector saturates and flags everything forever.

`SensorResolution.floor()` returns `3 × step / √12` — the standard deviation of a
uniformly quantised variable. This is a physical fact about the instrument, not
a fudge factor. Do not remove it, and do not replace it with a generic epsilon.

### 3.3 Repairs must be damped, or they run away

Naive repair is "last good value plus local slope". During a long fault (an
offset step runs up to 400 samples) every reading is flagged, so every repair
extrapolates from the *previous repair*. Station pressure walked from 985 hPa to
877 hPa inside one episode and poisoned every downstream layer.

`Repairer` uses geometric damping (`TREND_DAMPING = 0.8`), so the total trend
contribution converges to four slopes regardless of outage length. Repairs are
also clamped to physical limits.

### 3.4 A flagged channel must have a recovery path

If only repairs ever enter the buffer, a channel that gets flagged — correctly
or not — can never come back. The repair drifts further from reality every step,
and that widening gap guarantees the next sample is flagged too. In testing this
locked humidity at a constant 100 % for the remainder of the stream.

`REBASELINE_AFTER = 6`: after six consecutive substitutions the pipeline stops
trusting its own model and re-seeds from what the sensor is reporting. Beyond
about an hour, persistence has no forecast skill, so the substitute carries no
more information than the raw reading anyway.

### 3.5 L0 must compare against as-reported values, never repaired ones

The rate-of-change gate originally compared each reading against
`last_clean` — the *accepted* value. A drifting repair therefore made the physics
gate fire, which inverts the entire layering premise: **L0 is the layer that
cannot be wrong.** This single bug produced a 60 % false-alarm rate on a clean
stream.

`rate_per_minute` reads `state.reported`. The test asks whether the instrument
physically slewed faster than it can; that is a property of the sensor's own
output.

### 3.6 The stuck detector must read as-reported values too

Once a frozen channel is flagged, the pipeline substitutes a varying repair,
which un-freezes the buffer and silences the detector that found it. The
`reported` ring buffer exists for exactly this.

### 3.7 Weak evidence must not pool into alerts

Fusion is noisy-OR. Four detectors each chattering at score 0.05 on a third of
all samples will, in company, manufacture alerts out of nothing. Every detector
has an **emission gate** — it stays silent unless its own statistic is genuinely
notable (typically 0.75× its decision threshold). If you find yourself lowering
an emission gate to "catch more", measure the clean-stream false-positive rate
first.

### 3.8 No train/serve skew in the learned layers

`MultivariateDetector._residual_matrix` deliberately **replays the online path**
during fitting rather than using a faster centred rolling window. A centred
window peeks at the future, so training residuals would be better behaved than
anything seen in production. That skew is the classic way a detector posts good
offline numbers and fails on the day.

---

## 4. Architecture

Five layers, ordered by **how easily each can be fooled by real weather**. That
ordering *is* the design rationale and it determines the fusion weights.

| Layer | File | Method | Catches | Fooled by weather? |
|---|---|---|---|---|
| L0 physics | `detect/l0_physics.py` | Range, slew rate, dewpoint ≤ temperature | The impossible | Never — weight 1.00 |
| L1 temporal | `detect/l1_temporal.py` | Hampel on forecast residual, stuck test | Spikes, steps, frozen sensors | Rarely — 0.85 |
| L2 multivariate | `detect/l2_multivariate.py` | Robust Mahalanobis + Isolation Forest | Individually plausible, jointly impossible | Sometimes — 0.75 |
| L3 sequence | `detect/l3_sequence.py` | PCA subspace reconstruction | Drift, noise bursts | Yes — 0.70 |
| L4 spatial | `detect/l4_spatial.py` | WMO buddy check against neighbours | Sustained bias | Almost never — 0.90 |

**L0's dewpoint check is the flagship.** The problem statement's own example —
55 °C with very high humidity — is individually in range on both channels but
jointly implies a dewpoint far above air temperature. That is arithmetic, not
inference: deterministic, explainable in one sentence, and it runs unchanged on
an ESP32.

**L4 exists because single-station layers plateau at ~0.50 point recall.** Once a
sensor has read 5 °C high for an hour, the station's own forecast has adapted and
every self-referential test goes quiet. A persistent bias is only identifiable
against an external reference, and the cheapest one is the rest of the network.

### Data flow, and why the order is fixed

```
observation
   ├─ record as-reported values            (state.reported)
   ├─ every detector sees the SAME history (no detector mutates state)
   ├─ fuse evidence  → per-channel confidence (noisy-OR)
   ├─ classify root cause, assign severity
   ├─ build attributions, compute repair
   └─ THEN update state, health, buddy statistics
```

Detectors must not mutate state mid-pass. A detector that pushed its own value
would give the next detector a different view of the past, making results depend
on registration order.

---

## 5. Code standards

**Comments explain *why*, never *what*.** `# increment counter` is noise.
`# Centred on zero rather than the residual median: a non-zero median is itself
evidence of drift, and re-centring would hide it` is the standard.

**Every threshold carries its provenance.** A number in `config.py` with no
comment saying where it came from — a WMO limit, a physical bound, or a
calibration sweep — is a magic number and does not belong there.

**Types cross module boundaries, not dicts.** Everything shared is a frozen
dataclass in `models.py`. If a field is needed downstream, add it to the type;
do not invent an ad-hoc dict shape.

**Do not name the file `types.py`.** It shadows the stdlib module and breaks any
absolute `import types` when the working directory is the package. It is
`models.py` for that reason.

**Bounded memory in the hot path.** Every online statistic comes from a ring
buffer in O(window) time. No pandas rolling windows, no unbounded lists. Real-
time capability is 15 % of the score and ESP32 deployment is an explicit ask.

**Cascade expensive tests behind cheap ones.** sklearn's per-row
`score_samples` costs milliseconds and alone would blow the latency budget, so
the cheap chi-square test gates it. Mean latency dropped ~15× with no change to
F1.

**Style:** Python 3.12, `from __future__ import annotations`, full type hints,
`ruff` clean, 4-space indent, 90-column soft limit. No emoji in code or output.

---

## 6. Honesty rules — these are not negotiable

This project will be judged by people who work with AWS data professionally.
They will notice.

1. **Never report a metric you have not measured.** No placeholder accuracies,
   no "~99 %" in a README, no aspirational numbers in slides.
2. **Always report clean-stream false-positive rate** alongside recall. A QC
   system that cries wolf gets muted in week two, and once muted its recall is
   zero. This is the number most submissions never measure.
3. **Report event-level recall as well as point-level.** A six-hour drift caught
   on sample 200 of 300 is an operational success and a point-recall failure.
   Reporting only one is misleading.
4. **Ablate before claiming.** A five-layer architecture invites the question
   "does each layer earn its place?". `evaluation/network_benchmark.py` answers
   it by measurement.
5. **Name the limitations out loud.** See §8. A known, stated weakness reads as
   competence; a discovered, hidden one reads as the opposite.
6. **Do not tune on the test split.** Calibration uses its own seeds and time
   origin (`evaluation/calibrate.py`).

---

## 7. Current measured state

Run `make bench`, `make calibrate`, `make ablation` to reproduce. Measured with
numpy 2.4 / scikit-learn 1.8: the learned-layer scores shift by a few points
across sklearn minor versions, so pin the toolchain if you need these figures to
the third decimal. As of the last run:

- Clean-stream false-positive rate, single station: **0.022**
- Calibrated operating point: `make calibrate` selects `alert_threshold = 0.60`
  at **0.010** clean FPR on the validation split (1 % budget). `config.py` still
  ships the older default of 0.55; the two need reconciling.
- Point-level F1 on injected faults (network, all layers): **0.567**
- Event-level recall: **1.000**, but across the **seven** fault types the
  injector actually places -- drift episodes are never injected, see §8.6
- Latency p50 **~5 ms**, p99 **~8 ms**, about 300 observations/second
- Spatial ablation: L4 contributes **+0.015 F1**, +0.020 point recall (since
  §8.2's spatial rebaseline hold; it was +0.005 / +0.008 before)

The benchmark is reproducible run to run as of this revision; it was not before.
`apply_faults` consumed the RNG in `frozenset` iteration order, which depends on
the process hash seed, so point F1 varied by about +/-0.02 between runs on
nothing else. Clean-stream FPR was always stable.

**Interpretation, stated plainly:** event recall is excellent, point recall is
mediocre, precision needs work. The system reliably notices every fault when it
starts and then partially loses long ones -- less so since §8.2: the pipeline no
longer re-seeds a channel's baseline from the sensor while the buddy check says
the neighbours still disagree, which lifted network point recall from 0.555 to
0.575 at unchanged clean-stream FPR and gave L4 a measurable reason to exist
(+0.015 F1 in the ablation, up from +0.005). Its buddy threshold was also
instrumented (§8 item 1) and found not to be the bottleneck. That is the honest
characterisation and it is what should appear on the slide.

## 8. Known weaknesses — work here first

Ordered by expected value.

1. **L4 spatial -- the threshold was never the bottleneck (resolved via item 2).**
   The ablation showed **+0.005 F1 / +0.008 point recall**, not enough to justify
   a layer. The buddy threshold (`spatial_sigma`, 4.0) was the suspect. It was
   instrumented -- `make spatial-calibrate`, which records the clean-stream
   departure distribution and sweeps the threshold end to end the way
   `evaluation/calibrate.py` sweeps the alert threshold. Findings:
   - The clean-stream median peer departure tops out near **3.6 sigma** (p99.9
     is 2.4-3.0 per channel), so 4.0 is indeed past the entire clean
     distribution and L4 emits on ~0.04 % of clean samples.
   - But lowering it barely moves the benchmark. sigma 3.0 buys +0.007 F1 and
     +0.007 recall for +0.0006 clean FPR; sigma 3.5 is within noise; sigma 2.5
     only trades precision for recall and lifts clean FPR from 0.105 to 0.130.
     Left at 4.0 -- chasing a 0.007 F1 gain is not worth a config change, and
     `CLAUDE.md` §3.7 is explicit about lowering gates to "catch more".
   - Why the threshold cannot help: event recall is already 1.000, so L4 only
     has room to contribute on the *sustained middle* of a long fault, and by
     then item 2's `REBASELINE_AFTER` has re-seeded the station's own baseline
     to the faulted value. `tests/test_spatial.py` confirms L4 fires correctly
     on a clear sustained single-station bias (the PS example scenario); it is
     the aggregate benchmark that does not move.

   The lever was **item 2**, not this threshold. `spatial_sigma` stays at 4.0.
2. **Point-level recall on long episodes -- DONE.** The `REBASELINE_AFTER`
   escape hatch traded sustained-fault recall for stability: after six flagged
   samples the pipeline re-seeded the channel from the sensor, which during a
   real fault means adopting the faulted value and going quiet.
   `pipeline._update_state` now consults the buddy check: while L4 says the
   neighbours still disagree with a channel (a `BUDDY_DISSENT_TTL`-sample
   window, so it survives L4's intermittency and releases about an hour after
   the disagreement stops), the re-baseline is held off -- the channel keeps
   getting the repair and stays flagged -- capped at `REBASELINE_CEILING`
   (3 hours) so a stale pairwise offset cannot lock a channel out.
   Measured on the network ablation: point recall 0.555 -> 0.575, F1
   0.552 -> 0.567, **clean-stream FPR unchanged at 0.100** (the §3.7 gate),
   single-station benchmark untouched (L4 has no peers there).
3. **Root-cause accuracy ~0.44, still the weak layer.** Was ~0.32; a targeted
   fix took it to **0.437 single-station / 0.433 network** with zero change to
   F1, recall or clean-stream FPR: `l2_multivariate` no longer suggests
   `physical_violation` unless the departure is genuinely spread across
   channels. A single sensor carrying the whole Mahalanobis distance is a spike
   or a step, not a cross-sensor inconsistency, so it was structurally
   mislabelling every large single-channel fault. The classifier still confuses
   `spike`/`offset_step` and `drift`/`noise_burst`; compare against a small
   gradient-boosted classifier on the evidence vector and record the result in
   `docs/ARCHITECTURE.md` either way.
4. **Network clean FPR (0.100) is worse than single-station (0.022).** Diagnose
   before adding features; something in the interleaved path is costing accuracy.
5. **No real-data validation.** Everything is synthetic. Wire
   `synth/replay.py` to an actual IMD or NOAA ISD archive and confirm the clean-
   stream FPR holds. Synthetic-only results are the weakest part of the story.

6. **The benchmark never injects drift.** `FaultInjector.plan_episodes` places
   the short, numerous fault types (spikes) before the long ones, then fits each
   drift episode (288-1440 samples) into a random gap, giving up after 20
   collisions. Against a spike-fragmented timeline every drift episode is
   rejected -- measured, zero drift episodes in both `bench` and `ablation`. So
   the "event recall 1.000 across all fault types" claim currently covers seven,
   and items 1 and 2 -- both about *long*-fault recall -- are measured only on
   `offset_step` (72-432 samples), never on `drift`. This gates honest
   measurement of the two items above it. The fix is not a one-liner: the
   injector's eight rates are not tuned to any target (the stream is ~23 %
   anomalous, not the ~4 % the `InjectionPlan` docstring claims), and letting
   drift in at its current rate would push that past 50 %. Re-tuning the mix to
   a defended anomalous fraction is its own task.

## 9. Deliberate non-goals

Do not add these without discussion — they cost time and win no marks:

- A torch/LSTM autoencoder for L3. PCA reconstruction is a linear autoencoder
  with a closed-form optimum; on this window size it is within a few F1 points,
  fits in under a second, has no hyperparameters to defend, and its forward pass
  is a matrix multiply that fits an ESP32 budget. If you do try one, it goes
  behind the same `Detector` interface and the benchmark decides.
- Runtime SHAP. The model is not a black box — every detector emits its own
  statistic and an exact factored contribution. Attribution costs microseconds
  instead of hundreds of model evaluations per alert. Keep the offline
  `shap_bridge` for validation and to show agreement on the slide; ship the exact
  path.
- Extra sensor channels. The PS says three. Using more is out of scope.
- A microservice architecture. One process, one pipeline object, many stations.

## 10. Layout

```
src/skyguard/
  models.py            frozen dataclasses crossing module boundaries
  config.py            every threshold, each with its provenance
  physics.py           Magnus/thermodynamic relations (shared by generator + L0)
  pipeline.py          orchestration; owns state-update ordering
  features/online.py   ring buffers, forecast, residual scale
  detect/              l0..l4 + fusion
  explain/             exact attribution + narrative
  health/              health index, Theil-Sen drift, maintenance forecast
  impute/              damped repair
  synth/               climate generator, fault injector, correlated network
  evaluation/          benchmark, calibration, spatial ablation
  api/                 FastAPI + WebSocket + dashboard
docs/                  ARCHITECTURE.md, DEMO_SCRIPT.md, USE_CASES.md
tests/
```

## 11. Before every commit

```bash
ruff check src tests
pytest -q
python -m skyguard.cli benchmark --quick
```

If the clean-stream false-positive rate has risen, the change is not ready,
whatever it did to recall.
