# Architecture

## Layering principle

Five detectors ordered by **how easily each can be fooled by genuine weather**.
That ordering is the design rationale and it determines the fusion weights.

```
observation
   ├─ record as-reported values            (state.reported)
   ├─ L0..L4 all see the SAME history      (no detector mutates state)
   ├─ fuse evidence -> per-channel confidence (weighted noisy-OR)
   ├─ classify root cause, assign severity
   ├─ build exact attributions, compute damped repair
   └─ THEN update state, health, buddy statistics
```

Detectors must not mutate state mid-pass. A detector that pushed its own value
would give the next detector a different view of the past, making results depend
on registration order.

## Why noisy-OR

`confidence = 1 - prod_i (1 - w_i * s_i)`

The layers are complements, not competitors — each looks for a different fault
signature. Two agreeing should raise confidence; one staying silent should not
veto another. An averaging rule would let three silent layers bury a certain
physics violation.

The weights express how much each layer can be fooled by real weather. L0 has
weight 1.00 and saturates confidence alone, which is correct: it only fires on
constraints weather cannot violate.

Consequence, learned the hard way: because pooling is multiplicative, detectors
that chatter at low scores manufacture alerts out of nothing. Every layer
therefore has an emission gate at roughly 0.75× its decision threshold.

## Decisions and the alternatives rejected

### Baseline: local linear forecast, not rolling median

A trailing median lags the diurnal cycle. Measured: **90 % false-positive rate**
on clean data, because every sunrise ramp put the current reading multiple sigma
above a 4-hour median. The forecast absorbs the ramp into its slope.

### Scale floor: sensor resolution, not an epsilon

Pressure is quantised to 0.1 hPa and moves slowly, so consecutive reports are
frequently identical, the residual MAD is exactly zero, and every score divides
by ~0. The floor is `3 × step / √12` — the standard deviation of a uniformly
quantised variable. A physical fact about the instrument.

### L3: PCA subspace, not an LSTM autoencoder

PCA reconstruction is a linear autoencoder with a closed-form optimum. At an
18-sample window it is within a few F1 points of a trained LSTM-AE, fits in under
a second, has no hyperparameters to defend to a judge, and its forward pass is a
matrix multiply inside an ESP32 budget. The `Detector` protocol makes swapping in
a torch model a one-line registration change, and the benchmark decides.

**Open:** the LSTM comparison has not actually been run. Do not claim a margin
until it has.

### Explainability: exact attribution, not runtime SHAP

Every detector already emits its own statistic, threshold and factored
contribution, so attribution is exact rather than sampled and costs microseconds
instead of hundreds of model evaluations per alert. The offline `shap_bridge`
exists to validate agreement for the judging slide.

### Root cause: scored rules, not a learned classifier

Eight classes with a labelled generator would support a trained model, but rules
are auditable, need no training data at deploy time, and a judge can read the
reasoning.

**Open:** current accuracy is ~0.32; spike/bias-step and drift/noise-burst are
confused. A gradient-boosted classifier on the evidence vector should be tried
and the result recorded here either way.

### L2: cascade the Isolation Forest behind Mahalanobis

sklearn's per-row `score_samples` costs milliseconds and alone would blow the
real-time budget. It only changes the verdict when the residual is already
unusual, so the cheap chi-square test gates it. Mean latency dropped ~15× with
no change to F1.

### L4 buddy threshold: instrumented, and left where it was

`spatial_sigma` (the median standardised peer departure above which a
disagreement is blamed on the station) was set to 4.0 by analogy with the
Hampel threshold, never measured. `evaluation/spatial_calibrate.py` (`make
spatial-calibrate`) now measures it: it records the clean-stream distribution of
the statistic and sweeps the threshold end to end on the four-station network,
the same way `calibrate.py` sweeps the alert threshold.

The clean-stream statistic tops out near **3.6 sigma** (p99.9 is 2.4–3.0 per
channel), so 4.0 sits past the entire clean distribution — the threshold really
is strict. But lowering it barely helps: sigma 3.0 buys **+0.007 F1 / +0.007
point recall** for +0.0006 clean FPR, sigma 3.5 is within noise, sigma 2.5 only
trades precision for recall and lifts clean FPR from 0.105 to 0.130. Kept at
4.0.

The threshold cannot be the lever because event recall is already 1.000: L4 only
has room to add value on the sustained middle of a long fault, and by then the
re-baseline escape below had adopted the faulted value. Fixing *that* (see next
section) took L4's ablation contribution from +0.005 F1 to **+0.015 F1 / +0.020
point recall**.

### Repair: damped trend, with a re-baseline escape — now buddy-aware

Undamped extrapolation compounds — pressure walked 985 → 877 hPa inside one
episode. Geometric damping (φ = 0.8) caps the total trend contribution at four
slopes. Separately, a channel that is flagged continuously can never recover if
only repairs enter its buffer, so after six consecutive substitutions the
pipeline re-seeds from the sensor.

Re-seeding blindly is a trade: it costs sustained-fault point recall to buy
stability, because during a real fault the value it adopts is the faulted one.
So `_update_state` now consults the buddy check. While L4 says the neighbours
still disagree with a channel — a `BUDDY_DISSENT_TTL`-sample window, so the
signal survives L4's intermittency and releases about an hour after the
disagreement stops — the re-baseline is held off and the channel keeps getting
the repair, capped at `REBASELINE_CEILING` (3 hours) so a stale pairwise offset
at a neighbour cannot lock a channel out.

Measured: network point recall 0.555 → 0.575, F1 0.552 → 0.567, **clean-stream
FPR unchanged at 0.100**, single-station benchmark untouched (no peers, so the
buddy check never speaks). It cost ~0.015 of root-cause accuracy — more
sustained-fault samples are now flagged and the rule classifier is weak on those
(the open root-cause item).

## Scalability

The learned components — subspace basis, Isolation Forest, physical limits —
model *sensor behaviour* and are shared across the whole network. Only the
fast-moving statistics are per-station, and those are a handful of bounded ring
buffers: a few kilobytes each.

Adding a station costs no retraining and about half a day of buffer fill. The
buddy board holds three floats and a timestamp per station plus pairwise
difference statistics, so a 500-station network is a few hundred kilobytes.

## Test strategy

`tests/` covers the bugs that actually happened, not a coverage target:

- `test_physics.py` — the round-trip and saturation invariants L0 depends on
- `test_features.py` — the lagging-baseline and zero-scale regressions
- `test_repair.py` — that a 300-sample outage stays bounded
- `test_detection.py` — clean-stream false-positive budget, frozen sensor,
  sentinel-as-dropout, JSON has no NaN, every alert carries an action

The clean-stream false-positive assertion is the important one. If it regresses,
the change is not ready, whatever it did to recall.
