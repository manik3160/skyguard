# Use cases

The problem statement asks for "a document explaining various use cases". These
are the operational situations the system is built for, with the mechanism that
handles each.

## 1. Pre-assimilation gate for numerical weather prediction

**Situation.** IMD's forecast models ingest AWS observations. A bad observation
propagates into the analysis and degrades the forecast over a wide area.

**Mechanism.** Every observation carries a confidence and a severity before it
reaches assimilation. Stations whose health index falls below
`HealthConfig.quarantine_index` (40) are excluded automatically. Because the
verdict includes a corrected estimate, a station with one bad channel can still
contribute its other two rather than being dropped entirely.

**Why it matters.** Current threshold-based quality control accepts frozen
sensors indefinitely — they violate no range and no step limit. That is the
single most damaging failure mode for assimilation, and layer one tests for it
explicitly.

## 2. Predictive maintenance scheduling

**Situation.** A technician visit to a remote AWS costs a day. Sending one on a
fixed calendar wastes visits; waiting for failure wastes data.

**Mechanism.** `SensorHealth.estimate_drift` fits a Theil-Sen slope over
multi-day history — median of pairwise slopes, so it is not dragged by the
spikes it has to tolerate. `days_to_maintenance` projects when drift will consume
the WMO calibration budget (0.5 °C, 0.3 hPa, 3 %). The dashboard surfaces this
as a countdown per channel.

**Why it matters.** It converts a maintenance calendar into a queue ordered by
when each station will actually go out of tolerance.

## 3. Duty-forecaster triage

**Situation.** A forecaster sees an implausible reading at 03:00 and has ninety
seconds to decide whether to trust it.

**Mechanism.** Each alert carries a plain-language narrative naming the
constraint that broke and by how much, an exact per-layer contribution
breakdown, a root-cause class, and a specific maintenance action —
"power-cycle the logger; inspect the sensor for icing", not "anomaly detected".

**Why it matters.** Explainability is 10 % of the marking scheme, but the real
argument is adoption: an alert a forecaster cannot act on is an alert they learn
to ignore.

## 4. Distinguishing extreme weather from sensor failure

**Situation.** A station reports 48 °C during a heatwave. Is it a heatwave or a
broken thermometer?

**Mechanism.** Layer four runs a WMO buddy check: the running difference between
this station and each neighbour, standardised against that pair's own history,
with pressure reduced to mean sea level so elevation does not confound it. Real
weather moves the whole region; a broken sensor moves one station. A departure
is only reported when every peer agrees on its sign.

**Why it matters.** This is the failure mode that makes operators distrust
automated quality control — a system that suppresses genuine extremes during the
exact events that matter most.

## 5. Edge deployment at the station

**Situation.** A station on a poor telemetry link should not spend bandwidth
uploading readings that are obviously wrong.

**Mechanism.** Layer zero needs no training, no history beyond one sample, and no
floating-point beyond an exponential. Range checks, slew-rate limits and the
dewpoint constraint run in constant time and constant memory on an ESP32. The
station can flag locally and mark its own uplink.

**Why it matters.** Energy efficiency is 5 % of the marking scheme, and layer
zero catches dropouts, power flickers and physical violations — a large share of
real faults — before they consume any radio time.

## 6. Network commissioning and audit

**Situation.** A new station comes online, or an old one is suspected of a
long-standing bias nobody has quantified.

**Mechanism.** Per-station state is independent, so a new station starts
producing verdicts after about half a day of buffer fill with no retraining —
the learned layers model sensor physics, not local climate, and are shared
network-wide. Buddy statistics accumulate the pair's normal offset, which *is*
the audit: a persistent offset against every neighbour is a quantified bias.

**Why it matters.** This is the scalability claim, and the network benchmark is
what tests it rather than asserting it.
