# Demo script — internal round

Six minutes. Rehearse until the timings are automatic; a fumbled live demo
costs more than a missing feature.

## Before you start

```bash
make serve
```

Open `http://127.0.0.1:8000` on the projector. It primes itself, so the charts
have history immediately and the baseline is quiet. Have a second terminal ready
with `make calibrate` and `make ablation`.

**Check the venue machine has no network dependency.** The console vendors
nothing from a CDN and the charts are hand-rolled canvas precisely so this works
on conference wifi. Verify anyway.

## 0:00 — The problem, in one sentence

> "An automatic weather station reports temperature, pressure and humidity every
> ten minutes. Some of those readings are wrong, and wrong readings go straight
> into forecasts and cyclone warnings. Spotting a wrong number is easy. Spotting
> it without also flagging a real heatwave is the actual problem — and you get
> three channels to do it with."

Do not open with architecture. Open with the constraint.

## 0:45 — The problem statement's own example

Select **Ambala** in the Network panel. Fault type **Impossible humidity
(108 %)**, press **Inject fault**.

> "This is the example in the problem statement. A humidity reading that high
> means the air holds more water than it physically can at this temperature —
> the pair implies a dewpoint above the air temperature, which is
> thermodynamically impossible. One of those two sensors is lying."

Point at the **Why it fired** panel and the **Detection layers** strip.

> "It doesn't just say anomaly. It says which constraint broke, by how much,
> which of the five layers found it, how much each contributed, and what to send
> the technician to do. And the station's health bar is already falling."

Press **Clear**, then move on.

## 1:45 — The fault nobody else catches

Channel **Pressure**, fault **Frozen sensor**, **Inject fault**. Wait for the
trace to flatten and the alert to appear.

> "A latched sensor is the one that does real damage. It passes every range
> check and every step check forever, while feeding a constant into
> assimilation. Threshold-based quality control never sees it. We test for it
> explicitly — and against what the logger actually reported, not against our
> own corrected estimate, because our correction would un-freeze the trace and
> hide the fault."

## 2:45 — Detection is only half of it

Clear, then channel **Temperature**, fault **Calibration drift**, **Inject
fault**. Point at the dashed line as it separates from the solid trace.

> "Solid line is what the sensor reported. Dashed is our corrected estimate.
> They're drawn differently on purpose — an estimate should never be mistaken
> for a measurement. And in the Network panel the station's health has dropped
> and it's been quarantined from assimilation."

## 3:45 — The numbers

Clear the console. Switch to the terminal, run `make calibrate`.

> "Threshold picked by sweeping a validation split, not by eye. And the rule
> isn't maximise F1 — it's maximise F1 subject to false alarms under one
> percent. A quality-control system that cries wolf gets switched off, and once
> it's off its recall is zero."

Then be straight about the rest (`make ablation`):

> "Event recall is one point zero — every injected fault is caught at onset.
> Point recall is about 0.58 on the network: we used to lose long faults once
> our own baseline adapted to them, and the fix was to stop re-seeding a
> channel's baseline while the neighbouring stations still disagree with it.
> That took point recall from 0.55 to 0.58 and gave the spatial layer a
> measurable reason to exist — at unchanged clean-stream false-positive rate.
> Precision still needs work. That's the honest characterisation."

Judges from IMD will trust a stated weakness far more than a suspiciously round
number.

## 4:45 — Scale and deployability

> "A few milliseconds an observation, single-threaded Python — the console
> shows the live latency in the Pipeline panel. India's AWS network is a few
> thousand stations reporting every ten minutes, so that's a few observations a
> second: the whole national network on one core with room to spare. Layer zero
> needs no training and no floating-point library beyond an exponential, so it
> runs unchanged on the ESP32 at the station itself."

## 5:30 — Close

> "Can AI build a self-aware, self-healing observation network? It can tell you
> which sensor is lying, why, how confident it is, what to send the technician
> to do, and what the reading should have been. That's the honest answer today."

## Questions to expect

**"Why not deep learning?"** PCA reconstruction is a linear autoencoder with a
closed-form optimum. At an 18-sample window it is within a few F1 points of a
trained LSTM autoencoder, fits in under a second, has no hyperparameters to
defend, and its forward pass is a matrix multiply that fits an ESP32 budget.
The interface is the same, so if a benchmark ever says the LSTM earns its cost,
it drops straight in.

**"You asked for SHAP."** SHAP approximates which input moved a black box. This
model is not a black box — every detector emits its own statistic and an exact
factored contribution, so attribution is exact and costs microseconds instead of
hundreds of model evaluations per alert. There is an offline SHAP bridge to
demonstrate the two agree.

**"Is this synthetic data?"** Yes, and that is the biggest limitation. The
generator derives humidity from temperature and dewpoint so it cannot be
thermodynamically inconsistent by construction, and the network shares a
synoptic driver with distance-dependent correlation. But real-archive validation
is the top item on the roadmap, not a claim we are making today.

**"What happens when a station is genuinely in extreme weather?"** That is the
whole design. Layer four checks the neighbours: a real front moves every station
in the region, a broken thermometer moves one.
