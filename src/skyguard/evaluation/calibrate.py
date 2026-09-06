"""Operating-point calibration.

`DetectorConfig.alert_threshold` decides whether a fused confidence becomes an
alert. Picking it by eye is the difference between a system an operator trusts
and one whose alerts get muted in week two, so it is chosen here by sweeping a
held-out validation split and reporting the whole trade-off curve.

The selection rule is deliberately not "maximise F1". A meteorological QC
system that cries wolf gets switched off, and once it is off its recall is zero
regardless of what the benchmark said. So the rule is:

    maximise F1  subject to  clean-stream false-positive rate <= budget

with the budget defaulting to 1 %, i.e. about one false alarm per station per
week at a 10-minute cadence. If no threshold meets the budget, the sweep says
so rather than silently returning the best F1 -- a failure to meet the budget
is information, not something to paper over.

The validation split uses different seeds and a different time origin from both
the training and the test splits, so the chosen threshold is not tuned on the
data it is later scored against.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..models import Observation
from ..pipeline import SkyGuardPipeline
from ..synth.climate import ClimateGenerator, StationProfile
from ..synth.faults import InjectionPlan, build_labelled_stream


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    clean_false_positive_rate: float

    @property
    def within(self) -> bool:
        return True


@dataclass
class CalibrationResult:
    curve: list[OperatingPoint]
    chosen: OperatingPoint | None
    budget: float

    def render(self) -> str:
        lines = [
            f"Threshold sweep (false-positive budget {self.budget:.1%})",
            "-" * 62,
            f"{'thresh':>8}{'precision':>12}{'recall':>10}{'F1':>10}{'clean FPR':>12}{'':>6}",
        ]
        for point in self.curve:
            mark = "  <-" if self.chosen and point.threshold == self.chosen.threshold else ""
            flag = " " if point.clean_false_positive_rate <= self.budget else "!"
            lines.append(
                f"{point.threshold:>8.2f}{point.precision:>12.3f}{point.recall:>10.3f}"
                f"{point.f1:>10.3f}{point.clean_false_positive_rate:>12.4f}{flag}{mark}"
            )
        if self.chosen is None:
            lines.append("")
            lines.append(
                "No threshold met the false-positive budget. Detector sensitivity "
                "needs work; do not ship by relaxing the budget."
            )
        else:
            lines.append("")
            lines.append(f"Selected alert_threshold = {self.chosen.threshold:.2f}")
        return "\n".join(lines)


def calibrate(
    profile: StationProfile,
    settings: Settings | None = None,
    train_samples: int = 5_000,
    validation_samples: int = 8_000,
    plan: InjectionPlan | None = None,
    seed: int = 31,
    warmup: int = 300,
    budget: float = 0.01,
    grid: tuple[float, ...] = (
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.93, 0.96, 0.98,
    ),
) -> CalibrationResult:
    """Sweep the alert threshold on a validation split and pick an operating point."""
    settings = settings or Settings()
    origin = 1_650_000_000.0

    train = ClimateGenerator(profile, origin, seed=seed).generate(train_samples)
    pipeline = SkyGuardPipeline(settings)
    pipeline.fit(train)

    # Faulted validation stream.
    val_gen = ClimateGenerator(profile, origin + train_samples * 600 + 86_400, seed=seed + 1)
    observations, labels, _episodes = build_labelled_stream(
        val_gen, validation_samples, plan=plan, seed=seed + 2
    )
    faulted_conf = _confidences(settings, pipeline, observations, "VAL")
    truth = [labels[i].is_anomalous for i in range(len(labels))]

    # Clean control stream, different origin again.
    clean_gen = ClimateGenerator(profile, origin + 9_000_000, seed=seed + 3)
    clean_obs = [
        Observation("VAL-CLEAN", o.timestamp, o.temperature, o.pressure, o.humidity)
        for o in clean_gen.generate(validation_samples // 2)
    ]
    clean_conf = _confidences(settings, pipeline, clean_obs, "VAL-CLEAN")

    curve: list[OperatingPoint] = []
    for threshold in grid:
        tp = fp = fn = 0
        for i in range(warmup, len(faulted_conf)):
            predicted = faulted_conf[i] >= threshold
            actual = truth[i]
            if predicted and actual:
                tp += 1
            elif predicted:
                fp += 1
            elif actual:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        tail = clean_conf[warmup:]
        clean_fpr = sum(c >= threshold for c in tail) / len(tail) if tail else 0.0

        curve.append(OperatingPoint(threshold, precision, recall, f1, clean_fpr))

    eligible = [p for p in curve if p.clean_false_positive_rate <= budget]
    chosen = max(eligible, key=lambda p: p.f1) if eligible else None
    return CalibrationResult(curve=curve, chosen=chosen, budget=budget)


def _confidences(
    settings: Settings,
    fitted: SkyGuardPipeline,
    observations: list[Observation],
    station_id: str,
) -> list[float]:
    """Run a stream through a fresh pipeline and return the fused confidence trace.

    A fresh pipeline per stream, reusing only the fitted layers: rolling buffers
    must start empty exactly as they would at a newly commissioned station.

    Note that the *verdict* still uses the configured threshold internally; only
    the confidence trace is used for the sweep, so one pass covers every
    candidate threshold instead of re-running the pipeline fourteen times.
    """
    pipeline = SkyGuardPipeline(settings)
    pipeline.l2 = fitted.l2
    pipeline.l3 = fitted.l3
    pipeline._fitted = True
    return [pipeline.process(o).confidence for o in observations]
