"""Benchmark harness -- the numbers that go on the judging slide.

Reports three families of metric because each answers a different question a
judge will ask:

Point-level        Of all anomalous *samples*, how many did we flag?
Event-level        Of all injected *episodes*, how many did we catch at all,
                   and how fast? A 6-hour drift caught on sample 200 of 300 is
                   a success operationally and a failure by point-level recall,
                   so reporting only one of the two is misleading.
Clean-stream FPR   How often do we cry wolf on data with no faults injected?
                   The single most important number for operational trust, and
                   the one most hackathon submissions never measure.

Also reported: per-fault-type recall (which faults are we bad at?), root-cause
classification accuracy, and latency percentiles.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from ..models import GroundTruth, Observation, Verdict
from ..pipeline import SkyGuardPipeline
from ..synth.climate import ClimateGenerator, StationProfile
from ..synth.faults import FaultEpisode, InjectionPlan, build_labelled_stream


@dataclass
class PointMetrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    @property
    def precision(self) -> float:
        d = self.true_positive + self.false_positive
        return self.true_positive / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positive + self.false_negative
        return self.true_positive / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        d = self.false_positive + self.true_negative
        return self.false_positive / d if d else 0.0


@dataclass
class EventMetrics:
    total: int = 0
    detected: int = 0
    delays: list[int] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.detected / self.total if self.total else 0.0

    @property
    def median_delay_samples(self) -> float:
        return statistics.median(self.delays) if self.delays else float("nan")


@dataclass
class BenchmarkReport:
    point: PointMetrics
    event: EventMetrics
    per_fault_recall: dict[str, float]
    root_cause_accuracy: float
    clean_false_positive_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_per_second: float
    samples: int
    episodes: int

    def render(self) -> str:
        """Fixed-width summary, suitable for pasting into the report."""
        lines = [
            "SkyGuard AI -- detection benchmark",
            "=" * 58,
            f"samples evaluated            {self.samples:>10,}",
            f"fault episodes injected      {self.episodes:>10,}",
            "",
            "Point level (per sample)",
            f"  precision                  {self.point.precision:>10.3f}",
            f"  recall                     {self.point.recall:>10.3f}",
            f"  F1                         {self.point.f1:>10.3f}",
            f"  false-positive rate        {self.point.false_positive_rate:>10.4f}",
            "",
            "Event level (per episode)",
            f"  episodes detected          {self.event.detected:>10,} / {self.event.total:,}",
            f"  event recall               {self.event.recall:>10.3f}",
            f"  median detection delay     {self.event.median_delay_samples:>10.1f} samples",
            "",
            f"root-cause accuracy          {self.root_cause_accuracy:>10.3f}",
            f"clean-stream FP rate         {self.clean_false_positive_rate:>10.4f}",
            "",
            "Latency",
            f"  p50                        {self.latency_p50_ms:>10.3f} ms",
            f"  p95                        {self.latency_p95_ms:>10.3f} ms",
            f"  p99                        {self.latency_p99_ms:>10.3f} ms",
            f"  throughput                 {self.throughput_per_second:>10,.0f} obs/s",
            "",
            "Recall by fault type",
        ]
        for name, value in sorted(self.per_fault_recall.items()):
            lines.append(f"  {name:<26} {value:>10.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "point": {
                **asdict(self.point),
                "precision": self.point.precision,
                "recall": self.point.recall,
                "f1": self.point.f1,
                "false_positive_rate": self.point.false_positive_rate,
            },
            "event": {
                "total": self.event.total,
                "detected": self.event.detected,
                "recall": self.event.recall,
                "median_delay_samples": self.event.median_delay_samples,
            },
            "per_fault_recall": self.per_fault_recall,
            "root_cause_accuracy": self.root_cause_accuracy,
            "clean_false_positive_rate": self.clean_false_positive_rate,
            "latency_ms": {
                "p50": self.latency_p50_ms,
                "p95": self.latency_p95_ms,
                "p99": self.latency_p99_ms,
            },
            "throughput_per_second": self.throughput_per_second,
            "samples": self.samples,
            "episodes": self.episodes,
        }


def run_benchmark(
    profile: StationProfile,
    train_samples: int = 6_000,
    test_samples: int = 12_000,
    plan: InjectionPlan | None = None,
    seed: int = 7,
    warmup: int = 300,
) -> BenchmarkReport:
    """Train on clean data, evaluate on injected faults plus a clean control.

    The train and test streams use different seeds and different time origins,
    so the subspace cannot have memorised the exact weather it is scored on.
    """
    start = 1_700_000_000.0

    train_gen = ClimateGenerator(profile, start, seed=seed)
    train = train_gen.generate(train_samples)

    pipeline = SkyGuardPipeline()
    pipeline.fit(train)

    test_gen = ClimateGenerator(
        profile, start + train_samples * 600 + 86_400, seed=seed + 101
    )
    observations, labels, episodes = build_labelled_stream(
        test_gen, test_samples, plan=plan, seed=seed + 202
    )

    verdicts = [pipeline.process(o) for o in observations]

    point, event, per_fault, root_acc = _score(
        verdicts, labels, episodes, warmup=warmup
    )
    clean_fpr = _clean_control(profile, pipeline, start, test_samples // 3, seed, warmup)

    latencies = sorted(v.latency_ms for v in verdicts[warmup:])
    perf = pipeline.throughput_summary()

    return BenchmarkReport(
        point=point,
        event=event,
        per_fault_recall=per_fault,
        root_cause_accuracy=root_acc,
        clean_false_positive_rate=clean_fpr,
        latency_p50_ms=_pct(latencies, 50),
        latency_p95_ms=_pct(latencies, 95),
        latency_p99_ms=_pct(latencies, 99),
        throughput_per_second=perf["throughput_per_second"],
        samples=len(observations) - warmup,
        episodes=len(episodes),
    )


def _score(
    verdicts: list[Verdict],
    labels: list[GroundTruth],
    episodes: list[FaultEpisode],
    warmup: int,
) -> tuple[PointMetrics, EventMetrics, dict[str, float], float]:
    point = PointMetrics()
    correct_cause = 0
    cause_total = 0

    for i in range(warmup, len(verdicts)):
        predicted = verdicts[i].is_anomalous
        actual = labels[i].is_anomalous
        if predicted and actual:
            point.true_positive += 1
            cause_total += 1
            if verdicts[i].fault_type is labels[i].fault_type:
                correct_cause += 1
        elif predicted and not actual:
            point.false_positive += 1
        elif not predicted and actual:
            point.false_negative += 1
        else:
            point.true_negative += 1

    event = EventMetrics()
    per_fault_hits: dict[str, list[int]] = {}
    for ep in episodes:
        if ep.end_index <= warmup:
            continue
        event.total += 1
        bucket = per_fault_hits.setdefault(ep.fault_type.value, [0, 0])
        bucket[1] += 1

        lo = max(ep.start_index, warmup)
        hit_at = next(
            (j for j in range(lo, min(ep.end_index, len(verdicts)))
             if verdicts[j].is_anomalous),
            None,
        )
        if hit_at is not None:
            event.detected += 1
            event.delays.append(hit_at - ep.start_index)
            bucket[0] += 1

    per_fault = {
        name: (hits / total if total else 0.0)
        for name, (hits, total) in per_fault_hits.items()
    }
    root_acc = correct_cause / cause_total if cause_total else 0.0
    return point, event, per_fault, root_acc


def _clean_control(
    profile: StationProfile,
    fitted_pipeline: SkyGuardPipeline,
    start: float,
    n: int,
    seed: int,
    warmup: int,
) -> float:
    """False-positive rate on a stream with no injected faults.

    Uses a fresh pipeline seeded from the same fitted layers, so per-station
    buffers start empty exactly as they would at a newly commissioned site.
    """
    control = SkyGuardPipeline(fitted_pipeline.settings)
    control.l2 = fitted_pipeline.l2
    control.l3 = fitted_pipeline.l3
    control._fitted = True

    gen = ClimateGenerator(profile, start + 5_000_000, seed=seed + 999)
    observations = gen.generate(n)
    # Distinct station id so the control gets its own fresh rolling buffers.
    observations = [
        Observation("CONTROL", o.timestamp, o.temperature, o.pressure, o.humidity)
        for o in observations
    ]
    flags = [control.process(o).is_anomalous for o in observations]
    tail = flags[warmup:]
    return sum(tail) / len(tail) if tail else 0.0


def _pct(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return float("nan")
    k = max(0, min(len(sorted_values) - 1, round(pct / 100 * (len(sorted_values) - 1))))
    return round(sorted_values[k], 4)
