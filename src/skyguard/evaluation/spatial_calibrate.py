"""Calibration of the L4 spatial buddy-check threshold.

`DetectorConfig.spatial_sigma` is the median standardised pairwise departure
above which the buddy check attributes a disagreement to one station rather than
to weather. It was set to 4.0 "to match the single-station Hampel threshold" --
by analogy, never against data -- and the network ablation shows L4 buying only
about +0.005 F1 at that setting (CLAUDE.md 7, 8 item 1).

This module does for `spatial_sigma` what `calibrate.py` does for
`alert_threshold`:

1. Instruments the buddy check on a clean four-station network and records the
   full distribution of the departure statistic it actually produces -- every
   channel that clears the peer and readiness filters, including the ones the
   emission gate later silences. 4.0 sigma only makes sense as a threshold if
   the clean departures genuinely sit well inside it.

2. Sweeps a grid of candidate thresholds end to end (a threshold change also
   moves the emission gate at 0.75x and the logistic score centre, so only a
   full-pipeline run is honest) and reports network point precision / recall /
   F1, event recall, clean-stream false-positive rate and root-cause accuracy at
   each -- the same columns as `network_benchmark`.

3. Selects by the same rule as the alert-threshold calibration: maximise F1
   subject to the network clean-stream FPR not exceeding what the current 4.0
   setting produces. The recall-maximising choice under the same constraint is
   reported alongside, because L4 exists to recover long-fault recall and the
   two may disagree.

Faults are injected into one target station only; the other three stay clean and
act as the buddy reference, which is the situation the problem statement's
example describes. Seeds and time origin are distinct from `benchmark.py`,
`network_benchmark.py` and `calibrate.py`, so the threshold is not tuned on data
it is later scored against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..config import DEFAULT, Settings
from ..models import CHANNELS, Channel, Observation
from ..pipeline import SkyGuardPipeline
from ..synth.climate import StationProfile
from ..synth.faults import FaultInjector, InjectionPlan, apply_faults
from ..synth.network import NetworkGenerator
from .benchmark import EventMetrics, PointMetrics
from .network_benchmark import _interleave

#: Grid of candidate `spatial_sigma` values. Brackets the current 4.0 and probes
#: below it, since the hypothesis (CLAUDE.md 8 item 1) is that 4.0 is too strict
#: for the pairwise-difference spread at inter-station correlation ~0.5. Each
#: point is a full end-to-end network run, so the grid is kept small.
DEFAULT_GRID: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 4.5)

#: Quantiles reported for the clean-stream departure distribution.
_QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 1.0)


@dataclass(frozen=True, slots=True)
class ChannelDistribution:
    channel: Channel
    count: int
    quantiles: dict[float, float]        # of abs(median departure) -- the statistic
    agree_fraction: float                # share where every peer shared a sign


@dataclass(frozen=True, slots=True)
class SweepRow:
    sigma: float
    precision: float
    recall: float
    f1: float
    event_recall: float
    clean_false_positive_rate: float
    root_cause_accuracy: float
    emits_fraction: float                # clean-stream share where L4 would emit


@dataclass
class SpatialCalibrationResult:
    distribution: list[ChannelDistribution]
    curve: list[SweepRow]
    baseline_sigma: float
    baseline_clean_fpr: float
    chosen_f1: SweepRow | None
    chosen_recall: SweepRow | None
    samples: int
    episodes: int
    stations: int

    def render(self) -> str:
        lines = [
            "SkyGuard AI -- L4 spatial-threshold calibration",
            "=" * 72,
            f"stations {self.stations}    samples/station {self.samples:,}    "
            f"episodes injected {self.episodes}",
            "",
            "Clean-stream departure distribution (statistic = |median peer departure|)",
            "-" * 72,
            f"{'channel':<14}{'n':>8}"
            + "".join(f"{f'p{q:g}':>10}" for q in _QUANTILES)
            + f"{'sign-agree':>12}",
        ]
        for d in self.distribution:
            lines.append(
                f"{d.channel.value:<14}{d.count:>8,}"
                + "".join(f"{d.quantiles[q]:>10.2f}" for q in _QUANTILES)
                + f"{d.agree_fraction:>12.3f}"
            )
        lines += [
            "",
            f"Threshold sweep (network clean-FPR budget {self.baseline_clean_fpr:.4f}, "
            f"the current sigma={self.baseline_sigma:g} rate)",
            "-" * 72,
            f"{'sigma':>7}{'prec':>9}{'recall':>9}{'F1':>9}{'evt rec':>10}"
            f"{'cleanFPR':>11}{'cause':>8}{'L4 emits':>10}{'':>5}",
        ]
        for row in self.curve:
            marks = []
            if self.chosen_f1 and row.sigma == self.chosen_f1.sigma:
                marks.append("F1")
            if self.chosen_recall and row.sigma == self.chosen_recall.sigma:
                marks.append("R")
            if row.sigma == self.baseline_sigma:
                marks.append("now")
            flag = " " if row.clean_false_positive_rate <= self.baseline_clean_fpr else "!"
            lines.append(
                f"{row.sigma:>7.2f}{row.precision:>9.3f}{row.recall:>9.3f}{row.f1:>9.3f}"
                f"{row.event_recall:>10.3f}{row.clean_false_positive_rate:>11.4f}"
                f"{row.root_cause_accuracy:>8.3f}{row.emits_fraction:>10.4f}"
                f"{flag}{('  <- ' + '/'.join(marks)) if marks else ''}"
            )
        lines.append("")
        if self.chosen_f1 is None:
            lines.append(
                "No candidate held the clean-FPR budget. Leave spatial_sigma at "
                f"{self.baseline_sigma:g}; do not relax the budget to move it."
            )
        else:
            f1 = self.chosen_f1
            lines.append(
                f"Max-F1 within budget:     spatial_sigma = {f1.sigma:g}  "
                f"(F1 {f1.f1:.3f}, recall {f1.recall:.3f}, clean FPR "
                f"{f1.clean_false_positive_rate:.4f})"
            )
            r = self.chosen_recall
            lines.append(
                f"Max-recall within budget: spatial_sigma = {r.sigma:g}  "
                f"(F1 {r.f1:.3f}, recall {r.recall:.3f}, clean FPR "
                f"{r.clean_false_positive_rate:.4f})"
            )
            base = next(x for x in self.curve if x.sigma == self.baseline_sigma)
            lines.append(
                f"Current sigma={self.baseline_sigma:g}:        "
                f"F1 {base.f1:.3f}, recall {base.recall:.3f}, clean FPR "
                f"{base.clean_false_positive_rate:.4f}"
            )
        return "\n".join(lines)


def _settings_for(sigma: float, base: Settings) -> Settings:
    return replace(base, detector=replace(base.detector, spatial_sigma=sigma))


def _build(
    profiles: tuple[StationProfile, ...],
    train: dict[str, list[Observation]],
    settings: Settings,
) -> SkyGuardPipeline:
    """A pipeline fitted on the target's clean history, with empty rolling
    buffers and an empty buddy board -- the state a newly commissioned network
    has, exactly as `network_benchmark._build` sets it up.

    Each sweep point gets its own fit rather than a shared one: L2's covariance
    EWMA and L3's error scale are *online* state that `process()` mutates, so a
    shared instance would carry one candidate's drift into the next.
    """
    pipeline = SkyGuardPipeline(settings)
    for profile in profiles:
        pipeline.register_station(profile.station_id, profile.altitude_m)
    pipeline.fit(train[profiles[0].station_id])
    return pipeline


def run_spatial_calibration(
    profiles: tuple[StationProfile, ...],
    train_samples: int = 3_000,
    test_samples: int = 6_000,
    plan: InjectionPlan | None = None,
    seed: int = 47,
    warmup: int = 400,
    grid: tuple[float, ...] = DEFAULT_GRID,
) -> SpatialCalibrationResult:
    origin = 1_620_000_000.0
    target = profiles[0].station_id
    order = [p.station_id for p in profiles]
    base_settings = Settings()

    train = NetworkGenerator(profiles, origin, seed=seed).generate(train_samples)
    clean = NetworkGenerator(
        profiles, origin + train_samples * 600 + 86_400, seed=seed + 5
    ).generate(test_samples)

    injector = FaultInjector(plan, seed=seed + 9)
    episodes = injector.plan_episodes(test_samples)
    faulted_target, labels = apply_faults(clean[target], episodes, seed=seed + 9)
    faulted = dict(clean)
    faulted[target] = faulted_target

    distribution = _measure_distribution(
        profiles, train, base_settings, clean, order, warmup
    )

    curve: list[SweepRow] = []
    for sigma in grid:
        settings = _settings_for(sigma, base_settings)
        point, event, cause = _score_faulted(
            profiles, train, settings, faulted, labels, episodes, target, order, warmup
        )
        clean_fpr = _clean_rate(
            profiles, train, settings, clean, target, order, warmup
        )
        emits = _emits_fraction(distribution, sigma)
        curve.append(
            SweepRow(
                sigma=sigma,
                precision=point.precision,
                recall=point.recall,
                f1=point.f1,
                event_recall=event.recall,
                clean_false_positive_rate=clean_fpr,
                root_cause_accuracy=cause,
                emits_fraction=emits,
            )
        )

    baseline_sigma = DEFAULT.detector.spatial_sigma
    baseline = next((r for r in curve if r.sigma == baseline_sigma), None)
    baseline_clean_fpr = baseline.clean_false_positive_rate if baseline else 1.0

    eligible = [r for r in curve if r.clean_false_positive_rate <= baseline_clean_fpr]
    chosen_f1 = max(eligible, key=lambda r: r.f1) if eligible else None
    chosen_recall = max(eligible, key=lambda r: r.recall) if eligible else None

    return SpatialCalibrationResult(
        distribution=distribution,
        curve=curve,
        baseline_sigma=baseline_sigma,
        baseline_clean_fpr=baseline_clean_fpr,
        chosen_f1=chosen_f1,
        chosen_recall=chosen_recall,
        samples=test_samples,
        episodes=len(episodes),
        stations=len(profiles),
    )


def _measure_distribution(
    profiles: tuple[StationProfile, ...],
    train: dict[str, list[Observation]],
    settings: Settings,
    clean: dict[str, list[Observation]],
    order: list[str],
    warmup: int,
) -> list[ChannelDistribution]:
    """Run the clean network through a fresh pipeline and record the buddy
    check's median peer departure for the target station, via the L4 `_observe`
    seam. Only the target matters: it is the station the sweep faults, so its
    clean departures are the null the threshold has to sit above."""
    target = profiles[0].station_id
    pipeline = _build(profiles, train, settings)
    seen: list[tuple[Channel, float, bool]] = []
    clock = [0]  # interleave position, so records before `warmup` can be dropped
    pipeline.l4._observe = (  # type: ignore[method-assign]
        lambda sid, ch, median, agree, n: seen.append((ch, median, agree))
        if sid == target and clock[0] >= len(order) * warmup
        else None
    )

    for position, obs in enumerate(_interleave(clean, order)):
        clock[0] = position
        pipeline.process(obs)

    out: list[ChannelDistribution] = []
    for channel in CHANNELS:
        stats = [abs(m) for (c, m, _a) in seen if c is channel]
        agree = [a for (c, _m, a) in seen if c is channel]
        if not stats:
            empty = {q: float("nan") for q in _QUANTILES}
            out.append(ChannelDistribution(channel, 0, empty, 0.0))
            continue
        stats.sort()
        quantiles = {q: _quantile(stats, q) for q in _QUANTILES}
        out.append(
            ChannelDistribution(
                channel=channel,
                count=len(stats),
                quantiles=quantiles,
                agree_fraction=sum(agree) / len(agree),
            )
        )
    return out


def _emits_fraction(distribution: list[ChannelDistribution], sigma: float) -> float:
    """Clean-stream share of buddy-check evaluations that would clear the
    emission gate (0.75 x sigma) -- an upper bound on L4's clean chatter,
    before fusion weighting and the alert threshold."""
    gate = sigma * 0.75
    total = hits = 0
    for d in distribution:
        if not d.count:
            continue
        total += d.count
        # Interpolate the survival fraction from the reported quantiles.
        hits += round(d.count * _tail_fraction(d, gate))
    return hits / total if total else 0.0


def _tail_fraction(d: ChannelDistribution, x: float) -> float:
    """Rough P(statistic >= x) from the stored quantile points."""
    points = sorted(d.quantiles.items())
    for q, value in points:
        if value >= x:
            return max(0.0, 1.0 - q)
    return 0.0


def _score_faulted(
    profiles, train, settings, faulted, labels, episodes, target, order, warmup
) -> tuple[PointMetrics, EventMetrics, float]:
    pipeline = _build(profiles, train, settings)
    # Process every station -- the peers are what feed the buddy board -- but
    # keep only the target's verdicts, which are the ones the labels describe.
    verdicts = []
    for obs in _interleave(faulted, order):
        verdict = pipeline.process(obs)
        if obs.station_id == target:
            verdicts.append(verdict)

    point = PointMetrics()
    correct = considered = 0
    for i in range(warmup, min(len(verdicts), len(labels))):
        predicted, actual = verdicts[i].is_anomalous, labels[i].is_anomalous
        if predicted and actual:
            point.true_positive += 1
            considered += 1
            correct += verdicts[i].fault_type is labels[i].fault_type
        elif predicted:
            point.false_positive += 1
        elif actual:
            point.false_negative += 1
        else:
            point.true_negative += 1

    event = EventMetrics()
    for episode in episodes:
        if episode.end_index <= warmup:
            continue
        event.total += 1
        lo = max(episode.start_index, warmup)
        hit = next(
            (j for j in range(lo, min(episode.end_index, len(verdicts)))
             if verdicts[j].is_anomalous),
            None,
        )
        if hit is not None:
            event.detected += 1
            event.delays.append(hit - episode.start_index)

    cause = correct / considered if considered else 0.0
    return point, event, cause


def _clean_rate(
    profiles, train, settings, clean, target, order, warmup
) -> float:
    pipeline = _build(profiles, train, settings)
    flags = []
    for obs in _interleave(clean, order):
        verdict = pipeline.process(obs)
        if obs.station_id == target:
            flags.append(verdict.is_anomalous)
    tail = flags[warmup:]
    return sum(tail) / len(tail) if tail else 0.0


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q >= 1.0:
        return sorted_values[-1]
    k = q * (len(sorted_values) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
