"""Fault injection with ground-truth labels.

The evaluation criteria in problem statement 26073 are scored on
anomaly-injected data, so the injector is a first-class component, not a test
fixture. Each fault is an *episode* with a start, a duration, an affected
channel set, and a type -- which is exactly the granularity the evaluation
harness needs to compute event-level recall and detection delay.

Design notes
------------
Faults are applied to a clean stream after generation, never inside the
generator. That separation means the same clean stream can be replayed with
different fault seeds, which is how the benchmark isolates detector variance
from data variance.

Magnitudes are drawn from ranges observed in IMD quality-control logs rather
than picked for detectability. A drift of 0.02 degC/day is genuinely hard to
see and it is supposed to be -- a benchmark that only contains obvious faults
reports a number that will not survive the finale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..models import CHANNELS, MISSING_SENTINEL, Channel, FaultType, GroundTruth, Observation


@dataclass(frozen=True, slots=True)
class FaultEpisode:
    """One injected fault, with the index range it covers."""

    episode_id: str
    fault_type: FaultType
    channels: frozenset[Channel]
    start_index: int
    length: int
    magnitude: float

    @property
    def end_index(self) -> int:
        return self.start_index + self.length

    def covers(self, index: int) -> bool:
        return self.start_index <= index < self.end_index


@dataclass(frozen=True, slots=True)
class InjectionPlan:
    """Relative frequency and magnitude ranges for each fault type.

    `rate` is episodes per 1000 samples. The defaults were meant to give roughly
    4 % anomalous samples (the fraction in IMD's published AWS quality-control
    statistics), but as measured they give about 23 %: the long fault types
    (`offset_step` especially) dominate, and `drift` never places at all because
    the short types are scheduled first and fragment the timeline. The eight
    rates need re-tuning against a measured target -- see CLAUDE.md 8.6 -- so
    treat these as provisional.
    """

    spike_rate: float = 6.0
    stuck_rate: float = 1.2
    drift_rate: float = 0.8
    offset_step_rate: float = 0.9
    noise_burst_rate: float = 1.3
    dropout_rate: float = 1.5
    power_flicker_rate: float = 0.5
    physical_violation_rate: float = 0.7


class FaultInjector:
    """Applies an :class:`InjectionPlan` to a clean series."""

    def __init__(self, plan: InjectionPlan | None = None, seed: int = 0) -> None:
        self.plan = plan or InjectionPlan()
        self._rng = np.random.default_rng(seed)

    # -- planning -----------------------------------------------------------

    def plan_episodes(self, n_samples: int) -> list[FaultEpisode]:
        """Draw a non-overlapping episode schedule.

        Overlaps are rejected rather than merged: a sample with two
        simultaneous faults has an ambiguous root-cause label, and scoring
        against an ambiguous label measures nothing.
        """
        rng = self._rng
        specs = [
            (FaultType.SPIKE, self.plan.spike_rate, (1, 3)),
            (FaultType.STUCK, self.plan.stuck_rate, (12, 72)),
            (FaultType.DRIFT, self.plan.drift_rate, (288, 1440)),
            (FaultType.OFFSET_STEP, self.plan.offset_step_rate, (72, 432)),
            (FaultType.NOISE_BURST, self.plan.noise_burst_rate, (18, 90)),
            (FaultType.DROPOUT, self.plan.dropout_rate, (1, 24)),
            (FaultType.POWER_FLICKER, self.plan.power_flicker_rate, (2, 12)),
            (FaultType.PHYSICAL_VIOLATION, self.plan.physical_violation_rate, (1, 6)),
        ]

        episodes: list[FaultEpisode] = []
        occupied = np.zeros(n_samples, dtype=bool)
        counter = 0

        for fault_type, rate, (min_len, max_len) in specs:
            n_episodes = rng.poisson(rate * n_samples / 1000.0)
            for _ in range(int(n_episodes)):
                length = int(rng.integers(min_len, max_len + 1))
                if length >= n_samples:
                    continue
                placed = False
                for _attempt in range(20):
                    start = int(rng.integers(0, n_samples - length))
                    if occupied[start : start + length].any():
                        continue
                    occupied[start : start + length] = True
                    placed = True
                    break
                if not placed:
                    continue

                counter += 1
                episodes.append(
                    FaultEpisode(
                        episode_id=f"{fault_type.value}-{counter:04d}",
                        fault_type=fault_type,
                        channels=self._pick_channels(fault_type),
                        start_index=start,
                        length=length,
                        magnitude=self._pick_magnitude(fault_type),
                    )
                )

        episodes.sort(key=lambda e: e.start_index)
        return episodes

    def _pick_channels(self, fault_type: FaultType) -> frozenset[Channel]:
        rng = self._rng
        # Comms and power faults take down the whole logger, so every channel
        # is affected. Sensor faults are per-instrument.
        if fault_type in (FaultType.DROPOUT, FaultType.POWER_FLICKER):
            return frozenset(CHANNELS)
        if fault_type is FaultType.PHYSICAL_VIOLATION:
            # Violations are a T/RH pair inconsistency by construction.
            return frozenset({Channel.TEMPERATURE, Channel.HUMIDITY})
        if fault_type is FaultType.SPIKE and rng.random() < 0.2:
            # Index rather than choosing from CHANNELS directly: numpy coerces
            # the str-backed enum into a fixed-width byte string and truncates it.
            picks = rng.choice(len(CHANNELS), size=2, replace=False)
            return frozenset(CHANNELS[int(i)] for i in picks)
        return frozenset({CHANNELS[int(rng.integers(0, len(CHANNELS)))]})

    def _pick_magnitude(self, fault_type: FaultType) -> float:
        rng = self._rng
        # Magnitudes are in "channel sigma" units and scaled per channel at
        # application time, so one number covers all three sensors.
        ranges = {
            FaultType.SPIKE: (3.5, 14.0),
            FaultType.STUCK: (0.0, 0.0),
            FaultType.DRIFT: (1.5, 6.0),
            FaultType.OFFSET_STEP: (2.0, 7.0),
            FaultType.NOISE_BURST: (4.0, 12.0),
            FaultType.DROPOUT: (0.0, 0.0),
            FaultType.POWER_FLICKER: (0.0, 0.0),
            FaultType.PHYSICAL_VIOLATION: (1.0, 1.0),
        }
        lo, hi = ranges[fault_type]
        return float(rng.uniform(lo, hi)) if hi > lo else lo


# Per-channel natural scale, used to convert a sigma magnitude into engineering
# units. These are the short-term (few-hour) standard deviations of the clean
# generator, measured once and hard-coded so the injector is deterministic.
_CHANNEL_SIGMA: dict[Channel, float] = {
    Channel.TEMPERATURE: 1.4,   # degC
    Channel.PRESSURE: 0.9,      # hPa
    Channel.HUMIDITY: 6.0,      # %
}


def apply_faults(
    clean: list[Observation],
    episodes: list[FaultEpisode],
    seed: int = 0,
) -> tuple[list[Observation], list[GroundTruth]]:
    """Return the faulted series and its per-sample ground truth."""
    rng = np.random.default_rng(seed)
    values = {c: np.array([o.value(c) for o in clean], dtype=float) for c in CHANNELS}
    labels = [
        GroundTruth(False, FaultType.NONE, frozenset(), None) for _ in clean
    ]

    for ep in episodes:
        sl = slice(ep.start_index, ep.end_index)
        # Iterate in fixed CHANNELS order, not frozenset order. `_apply_one`
        # draws from `rng`, so for a multi-channel episode (dropout, power
        # flicker, physical violation) the order channels are visited decides
        # which channel consumes which draw. `ep.channels` is a frozenset whose
        # iteration order depends on the process hash seed, which made the whole
        # benchmark non-reproducible run to run -- point F1 swung about +/-0.02
        # on nothing but PYTHONHASHSEED.
        for channel in CHANNELS:
            if channel not in ep.channels:
                continue
            sigma = _CHANNEL_SIGMA[channel]
            series = values[channel]
            _apply_one(ep, channel, series, sl, sigma, rng)

        for i in range(ep.start_index, min(ep.end_index, len(clean))):
            labels[i] = GroundTruth(True, ep.fault_type, ep.channels, ep.episode_id)

    faulted = [
        replace(
            obs,
            temperature=float(values[Channel.TEMPERATURE][i]),
            pressure=float(values[Channel.PRESSURE][i]),
            humidity=float(values[Channel.HUMIDITY][i]),
        )
        for i, obs in enumerate(clean)
    ]
    return faulted, labels


def _apply_one(
    ep: FaultEpisode,
    channel: Channel,
    series: np.ndarray,
    sl: slice,
    sigma: float,
    rng: np.random.Generator,
) -> None:
    """Mutate `series[sl]` in place according to the episode type."""
    n = len(series[sl])
    if n == 0:
        return
    sign = 1.0 if rng.random() < 0.5 else -1.0

    match ep.fault_type:
        case FaultType.SPIKE:
            series[sl] += sign * ep.magnitude * sigma

        case FaultType.STUCK:
            # Latch at the last good value. This is what an ADC hold-fault or
            # an iced-over sensor produces.
            series[sl] = series[max(sl.start - 1, 0)]

        case FaultType.DRIFT:
            # Linear ramp reaching `magnitude` sigma by the end of the episode.
            ramp = np.linspace(0.0, sign * ep.magnitude * sigma, n)
            series[sl] += ramp

        case FaultType.OFFSET_STEP:
            series[sl] += sign * ep.magnitude * sigma

        case FaultType.NOISE_BURST:
            series[sl] += rng.normal(0.0, ep.magnitude * sigma * 0.35, size=n)

        case FaultType.DROPOUT:
            series[sl] = MISSING_SENTINEL

        case FaultType.POWER_FLICKER:
            # Brown-out: readings collapse toward the ADC zero rail with a
            # little residual noise, rather than going cleanly missing.
            series[sl] = rng.normal(0.0, 0.5, size=n)

        case FaultType.PHYSICAL_VIOLATION:
            # Drive RH above saturation and temperature down, so the pair
            # implies a dewpoint above the air temperature.
            if channel is Channel.HUMIDITY:
                series[sl] = rng.uniform(100.5, 118.0, size=n)
            else:
                series[sl] -= rng.uniform(1.0, 4.0, size=n)

        case FaultType.NONE:
            return


def build_labelled_stream(
    profile_generator,
    n_samples: int,
    plan: InjectionPlan | None = None,
    seed: int = 0,
) -> tuple[list[Observation], list[GroundTruth], list[FaultEpisode]]:
    """Convenience wrapper: generate clean data, inject faults, return all three.

    `profile_generator` is a :class:`~skyguard.synth.climate.ClimateGenerator`.
    """
    clean = profile_generator.generate(n_samples)
    injector = FaultInjector(plan, seed=seed)
    episodes = injector.plan_episodes(n_samples)
    faulted, labels = apply_faults(clean, episodes, seed=seed)
    return faulted, labels, episodes
