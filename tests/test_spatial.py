"""L4 spatial buddy check -- the layer with no direct coverage before this.

Every other test in the suite registers a single station, so `SpatialDetector`
always early-returns for want of peers. These drive a real three-station network
through the pipeline: a sustained single-station bias must be caught, a genuine
network-wide swing must not, and neither can happen before the pairwise
difference statistics have enough history to mean anything.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from skyguard.detect.l4_spatial import MIN_PAIR_SAMPLES
from skyguard.models import Channel, Observation
from skyguard.pipeline import REBASELINE_AFTER, REBASELINE_CEILING, SkyGuardPipeline
from skyguard.synth.climate import DEMO_NETWORK
from skyguard.synth.network import NetworkGenerator

# Ambala / Karnal / Hisar -- three plains stations ~80-130 km apart, so they
# share weather closely. Shimla is left out: at 2.2 km elevation its pairwise
# spread is wide and it is a deliberately poor buddy.
PROFILES = DEMO_NETWORK[:3]
TARGET = PROFILES[0].station_id
ORIGIN = 1_705_000_000.0
WARMUP_STEPS = MIN_PAIR_SAMPLES + 120  # past the point where pairs are `ready()`


def _warm_network(
    future_steps: int = 20, seed: int = 17
) -> tuple[SkyGuardPipeline, list[list[Observation]]]:
    """A pipeline with warm buddy statistics, plus `future_steps` of coherent
    clean weather ahead of the generator for a test to keep pulling."""
    pipeline = SkyGuardPipeline()
    for profile in PROFILES:
        pipeline.register_station(profile.station_id, profile.altitude_m)

    generator = NetworkGenerator(PROFILES, ORIGIN, seed=seed)
    pipeline.fit(generator.generate(1_500)[TARGET])

    for _ in range(WARMUP_STEPS):
        for obs in generator.step():
            pipeline.process(obs)

    future = [generator.step() for _ in range(future_steps)]
    return pipeline, future


def _run_sustained_bias(
    pipeline: SkyGuardPipeline,
    future: list[list[Observation]],
    channel: Channel,
    delta: float,
) -> list[bool]:
    """Inject a constant single-station bias on `channel` for the whole of
    `future`, peers untouched. Returns whether the target flagged that channel
    at each step."""
    flagged_history: list[bool] = []
    for step in future:
        for obs in step:
            if obs.station_id == TARGET:
                obs = _bias(obs, channel, delta)
            verdict = pipeline.process(obs)
            if obs.station_id == TARGET:
                flagged_history.append(channel in verdict.flagged)
    return flagged_history


def _bias(obs: Observation, channel: Channel, delta: float) -> Observation:
    return replace(obs, **{channel.value: obs.value(channel) + delta})


def test_sustained_single_station_bias_is_caught_by_the_buddy_check():
    pipeline, future = _warm_network()

    l4_fired = False
    for step in future:
        for obs in step:
            if obs.station_id == TARGET:
                obs = _bias(obs, Channel.TEMPERATURE, 5.0)
            verdict = pipeline.process(obs)
            if obs.station_id == TARGET and any(
                e.detector == "l4_spatial" and e.channel is Channel.TEMPERATURE
                for e in verdict.evidence
            ):
                l4_fired = True
    assert l4_fired, "buddy check missed a sustained 5 degC single-station bias"


def test_network_wide_swing_does_not_trigger_the_buddy_check():
    """A real front moves every station together; the pairwise differences do
    not change, so L4 must stay silent -- this is the whole point of the layer.

    The swing is ramped, not stepped: an instantaneous jump trips L1 at every
    station, nothing gets published, and the buddy check would then compare a
    moved target against stale peers. A front arriving over a few hours is
    tracked by each station's forecast and stays out of L1's way.
    """
    pipeline, future = _warm_network()

    accumulated = 0.0
    for step in future:
        accumulated += 0.25  # ~0.025 degC/min, well under L1's slew gate
        for obs in step:
            obs = _bias(obs, Channel.TEMPERATURE, accumulated)  # every station
            verdict = pipeline.process(obs)
            if obs.station_id == TARGET:
                assert not any(
                    e.detector == "l4_spatial" for e in verdict.evidence
                ), "buddy check fired on a coherent network-wide swing"


def test_buddy_dissent_holds_off_the_rebaseline():
    """A sustained single-station bias: without the spatial check the pipeline
    re-seeds its baseline from the sensor after REBASELINE_AFTER samples and
    goes quiet. With neighbours disagreeing, it must keep the channel flagged."""
    pipeline, future = _warm_network(future_steps=REBASELINE_AFTER + 8)
    history = _run_sustained_bias(pipeline, future, Channel.TEMPERATURE, 5.0)

    assert pipeline.stats["rebaseline_suppressed"] > 0
    # Still flagged well past the point a lone station would have re-baselined.
    assert all(history[REBASELINE_AFTER + 2 :])


def test_rebaseline_ceiling_fires_even_with_buddy_dissent():
    """A stale pairwise offset must not be able to lock a channel forever: past
    REBASELINE_CEILING the re-baseline goes through regardless of the buddies."""
    pipeline, future = _warm_network(future_steps=REBASELINE_CEILING + 10)
    _run_sustained_bias(pipeline, future, Channel.TEMPERATURE, 5.0)

    assert pipeline.stats["rebaselines"] > 0


def test_single_station_never_suppresses_a_rebaseline():
    """No peers means no buddy check means the recovery path is untouched --
    this is why the single-station benchmark is the control for the change."""
    pipeline = SkyGuardPipeline()
    profile = PROFILES[0]
    pipeline.register_station(profile.station_id, profile.altitude_m)
    generator = NetworkGenerator(PROFILES[:1], ORIGIN, seed=31)
    pipeline.fit(generator.generate(1_500)[TARGET])
    for _ in range(400):
        pipeline.process(generator.step()[0])
    for _ in range(40):
        obs = _bias(generator.step()[0], Channel.PRESSURE, 4.0)
        pipeline.process(obs)

    assert pipeline.stats["rebaseline_suppressed"] == 0


def test_buddy_check_is_silent_until_pairs_have_history():
    """A bias present from commissioning: the pairwise stats are not `ready()`
    for the first MIN_PAIR_SAMPLES steps, so L4 cannot speak yet."""
    pipeline = SkyGuardPipeline()
    for profile in PROFILES:
        pipeline.register_station(profile.station_id, profile.altitude_m)
    generator = NetworkGenerator(PROFILES, ORIGIN, seed=23)
    pipeline.fit(generator.generate(1_500)[TARGET])

    for _ in range(MIN_PAIR_SAMPLES - 30):
        for obs in generator.step():
            if obs.station_id == TARGET:
                obs = _bias(obs, Channel.TEMPERATURE, 6.0)
            verdict = pipeline.process(obs)
            if obs.station_id == TARGET:
                assert not any(e.detector == "l4_spatial" for e in verdict.evidence)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
