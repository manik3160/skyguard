"""L1 -- per-channel temporal consistency.

Three tests, each aimed at a different fault signature:

Hampel     robust z of the one-step forecast residual. Catches spikes and
           bias steps.
Persistence  whether large residuals of the same sign are stacking up. This is
           what separates a spike (one big residual, then back to normal) from
           a step or a drift (many same-sign residuals) -- available online,
           with no lookahead.
Stuck      peak-to-peak range near zero over a window of *as-reported* values.
           Catches latched ADCs and iced sensors, which the Hampel test cannot
           see because a frozen value is perfectly predicted by a flat trend
           fit, giving it a residual of zero.

The stuck test is the one most quality-control systems omit, and it is the
fault that does the most damage in practice: a frozen sensor passes every range
and step check indefinitely while feeding a constant into assimilation.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import DetectorConfig
from ..features.online import StationState, is_missing, squash
from ..models import CHANNELS, Channel, Evidence, FaultType, Observation
from .base import StatelessDetector


class TemporalDetector(StatelessDetector):
    name = "l1_temporal"

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def inspect(self, obs: Observation, state: StationState) -> tuple[Evidence, ...]:
        out: list[Evidence] = []
        for channel in CHANNELS:
            # The stuck test needs no warm-up beyond its own window and must run
            # even during the residual buffer's fill period, because a sensor
            # can be frozen from the moment it is commissioned.
            stuck = self._stuck(channel, state)
            if stuck is not None:
                out.append(stuck)
                continue

            if not state.warm:
                continue
            value = obs.value(channel)
            if is_missing(value):
                continue  # L0 owns missing data
            hampel = self._hampel(channel, value, state)
            if hampel is not None:
                out.append(hampel)
        return tuple(out)

    def _hampel(
        self, channel: Channel, value: float, state: StationState
    ) -> Evidence | None:
        z = state.residual_z(channel, value)
        magnitude = abs(z)
        threshold = self.config.hampel_sigma
        # Emit slightly below the decision threshold so fusion can combine
        # near-misses that agree across layers -- but not far below. Noisy-OR
        # pools evidence multiplicatively, so a detector that chatters at score
        # 0.05 on a third of all samples will, in company with two other
        # chattering detectors, manufacture an alert out of nothing. Measured:
        # dropping this gate from 0.5 to 0.75 cut the clean-stream false-alarm
        # rate by an order of magnitude with no loss of event recall.
        if magnitude < threshold * 0.75:
            return None

        prediction = state.predict(channel)
        scale = state.residual_scale(channel)
        direction = "above" if z > 0 else "below"
        suggests = self._shape_from_persistence(channel, z, state)

        return Evidence(
            detector=self.name,
            channel=channel,
            score=squash(magnitude, threshold),
            statistic=magnitude,
            threshold=threshold,
            detail=(
                f"Reading {value:.1f} {channel.unit} is {magnitude:.1f} sigma "
                f"{direction} the forecast of {prediction:.1f} {channel.unit} "
                f"(typical error {scale:.2f} {channel.unit})."
            ),
            suggests=suggests,
        )

    def _shape_from_persistence(
        self, channel: Channel, z: float, state: StationState
    ) -> FaultType:
        """Spike or step? Decided by whether recent residuals agree in sign.

        A spike is one outlier against a stable forecast, so the residuals
        around it alternate. A bias step or a drift pushes the forecast the same
        way for several samples in a row.
        """
        buf = state.residual[channel]
        if len(buf) < 4:
            return FaultType.SPIKE
        recent = buf.as_array()[-4:]
        scale = state.residual_scale(channel)
        if not math.isfinite(scale) or scale <= 0:
            return FaultType.SPIKE
        significant = recent[np.abs(recent) > 1.5 * scale]
        if significant.size >= 2 and np.all(np.sign(significant) == np.sign(z)):
            return FaultType.OFFSET_STEP
        return FaultType.SPIKE

    def _stuck(self, channel: Channel, state: StationState) -> Evidence | None:
        buf = state.reported[channel]
        window = self.config.stuck_window
        if len(buf) < window:
            return None

        span = buf.span(window)
        epsilon = self.config.stuck_epsilon[channel.value]
        if not math.isfinite(span) or span > epsilon:
            return None

        minutes = window * 10  # nominal cadence; the figure is cosmetic
        # Confidence grows the flatter the window is. A brief plateau in calm
        # conditions is normal; bit-identical readings for two hours are not.
        tightness = epsilon / max(span, epsilon / 20.0)
        return Evidence(
            detector=self.name,
            channel=channel,
            score=min(0.60 + 0.04 * tightness, 1.0),
            statistic=float(tightness),
            threshold=1.0,
            detail=(
                f"Value has moved less than {span:.3f} {channel.unit} across "
                f"{window} consecutive reports (about {minutes} minutes). "
                "The sensor appears latched."
            ),
            suggests=FaultType.STUCK,
        )
