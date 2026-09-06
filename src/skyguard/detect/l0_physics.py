"""L0 -- deterministic physics gate.

Catches readings that cannot be real regardless of weather: out-of-range
values, sentinel codes, impossible rates of change, and thermodynamically
inconsistent T/RH pairs. No training, no history beyond one sample, constant
time. This is the layer that runs unchanged on an ESP32.

Everything here fires with score 1.0 -- there is no probabilistic judgement to
make. If the dewpoint exceeds the air temperature by more than instrument
tolerance, one of the two sensors is wrong; that is arithmetic, not inference.
"""

from __future__ import annotations

from ..config import PhysicalLimits
from ..features.online import StationState, is_missing
from ..models import MISSING_SENTINEL, Channel, Evidence, FaultType, Observation
from ..physics import dewpoint
from .base import StatelessDetector


class PhysicsGate(StatelessDetector):
    name = "l0_physics"

    def __init__(self, limits: PhysicalLimits) -> None:
        self.limits = limits
        self._bounds = {
            Channel.TEMPERATURE: (limits.temperature_min_c, limits.temperature_max_c),
            Channel.PRESSURE: (limits.pressure_min_hpa, limits.pressure_max_hpa),
            Channel.HUMIDITY: (limits.humidity_min_pct, limits.humidity_max_pct),
        }
        self._rates = {
            Channel.TEMPERATURE: limits.max_temperature_rate_c_per_min,
            Channel.PRESSURE: limits.max_pressure_rate_hpa_per_min,
            Channel.HUMIDITY: limits.max_humidity_rate_pct_per_min,
        }

    def inspect(self, obs: Observation, state: StationState) -> tuple[Evidence, ...]:
        out: list[Evidence] = []

        for channel, (lo, hi) in self._bounds.items():
            value = obs.value(channel)

            # Sentinel first: -999 would otherwise be reported as a range
            # violation, which is technically true but useless to an operator
            # deciding whether to send a technician or check the modem.
            if value == MISSING_SENTINEL:
                out.append(
                    Evidence(
                        detector=self.name,
                        channel=channel,
                        score=1.0,
                        statistic=1.0,
                        threshold=0.0,
                        detail="Logger reported the no-data sentinel (-999).",
                        suggests=FaultType.DROPOUT,
                    )
                )
                continue

            if is_missing(value):
                out.append(
                    Evidence(
                        detector=self.name,
                        channel=channel,
                        score=1.0,
                        statistic=1.0,
                        threshold=0.0,
                        detail="No value received for this channel.",
                        suggests=FaultType.DROPOUT,
                    )
                )
                continue

            if value < lo or value > hi:
                excess = max(lo - value, value - hi)
                out.append(
                    Evidence(
                        detector=self.name,
                        channel=channel,
                        score=1.0,
                        statistic=abs(excess),
                        threshold=0.0,
                        detail=(
                            f"{value:.1f} {channel.unit} is outside the physical "
                            f"range {lo:g} to {hi:g}."
                        ),
                        suggests=FaultType.PHYSICAL_VIOLATION,
                    )
                )
                continue

            rate = abs(state.rate_per_minute(channel, value, obs.timestamp))
            limit = self._rates[channel]
            if state.last_timestamp is not None and rate > limit:
                out.append(
                    Evidence(
                        detector=self.name,
                        channel=channel,
                        score=1.0,
                        statistic=rate,
                        threshold=limit,
                        detail=(
                            f"Changed at {rate:.2f} {channel.unit}/min, above the "
                            f"{limit:g} {channel.unit}/min limit for this sensor."
                        ),
                        suggests=FaultType.SPIKE,
                    )
                )

        out.extend(self._dewpoint_check(obs))
        return tuple(out)

    def _dewpoint_check(self, obs: Observation) -> list[Evidence]:
        """Dewpoint may not exceed air temperature.

        This is the cross-channel constraint that makes the two sensors check
        each other. It is also the check that resolves the example use case in
        the problem statement: 55 degC with very high humidity is individually
        in range for both channels but jointly implies a dewpoint far above the
        air temperature.
        """
        t, rh = obs.temperature, obs.humidity
        if is_missing(t) or is_missing(rh):
            return []
        if not (-60.0 < t < 70.0) or not (0.0 < rh <= 100.0):
            # Out-of-range inputs already produced evidence above; computing a
            # dewpoint from them would report a second, derived failure for the
            # same root cause.
            return []

        td = dewpoint(t, rh)
        excess = td - t
        tol = self.limits.dewpoint_excess_tolerance_c
        if excess <= tol:
            return []

        return [
            Evidence(
                detector=self.name,
                channel=channel,
                score=1.0,
                statistic=excess,
                threshold=tol,
                detail=(
                    f"T={t:.1f} degC with RH={rh:.0f} % implies a dewpoint of "
                    f"{td:.1f} degC, which is {excess:.1f} degC above the air "
                    "temperature. The T and RH sensors disagree."
                ),
                suggests=FaultType.PHYSICAL_VIOLATION,
            )
            for channel in (Channel.TEMPERATURE, Channel.HUMIDITY)
        ]
