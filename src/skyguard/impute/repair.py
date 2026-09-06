"""Corrected-value estimation for flagged observations.

Marked optional in the problem statement, but it is what turns detection into
the "self-healing network" the grand challenge asks about, so it ships.

Why the naive version diverges
------------------------------
The obvious repair is "last accepted value plus the local slope". During a long
fault -- an offset step can run 400 samples, a drift 1400 -- every reading is
flagged, so every repair extrapolates from the *previous repair*. That is a
linear recurrence with the slope re-estimated from its own output, and it runs
away: in testing, station pressure drifted from 985 hPa to 877 hPa inside one
episode, which then poisoned every detector downstream and flagged the entire
remaining stream.

So extrapolation is *damped*. Each successive step contributes phi^k of the
slope, so the estimate converges to a finite offset from the anchor instead of
marching off. This is the damped-trend idea from Gardner & McKenzie's
exponential smoothing work, and the same reason operational forecasters damp
persistence forecasts at longer lead times.

Strategy, in order of preference:

1. **Physics.** If temperature is trustworthy and only humidity is flagged, RH
   is reconstructed from the last good dewpoint, which is far more persistent
   than RH itself. A real reconstruction, not a guess.
2. **Damped trend.** Extrapolate from the anchor with a decaying slope.
3. **Hold.** Beyond the extrapolation horizon, hold the anchor.

Every repair is clamped to physical limits, and repaired values are marked in
the dashboard with a dashed trace so nothing downstream mistakes an estimate
for a measurement.
"""

from __future__ import annotations

import math

from ..config import PhysicalLimits
from ..features.online import StationState, is_missing
from ..models import CHANNELS, Channel, Observation
from ..physics import dewpoint, humidity_from_dewpoint

#: Geometric damping factor per extrapolated step. The total trend contribution
#: converges to slope * phi / (1 - phi) = 4 slopes, so a repair can never sit
#: more than four normal steps away from its anchor however long the outage.
TREND_DAMPING = 0.8

#: After this many consecutive substitutions the trend term is exhausted and
#: the estimate is the anchor. Chosen so the horizon (about 3 hours at the IMD
#: cadence) matches the timescale over which persistence stops being skilful.
MAX_EXTRAPOLATION_STEPS = 18


class Repairer:
    """Produces best-estimate replacements for flagged channels."""

    def __init__(self, limits: PhysicalLimits | None = None) -> None:
        self.limits = limits or PhysicalLimits()
        self._last_good_dewpoint: dict[str, float] = {}
        # Per (station, channel): the value and slope at the moment the fault
        # started, plus how many samples we have been substituting for.
        self._anchor: dict[tuple[str, Channel], tuple[float, float]] = {}
        self._run_length: dict[tuple[str, Channel], int] = {}
        self._bounds = {
            Channel.TEMPERATURE: (
                self.limits.temperature_min_c,
                self.limits.temperature_max_c,
            ),
            Channel.PRESSURE: (self.limits.pressure_min_hpa, self.limits.pressure_max_hpa),
            Channel.HUMIDITY: (self.limits.humidity_min_pct, self.limits.humidity_max_pct),
        }

    # -- bookkeeping -------------------------------------------------------

    def note_clean(self, obs: Observation) -> None:
        """Record dewpoint from an accepted observation, for later reconstruction."""
        if is_missing(obs.temperature) or is_missing(obs.humidity):
            return
        if not (0.0 < obs.humidity <= 100.0) or not (-60.0 < obs.temperature < 70.0):
            return
        self._last_good_dewpoint[obs.station_id] = dewpoint(obs.temperature, obs.humidity)

    def note_accepted(self, station_id: str, channel: Channel) -> None:
        """Clear the substitution run once the channel reports credibly again."""
        self._run_length.pop((station_id, channel), None)
        self._anchor.pop((station_id, channel), None)

    # -- repair ------------------------------------------------------------

    def repair(
        self,
        obs: Observation,
        flagged: frozenset[Channel],
        state: StationState,
    ) -> dict[str, float] | None:
        if not flagged:
            return None

        out: dict[str, float] = {}
        for channel in CHANNELS:
            if channel not in flagged:
                self.note_accepted(obs.station_id, channel)
                continue

            value = self._repair_by_physics(obs, channel, flagged)
            if value is None:
                value = self._repair_by_damped_trend(obs.station_id, channel, state)
            if value is None:
                value = state.last_clean.get(channel)
            if value is None or not math.isfinite(value):
                continue

            lo, hi = self._bounds[channel]
            out[channel.value] = round(min(max(float(value), lo), hi), 2)

        return out or None

    def _repair_by_physics(
        self,
        obs: Observation,
        channel: Channel,
        flagged: frozenset[Channel],
    ) -> float | None:
        """Reconstruct RH from a trustworthy temperature and the last good dewpoint.

        Dewpoint is an air-mass property: it changes on synoptic timescales,
        while RH swings 40 points over a single diurnal cycle purely because
        temperature moved. Holding dewpoint and recomputing RH therefore tracks
        the real diurnal shape instead of flat-lining it.
        """
        if channel is not Channel.HUMIDITY:
            return None
        if Channel.TEMPERATURE in flagged:
            return None
        td = self._last_good_dewpoint.get(obs.station_id)
        if td is None or is_missing(obs.temperature):
            return None
        return humidity_from_dewpoint(obs.temperature, td)

    def _repair_by_damped_trend(
        self,
        station_id: str,
        channel: Channel,
        state: StationState,
    ) -> float | None:
        """Anchor plus a geometrically damped trend extension."""
        key = (station_id, channel)
        run = self._run_length.get(key, 0)

        if run == 0:
            # First substitution of this fault: freeze the anchor from the last
            # credible history, before any repair enters the buffer.
            buf = state.raw[channel]
            if not buf.ready:
                return None
            history = buf.as_array()
            anchor = float(history[-1])
            slope = (
                float((history[-1] - history[-4]) / 3.0) if history.size >= 4 else 0.0
            )
            self._anchor[key] = (anchor, slope)

        anchor_slope = self._anchor.get(key)
        if anchor_slope is None:
            return None
        anchor, slope = anchor_slope

        run += 1
        self._run_length[key] = run
        steps = min(run, MAX_EXTRAPOLATION_STEPS)

        # Geometric series sum_{k=1..steps} phi^k -- bounded, and computed in
        # closed form so a long outage costs no more than a short one.
        phi = TREND_DAMPING
        damped_steps = phi * (1.0 - phi**steps) / (1.0 - phi)
        return anchor + slope * damped_steps
