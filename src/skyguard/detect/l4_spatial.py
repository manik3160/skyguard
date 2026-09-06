"""L4 -- spatial consistency (buddy check) across neighbouring stations.

The layer that makes sustained faults detectable at all.

Why it is necessary
-------------------
L1 through L3 all measure a station against its own recent history. That works
for spikes and freezes, but it cannot see a *persistent* bias: once a sensor has
been reading 5 degC high for an hour, the station's own forecast has adapted, the
residuals are back to zero, and every single-station test goes quiet. Measured on
the benchmark, single-station layers plateaued at 0.50 point-level recall for
exactly this reason -- they caught the onset of every step and drift, then lost
them.

A biased sensor is only identifiable relative to an external reference. The
cheapest reference available is the rest of the network: weather is spatially
coherent over tens of kilometres, so a real cold front moves every nearby
station together, while a broken thermometer moves one.

Method
------
The classical WMO buddy check, run online. For each peer, track the running
difference `station - peer` with an EWMA mean and robust scale. That difference
is slowly varying -- both stations feel the same synoptic weather, so the shared
signal cancels and what remains is the pair's climatological offset plus noise.
A bias at the target station shifts *every* pair it participates in by the same
amount, which is both detectable and attributable.

Pressure is reduced to mean sea level before differencing, otherwise a hill
station is permanently "anomalous" relative to a plains station by 200 hPa.

Reference: WMO-No. 488, *Guide to the Global Observing System*, section on
spatial consistency checks; and Section 4 of the WMO-No. 8 quality-control
annex.
"""

from __future__ import annotations

import math

from ..config import DetectorConfig
from ..features.online import StationState, is_missing, squash
from ..models import CHANNELS, Channel, Evidence, FaultType, Observation
from ..physics import reduce_to_sea_level
from .base import StatelessDetector

#: EWMA rate for the pairwise difference statistics. 0.002 gives a half-life of
#: about 350 samples (two and a half days), long enough that a fault lasting
#: hours cannot be absorbed into the "normal" offset before it is reported.
PAIR_ALPHA = 0.002

#: Peers must have reported within this many seconds to be comparable. Two
#: hours: wide enough to tolerate ragged telemetry, narrow enough that the
#: shared weather signal still cancels.
PEER_MAX_AGE_S = 7200.0

#: Minimum peers before the check is trusted. With one peer there is no way to
#: tell which of the two stations is wrong.
MIN_PEERS = 2

#: Updates a station pair needs before its difference statistics are used. Two
#: days of reports: enough to have seen a full diurnal cycle twice, so the
#: pair's normal offset is established rather than guessed.
MIN_PAIR_SAMPLES = 288


class NetworkBoard:
    """Shared last-known state for every station in the network.

    Deliberately tiny: three floats and a timestamp per station, plus pairwise
    difference statistics. A 500-station network costs a few hundred kilobytes,
    which is what makes the spatial check affordable at national scale.
    """

    def __init__(self) -> None:
        self.latest: dict[str, tuple[float, dict[Channel, float]]] = {}
        self.altitude: dict[str, float] = {}
        # (station, peer, channel) -> (ewma mean, ewma mean absolute deviation)
        self._pair: dict[tuple[str, str, Channel], tuple[float, float]] = {}
        self._pair_count: dict[tuple[str, str, Channel], int] = {}

    def register(self, station_id: str, altitude_m: float) -> None:
        self.altitude[station_id] = altitude_m

    def publish(self, obs: Observation, accepted: dict[Channel, float]) -> None:
        """Record a station's accepted values so peers can compare against them.

        Only accepted values are published. Publishing a value the pipeline just
        rejected would let one broken station drag the consensus its way, which
        is how buddy checks fail in networks with correlated hardware faults.
        """
        if not accepted:
            return
        self.latest[obs.station_id] = (obs.timestamp, dict(accepted))

    def peers(self, station_id: str, timestamp: float) -> list[str]:
        return [
            other
            for other, (ts, _values) in self.latest.items()
            if other != station_id and abs(timestamp - ts) <= PEER_MAX_AGE_S
        ]

    def normalised(self, station_id: str, channel: Channel, value: float) -> float:
        """Altitude-corrected value, so stations at different elevations compare."""
        if channel is not Channel.PRESSURE:
            return value
        altitude = self.altitude.get(station_id, 0.0)
        _ts, values = self.latest.get(station_id, (0.0, {}))
        temperature = values.get(Channel.TEMPERATURE, 15.0)
        return reduce_to_sea_level(value, altitude, temperature)

    def difference_stats(
        self, station_id: str, peer: str, channel: Channel
    ) -> tuple[float, float] | None:
        return self._pair.get((station_id, peer, channel))

    def update_difference(
        self, station_id: str, peer: str, channel: Channel, difference: float
    ) -> None:
        """Fold one pairwise difference into its running mean and scale.

        The EWMA uses a bias-corrected rate: `max(PAIR_ALPHA, 1/n)` for the
        n-th update. With the fixed rate alone the estimate stays anchored to
        its seed for hundreds of samples, so at demo start-up the scale was far
        smaller than the true pairwise spread, every standardised departure was
        enormous, and the buddy check flagged half the network. Bias correction
        is the textbook remedy and converges in a few dozen samples while still
        settling to the slow rate for the long run.
        """
        key = (station_id, peer, channel)
        count = self._pair_count.get(key, 0) + 1
        self._pair_count[key] = count

        current = self._pair.get(key)
        if current is None:
            self._pair[key] = (difference, 0.5)
            return

        alpha = max(PAIR_ALPHA, 1.0 / count)
        mean, scale = current
        new_mean = (1 - alpha) * mean + alpha * difference
        deviation = abs(difference - new_mean)
        new_scale = (1 - alpha) * scale + alpha * deviation
        self._pair[key] = (new_mean, max(new_scale, 1e-3))

    def ready(self, station_id: str, peer: str, channel: Channel) -> bool:
        """Whether a pair has enough history for its statistics to mean anything."""
        return self._pair_count.get((station_id, peer, channel), 0) >= MIN_PAIR_SAMPLES


class SpatialDetector(StatelessDetector):
    name = "l4_spatial"

    def __init__(self, config: DetectorConfig, board: NetworkBoard) -> None:
        self.config = config
        self.board = board
        # Threshold on the median standardised pairwise departure. Provenance
        # and calibration status live with the value in `config.py`.
        self.threshold = config.spatial_sigma

    def inspect(self, obs: Observation, state: StationState) -> tuple[Evidence, ...]:
        peers = self.board.peers(obs.station_id, obs.timestamp)
        if len(peers) < MIN_PEERS:
            return ()

        out: list[Evidence] = []
        for channel in CHANNELS:
            value = obs.value(channel)
            if is_missing(value):
                continue

            evidence = self._check_channel(obs, channel, value, peers)
            if evidence is not None:
                out.append(evidence)
        return tuple(out)

    def _check_channel(
        self,
        obs: Observation,
        channel: Channel,
        value: float,
        peers: list[str],
    ) -> Evidence | None:
        own = self.board.normalised(obs.station_id, channel, value)
        departures: list[float] = []

        for peer in peers:
            _ts, peer_values = self.board.latest[peer]
            peer_value = peer_values.get(channel)
            if peer_value is None or is_missing(peer_value):
                continue
            peer_norm = self.board.normalised(peer, channel, peer_value)
            difference = own - peer_norm

            if not self.board.ready(obs.station_id, peer, channel):
                continue
            stats = self.board.difference_stats(obs.station_id, peer, channel)
            if stats is None:
                continue
            mean, scale = stats
            # Mean absolute deviation to sigma, same constant as elsewhere.
            departures.append((difference - mean) / (scale * 1.4826))

        if len(departures) < MIN_PEERS:
            return None

        departures.sort()
        median = departures[len(departures) // 2]
        magnitude = abs(median)
        same_sign = all(d > 0 for d in departures) or all(d < 0 for d in departures)

        # Instrumentation seam. `median` is exactly the statistic `self.threshold`
        # is compared against, and its clean-stream distribution across every
        # channel that clears the peer/readiness filters -- not just the ones
        # that survive the emission gate below -- is what
        # `skyguard.evaluation.spatial_calibrate` measures to set the threshold.
        # The default is a no-op; only the calibrator overrides it.
        self._observe(obs.station_id, channel, median, same_sign, len(departures))

        if magnitude < self.threshold * 0.75:
            return None

        # Every peer must agree on the sign. If the network disagrees about
        # which way this station is off, it is weather, not a sensor.
        if not same_sign:
            return None

        direction = "higher" if median > 0 else "lower"
        return Evidence(
            detector=self.name,
            channel=channel,
            score=squash(magnitude, self.threshold),
            statistic=magnitude,
            threshold=self.threshold,
            detail=(
                f"Reads {magnitude:.1f} sigma {direction} than all "
                f"{len(departures)} neighbouring stations relative to their "
                "usual offset, so the departure is local to this sensor rather "
                "than a weather event."
            ),
            suggests=FaultType.OFFSET_STEP,
        )

    def _observe(
        self,
        station_id: str,
        channel: Channel,
        median_departure: float,
        peers_agree: bool,
        peer_count: int,
    ) -> None:
        """Called once per channel that has at least MIN_PEERS ready buddies,
        before the emission gate and sign-agreement checks. No-op in production;
        the calibration harness replaces it with a recorder so the clean-stream
        distribution of `median_departure` can be measured directly."""

    def learn(
        self,
        obs: Observation,
        accepted: dict[Channel, float],
        peers: list[str],
    ) -> None:
        """Fold accepted values into the pairwise difference statistics.

        Called by the pipeline only for observations that survived detection,
        so a fault cannot teach the network that it is normal.
        """
        for channel, value in accepted.items():
            own = self.board.normalised(obs.station_id, channel, value)
            for peer in peers:
                _ts, peer_values = self.board.latest.get(peer, (0.0, {}))
                peer_value = peer_values.get(channel)
                if peer_value is None or is_missing(peer_value):
                    continue
                peer_norm = self.board.normalised(peer, channel, peer_value)
                difference = own - peer_norm
                if math.isfinite(difference):
                    self.board.update_difference(obs.station_id, peer, channel, difference)
