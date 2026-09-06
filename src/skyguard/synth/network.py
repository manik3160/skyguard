"""Spatially coherent multi-station generation.

The spatial buddy check in L4 is only meaningful if neighbouring stations
actually share weather. If each station were driven by an independent random
walk, every genuine synoptic swing would look like a one-station departure and
the buddy check would fire constantly -- so a benchmark built on independent
stations would report a false-alarm rate that says nothing about the field.

So the synoptic departure is decomposed:

    departure_i = sqrt(rho) * shared + sqrt(1 - rho) * local_i

`shared` is one AR(1) process felt by the whole network; `local_i` is that
station's own. The weights preserve unit variance, so each station's marginal
statistics are unchanged from the single-station generator -- only the
correlation between stations is added.

`rho` falls off with separation, which is what makes the network realistic:
Ambala and Karnal (about 80 km apart) move together closely; Shimla, across the
Siwaliks and 2 km higher, does not. A buddy check that treats those two as
equally good references would be wrong, and this generator lets the benchmark
demonstrate that rather than assume it.
"""

from __future__ import annotations

import math

import numpy as np

from ..models import Observation
from .climate import ClimateGenerator, StationProfile

#: Correlation e-folding distance in kilometres. Synoptic pressure and
#: temperature fields decorrelate over a few hundred kilometres in the plains;
#: 250 km is a conventional working value for the Indo-Gangetic region.
CORRELATION_LENGTH_KM = 250.0

#: Extra decorrelation per kilometre of elevation difference. Stations at very
#: different heights sample different air masses even when they are close on a
#: map, which is exactly the Shimla-versus-Ambala case.
ELEVATION_PENALTY_PER_KM = 0.55


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def correlation(a: StationProfile, b: StationProfile) -> float:
    """Expected synoptic correlation between two stations, in [0, 1]."""
    distance = haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
    elevation_km = abs(a.altitude_m - b.altitude_m) / 1000.0
    exponent = distance / CORRELATION_LENGTH_KM + ELEVATION_PENALTY_PER_KM * elevation_km
    return float(math.exp(-exponent))


class NetworkGenerator:
    """Generates a whole AWS network with shared synoptic weather.

    Each station's marginal distribution matches what
    :class:`~skyguard.synth.climate.ClimateGenerator` produces alone; the only
    difference is that departures are now correlated between stations.
    """

    def __init__(
        self,
        profiles: tuple[StationProfile, ...],
        start_timestamp: float,
        interval_seconds: int = 600,
        seed: int = 0,
    ) -> None:
        self.profiles = profiles
        self.interval = interval_seconds
        self._rng = np.random.default_rng(seed)
        self._generators = {
            p.station_id: ClimateGenerator(p, start_timestamp, interval_seconds, seed=seed + i)
            for i, p in enumerate(profiles)
        }
        # Correlation is taken against the first profile, which acts as the
        # network's reference point. For a compact regional network this is a
        # good approximation to a full covariance model and costs nothing.
        anchor = profiles[0]
        self._rho = {p.station_id: correlation(anchor, p) for p in profiles}

        phi = anchor.synoptic_persistence
        self._phi = phi
        self._innovation = math.sqrt(1.0 - phi * phi)
        self._shared = {
            "temperature": float(self._rng.normal()),
            "pressure": float(self._rng.normal()),
            "dewpoint": float(self._rng.normal()),
        }

    def step(self) -> list[Observation]:
        """Advance every station by one interval, sharing the synoptic driver."""
        rng = self._rng
        for key in self._shared:
            self._shared[key] = (
                self._phi * self._shared[key] + self._innovation * float(rng.normal())
            )

        out: list[Observation] = []
        for profile in self.profiles:
            gen = self._generators[profile.station_id]
            rho = self._rho[profile.station_id]
            w_shared, w_local = math.sqrt(rho), math.sqrt(1.0 - rho)

            observation = gen.step()

            # Blend the station's own (already advanced) local departure with
            # the shared one. Reaching into the generator's departure state
            # keeps a single implementation of the climatology instead of
            # duplicating it here, which is why those attributes are not
            # name-mangled.
            blend = {
                "_d_temp": ("temperature", profile.synoptic_temp_sigma_c),
                "_d_pres": ("pressure", profile.synoptic_pressure_sigma_hpa),
                "_d_dewp": ("dewpoint", profile.synoptic_dewpoint_sigma_c),
            }
            for attribute, (shared_key, sigma) in blend.items():
                local = getattr(gen, attribute)
                shared = self._shared[shared_key] * sigma
                setattr(gen, attribute, w_local * local + w_shared * shared)

            out.append(observation)
        return out

    def generate(self, count: int) -> dict[str, list[Observation]]:
        """Return `count` samples for every station, keyed by station id."""
        series: dict[str, list[Observation]] = {p.station_id: [] for p in self.profiles}
        for _ in range(count):
            for observation in self.step():
                series[observation.station_id].append(observation)
        return series

    def interleaved(self, count: int) -> list[Observation]:
        """Return all observations in timestamp order, as the network would report."""
        rows: list[Observation] = []
        for _ in range(count):
            rows.extend(self.step())
        return rows
