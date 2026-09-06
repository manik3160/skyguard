"""Synthetic AWS generator producing physically coherent T / P / RH series.

Why synthesise rather than download: the problem statement ships no dataset and
states that evaluation happens on *anomaly-injected* data. To measure detection
accuracy we need ground-truth labels, which no public archive provides. So we
generate a stream whose statistics match Indian AWS records, inject faults with
known extents, and score against them. `skyguard.synth.replay` loads real
archives when they are available; the detector cannot tell the difference
because both emit `Observation`.

Construction
------------
Temperature and dewpoint are the free variables; humidity is *derived* from
them. That ordering matters: it guarantees RH stays thermodynamically
consistent with T, so any inconsistency the detector later finds is a fault we
injected rather than an artefact of the generator.

    T(t)  = annual + diurnal + synoptic AR(1) + turbulence
    Td(t) = airmass base + slow AR(1), clipped to Td <= T
    RH(t) = f(T, Td)                             [physics.humidity_from_dewpoint]
    P(t)  = altitude base + synoptic AR(1) + semidiurnal tide + noise
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..models import Observation
from ..physics import humidity_from_dewpoint, pressure_at_altitude

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class StationProfile:
    """Climatological identity of one station.

    Defaults describe a semi-arid station on the Indo-Gangetic plain, which is
    what most IMD AWS sites in Haryana / Punjab look like.
    """

    station_id: str
    name: str
    latitude: float
    longitude: float
    altitude_m: float = 220.0

    annual_mean_c: float = 25.0
    annual_amplitude_c: float = 11.0      # summer-winter half-range
    diurnal_amplitude_c: float = 7.5      # day-night half-range
    #: Hour of local solar time at which temperature peaks.
    diurnal_peak_hour: float = 15.0
    #: Day-of-year at which the annual cycle peaks (mid-May pre-monsoon).
    annual_peak_doy: float = 135.0

    dewpoint_base_c: float = 12.0
    dewpoint_amplitude_c: float = 9.0     # monsoon moisture swing

    #: Persistence of synoptic-scale departures, per 10-minute step.
    synoptic_persistence: float = 0.995
    synoptic_temp_sigma_c: float = 2.4
    synoptic_pressure_sigma_hpa: float = 4.5
    synoptic_dewpoint_sigma_c: float = 2.0

    #: Sensor resolution -- readings are quantised to these steps.
    temperature_resolution_c: float = 0.1
    pressure_resolution_hpa: float = 0.1
    humidity_resolution_pct: float = 1.0

    #: White measurement noise, one standard deviation.
    temperature_noise_c: float = 0.08
    pressure_noise_hpa: float = 0.05
    humidity_noise_pct: float = 0.6


#: A small network with contrasting climates, used for spatial-consistency demos.
DEMO_NETWORK: tuple[StationProfile, ...] = (
    StationProfile(
        "AWS-HR-AMB", "Ambala", 30.38, 76.78, altitude_m=272.0,
        annual_mean_c=24.5, annual_amplitude_c=11.5, diurnal_amplitude_c=8.0,
        dewpoint_base_c=12.0, dewpoint_amplitude_c=9.5,
    ),
    StationProfile(
        "AWS-HR-KNL", "Karnal", 29.69, 76.99, altitude_m=245.0,
        annual_mean_c=25.0, annual_amplitude_c=11.2, diurnal_amplitude_c=7.8,
        dewpoint_base_c=12.5, dewpoint_amplitude_c=9.2,
    ),
    StationProfile(
        "AWS-HR-HSR", "Hisar", 29.15, 75.72, altitude_m=215.0,
        annual_mean_c=26.0, annual_amplitude_c=12.5, diurnal_amplitude_c=9.5,
        dewpoint_base_c=10.0, dewpoint_amplitude_c=10.5,
    ),
    StationProfile(
        "AWS-HP-SML", "Shimla", 31.10, 77.17, altitude_m=2202.0,
        annual_mean_c=13.0, annual_amplitude_c=8.5, diurnal_amplitude_c=5.0,
        dewpoint_base_c=5.0, dewpoint_amplitude_c=7.0,
    ),
)


class ClimateGenerator:
    """Generates one station's clean signal.

    Stateful and sequential: each call to :meth:`step` advances the AR(1)
    processes. That mirrors how a real logger emits data and lets the same code
    drive both the offline benchmark and the live demo.
    """

    def __init__(
        self,
        profile: StationProfile,
        start_timestamp: float,
        interval_seconds: int = 600,
        seed: int = 0,
    ) -> None:
        self.profile = profile
        self.interval = interval_seconds
        self._t = start_timestamp
        self._rng = np.random.default_rng(seed)

        # AR(1) departure states, initialised at their stationary variance so
        # the series does not need a burn-in.
        self._d_temp = float(self._rng.normal(0.0, profile.synoptic_temp_sigma_c))
        self._d_pres = float(self._rng.normal(0.0, profile.synoptic_pressure_sigma_hpa))
        self._d_dewp = float(self._rng.normal(0.0, profile.synoptic_dewpoint_sigma_c))

    # -- deterministic climatology -----------------------------------------

    def _annual_phase(self, ts: float) -> float:
        doy = (ts % SECONDS_PER_YEAR) / SECONDS_PER_DAY
        return 2 * math.pi * (doy - self.profile.annual_peak_doy) / 365.25

    def _diurnal_phase(self, ts: float) -> float:
        # Local solar time from longitude: 15 degrees of longitude per hour.
        solar_offset = self.profile.longitude / 15.0 * 3600.0
        local = (ts + solar_offset) % SECONDS_PER_DAY
        hour = local / 3600.0
        return 2 * math.pi * (hour - self.profile.diurnal_peak_hour) / 24.0

    def climatological_temperature(self, ts: float) -> float:
        p = self.profile
        return (
            p.annual_mean_c
            + p.annual_amplitude_c * math.cos(self._annual_phase(ts))
            + p.diurnal_amplitude_c * math.cos(self._diurnal_phase(ts))
        )

    def _climatological_dewpoint(self, ts: float) -> float:
        p = self.profile
        # Moisture peaks in the monsoon, roughly 45 days after peak heat.
        phase = self._annual_phase(ts) - 2 * math.pi * 45.0 / 365.25
        return p.dewpoint_base_c + p.dewpoint_amplitude_c * math.cos(phase)

    def _climatological_pressure(self, ts: float) -> float:
        p = self.profile
        base = pressure_at_altitude(1013.25, p.altitude_m)
        # Annual cycle: winter high, monsoon low, about 8 hPa peak-to-peak.
        annual = -4.0 * math.cos(self._annual_phase(ts))
        # Atmospheric semidiurnal tide, maxima near 10 and 22 local solar time.
        tide = 1.1 * math.cos(2 * (self._diurnal_phase(ts) + 2 * math.pi * 5.0 / 24.0))
        return base + annual + tide

    # -- stepping -----------------------------------------------------------

    def step(self) -> Observation:
        p = self.profile
        rng = self._rng
        ts = self._t
        self._t += self.interval

        # Advance the AR(1) departures. The innovation is scaled so the
        # stationary variance equals the configured sigma regardless of phi.
        phi = p.synoptic_persistence
        inn = math.sqrt(1.0 - phi * phi)
        self._d_temp = phi * self._d_temp + inn * rng.normal(0.0, p.synoptic_temp_sigma_c)
        self._d_pres = phi * self._d_pres + inn * rng.normal(
            0.0, p.synoptic_pressure_sigma_hpa
        )
        self._d_dewp = phi * self._d_dewp + inn * rng.normal(0.0, p.synoptic_dewpoint_sigma_c)

        temperature = (
            self.climatological_temperature(ts)
            + self._d_temp
            + rng.normal(0.0, p.temperature_noise_c)
        )
        pressure = (
            self._climatological_pressure(ts)
            + self._d_pres
            + rng.normal(0.0, p.pressure_noise_hpa)
        )

        # Dewpoint cannot exceed air temperature. Saturating rather than
        # clipping keeps the transition smooth, which is what fog looks like.
        dewpoint = self._climatological_dewpoint(ts) + self._d_dewp
        if dewpoint > temperature - 0.2:
            dewpoint = temperature - 0.2 * math.exp(-(dewpoint - temperature))

        humidity = humidity_from_dewpoint(temperature, dewpoint)
        humidity += rng.normal(0.0, p.humidity_noise_pct)
        humidity = min(max(humidity, 1.0), 100.0)

        return Observation(
            station_id=p.station_id,
            timestamp=ts,
            temperature=_quantise(temperature, p.temperature_resolution_c),
            pressure=_quantise(pressure, p.pressure_resolution_hpa),
            humidity=_quantise(humidity, p.humidity_resolution_pct),
        )

    def generate(self, count: int) -> list[Observation]:
        return [self.step() for _ in range(count)]


def _quantise(value: float, step: float) -> float:
    """Round to the sensor's reporting resolution.

    Real loggers report quantised values. Skipping this makes the data
    unrealistically smooth and inflates detector accuracy.
    """
    return round(round(value / step) * step, 4)
