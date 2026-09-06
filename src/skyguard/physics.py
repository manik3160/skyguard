"""Thermodynamic relations for temperature / pressure / humidity.

The same functions are used by the synthetic generator (to produce physically
coherent data) and by the L0 physics gate (to detect incoherent data). Keeping
one implementation means the gate cannot be accidentally "tuned" to a
generator quirk -- if a relation is wrong, both sides are wrong together and
the tests catch it.

References
----------
Magnus-Tetens coefficients: World Meteorological Organization, *Guide to
Meteorological Instruments and Methods of Observation* (WMO-No. 8), Annex 4.B.
"""

from __future__ import annotations

import math

# Magnus coefficients over water, valid -45 degC .. +60 degC
_MAGNUS_A = 17.62
_MAGNUS_B = 243.12  # degC
_E0 = 6.112  # hPa, saturation vapour pressure at 0 degC


def saturation_vapour_pressure(temperature_c: float) -> float:
    """Saturation vapour pressure over water, in hPa."""
    return _E0 * math.exp(_MAGNUS_A * temperature_c / (_MAGNUS_B + temperature_c))


def vapour_pressure(temperature_c: float, humidity_pct: float) -> float:
    """Actual vapour pressure, in hPa."""
    return saturation_vapour_pressure(temperature_c) * (humidity_pct / 100.0)


def dewpoint(temperature_c: float, humidity_pct: float) -> float:
    """Dewpoint in degC from temperature and relative humidity.

    Humidity is clamped to a small positive floor because RH == 0 sends the
    logarithm to negative infinity, and real hygrometers never report a true
    zero.
    """
    rh = max(min(humidity_pct, 100.0), 0.1)
    gamma = math.log(rh / 100.0) + _MAGNUS_A * temperature_c / (_MAGNUS_B + temperature_c)
    return _MAGNUS_B * gamma / (_MAGNUS_A - gamma)


def humidity_from_dewpoint(temperature_c: float, dewpoint_c: float) -> float:
    """Relative humidity (%) implied by a temperature / dewpoint pair."""
    ratio = saturation_vapour_pressure(dewpoint_c) / saturation_vapour_pressure(temperature_c)
    return max(min(100.0 * ratio, 100.0), 0.0)


def pressure_at_altitude(sea_level_hpa: float, altitude_m: float) -> float:
    """Barometric formula for a standard atmosphere (troposphere)."""
    return sea_level_hpa * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588


def reduce_to_sea_level(station_hpa: float, altitude_m: float, temperature_c: float) -> float:
    """Reduce a station pressure to its mean-sea-level equivalent.

    Used so that stations at different elevations can be compared directly in
    the spatial-consistency check.
    """
    return station_hpa * math.exp(altitude_m / (29.271 * (temperature_c + 273.15)))
