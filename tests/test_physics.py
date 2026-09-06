"""Thermodynamic relations must round-trip and respect saturation."""

import itertools

import pytest

from skyguard.physics import (
    dewpoint,
    humidity_from_dewpoint,
    pressure_at_altitude,
    saturation_vapour_pressure,
)


@pytest.mark.parametrize("temperature", [-20.0, 0.0, 15.0, 30.0, 45.0])
@pytest.mark.parametrize("humidity", [5.0, 35.0, 70.0, 99.0])
def test_dewpoint_round_trips(temperature, humidity):
    """RH -> Td -> RH must return the original within instrument tolerance."""
    td = dewpoint(temperature, humidity)
    assert humidity_from_dewpoint(temperature, td) == pytest.approx(humidity, abs=0.1)


@pytest.mark.parametrize("temperature", [-10.0, 10.0, 40.0])
def test_dewpoint_never_exceeds_temperature_below_saturation(temperature):
    """This invariant is the whole basis of the L0 cross-sensor check."""
    assert dewpoint(temperature, 99.9) <= temperature + 1e-6


def test_saturation_pressure_increases_with_temperature():
    values = [saturation_vapour_pressure(t) for t in range(-30, 50, 5)]
    assert all(b > a for a, b in itertools.pairwise(values))


def test_saturation_pressure_at_zero_matches_reference():
    """6.112 hPa at 0 degC is the WMO reference value."""
    assert saturation_vapour_pressure(0.0) == pytest.approx(6.112, abs=1e-3)


def test_pressure_falls_with_altitude():
    sea = pressure_at_altitude(1013.25, 0.0)
    hill = pressure_at_altitude(1013.25, 2200.0)
    assert sea == pytest.approx(1013.25, abs=0.01)
    assert 760.0 < hill < 790.0  # Shimla sits near 775 hPa
