"""Regression tests for the two feature-layer bugs that cost the most time."""

import math

import pytest

from skyguard.features.online import StationState, squash
from skyguard.models import CHANNELS, Channel


def _state(floor=0.0866):
    return StationState(
        robust_window=24,
        sequence_window=18,
        drift_window=200,
        scale_floor={c: floor for c in CHANNELS},
    )


def test_forecast_tracks_a_ramp():
    """A trailing median lags a ramp; a local linear forecast must not.

    This is the bug that produced a 90 % false-positive rate: every clean
    sunrise scored as a multi-sigma anomaly against a lagging baseline.
    """
    state = _state()
    for i in range(12):
        state.observe(Channel.TEMPERATURE, 10.0 + 0.5 * i)
    prediction = state.predict(Channel.TEMPERATURE)
    assert prediction == pytest.approx(16.0, abs=0.2)


def test_scale_floor_survives_identical_readings():
    """A quantised, slow-moving channel reports identical values.

    Without a resolution floor the residual MAD is exactly zero, every score
    divides by ~0 and the detector saturates permanently.
    """
    state = _state(floor=0.0866)
    for _ in range(200):
        state.observe(Channel.PRESSURE, 1004.2)
    scale = state.residual_scale(Channel.PRESSURE)
    assert scale >= 0.0866
    assert math.isfinite(state.residual_z(Channel.PRESSURE, 1004.3))


def test_squash_is_centred_on_its_threshold():
    assert squash(4.0, 4.0) == pytest.approx(0.5, abs=1e-9)
    assert squash(0.0, 4.0) < 0.01
    assert squash(400.0, 4.0) > 0.99


def test_squash_does_not_overflow():
    assert squash(1e12, 1.0) == pytest.approx(1.0)
    assert squash(-1e12, 1.0) == pytest.approx(0.0)
