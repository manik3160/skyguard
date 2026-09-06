"""The repair loop diverged in testing; these tests keep it bounded."""


from skyguard.config import PhysicalLimits
from skyguard.features.online import StationState
from skyguard.impute.repair import Repairer
from skyguard.models import CHANNELS, Channel, Observation


def _state():
    state = StationState(24, 18, 200, {c: 0.1 for c in CHANNELS})
    for i in range(24):
        state.observe(Channel.PRESSURE, 1000.0 + 0.1 * i)
        state.observe(Channel.TEMPERATURE, 20.0 + 0.05 * i)
        state.observe(Channel.HUMIDITY, 50.0)
    return state


def test_long_outage_does_not_run_away():
    """Undamped repair walked pressure from 985 to 877 hPa inside one episode."""
    state = _state()
    repairer = Repairer(PhysicalLimits())
    flagged = frozenset({Channel.PRESSURE})
    values = []
    for i in range(300):
        obs = Observation("S", 1_700_000_000.0 + 600 * i, 20.0, 1002.3, 50.0)
        result = repairer.repair(obs, flagged, state)
        values.append(result["pressure"])
        state.observe(Channel.PRESSURE, result["pressure"])

    # Damping caps the total trend contribution, so the estimate must stay
    # close to its anchor no matter how long the fault lasts.
    assert max(values) - min(values) < 5.0
    assert all(900.0 < v < 1100.0 for v in values)


def test_repairs_are_clamped_to_physical_limits():
    state = _state()
    repairer = Repairer(PhysicalLimits())
    obs = Observation("S", 1_700_000_000.0, 20.0, 1002.3, 50.0)
    result = repairer.repair(obs, frozenset({Channel.HUMIDITY}), state)
    assert 0.0 <= result["humidity"] <= 100.0


def test_humidity_is_reconstructed_from_dewpoint_when_temperature_is_sound():
    """Dewpoint is an air-mass property and far more persistent than RH."""
    state = _state()
    repairer = Repairer(PhysicalLimits())
    repairer.note_clean(Observation("S", 0.0, 20.0, 1000.0, 60.0))
    obs = Observation("S", 600.0, 25.0, 1000.0, 5.0)
    result = repairer.repair(obs, frozenset({Channel.HUMIDITY}), state)
    # Warmer air, same dewpoint -> lower RH, and it must track rather than hold.
    assert 40.0 < result["humidity"] < 50.0
