"""Sensor health index and predictive maintenance.

Two questions an operator actually asks:

  "Can I trust this station right now?"   -> health index, 0 to 100
  "When does it need a technician?"       -> days until drift exceeds budget

The health index blends a decaying anomaly rate with the magnitude of estimated
calibration drift. Drift is estimated with the Theil-Sen slope rather than
least squares: least squares is dragged by the very spikes we are trying to
tolerate, so a station with a few impulse faults would appear to be drifting.
"""

from __future__ import annotations

import numpy as np

from ..config import HealthConfig
from ..models import CHANNELS, Channel


class SensorHealth:
    """Per-channel health state for one station."""

    def __init__(self, config: HealthConfig, interval_seconds: int = 600) -> None:
        self.config = config
        self.interval = interval_seconds
        self._alpha = 1.0 - 0.5 ** (1.0 / config.anomaly_halflife)
        self._anomaly_rate: dict[Channel, float] = {c: 0.0 for c in CHANNELS}
        self._drift_per_day: dict[Channel, float] = {c: 0.0 for c in CHANNELS}
        self._samples = 0

    def update(self, flagged: frozenset[Channel]) -> None:
        for channel in CHANNELS:
            hit = 1.0 if channel in flagged else 0.0
            r = self._anomaly_rate[channel]
            self._anomaly_rate[channel] = (1 - self._alpha) * r + self._alpha * hit
        self._samples += 1

    def estimate_drift(self, channel: Channel, history: np.ndarray) -> float:
        """Theil-Sen slope in engineering units per day.

        The diurnal cycle is removed first by differencing at the daily lag;
        without that, the slope over a sub-daily window measures the time of
        day rather than the instrument.
        """
        n = history.size
        per_day = max(int(86400 / self.interval), 1)
        if n < per_day * 3:
            return 0.0

        daily = history[::per_day]
        if daily.size < 3:
            return 0.0

        slope = _theil_sen(daily)
        self._drift_per_day[channel] = float(slope)
        return float(slope)

    def index(self, channel: Channel) -> float:
        """Health on a 0-100 scale.

        Anomaly rate and drift are combined multiplicatively: a station can be
        condemned by either, and a station failing both should score worse than
        one failing either alone.
        """
        rate = self._anomaly_rate[channel]
        rate_term = max(0.0, 1.0 - rate / self.config.saturating_anomaly_rate)

        budget = self.config.drift_budget[channel.value]
        drift_30d = abs(self._drift_per_day[channel]) * 30.0
        drift_term = max(0.0, 1.0 - drift_30d / max(budget, 1e-9))

        return round(100.0 * rate_term * drift_term, 1)

    def station_index(self) -> float:
        """Worst channel governs the station. An AWS is only as good as its weakest sensor."""
        return min(self.index(c) for c in CHANNELS)

    def days_to_maintenance(self, channel: Channel) -> float | None:
        """Days until drift consumes the calibration budget, or None if stable."""
        slope = abs(self._drift_per_day[channel])
        if slope < 1e-6:
            return None
        budget = self.config.drift_budget[channel.value]
        remaining = budget - slope * 30.0
        if remaining <= 0:
            return 0.0
        return round(remaining / slope, 1)

    def quarantined(self) -> bool:
        return self.station_index() < self.config.quarantine_index

    def snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {"station": self.station_index()}
        for channel in CHANNELS:
            out[channel.value] = self.index(channel)
            out[f"{channel.value}_drift_per_day"] = round(self._drift_per_day[channel], 5)
        return out


def _theil_sen(series: np.ndarray) -> float:
    """Median of pairwise slopes. O(n^2) but n is the number of days, so small."""
    n = series.size
    if n < 2:
        return 0.0
    if n > 120:
        series = series[-120:]
        n = 120
    idx = np.arange(n, dtype=float)
    slopes = []
    for i in range(n - 1):
        dx = idx[i + 1 :] - idx[i]
        dy = series[i + 1 :] - series[i]
        valid = (dx != 0) & np.isfinite(dy)
        if valid.any():
            slopes.append(dy[valid] / dx[valid])
    if not slopes:
        return 0.0
    return float(np.median(np.concatenate(slopes)))
