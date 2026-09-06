"""Streaming feature extraction.

Every statistic here is computed from a bounded ring buffer in O(window) time
with no allocation in the hot path. That is a hard requirement: the problem
statement weights real-time capability at 15 % and asks for deployment on an
ESP32, where an unbounded pandas rolling window is not an option.

The baseline is a one-step-ahead forecast, not a rolling median
--------------------------------------------------------------
The obvious design -- compare each reading to the median of a trailing window --
fails on this data, and the failure is instructive. Temperature has an 8 degC
diurnal swing, so during the morning ramp a 4-hour trailing median sits several
degrees below the current value. Every clean sunrise then scores as a multi-
sigma anomaly. Measured on the clean control stream, that design produced a
90 % false-positive rate before it was replaced.

So the baseline is a local *linear* extrapolation: fit a line to the last few
samples, predict the next one, and test the residual. The diurnal ramp is
absorbed into the slope, leaving a residual that is near zero-mean under normal
conditions. The scale is estimated from the residual's own MAD, so the detector
adapts to an intrinsically noisy station without per-station tuning.

Residuals, not raw values, are also what the multivariate layer consumes --
which is why that layer sees genuine cross-sensor inconsistency rather than
"it is 3 p.m. at all three sensors simultaneously".
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..models import CHANNELS, MISSING_SENTINEL, Channel

#: Consistency constant making MAD an unbiased estimator of sigma for Gaussians.
MAD_TO_SIGMA = 1.4826

#: Samples used for the local linear fit. Six samples = one hour at the IMD
#: cadence: long enough to average out measurement noise, short enough that a
#: straight line is a good local model of a sinusoidal diurnal cycle.
TREND_SPAN = 6


def is_missing(value: float) -> bool:
    return value is None or value != value or value == MISSING_SENTINEL


def median_absolute_deviation(window: np.ndarray) -> tuple[float, float]:
    """Return (median, MAD-scaled sigma) for a 1-D window.

    Chosen over mean/stdev because a single spike moves the mean and inflates
    the stdev enough to hide itself -- the classic masking failure that makes
    threshold quality control miss the very events it exists to catch.
    """
    if window.size == 0:
        return math.nan, math.nan
    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med)))
    return med, mad * MAD_TO_SIGMA


class RingBuffer:
    """Bounded history with a cached robust median and scale."""

    __slots__ = ("_buf", "_capacity", "_dirty", "_median", "_sigma")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf: deque[float] = deque(maxlen=capacity)
        self._dirty = True
        self._median = math.nan
        self._sigma = math.nan

    def push(self, value: float) -> None:
        if is_missing(value):
            return
        self._buf.append(float(value))
        self._dirty = True

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def ready(self) -> bool:
        # Half a window gives a usable estimate and avoids a long dead zone at
        # station commissioning.
        return len(self._buf) >= max(4, self._capacity // 2)

    def as_array(self) -> np.ndarray:
        return np.fromiter(self._buf, dtype=float, count=len(self._buf))

    def _refresh(self) -> None:
        if not self._dirty:
            return
        self._median, self._sigma = median_absolute_deviation(self.as_array())
        self._dirty = False

    @property
    def median(self) -> float:
        self._refresh()
        return self._median

    @property
    def sigma(self) -> float:
        """Robust scale, floored so a quantised-flat window cannot divide by zero."""
        self._refresh()
        if not math.isfinite(self._sigma) or self._sigma < 1e-9:
            return 1e-9
        return self._sigma

    @property
    def last(self) -> float:
        return self._buf[-1] if self._buf else math.nan

    def span(self, n: int) -> float:
        """Peak-to-peak range over the most recent `n` values."""
        if len(self._buf) < 2:
            return math.nan
        recent = list(self._buf)[-n:]
        return max(recent) - min(recent)


class StationState:
    """All rolling state for one station, across all three channels.

    Buffer families per channel:

    ``raw``       recent accepted values, used to build the forecast
    ``residual``  forecast errors, used to estimate the noise scale
    ``sequence``  window fed to the L3 subspace detector
    ``long``      multi-day history, used for drift estimation
    """

    def __init__(
        self,
        robust_window: int,
        sequence_window: int,
        drift_window: int,
        scale_floor: dict[Channel, float] | None = None,
    ) -> None:
        # Per-channel lower bound on the residual scale, derived from sensor
        # resolution. Without it, a slowly-varying quantised channel produces
        # identical consecutive reports, a zero median absolute residual, and
        # an infinite standardised score for every subsequent sample.
        self.scale_floor: dict[Channel, float] = scale_floor or {c: 1e-3 for c in CHANNELS}
        self.raw: dict[Channel, RingBuffer] = {
            c: RingBuffer(robust_window) for c in CHANNELS
        }
        # Four windows of residuals: the scale estimate should be steadier than
        # the forecast it normalises, or the detector chases its own noise.
        self.residual: dict[Channel, RingBuffer] = {
            c: RingBuffer(robust_window * 4) for c in CHANNELS
        }
        self.sequence: dict[Channel, RingBuffer] = {
            c: RingBuffer(sequence_window) for c in CHANNELS
        }
        self.long: dict[Channel, RingBuffer] = {
            c: RingBuffer(drift_window) for c in CHANNELS
        }
        # As-reported values, including ones the pipeline rejected. The stuck
        # detector must read this rather than `raw`: once a frozen channel is
        # flagged, the pipeline substitutes a repaired estimate into `raw`,
        # which un-freezes it and would silence the very detector that found it.
        self.reported: dict[Channel, RingBuffer] = {
            c: RingBuffer(robust_window) for c in CHANNELS
        }
        self.last_timestamp: float | None = None
        self.last_clean: dict[Channel, float] = {}
        self.samples_seen = 0

    # -- updates -----------------------------------------------------------

    def record_reported(self, channel: Channel, value: float) -> None:
        """Log the value the logger actually sent, accepted or not."""
        self.reported[channel].push(value)

    def observe(self, channel: Channel, value: float) -> None:
        """Record an accepted value and the residual it produced.

        Order matters: the residual is measured against the forecast made
        *before* this value enters the buffer. Reversing it would shrink the
        scale estimate toward zero and make every subsequent reading look
        anomalous.
        """
        if is_missing(value):
            return
        prediction = self.predict(channel)
        if math.isfinite(prediction):
            self.residual[channel].push(float(value) - prediction)
        self.raw[channel].push(value)
        self.sequence[channel].push(value)
        self.long[channel].push(value)
        self.last_clean[channel] = float(value)

    # -- forecasting -------------------------------------------------------

    def predict(self, channel: Channel) -> float:
        """One-step-ahead forecast by local linear extrapolation.

        Falls back to persistence (repeat the last value) while the buffer is
        too short to fit a slope. Persistence is a strong baseline for
        meteorological series at a 10-minute cadence, so the fallback is a
        slightly noisier mode rather than a broken one.
        """
        buf = self.raw[channel]
        n = len(buf)
        if n == 0:
            return math.nan
        if n < 3:
            return buf.last

        span = min(TREND_SPAN, n)
        recent = buf.as_array()[-span:]
        x = np.arange(span, dtype=float)
        # Closed-form OLS slope: cheaper than polyfit and allocation-light.
        x_mean = x.mean()
        y_mean = float(recent.mean())
        denom = float(((x - x_mean) ** 2).sum())
        if denom <= 0:
            return y_mean
        slope = float(((x - x_mean) * (recent - y_mean)).sum() / denom)
        intercept = y_mean - slope * x_mean
        return float(intercept + slope * span)

    def residual_scale(self, channel: Channel) -> float:
        """Robust standard deviation of recent forecast errors."""
        buf = self.residual[channel]
        if not buf.ready:
            return math.nan
        # Centred on zero rather than on the residual median: a non-zero median
        # is itself evidence of bias (drift), and re-centring would hide it.
        arr = buf.as_array()
        mad = float(np.median(np.abs(arr)))
        return max(mad * MAD_TO_SIGMA, self.scale_floor[channel])

    def residual_z(self, channel: Channel, value: float) -> float:
        """Signed standardised forecast error -- the core anomaly statistic."""
        if is_missing(value):
            return 0.0
        prediction = self.predict(channel)
        scale = self.residual_scale(channel)
        if not math.isfinite(prediction) or not math.isfinite(scale):
            return 0.0
        return (float(value) - prediction) / scale

    @property
    def warm(self) -> bool:
        return all(b.ready for b in self.residual.values())

    def rate_per_minute(self, channel: Channel, value: float, timestamp: float) -> float:
        """First difference normalised to units per minute.

        Measured against the previous *as-reported* value, deliberately, not
        against the pipeline's accepted estimate. This test asks whether the
        instrument physically slewed faster than it can. That is a property of
        the sensor's own output; comparing against a repaired estimate would
        mean a drifting repair makes the physics gate fire, which inverts the
        entire layering premise -- L0 is supposed to be the layer that cannot
        be wrong. This bug flagged 60 % of a clean stream before it was found.
        """
        buf = self.reported[channel]
        # Two entries needed: the last one is the sample under test.
        prev = buf.as_array()[-2] if len(buf) >= 2 else None
        if prev is None or self.last_timestamp is None or is_missing(value):
            return 0.0
        dt_min = (timestamp - self.last_timestamp) / 60.0
        if dt_min <= 0:
            return 0.0
        return (float(value) - prev) / dt_min


def squash(statistic: float, threshold: float, sharpness: float = 1.5) -> float:
    """Map an unbounded test statistic to a calibrated score in [0, 1].

    A logistic centred on the threshold, so `statistic == threshold` scores
    exactly 0.5. Fusion can then treat every detector's output as comparable
    evidence without knowing whether it came from a z-score or a chi-square.
    """
    if not math.isfinite(statistic) or threshold <= 0:
        return 0.0
    x = sharpness * (statistic / threshold - 1.0)
    # Clamp the exponent: exp(700) overflows float64 and a saturated score is
    # the correct answer anyway.
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-4.0 * x))
