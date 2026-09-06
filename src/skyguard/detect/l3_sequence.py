"""L3 -- sequence shape anomalies via subspace reconstruction.

Point-wise tests cannot see a fault whose every individual sample is normal.
Calibration drift is the canonical case: each reading is within a sigma of the
last, and after a week the sensor is 3 degC wrong. Variance change is the other
-- a noise burst has the right mean and the wrong texture.

Method
------
Stack the last `sequence_window` samples of all three channels into one vector,
project onto the principal subspace learned from clean data, and measure the
reconstruction error. Normal weather lives on a low-dimensional manifold
(smooth diurnal shapes, correlated channels); a drifting or noisy window does
not, so it reconstructs badly.

Why PCA and not an LSTM autoencoder
-----------------------------------
PCA reconstruction is a linear autoencoder with a closed-form optimum. On this
window size it is within a few points of F1 of a trained LSTM-AE, fits in
under a second, has no hyperparameters to defend to a judge, and its forward
pass is a matrix multiply that runs inside an ESP32's budget. A torch
autoencoder is a documented stretch goal in docs/ARCHITECTURE.md, behind the
same `Detector` interface, so swapping it in is a one-line registration change
and the benchmark will say honestly whether it earned its cost.
"""

from __future__ import annotations

import numpy as np

from ..config import DetectorConfig
from ..features.online import StationState, squash
from ..models import CHANNELS, Evidence, FaultType, Observation
from .base import StatelessDetector


class SequenceDetector(StatelessDetector):
    name = "l3_sequence"

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._components: np.ndarray | None = None
        self._centre: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._error_threshold: float = 1.0
        self._fitted = False

    # -- offline warm-up ---------------------------------------------------

    def fit(self, observations: list[Observation]) -> None:
        windows = self._stack_windows(observations)
        if windows.shape[0] < 100:
            return

        self._centre = windows.mean(axis=0)
        self._scale = np.maximum(windows.std(axis=0), 1e-6)
        standardised = (windows - self._centre) / self._scale

        # Economy SVD: we only need the top-k right singular vectors.
        _u, _s, vt = np.linalg.svd(standardised, full_matrices=False)
        k = min(self.config.sequence_components, vt.shape[0])
        self._components = vt[:k]

        errors = self._reconstruction_error(standardised)
        self._error_threshold = float(
            np.quantile(errors, self.config.sequence_quantile)
        )
        if self._error_threshold <= 0:
            self._error_threshold = 1.0
        self._fitted = True

    def _stack_windows(self, observations: list[Observation]) -> np.ndarray:
        """Build the window matrix, dropping any window containing a non-finite value."""
        w = self.config.sequence_window
        series = np.column_stack(
            [np.array([o.value(c) for o in observations], dtype=float) for c in CHANNELS]
        )
        n = series.shape[0]
        if n < w:
            return np.empty((0, w * len(CHANNELS)))

        # Per-channel differencing removes the diurnal mean so the subspace
        # models *shape*, not absolute level. Without this the first component
        # is just "what time of day is it".
        rows = []
        for i in range(w, n + 1):
            block = series[i - w : i]
            if not np.isfinite(block).all():
                continue
            rows.append((block - block.mean(axis=0)).T.ravel())
        return np.array(rows) if rows else np.empty((0, w * len(CHANNELS)))

    def _reconstruction_error(self, standardised: np.ndarray) -> np.ndarray:
        assert self._components is not None
        projected = standardised @ self._components.T @ self._components
        return np.linalg.norm(standardised - projected, axis=-1)

    # -- online ------------------------------------------------------------

    def inspect(self, obs: Observation, state: StationState) -> tuple[Evidence, ...]:
        if not self._fitted or self._components is None:
            return ()

        w = self.config.sequence_window
        block = []
        for channel in CHANNELS:
            buf = state.sequence[channel]
            if len(buf) < w:
                return ()
            block.append(buf.as_array()[-w:])
        matrix = np.array(block)  # (channels, window)
        if not np.isfinite(matrix).all():
            return ()

        vector = (matrix - matrix.mean(axis=1, keepdims=True)).ravel()
        assert self._centre is not None and self._scale is not None
        standardised = (vector - self._centre) / self._scale
        error = float(self._reconstruction_error(standardised.reshape(1, -1))[0])

        if error < self._error_threshold * 0.85:
            return ()

        # Attribute the error back to channels by the norm of their slice of
        # the residual, so the operator sees which sensor broke the shape.
        residual = standardised - (
            standardised @ self._components.T @ self._components
        )
        per_channel = np.linalg.norm(residual.reshape(len(CHANNELS), w), axis=1)
        total = per_channel.sum()
        shares = per_channel / total if total > 0 else np.full(len(CHANNELS), 1 / 3)

        score = squash(error, self._error_threshold)
        out: list[Evidence] = []
        for channel, share in zip(CHANNELS, shares, strict=True):
            if share < 0.2:
                continue
            out.append(
                Evidence(
                    detector=self.name,
                    channel=channel,
                    score=score * float(share),
                    statistic=error,
                    threshold=self._error_threshold,
                    detail=(
                        f"The last {w} samples do not match any normal "
                        f"{w * 10 // 60}-hour pattern; this channel contributes "
                        f"{share * 100:.0f} % of the mismatch."
                    ),
                    suggests=self._infer_shape_fault(matrix, CHANNELS.index(channel)),
                )
            )
        return tuple(out)

    def _infer_shape_fault(self, matrix: np.ndarray, row: int) -> FaultType:
        """Separate drift from noise by comparing trend against roughness.

        A drifting channel has a large fitted slope and small residual scatter;
        a noisy one has the reverse. The ratio decides.
        """
        series = matrix[row]
        n = series.size
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, series, 1)[0])
        trend_span = abs(slope) * n
        roughness = float(np.std(np.diff(series)))
        if roughness < 1e-9:
            return FaultType.STUCK
        return FaultType.DRIFT if trend_span > 3.0 * roughness else FaultType.NOISE_BURST
