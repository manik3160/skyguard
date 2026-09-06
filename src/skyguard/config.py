"""Tunable parameters, in one place, with the reason for each value.

Rule for this file: every number carries a comment saying where it came from --
a WMO limit, a physical bound, or a threshold calibrated on the validation
split by `skyguard.evaluation.calibrate`. A number with no provenance is a
magic number and does not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PhysicalLimits:
    """Hard bounds. Outside these, the reading cannot be real anywhere on Earth."""

    # WMO-No. 8 gross-error limits, widened to cover Indian extremes
    # (Phalodi 51.0 degC in 2016; Dras -45 degC).
    temperature_min_c: float = -50.0
    temperature_max_c: float = 58.0

    # Station pressure at sea level vs. Himalayan AWS sites above 5000 m.
    pressure_min_hpa: float = 450.0
    pressure_max_hpa: float = 1085.0

    humidity_min_pct: float = 0.0
    humidity_max_pct: float = 100.0

    # Step limits per WMO temporal-consistency check, for a 10-minute report.
    # Values are per minute so the gate stays correct if the interval changes.
    max_temperature_rate_c_per_min: float = 0.5    # 5 degC / 10 min
    max_pressure_rate_hpa_per_min: float = 0.3     # 3 hPa / 10 min
    max_humidity_rate_pct_per_min: float = 5.0     # 50 % / 10 min

    # Dewpoint can exceed air temperature only by instrument tolerance.
    dewpoint_excess_tolerance_c: float = 0.5


@dataclass(frozen=True, slots=True)
class SensorResolution:
    """Reporting resolution of each sensor, in engineering units.

    This is not a cosmetic detail. It is the floor on any scale estimate: a
    forecast residual smaller than the sensor's own quantisation step is not a
    measurable quantity. Barometers report to 0.1 hPa and pressure moves slowly,
    so consecutive reports are frequently bit-identical; the median absolute
    residual then evaluates to exactly zero and every standardised score
    divides by ~0 and saturates. Flooring the scale here is what prevents that,
    and the floor is a physical fact about the instrument rather than a
    fudge factor.
    """

    temperature: float = 0.1   # degC
    pressure: float = 0.1      # hPa
    humidity: float = 1.0      # %

    def floor(self, channel: str) -> float:
        # A uniformly quantised variable has standard deviation step/sqrt(12).
        # Three times that is a conservative floor which still lets a genuine
        # multi-count departure register.
        step = getattr(self, channel)
        return 3.0 * step / (12.0 ** 0.5)


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Window sizes and decision thresholds for the four detection layers."""

    # ---- L1 temporal ----
    # 24 samples = 4 hours at 10-minute cadence: long enough for a robust
    # median, short enough to track the diurnal cycle.
    robust_window: int = 24
    # Hampel filter: 3 MAD is the textbook default; raised to 4.0 after the
    # validation sweep because 3.0 fired on genuine frontal passages.
    hampel_sigma: float = 4.0
    # A channel that has not moved by more than this over `stuck_window`
    # samples is latched. Floor is set by each sensor's quantisation step.
    stuck_window: int = 12
    stuck_epsilon: dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 0.05,  # 0.1 degC resolution -> half a count
            "pressure": 0.05,     # 0.1 hPa resolution
            "humidity": 0.2,      # 1 % resolution, hygrometers are noisier
        }
    )

    # ---- L2 multivariate ----
    # EWMA half-life for the residual covariance, in samples. 288 = two days.
    covariance_halflife: int = 288
    # Chi-square(3) upper tail: 16.27 is p=0.001, the operating point chosen to
    # hold clean-stream false-positive rate under 0.5 %.
    mahalanobis_threshold: float = 16.27
    isolation_forest_trees: int = 120
    isolation_forest_contamination: float = 0.02

    # ---- L3 sequence ----
    # 18 samples = 3 hours. Shape anomalies (drift, noise burst) need a window
    # longer than the fault's own timescale to be visible.
    sequence_window: int = 18
    sequence_components: int = 8
    # Reconstruction error is squashed with a soft threshold at this quantile
    # of the training-set error distribution.
    sequence_quantile: float = 0.995

    # ---- L4 spatial ----
    # Median standardised pairwise departure (in sigma) above which a buddy-check
    # disagreement is attributed to this station rather than to weather.
    # Instrumented by `skyguard.evaluation.spatial_calibrate` (`make
    # spatial-calibrate`): the clean-stream statistic tops out near 3.6 sigma, so
    # 4.0 is past the whole clean distribution -- but the sweep also shows that
    # lowering it barely moves the benchmark (sigma 3.0 buys +0.007 F1 / +0.007
    # recall for +0.0006 clean FPR; sigma 2.5 trades precision for recall and
    # lifts clean FPR to 0.13). The threshold is not the bottleneck; see
    # CLAUDE.md 8 items 1-2. Kept at 4.0 rather than chase a 0.007 F1 gain.
    spatial_sigma: float = 4.0

    # ---- Fusion ----
    # Fire an alert when fused confidence exceeds this. Calibrated on the
    # validation split to maximise F1 at a false-positive rate below 1 %.
    alert_threshold: float = 0.55
    severity_warning: float = 0.70
    severity_critical: float = 0.88
    # Detector weights in the noisy-OR pool. Physics is trusted absolutely;
    # the learned layers are discounted because they are the ones that can be
    # fooled by an unusual but genuine weather event.
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "l0_physics": 1.00,
            "l1_temporal": 0.85,
            "l2_multivariate": 0.75,
            "l3_sequence": 0.70,
            # Spatial evidence is trusted highly: a departure that every
            # neighbour agrees on cannot be explained by weather, which is the
            # one thing the single-station layers can never rule out.
            "l4_spatial": 0.90,
        }
    )


@dataclass(frozen=True, slots=True)
class HealthConfig:
    """Sensor health index and degradation forecasting."""

    # Anomaly-rate EWMA half-life, in samples. 1008 = one week at 10 minutes.
    anomaly_halflife: int = 1008
    # Health index drops to 0 at this sustained anomaly rate.
    saturating_anomaly_rate: float = 0.25
    # Drift is estimated by Theil-Sen over this many samples (about a week).
    drift_window: int = 1008
    # Recalibration is due when estimated drift reaches this magnitude.
    drift_budget: dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 0.5,   # degC, WMO uncertainty requirement
            "pressure": 0.3,      # hPa
            "humidity": 3.0,      # %
        }
    )
    # Below this index a station is quarantined from downstream assimilation.
    quarantine_index: float = 40.0


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Cadence and network shape."""

    interval_seconds: int = 600  # 10 minutes, the IMD AWS standard reporting cadence
    warmup_samples: int = 288    # two days before detection is trusted


@dataclass(frozen=True, slots=True)
class Settings:
    limits: PhysicalLimits = field(default_factory=PhysicalLimits)
    resolution: SensorResolution = field(default_factory=SensorResolution)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)


DEFAULT = Settings()
