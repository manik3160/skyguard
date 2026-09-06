"""Domain types shared across the SkyGuard pipeline.

Everything that crosses a module boundary is defined here as a frozen dataclass.
Detectors never invent their own dict shapes -- if a field is needed downstream,
it belongs in one of these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------

class Channel(StrEnum):
    """The three sensor channels mandated by problem statement 26073."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    HUMIDITY = "humidity"

    @property
    def unit(self) -> str:
        return {"temperature": "degC", "pressure": "hPa", "humidity": "%"}[self.value]

    @property
    def short(self) -> str:
        return {"temperature": "T", "pressure": "P", "humidity": "RH"}[self.value]


CHANNELS: tuple[Channel, ...] = (
    Channel.TEMPERATURE,
    Channel.PRESSURE,
    Channel.HUMIDITY,
)

#: Sentinel emitted by real AWS loggers when a reading is unavailable.
MISSING_SENTINEL = -999.0


# --------------------------------------------------------------------------
# Fault taxonomy
# --------------------------------------------------------------------------

class FaultType(StrEnum):
    """Root-cause classes the system reports.

    This taxonomy is the contract between the fault injector (which produces
    ground truth), the classifier (which predicts), and the evaluation harness
    (which scores per-class recall). Adding a member here means updating all
    three.
    """

    NONE = "none"
    SPIKE = "spike"                      # isolated impulse, sensor glitch / EMI
    STUCK = "stuck"                      # frozen value, ADC or logger latch-up
    DRIFT = "drift"                      # slow calibration ramp
    OFFSET_STEP = "offset_step"          # abrupt bias, often post-maintenance
    NOISE_BURST = "noise_burst"          # variance inflation, loose wiring
    DROPOUT = "dropout"                  # missing / sentinel, comms failure
    POWER_FLICKER = "power_flicker"      # values collapse toward zero
    PHYSICAL_VIOLATION = "physical_violation"  # breaks thermodynamic constraint

    @property
    def label(self) -> str:
        return {
            "none": "Nominal",
            "spike": "Impulse spike",
            "stuck": "Frozen sensor",
            "drift": "Calibration drift",
            "offset_step": "Bias step",
            "noise_burst": "Noise burst",
            "dropout": "Data dropout",
            "power_flicker": "Power flicker",
            "physical_violation": "Physical violation",
        }[self.value]

    @property
    def maintenance_action(self) -> str:
        """Operator-facing next step. Shown verbatim in the alert drawer."""
        return {
            "none": "No action required.",
            "spike": "Check sensor cabling and grounding for intermittent contact.",
            "stuck": "Power-cycle the logger; inspect the sensor for icing or blockage.",
            "drift": "Schedule recalibration against the reference instrument.",
            "offset_step": (
                "Verify the last maintenance entry; re-apply calibration coefficients."
            ),
            "noise_burst": "Inspect terminal block and shield continuity.",
            "dropout": "Check the telemetry link and logger storage.",
            "power_flicker": "Test battery voltage and solar charge controller.",
            "physical_violation": "Cross-check T and RH sensors; the pair is inconsistent.",
        }[self.value]


class Severity(StrEnum):
    NOMINAL = "nominal"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"nominal": 0, "info": 1, "warning": 2, "critical": 3}[self.value]


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Observation:
    """A single AWS report.

    `timestamp` is epoch seconds (UTC). Values may be `MISSING_SENTINEL` or NaN;
    detectors must handle both rather than assuming clean input.
    """

    station_id: str
    timestamp: float
    temperature: float
    pressure: float
    humidity: float

    def value(self, channel: Channel) -> float:
        return getattr(self, channel.value)

    def as_vector(self) -> tuple[float, float, float]:
        return (self.temperature, self.pressure, self.humidity)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Injected-fault labels. Present only in synthetic and replay streams."""

    is_anomalous: bool
    fault_type: FaultType
    affected: frozenset[Channel]
    episode_id: str | None = None


@dataclass(frozen=True, slots=True)
class Evidence:
    """One detector's opinion about one channel at one instant.

    `score` is normalised to [0, 1] where 0 means "indistinguishable from
    normal" and 1 means "certainly anomalous". Detectors that produce unbounded
    statistics (z-scores, Mahalanobis distances) are responsible for squashing
    them before emitting Evidence, so fusion never has to know the scale.
    """

    detector: str
    channel: Channel
    score: float
    statistic: float
    threshold: float
    detail: str
    suggests: FaultType = FaultType.NONE

    @property
    def exceeded(self) -> bool:
        return self.statistic > self.threshold


@dataclass(frozen=True, slots=True)
class Attribution:
    """Explainability payload for a single flagged channel."""

    channel: Channel
    observed: float
    expected: float
    deviation_sigma: float
    # (feature name, share of the channel score)
    contributions: tuple[tuple[str, float], ...]
    narrative: str


@dataclass(frozen=True, slots=True)
class Verdict:
    """Pipeline output for one observation. This is what the API serialises."""

    station_id: str
    timestamp: float
    observation: Observation
    is_anomalous: bool
    confidence: float
    severity: Severity
    fault_type: FaultType
    flagged: frozenset[Channel]
    evidence: tuple[Evidence, ...]
    attributions: tuple[Attribution, ...]
    repaired: dict[str, float] | None
    latency_ms: float
    health: dict[str, float] = field(default_factory=dict)
    ground_truth: GroundTruth | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp,
            "observation": {
                "temperature": _clean(self.observation.temperature),
                "pressure": _clean(self.observation.pressure),
                "humidity": _clean(self.observation.humidity),
            },
            "is_anomalous": self.is_anomalous,
            "confidence": round(self.confidence, 4),
            "severity": self.severity.value,
            "fault_type": self.fault_type.value,
            "fault_label": self.fault_type.label,
            "action": self.fault_type.maintenance_action,
            "flagged": sorted(c.value for c in self.flagged),
            "evidence": [
                {
                    "detector": e.detector,
                    "channel": e.channel.value,
                    "score": round(e.score, 4),
                    "statistic": round(float(e.statistic), 4),
                    "threshold": round(float(e.threshold), 4),
                    "detail": e.detail,
                }
                for e in self.evidence
                if e.score > 0.01
            ],
            "attributions": [
                {
                    "channel": a.channel.value,
                    "observed": _clean(a.observed),
                    "expected": _clean(a.expected),
                    "deviation_sigma": round(a.deviation_sigma, 3),
                    "contributions": [
                        {"feature": name, "share": round(share, 4)}
                        for name, share in a.contributions
                    ],
                    "narrative": a.narrative,
                }
                for a in self.attributions
            ],
            "repaired": self.repaired,
            "latency_ms": round(self.latency_ms, 3),
            "health": {k: round(v, 2) for k, v in self.health.items()},
        }


def _clean(x: float) -> float | None:
    """JSON has no NaN. Missing values travel as null."""
    if x is None:
        return None
    fx = float(x)
    if fx != fx or fx == MISSING_SENTINEL:
        return None
    return round(fx, 3)
