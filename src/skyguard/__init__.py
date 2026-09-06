"""SkyGuard AI -- real-time anomaly detection for Automatic Weather Stations.

Smart India Hackathon 2026, problem statement 26073.
Ministry of Earth Sciences / India Meteorological Department.

Public surface is deliberately small: build a pipeline, feed it observations,
read verdicts.

    from skyguard import SkyGuardPipeline, Observation

    pipeline = SkyGuardPipeline()
    pipeline.fit(clean_history)
    verdict = pipeline.process(observation)
"""

from .models import Channel, FaultType, Observation, Severity, Verdict
from .pipeline import SkyGuardPipeline

__version__ = "0.3.0"
__all__ = [
    "Channel",
    "FaultType",
    "Observation",
    "Severity",
    "SkyGuardPipeline",
    "Verdict",
]
