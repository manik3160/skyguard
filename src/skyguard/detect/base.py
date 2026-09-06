"""The detector contract.

Four layers run on every observation. Each is independent, stateless with
respect to the others, and returns a tuple of :class:`Evidence`. Fusion is the
only place where their opinions meet.

Layering rationale
------------------
L0 physics   deterministic, zero training, catches the impossible
L1 temporal  per-channel, catches spikes / steps / frozen values
L2 joint     cross-channel, catches "each value plausible, combination is not"
L3 sequence  window-shaped, catches drift and variance change

The layers are ordered by how much they can be fooled. L0 is never wrong about
a violated physical constraint; L3 is a learned model and can mistake an
unusual-but-real event for a fault. Fusion weights them accordingly, and the
explanation shown to the operator always names which layer fired.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..features.online import StationState
from ..models import Evidence, Observation


@runtime_checkable
class Detector(Protocol):
    """Anything that can look at one observation and produce evidence."""

    name: str

    def inspect(self, obs: Observation, state: StationState) -> tuple[Evidence, ...]:
        """Score a single observation.

        Must not mutate `state` -- the pipeline owns buffer updates and applies
        them only after every detector has seen the same history. A detector
        that pushed its own value would give the next detector a different view
        of the past, making results depend on registration order.
        """
        ...

    def fit(self, observations: list[Observation]) -> None:
        """Optional offline warm-up. Layers with no learned state no-op."""
        ...


class StatelessDetector:
    """Base class for layers that need no training."""

    name = "unnamed"

    def fit(self, observations: list[Observation]) -> None:
        return None
