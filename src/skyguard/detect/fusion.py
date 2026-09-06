"""Evidence fusion and root-cause classification.

Fusion combines the four layers' opinions into one confidence per channel, then
one verdict per observation. The combiner is a weighted noisy-OR:

    confidence = 1 - prod_i (1 - w_i * s_i)

Noisy-OR is the right shape because the layers are complements, not competitors.
Each looks for a different fault signature, so two layers agreeing should raise
confidence, but one layer staying silent should not veto another -- an averaging
rule would let three silent layers bury a certain physics violation.

The weights `w_i` express how much each layer can be fooled by real weather.
L0 has weight 1.0 and therefore saturates confidence on its own; that is
intended, because L0 only fires on constraints that cannot be violated by
weather.

Root-cause classification is a scored rule set over the evidence signature, not
a learned classifier. With eight classes and a labelled synthetic generator we
could train one, but the rules are auditable, need no training data at deploy
time, and a judge can read the reasoning. `docs/ARCHITECTURE.md` records the
comparison against a gradient-boosted alternative.
"""

from __future__ import annotations

from collections import defaultdict

from ..config import DetectorConfig
from ..models import Channel, Evidence, FaultType, Severity


def fuse_channel_confidence(
    evidence: tuple[Evidence, ...],
    weights: dict[str, float],
) -> dict[Channel, float]:
    """Weighted noisy-OR pooling, per channel."""
    per_channel: dict[Channel, float] = defaultdict(lambda: 1.0)
    for ev in evidence:
        w = weights.get(ev.detector, 0.5)
        contribution = max(0.0, min(1.0, w * ev.score))
        per_channel[ev.channel] *= 1.0 - contribution
    return {channel: 1.0 - product for channel, product in per_channel.items()}


def classify_root_cause(evidence: tuple[Evidence, ...]) -> FaultType:
    """Pick the fault class best supported by the evidence signature.

    Each piece of evidence votes for the type its detector suggests, weighted
    by its own score. Ties break toward the more specific diagnosis -- a
    dropout is a more actionable answer than a generic physical violation, so
    it wins an equal vote.
    """
    if not evidence:
        return FaultType.NONE

    votes: dict[FaultType, float] = defaultdict(float)
    for ev in evidence:
        if ev.suggests is FaultType.NONE:
            continue
        votes[ev.suggests] += ev.score

    if not votes:
        return FaultType.NONE

    # Specificity ordering used only to break ties.
    specificity = {
        FaultType.DROPOUT: 6,
        FaultType.POWER_FLICKER: 6,
        FaultType.STUCK: 5,
        FaultType.PHYSICAL_VIOLATION: 4,
        FaultType.DRIFT: 3,
        FaultType.OFFSET_STEP: 3,
        FaultType.NOISE_BURST: 2,
        FaultType.SPIKE: 1,
        FaultType.NONE: 0,
    }
    return max(votes.items(), key=lambda kv: (round(kv[1], 6), specificity[kv[0]]))[0]


def assign_severity(confidence: float, config: DetectorConfig) -> Severity:
    if confidence < config.alert_threshold:
        return Severity.NOMINAL
    if confidence < config.severity_warning:
        return Severity.INFO
    if confidence < config.severity_critical:
        return Severity.WARNING
    return Severity.CRITICAL
