"""The pipeline: one observation in, one verdict out.

Ordering matters and is enforced here rather than left to each detector:

1. every detector sees the *same* history (no detector updates state mid-pass)
2. fusion combines their evidence
3. the verdict is formed
4. only then is state updated -- and only with accepted or repaired values

Step 4 is the part that is easy to get wrong. If a spike enters the rolling
median, the baseline moves toward the fault and the next spike looks smaller.
Over a long stuck episode the system would gradually accept the frozen value as
normal. Feeding back the repaired estimate instead keeps the baseline anchored
to what the sensor *should* be reading.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .config import DEFAULT, Settings
from .detect.fusion import assign_severity, classify_root_cause, fuse_channel_confidence
from .detect.l0_physics import PhysicsGate
from .detect.l1_temporal import TemporalDetector
from .detect.l2_multivariate import MultivariateDetector
from .detect.l3_sequence import SequenceDetector
from .detect.l4_spatial import NetworkBoard, SpatialDetector
from .explain.attribution import build_attributions
from .features.online import StationState, is_missing
from .health.index import SensorHealth
from .impute.repair import Repairer
from .models import CHANNELS, Channel, Evidence, FaultType, Observation, Severity, Verdict

#: Consecutive substitutions on one channel before the pipeline stops
#: trusting its own repairs and re-seeds from the sensor. Matches the
#: horizon over which persistence has forecast skill at a 10-minute cadence.
#: Past about an hour our substitute carries no more information than the
#: sensor's own reading, so continuing to prefer it only widens the gap that
#: keeps the channel flagged.
REBASELINE_AFTER = 6

#: Absolute ceiling on how long the buddy check can hold off a re-baseline
#: (see `_update_state`). 18 samples = 3 hours, the same horizon as the repair
#: extrapolator's `MAX_EXTRAPOLATION_STEPS`: beyond it the substitute has no
#: forecast skill anyway, so a stale pairwise offset at a neighbour cannot lock
#: a channel out indefinitely.
REBASELINE_CEILING = 18

#: How many flagged samples a buddy-check disagreement keeps suppressing the
#: re-baseline after it was last seen. The buddy check fires intermittently
#: through a sustained fault, so a single-sample signal would leak; a window
#: rather than a latch also means suppression *releases* about an hour after the
#: neighbours stop disagreeing, which is how a recovered sensor gets its
#: baseline back.
BUDDY_DISSENT_TTL = REBASELINE_AFTER


class SkyGuardPipeline:
    """Stateful, per-network anomaly detection.

    One instance handles many stations; per-station state lives in dictionaries
    keyed by station id, so adding a station costs a few kilobytes and no
    retraining. That is the scalability story: the learned components (subspace
    basis, forest) are shared across the network because they model *sensor
    physics*, while the fast-moving statistics are per-station.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or DEFAULT
        cfg = self.settings.detector

        floors = {c: self.settings.resolution.floor(c.value) for c in CHANNELS}
        self.l0 = PhysicsGate(self.settings.limits)
        self.l1 = TemporalDetector(cfg)
        self.l2 = MultivariateDetector(cfg, scale_floor=floors)
        self.l3 = SequenceDetector(cfg)
        self.board = NetworkBoard()
        self.l4 = SpatialDetector(cfg, self.board)
        self.detectors = (self.l0, self.l1, self.l2, self.l3, self.l4)

        self._states: dict[str, StationState] = {}
        self._flag_run: dict[tuple[str, Channel], int] = {}
        # (station, channel) -> flagged samples of buddy-check suppression still
        # owed. Refreshed to BUDDY_DISSENT_TTL whenever L4 disagrees with the
        # channel, counted down otherwise, cleared on recovery or re-baseline.
        self._buddy_dissent: dict[tuple[str, Channel], int] = {}
        self._health: dict[str, SensorHealth] = {}
        self.repairer = Repairer(self.settings.limits)
        self._fitted = False

        self.stats: dict[str, float] = defaultdict(float)

    # -- lifecycle ---------------------------------------------------------

    def register_station(self, station_id: str, altitude_m: float) -> None:
        """Declare a station's elevation so pressure can be compared across the network."""
        self.board.register(station_id, altitude_m)
        self._state_for(station_id)

    def fit(self, clean_observations: list[Observation]) -> None:
        """Warm-start the learned layers on a clean reference stream.

        "Clean" means no injected faults. In deployment this is a quality-
        controlled archive; in the demo it is generator output before injection.
        Fitting on contaminated data teaches the subspace that drift is normal,
        which is the single fastest way to make this system useless.
        """
        for detector in self.detectors:
            detector.fit(clean_observations)
        self._fitted = True

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _state_for(self, station_id: str) -> StationState:
        if station_id not in self._states:
            cfg = self.settings.detector
            self._states[station_id] = StationState(
                robust_window=cfg.robust_window,
                sequence_window=cfg.sequence_window,
                drift_window=self.settings.health.drift_window,
                scale_floor={
                    c: self.settings.resolution.floor(c.value) for c in CHANNELS
                },
            )
            self._health[station_id] = SensorHealth(
                self.settings.health, self.settings.stream.interval_seconds
            )
        return self._states[station_id]

    def health_for(self, station_id: str) -> SensorHealth:
        self._state_for(station_id)
        return self._health[station_id]

    # -- main entry point --------------------------------------------------

    def process(self, obs: Observation) -> Verdict:
        started = time.perf_counter()
        state = self._state_for(obs.station_id)
        health = self._health[obs.station_id]
        cfg = self.settings.detector

        # Log what the logger actually sent, before any detector or repair
        # touches it. The stuck detector reads this buffer.
        for channel in CHANNELS:
            state.record_reported(channel, obs.value(channel))

        evidence: list[Evidence] = []
        for detector in self.detectors:
            evidence.extend(detector.inspect(obs, state))
        evidence_t = tuple(evidence)

        confidences = fuse_channel_confidence(evidence_t, cfg.weights)
        flagged = frozenset(
            channel
            for channel, conf in confidences.items()
            if conf >= cfg.alert_threshold
        )
        confidence = max(confidences.values(), default=0.0)

        is_anomalous = bool(flagged)
        fault_type = (
            classify_root_cause(tuple(e for e in evidence_t if e.channel in flagged))
            if is_anomalous
            else FaultType.NONE
        )
        severity = assign_severity(confidence, cfg) if is_anomalous else Severity.NOMINAL

        attributions = (
            build_attributions(obs, evidence_t, flagged, state, fault_type)
            if is_anomalous
            else ()
        )
        repaired = self.repairer.repair(obs, flagged, state) if is_anomalous else None

        # Channels the buddy check says are departing from the neighbours right
        # now. `_update_state` uses this to hold off on re-baselining: if the
        # rest of the network still disagrees with a channel, re-seeding the
        # baseline from that channel's sensor would just adopt the fault.
        spatial_dissent = frozenset(
            e.channel for e in evidence_t if e.detector == self.l4.name
        )

        # --- state update, after every decision is made ---
        rebaselined = self._update_state(obs, state, flagged, repaired, spatial_dissent)
        if rebaselined:
            self.stats["rebaselines"] += len(rebaselined)
        health.update(flagged)
        self._refresh_drift(state, health)
        if not is_anomalous:
            self.repairer.note_clean(obs)
            residual = np.array(
                [state.residual_z(c, obs.value(c)) for c in CHANNELS], dtype=float
            )
            self.l2.update(residual)

        # Publish only what survived detection, and only then teach the buddy
        # statistics. Publishing a rejected value would let one broken station
        # drag the network consensus toward its own fault.
        accepted = {
            c: obs.value(c)
            for c in CHANNELS
            if c not in flagged and not is_missing(obs.value(c))
        }
        peers_before = self.board.peers(obs.station_id, obs.timestamp)
        if accepted:
            self.l4.learn(obs, accepted, peers_before)
            self.board.publish(obs, accepted)

        state.last_timestamp = obs.timestamp
        state.samples_seen += 1

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.stats["processed"] += 1
        self.stats["total_latency_ms"] += latency_ms
        if is_anomalous:
            self.stats["alerts"] += 1

        return Verdict(
            station_id=obs.station_id,
            timestamp=obs.timestamp,
            observation=obs,
            is_anomalous=is_anomalous,
            confidence=confidence,
            severity=severity,
            fault_type=fault_type,
            flagged=flagged,
            evidence=evidence_t,
            attributions=attributions,
            repaired=repaired,
            latency_ms=latency_ms,
            health=health.snapshot(),
        )

    def _update_state(
        self,
        obs: Observation,
        state: StationState,
        flagged: frozenset[Channel],
        repaired: dict[str, float] | None,
        spatial_dissent: frozenset[Channel] = frozenset(),
    ) -> list[Channel]:
        """Push accepted values into history; substitute repairs for flagged ones.

        Returns the channels that were force-rebaselined this step.

        The subtlety is the recovery path. If only repairs ever enter the
        buffer, a channel that gets flagged -- correctly or not -- can never
        come back: the repair is anchored to the moment the fault began, it
        drifts further from reality with every step, and that widening gap
        guarantees the next sample is flagged too. In testing this locked
        humidity at a constant 100 % for the remainder of the stream.

        So after `REBASELINE_AFTER` consecutive substitutions the pipeline stops
        trusting its own model and re-seeds the channel from what the sensor is
        actually reporting. Either the sensor really is broken -- in which case
        the health index has already quarantined the station and re-baselining
        costs nothing -- or the detector was wrong, and re-baselining is exactly
        the correction required. Both branches are better than locking in.

        The exception is a channel in `spatial_dissent`: the buddy check says
        the rest of the network still disagrees with this reading, which is
        exactly the evidence a single station cannot produce about its own
        sustained bias. Re-seeding from the sensor there would adopt the fault
        and silence every downstream layer, so the re-baseline is held off --
        the channel keeps getting the repair and stays flagged -- until either
        the neighbours stop disagreeing or `REBASELINE_CEILING` is reached, past
        which the substitute has no forecast skill anyway.
        """
        rebaselined: list[Channel] = []
        for channel in CHANNELS:
            key = (obs.station_id, channel)
            if channel not in flagged:
                self._flag_run[key] = 0
                self._buddy_dissent.pop(key, None)
                state.observe(channel, obs.value(channel))
                continue

            run = self._flag_run.get(key, 0) + 1
            self._flag_run[key] = run

            if channel in spatial_dissent:
                self._buddy_dissent[key] = BUDDY_DISSENT_TTL
            dissent_ttl = self._buddy_dissent.get(key, 0)
            # The buddy check disagreed with this channel within the last
            # BUDDY_DISSENT_TTL flagged samples: hold off re-baselining -- until
            # it stays quiet, or the 3-hour ceiling -- rather than re-seed the
            # baseline from a reading the rest of the network says is wrong.
            held_by_buddies = dissent_ttl > 0 and run <= REBASELINE_CEILING
            if dissent_ttl > 0:
                self._buddy_dissent[key] = dissent_ttl - 1

            if run > REBASELINE_AFTER and not is_missing(obs.value(channel)):
                if held_by_buddies:
                    self.stats["rebaseline_suppressed"] += 1
                else:
                    state.observe(channel, obs.value(channel))
                    self.repairer.note_accepted(obs.station_id, channel)
                    self._flag_run[key] = 0
                    self._buddy_dissent.pop(key, None)
                    rebaselined.append(channel)
                    continue

            if repaired and channel.value in repaired:
                state.observe(channel, repaired[channel.value])
            # No repair available: skip entirely rather than poison the buffer.
            # A short gap is recoverable; a corrupted baseline is not.
        return rebaselined

    def _refresh_drift(self, state: StationState, health: SensorHealth) -> None:
        """Re-estimate drift once per simulated day, not per sample.

        Theil-Sen is O(n^2). Running it every sample would dominate the latency
        budget for no benefit -- drift does not change meaningfully in ten
        minutes.
        """
        per_day = max(int(86400 / self.settings.stream.interval_seconds), 1)
        if state.samples_seen % per_day != 0:
            return
        for channel in CHANNELS:
            history = state.long[channel].as_array()
            if history.size:
                health.estimate_drift(channel, history)

    # -- reporting ---------------------------------------------------------

    def throughput_summary(self) -> dict[str, float]:
        processed = self.stats["processed"] or 1.0
        mean_latency = self.stats["total_latency_ms"] / processed
        return {
            "processed": self.stats["processed"],
            "alerts": self.stats["alerts"],
            "mean_latency_ms": round(mean_latency, 4),
            "throughput_per_second": round(1000.0 / mean_latency, 1) if mean_latency else 0.0,
        }
