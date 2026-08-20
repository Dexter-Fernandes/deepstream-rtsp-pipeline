"""CPU-only cross-camera tracklet association."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import hypot, sqrt

import numpy as np

TrackletKey = tuple[int, int, int]
PublicTrackletKey = tuple[int, int]


@dataclass(frozen=True)
class GroundObservation:
    timestamp_ns: int
    source_id: int
    object_id: int
    x: float
    y: float
    sigma: float
    embedding: np.ndarray | None = None
    generation: int = 0
    association_bucket: int | None = None


@dataclass(frozen=True)
class MtmcFrame:
    """One source frame and its projected observations from a muxed batch."""

    timestamp_ns: int
    source_id: int
    observations: tuple[GroundObservation, ...]


@dataclass(frozen=True)
class _QualifiedObservation:
    timestamp_ns: int
    source_id: int
    generation: int
    object_id: int
    x: float
    y: float
    sigma: float
    embedding: np.ndarray | None = None
    association_bucket: int | None = None
    observation_age_epoch: int | None = None


@dataclass
class _BucketState:
    earliest_timestamp_ns: int
    latest_timestamp_ns: int
    by_source: dict[int, list[_QualifiedObservation]] = field(default_factory=dict)


@dataclass
class MtmcConfig:
    z_gate: float = 3.0
    max_distance_m: float = 2.0
    max_sigma_m: float = 1.5
    w_geo: float = 1.0
    w_app: float = 0.0
    min_cooccurrence: int = 5
    min_support: int = 3
    min_affinity: float = 0.5
    reassign_interval: int = 10
    tracklet_ttl: int = 60
    sync_bucket_ns: int = 33_000_000
    max_skew_ns: int = 100_000_000


def atomic_batch_rejection_reason(
    frames: Iterable[MtmcFrame],
    *,
    expected_sources: frozenset[int] | None,
    max_skew_ns: int,
) -> str | None:
    """Return the atomic mux-batch rejection reason without mutating fuser state."""
    frames = tuple(frames)
    source_ids = [int(frame.source_id) for frame in frames]
    observed_sources = set(source_ids)
    complete = len(observed_sources) == len(source_ids)
    if expected_sources is not None:
        complete = complete and observed_sources == expected_sources
    if not complete:
        return "missing_sources"

    timestamps = [int(frame.timestamp_ns) for frame in frames]
    if not timestamps or any(timestamp < 0 for timestamp in timestamps):
        return "missing_sources"
    if max(timestamps) - min(timestamps) > max_skew_ns:
        return "timestamp_skew"
    return None


class MtmcFuser:
    """Collect timestamp buckets and publish cross-camera identity snapshots.

    ``observe`` and ``flush_ready`` have one streaming-thread writer. Readers may call
    ``global_id`` or ``id_map`` from other Python threads: publication always rebinds a complete
    dictionary under the GIL and never mutates the visible dictionary in place. TTL is measured
    in liveness attempts; periodic reassignment uses accepted association buckets.
    """

    def __init__(
        self,
        config: MtmcConfig | None = None,
        *,
        expected_sources: Iterable[int] | None = None,
    ) -> None:
        self.config = config or MtmcConfig()
        self._expected_sources = (
            frozenset(expected_sources) if expected_sources is not None else None
        )
        self._pending: dict[int, _BucketState] = {}
        self._observations: list[_QualifiedObservation] = []
        self._all_tracklets: set[TrackletKey] = set()
        self._historical_tracklets: set[TrackletKey] = set()
        self._historical_support: dict[tuple[TrackletKey, TrackletKey], int] = defaultdict(int)
        self._historical_cooccurrence: dict[
            tuple[TrackletKey, TrackletKey], int
        ] = defaultdict(int)
        self._historical_lifetimes: dict[TrackletKey, tuple[int, int]] = {}
        self._id_map: dict[PublicTrackletKey, int] = {}
        self._qualified_id_map: dict[TrackletKey, int] = {}
        self._generations: dict[int, int] = defaultdict(int)
        self._last_seen: dict[TrackletKey, int] = {}
        self._embedding_cache: dict[TrackletKey, np.ndarray] = {}
        self._ingestion_mode: str | None = None
        self._processed_buckets = 0
        self._attempted_buckets = 0
        self._newest_timestamp_ns: int | None = None
        self._desync_buckets = 0
        self._skew_refusals = 0
        self._next_global_id = 1

    def observe(self, timestamp_ns: int, source_id: int, detections: Iterable[object]) -> None:
        self._select_ingestion_mode("legacy")
        bucket = timestamp_ns // self.config.sync_bucket_ns
        if self._newest_timestamp_ns is None or timestamp_ns > self._newest_timestamp_ns:
            self._newest_timestamp_ns = timestamp_ns
        state = self._pending.setdefault(bucket, _BucketState(timestamp_ns, timestamp_ns))
        state.earliest_timestamp_ns = min(timestamp_ns, state.earliest_timestamp_ns)
        state.latest_timestamp_ns = max(timestamp_ns, state.latest_timestamp_ns)
        source_observations = state.by_source.setdefault(source_id, [])
        source_observations.extend(
            self._qualify_observations(timestamp_ns, source_id, detections)
        )

    def observe_batch(
        self,
        frames: Iterable[MtmcFrame],
        *,
        attempt_epoch: int | None = None,
        association_bucket: int | None = None,
    ) -> bool:
        """Atomically validate and accept one mux batch using its true source clocks."""
        self._start_atomic_attempt(attempt_epoch)
        observation_age_epoch = self._attempted_buckets
        frames = tuple(frames)
        rejection_reason = atomic_batch_rejection_reason(
            frames,
            expected_sources=self._expected_sources,
            max_skew_ns=self.config.max_skew_ns,
        )
        if rejection_reason == "missing_sources":
            self._desync_buckets += 1
            self._evict_stale_tracklets()
            return False
        if rejection_reason == "timestamp_skew":
            self._skew_refusals += 1
            self._evict_stale_tracklets()
            return False

        timestamps = [int(frame.timestamp_ns) for frame in frames]
        expected_bucket = self._processed_buckets + 1
        processed_bucket = (
            expected_bucket if association_bucket is None else int(association_bucket)
        )
        if processed_bucket != expected_bucket:
            raise RuntimeError(
                "atomic association bucket is out of order: "
                f"expected {expected_bucket}, got {processed_bucket}"
            )
        batch_observations = []
        for frame in frames:
            timestamp_ns = int(frame.timestamp_ns)
            source_id = int(frame.source_id)
            batch_observations.extend(
                self._qualify_observations(
                    timestamp_ns,
                    source_id,
                    frame.observations,
                    association_bucket=processed_bucket,
                    observation_age_epoch=observation_age_epoch,
                )
            )

        self._newest_timestamp_ns = max(
            max(timestamps),
            self._newest_timestamp_ns if self._newest_timestamp_ns is not None else max(timestamps),
        )
        self._all_tracklets.update(
            _tracklet_key(observation) for observation in batch_observations
        )
        self._record_historical_evidence(batch_observations)
        for observation in batch_observations:
            self._last_seen[_tracklet_key(observation)] = observation_age_epoch
        self._observations.extend(batch_observations)
        self._processed_buckets = processed_bucket
        self._evict_stale_tracklets()
        if self._processed_buckets % self.config.reassign_interval == 0:
            self._publish_assignments()
        return True

    def advance_liveness(self, *, attempt_epoch: int | None = None) -> None:
        """Age atomic live state for one attempt without recording association evidence."""
        self._start_atomic_attempt(attempt_epoch)
        self._evict_stale_tracklets()

    def _start_atomic_attempt(self, attempt_epoch: int | None = None) -> None:
        self._select_ingestion_mode("atomic")
        expected_epoch = self._attempted_buckets + 1
        submitted_epoch = expected_epoch if attempt_epoch is None else int(attempt_epoch)
        if submitted_epoch != expected_epoch:
            raise RuntimeError(
                "atomic attempt epoch is out of order: "
                f"expected {expected_epoch}, got {submitted_epoch}"
            )
        self._attempted_buckets = submitted_epoch

    def _qualify_observations(
        self,
        timestamp_ns: int,
        source_id: int,
        detections: Iterable[object],
        *,
        association_bucket: int | None = None,
        observation_age_epoch: int | None = None,
    ) -> list[_QualifiedObservation]:
        observations = []
        for detection in detections:
            sigma = float(detection.sigma)
            if not np.isfinite(sigma) or sigma <= 0.0 or sigma > self.config.max_sigma_m:
                continue
            generation = self._generations[source_id]
            object_id = int(detection.object_id)
            tracklet = (source_id, generation, object_id)
            embedding = _normalise_embedding(getattr(detection, "embedding", None))
            if embedding is not None:
                self._embedding_cache[tracklet] = embedding
            observation = _QualifiedObservation(
                timestamp_ns=timestamp_ns,
                source_id=source_id,
                generation=generation,
                object_id=object_id,
                x=float(detection.x),
                y=float(detection.y),
                sigma=sigma,
                embedding=self._embedding_cache.get(tracklet),
                association_bucket=association_bucket,
                observation_age_epoch=observation_age_epoch,
            )
            observations.append(observation)
        return observations

    def flush_ready(self, *, force: bool = False) -> list[int]:
        flushed: list[int] = []
        for bucket in sorted(self._pending):
            state = self._pending[bucket]
            source_rows = state.by_source
            complete = self._expected_sources is None or self._expected_sources.issubset(source_rows)
            stale = (
                self._newest_timestamp_ns is not None
                and self._newest_timestamp_ns - state.latest_timestamp_ns
                > self.config.max_skew_ns
            )
            if not force and not complete and not stale:
                continue
            if self._expected_sources is not None and not complete:
                self._desync_buckets += 1
            skew_refused = (
                state.latest_timestamp_ns - state.earliest_timestamp_ns
                > self.config.max_skew_ns
            )
            bucket_observations = []
            for observations in source_rows.values():
                bucket_observations.extend(observations)
            processed_bucket = self._processed_buckets + 1
            self._attempted_buckets += 1
            observation_age_epoch = self._attempted_buckets
            bucket_observations = [
                replace(
                    observation,
                    observation_age_epoch=observation_age_epoch,
                )
                for observation in bucket_observations
            ]
            self._all_tracklets.update(
                _tracklet_key(observation)
                for observation in bucket_observations
            )
            self._record_historical_evidence(bucket_observations)
            for observation in bucket_observations:
                key = _tracklet_key(observation)
                self._last_seen[key] = observation_age_epoch
            if skew_refused:
                self._skew_refusals += 1
            self._observations.extend(bucket_observations)
            del self._pending[bucket]
            flushed.append(bucket)
            self._processed_buckets = processed_bucket
            self._evict_stale_tracklets()
            if self._processed_buckets % self.config.reassign_interval == 0:
                self._publish_assignments()
        if force and self._all_tracklets and self._processed_buckets % self.config.reassign_interval:
            self._publish_assignments()
        return flushed

    def _record_historical_evidence(
        self,
        observations: list[_QualifiedObservation],
    ) -> None:
        """Retain bounded sufficient statistics for the shutdown assignment."""
        support, cooccurrence = _evidence(observations, self.config)
        for pair, count in support.items():
            self._historical_support[pair] += count
        for pair, count in cooccurrence.items():
            self._historical_cooccurrence[pair] += count
        for observation in observations:
            key = _tracklet_key(observation)
            bucket = _association_bucket(observation, self.config)
            self._historical_tracklets.add(key)
            first, last = self._historical_lifetimes.get(key, (bucket, bucket))
            self._historical_lifetimes[key] = min(first, bucket), max(last, bucket)

    def _evict_stale_tracklets(self) -> None:
        if not self._attempted_buckets:
            return
        oldest_live_epoch = self._attempted_buckets - self.config.tracklet_ttl + 1
        self._observations = [
            observation
            for observation in self._observations
            if observation.observation_age_epoch is not None
            and observation.observation_age_epoch >= oldest_live_epoch
        ]
        stale = {
            key
            for key, last_seen in self._last_seen.items()
            if self._attempted_buckets - last_seen >= self.config.tracklet_ttl
        }
        if not stale:
            return
        self._all_tracklets.difference_update(stale)
        for key in stale:
            del self._last_seen[key]
            self._embedding_cache.pop(key, None)
        self._qualified_id_map = {
            key: global_id
            for key, global_id in self._qualified_id_map.items()
            if key not in stale
        }
        self._publish_current_generation()

    def _select_ingestion_mode(self, mode: str) -> None:
        if self._ingestion_mode is None:
            self._ingestion_mode = mode
        elif self._ingestion_mode != mode:
            raise RuntimeError("cannot mix legacy and atomic MTMC ingestion modes")

    def _publish_assignments(self) -> None:
        candidate_map = _fuse_qualified(self._observations, self.config)
        next_cluster_id = max(candidate_map.values(), default=0) + 1
        for tracklet in sorted(self._all_tracklets):
            if tracklet not in candidate_map:
                candidate_map[tracklet] = next_cluster_id
                next_cluster_id += 1

        cluster_members: dict[int, set[TrackletKey]] = defaultdict(set)
        for tracklet, cluster_id in candidate_map.items():
            cluster_members[cluster_id].add(tracklet)

        ranked_clusters = []
        for members in cluster_members.values():
            votes = Counter(
                self._qualified_id_map[member]
                for member in members
                if member in self._qualified_id_map
            )
            candidates = sorted(votes, key=lambda global_id: (-votes[global_id], global_id))
            top_count = votes[candidates[0]] if candidates else 0
            top_id = candidates[0] if candidates else self._next_global_id
            ranked_clusters.append((-top_count, top_id, min(members), members, candidates))

        new_map: dict[TrackletKey, int] = {}
        used_global_ids: set[int] = set()
        for _, _, _, members, candidates in sorted(ranked_clusters):
            inherited = next(
                (global_id for global_id in candidates if global_id not in used_global_ids),
                None,
            )
            if inherited is None:
                while self._next_global_id in used_global_ids:
                    self._next_global_id += 1
                inherited = self._next_global_id
                self._next_global_id += 1
            used_global_ids.add(inherited)
            for member in members:
                new_map[member] = inherited

        self._qualified_id_map = new_map
        self._publish_current_generation()

    def _publish_current_generation(self) -> None:
        # Atomic rebinding is the cross-thread publication seam. Never mutate this dict in place.
        self._id_map = {
            (source_id, object_id): global_id
            for (source_id, generation, object_id), global_id in self._qualified_id_map.items()
            if generation == self._generations[source_id]
        }

    def bump_generation(self, source_id: int) -> int:
        """Advance a source generation after reconnecting its local tracker."""
        self._generations[source_id] += 1
        self._publish_current_generation()
        return self._generations[source_id]

    def global_id(self, source_id: int, object_id: int) -> int | None:
        return self._id_map.get((source_id, object_id))

    def id_map(self) -> dict[PublicTrackletKey, int]:
        """Return a detached snapshot for current source generations."""
        return dict(self._id_map)

    def qualified_id_map(self) -> dict[TrackletKey, int]:
        """Return active assignments without hiding source generations."""
        return dict(self._qualified_id_map)

    def authoritative_id_map(self) -> dict[TrackletKey, int]:
        """Fuse whole-run evidence without retaining every observation row.

        Live publication remains TTL-bounded. This final view uses accumulated
        support, co-occurrence, and lifetime statistics, so expiration cannot
        fragment the shutdown artifact.
        """
        return _cluster_qualified(
            sorted(self._historical_tracklets),
            self._historical_support,
            self._historical_cooccurrence,
            self._historical_lifetimes,
            self.config,
        )

    def stats(self) -> dict[str, int]:
        return {
            "processed_buckets": self._processed_buckets,
            "attempted_buckets": self._attempted_buckets,
            "pending_buckets": len(self._pending),
            "published_tracklets": len(self._id_map),
            "desync_buckets": self._desync_buckets,
            "skew_refusals": self._skew_refusals,
            "active_tracklets": len(self._last_seen),
            "retained_observations": len(self._observations),
        }


def _tracklet_key(observation: GroundObservation | _QualifiedObservation) -> TrackletKey:
    return (
        observation.source_id,
        getattr(observation, "generation", 0),
        observation.object_id,
    )


def _association_bucket(
    observation: GroundObservation | _QualifiedObservation,
    config: MtmcConfig,
) -> int:
    explicit_bucket = getattr(observation, "association_bucket", None)
    if explicit_bucket is not None:
        return int(explicit_bucket)
    return observation.timestamp_ns // config.sync_bucket_ns


def _normalise_embedding(value: object) -> np.ndarray | None:
    if value is None:
        return None
    embedding = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    norm = float(np.linalg.norm(embedding))
    if embedding.size == 0 or not np.isfinite(norm) or norm <= 0.0:
        return None
    embedding /= norm
    return embedding


def _candidate_cost(
    left: GroundObservation | _QualifiedObservation,
    right: GroundObservation | _QualifiedObservation,
    config: MtmcConfig,
) -> float | None:
    distance = hypot(left.x - right.x, left.y - right.y)
    z = distance / sqrt(left.sigma**2 + right.sigma**2)
    if distance > config.max_distance_m or z > config.z_gate:
        return None
    appearance_cost = 1.0  # neutral cosine distance when either feature is unavailable
    if config.w_app and left.embedding is not None and right.embedding is not None:
        left_embedding = np.asarray(left.embedding).reshape(-1)
        right_embedding = np.asarray(right.embedding).reshape(-1)
        left_norm = float(np.linalg.norm(left_embedding))
        right_norm = float(np.linalg.norm(right_embedding))
        if (
            left_embedding.shape == right_embedding.shape
            and np.isfinite((left_norm, right_norm)).all()
            and left_norm > 0.0
            and right_norm > 0.0
        ):
            cosine = float(
                np.dot(left_embedding, right_embedding) / (left_norm * right_norm)
            )
            appearance_cost = 1.0 - max(-1.0, min(1.0, cosine))
    return config.w_geo * z + config.w_app * appearance_cost


def _evidence(
    observations: Iterable[GroundObservation | _QualifiedObservation], config: MtmcConfig
) -> tuple[dict[tuple[TrackletKey, TrackletKey], int], dict[tuple[TrackletKey, TrackletKey], int]]:
    buckets: dict[
        tuple[bool, int], list[GroundObservation | _QualifiedObservation]
    ] = defaultdict(list)
    for observation in observations:
        explicit_bucket = getattr(observation, "association_bucket", None)
        buckets[(
            explicit_bucket is not None,
            _association_bucket(observation, config),
        )].append(observation)

    support: dict[tuple[TrackletKey, TrackletKey], int] = defaultdict(int)
    cooccurrence: dict[tuple[TrackletKey, TrackletKey], int] = defaultdict(int)
    for (explicit_bucket, _), bucket in buckets.items():
        timestamps = [observation.timestamp_ns for observation in bucket]
        if (
            not explicit_bucket
            and max(timestamps) - min(timestamps) > config.max_skew_ns
        ):
            continue
        by_source: dict[int, list[GroundObservation | _QualifiedObservation]] = defaultdict(list)
        for observation in bucket:
            by_source[observation.source_id].append(observation)
        for source_a, source_b in combinations(sorted(by_source), 2):
            left = sorted(by_source[source_a], key=_tracklet_key)
            right = sorted(by_source[source_b], key=_tracklet_key)
            for observation_a in left:
                for observation_b in right:
                    cooccurrence[(_tracklet_key(observation_a), _tracklet_key(observation_b))] += 1

            nearest_right: dict[TrackletKey, TrackletKey] = {}
            nearest_left: dict[TrackletKey, TrackletKey] = {}
            costs: dict[tuple[TrackletKey, TrackletKey], float] = {}
            for observation_a in left:
                for observation_b in right:
                    cost = _candidate_cost(observation_a, observation_b, config)
                    if cost is not None:
                        costs[(_tracklet_key(observation_a), _tracklet_key(observation_b))] = cost
            for key_a in {_tracklet_key(item) for item in left}:
                choices = [
                    (cost, key_b)
                    for (candidate_a, key_b), cost in costs.items()
                    if candidate_a == key_a
                ]
                if choices:
                    nearest_right[key_a] = min(choices)[1]
            for key_b in {_tracklet_key(item) for item in right}:
                choices = [
                    (cost, key_a)
                    for (key_a, candidate_b), cost in costs.items()
                    if candidate_b == key_b
                ]
                if choices:
                    nearest_left[key_b] = min(choices)[1]
            for key_a, key_b in nearest_right.items():
                if nearest_left.get(key_b) == key_a:
                    support[(key_a, key_b)] += 1
    return support, cooccurrence


def _cluster_qualified(
    keys: Iterable[TrackletKey],
    support: dict[tuple[TrackletKey, TrackletKey], int],
    cooccurrence: dict[tuple[TrackletKey, TrackletKey], int],
    lifetimes: dict[TrackletKey, tuple[int, int]],
    config: MtmcConfig,
) -> dict[TrackletKey, int]:
    keys = sorted(keys)
    parent = {key: key for key in keys}
    members = {key: {key} for key in keys}

    def find(key: TrackletKey) -> TrackletKey:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: TrackletKey, right: TrackletKey) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        for member_left in members[root_left]:
            for member_right in members[root_right]:
                if member_left[0] != member_right[0]:
                    continue
                first_left, last_left = lifetimes[member_left]
                first_right, last_right = lifetimes[member_right]
                if first_left <= last_right and first_right <= last_left:
                    return
        kept_root = min(root_left, root_right)
        removed_root = max(root_left, root_right)
        parent[removed_root] = kept_root
        members[kept_root].update(members.pop(removed_root))

    edges = []
    for pair, matches in support.items():
        opportunities = cooccurrence[pair]
        affinity = matches / opportunities
        if (
            opportunities >= config.min_cooccurrence
            and matches >= config.min_support
            and affinity >= config.min_affinity
        ):
            edges.append((-affinity, -matches, pair))
    for _, _, (left, right) in sorted(edges):
        union(left, right)

    roots = sorted({find(key) for key in keys})
    global_ids = {root: index + 1 for index, root in enumerate(roots)}
    return {key: global_ids[find(key)] for key in keys}


def _fuse_qualified(
    observations: Iterable[GroundObservation | _QualifiedObservation], config: MtmcConfig
) -> dict[TrackletKey, int]:
    rows = [observation for observation in observations if observation.sigma <= config.max_sigma_m]
    support, cooccurrence = _evidence(rows, config)
    lifetimes: dict[TrackletKey, tuple[int, int]] = {}
    for observation in rows:
        key = _tracklet_key(observation)
        bucket = _association_bucket(observation, config)
        first, last = lifetimes.get(key, (bucket, bucket))
        lifetimes[key] = min(first, bucket), max(last, bucket)
    return _cluster_qualified(lifetimes, support, cooccurrence, lifetimes, config)


def fuse_offline(
    observations: Iterable[GroundObservation], config: MtmcConfig
) -> dict[PublicTrackletKey, int]:
    """Associate a finite observation stream and return source-local to global IDs."""
    qualified = fuse_offline_qualified(observations, config)
    return {
        (source_id, object_id): global_id
        for (source_id, _, object_id), global_id in qualified.items()
    }


def fuse_offline_qualified(
    observations: Iterable[GroundObservation], config: MtmcConfig
) -> dict[TrackletKey, int]:
    """Associate a finite stream without collapsing reconnect generations.

    An observation's explicit ``association_bucket`` takes precedence over its
    timestamp floor, allowing offline evidence to reproduce accepted mux batches.
    """
    return _fuse_qualified(observations, config)
