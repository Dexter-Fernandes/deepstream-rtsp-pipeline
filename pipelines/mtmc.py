"""CPU-only cross-camera tracklet association."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
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


class MtmcFuser:
    """Collect timestamp buckets and publish cross-camera identity snapshots.

    ``observe`` and ``flush_ready`` have one streaming-thread writer. Readers may call
    ``global_id`` or ``id_map`` from other Python threads: publication always rebinds a complete
    dictionary under the GIL and never mutates the visible dictionary in place. TTL and periodic
    reassignment intervals are measured in processed timestamp buckets.
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
        self._id_map: dict[PublicTrackletKey, int] = {}
        self._qualified_id_map: dict[TrackletKey, int] = {}
        self._generations: dict[int, int] = defaultdict(int)
        self._last_seen: dict[TrackletKey, int] = {}
        self._latest_processed_bucket: int | None = None
        self._processed_buckets = 0
        self._newest_timestamp_ns: int | None = None
        self._desync_buckets = 0
        self._skew_refusals = 0
        self._next_global_id = 1

    def observe(self, timestamp_ns: int, source_id: int, detections: Iterable[object]) -> None:
        bucket = timestamp_ns // self.config.sync_bucket_ns
        if self._newest_timestamp_ns is None or timestamp_ns > self._newest_timestamp_ns:
            self._newest_timestamp_ns = timestamp_ns
        state = self._pending.setdefault(bucket, _BucketState(timestamp_ns, timestamp_ns))
        state.earliest_timestamp_ns = min(timestamp_ns, state.earliest_timestamp_ns)
        state.latest_timestamp_ns = max(timestamp_ns, state.latest_timestamp_ns)
        source_observations = state.by_source.setdefault(source_id, [])
        for detection in detections:
            observation = _QualifiedObservation(
                timestamp_ns=timestamp_ns,
                source_id=source_id,
                generation=self._generations[source_id],
                object_id=int(detection.object_id),
                x=float(detection.x),
                y=float(detection.y),
                sigma=float(detection.sigma),
                embedding=getattr(detection, "embedding", None),
            )
            if observation.sigma <= self.config.max_sigma_m:
                source_observations.append(observation)

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
            self._all_tracklets.update(
                _tracklet_key(observation)
                for observation in bucket_observations
            )
            for observation in bucket_observations:
                key = _tracklet_key(observation)
                self._last_seen[key] = max(bucket, self._last_seen.get(key, bucket))
            if skew_refused:
                self._skew_refusals += 1
            self._observations.extend(bucket_observations)
            del self._pending[bucket]
            flushed.append(bucket)
            self._processed_buckets += 1
            self._latest_processed_bucket = max(
                bucket,
                self._latest_processed_bucket if self._latest_processed_bucket is not None else bucket,
            )
            self._evict_stale_tracklets()
            if self._processed_buckets % self.config.reassign_interval == 0:
                self._publish_assignments()
        if force and self._all_tracklets and self._processed_buckets % self.config.reassign_interval:
            self._publish_assignments()
        return flushed

    def _evict_stale_tracklets(self) -> None:
        if self._latest_processed_bucket is None:
            return
        stale = {
            key
            for key, last_seen in self._last_seen.items()
            if self._latest_processed_bucket - last_seen >= self.config.tracklet_ttl
        }
        if not stale:
            return
        self._observations = [
            observation
            for observation in self._observations
            if _tracklet_key(observation) not in stale
        ]
        self._all_tracklets.difference_update(stale)
        for key in stale:
            del self._last_seen[key]
        self._qualified_id_map = {
            key: global_id
            for key, global_id in self._qualified_id_map.items()
            if key not in stale
        }
        self._publish_current_generation()

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

    def stats(self) -> dict[str, int]:
        return {
            "processed_buckets": self._processed_buckets,
            "pending_buckets": len(self._pending),
            "published_tracklets": len(self._id_map),
            "desync_buckets": self._desync_buckets,
            "skew_refusals": self._skew_refusals,
            "active_tracklets": len(self._last_seen),
        }


def _tracklet_key(observation: GroundObservation | _QualifiedObservation) -> TrackletKey:
    return (
        observation.source_id,
        getattr(observation, "generation", 0),
        observation.object_id,
    )


def _candidate_cost(
    left: GroundObservation | _QualifiedObservation,
    right: GroundObservation | _QualifiedObservation,
    config: MtmcConfig,
) -> float | None:
    distance = hypot(left.x - right.x, left.y - right.y)
    z = distance / sqrt(left.sigma**2 + right.sigma**2)
    if distance > config.max_distance_m or z > config.z_gate:
        return None
    appearance_cost = 0.0
    if config.w_app and left.embedding is not None and right.embedding is not None:
        left_norm = float(np.linalg.norm(left.embedding))
        right_norm = float(np.linalg.norm(right.embedding))
        if left_norm > 0.0 and right_norm > 0.0:
            cosine = float(np.dot(left.embedding, right.embedding) / (left_norm * right_norm))
            appearance_cost = 1.0 - max(-1.0, min(1.0, cosine))
    return config.w_geo * z + config.w_app * appearance_cost


def _evidence(
    observations: Iterable[GroundObservation | _QualifiedObservation], config: MtmcConfig
) -> tuple[dict[tuple[TrackletKey, TrackletKey], int], dict[tuple[TrackletKey, TrackletKey], int]]:
    buckets: dict[int, list[GroundObservation | _QualifiedObservation]] = defaultdict(list)
    for observation in observations:
        buckets[observation.timestamp_ns // config.sync_bucket_ns].append(observation)

    support: dict[tuple[TrackletKey, TrackletKey], int] = defaultdict(int)
    cooccurrence: dict[tuple[TrackletKey, TrackletKey], int] = defaultdict(int)
    for bucket in buckets.values():
        timestamps = [observation.timestamp_ns for observation in bucket]
        if max(timestamps) - min(timestamps) > config.max_skew_ns:
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


def _fuse_qualified(
    observations: Iterable[GroundObservation | _QualifiedObservation], config: MtmcConfig
) -> dict[TrackletKey, int]:
    rows = [observation for observation in observations if observation.sigma <= config.max_sigma_m]
    keys = sorted({_tracklet_key(observation) for observation in rows})
    support, cooccurrence = _evidence(rows, config)
    lifetimes: dict[TrackletKey, tuple[int, int]] = {}
    for observation in rows:
        key = _tracklet_key(observation)
        bucket = observation.timestamp_ns // config.sync_bucket_ns
        first, last = lifetimes.get(key, (bucket, bucket))
        lifetimes[key] = min(first, bucket), max(last, bucket)

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


def fuse_offline(
    observations: Iterable[GroundObservation], config: MtmcConfig
) -> dict[PublicTrackletKey, int]:
    """Associate a finite observation stream and return source-local to global IDs."""
    qualified = _fuse_qualified(observations, config)
    return {
        (source_id, object_id): global_id
        for (source_id, _, object_id), global_id in qualified.items()
    }
