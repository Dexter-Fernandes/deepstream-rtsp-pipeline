"""CPU-testable adapter between DeepStream metadata and online MTMC fusion."""

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from types import MappingProxyType

import numpy as np

from pipelines.ground_plane import (
    Homography,
    foot_pixel_sigma,
    foot_point,
    is_foot_reliable,
    load_homography,
    project_point,
    projection_sigma,
)
from pipelines.metadata_parser import Detection
from pipelines.mtmc import GroundObservation, MtmcConfig, MtmcFuser

_FILE_FRAME_INTERVAL_NS = 500_000_000
_STOP = object()


@dataclass(frozen=True)
class _ObservationBatch:
    timestamp_ns: int
    source_id: int
    observations: tuple[GroundObservation, ...]


@dataclass(frozen=True)
class _GenerationBump:
    source_id: int
    generation: int


class MtmcRuntime:
    """Own an online fuser on one worker and publish read-only ID snapshots."""

    def __init__(
        self,
        config: MtmcConfig,
        *,
        expected_sources: Iterable[int],
        on_desync: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        expected = frozenset(int(source_id) for source_id in expected_sources)
        self._fuser = MtmcFuser(config, expected_sources=expected)
        self._on_desync = on_desync
        self._queue: Queue[object] = Queue()
        self._id_map: Mapping[tuple[int, int], int] = MappingProxyType({})
        self._qualified_id_map: Mapping[tuple[int, int, int], int] = MappingProxyType({})
        self._assignment_history: dict[tuple[int, int, int], int] = {}
        self._stats: Mapping[str, int] = MappingProxyType({})
        self._generations = {source_id: 0 for source_id in expected}
        self._submission_lock = Lock()
        self._thread: Thread | None = None
        self._started = False
        self._closed = False
        self._failure: Exception | None = None

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("MTMC runtime is closed")
        if self._started:
            return
        self._thread = Thread(target=self._run_worker, name="mtmc-fuser", daemon=True)
        self._thread.start()
        self._started = True

    def submit(
        self,
        timestamp_ns: int,
        source_id: int,
        observations: Iterable[GroundObservation],
    ) -> None:
        """Queue an immutable frame snapshot without doing fusion on the caller."""
        with self._submission_lock:
            self._require_running()
            if source_id not in self._generations:
                raise ValueError(f"unknown MTMC source {source_id}")
            self._queue.put(
                _ObservationBatch(
                    timestamp_ns=int(timestamp_ns),
                    source_id=int(source_id),
                    observations=tuple(observations),
                )
            )

    def bump_generation(self, source_id: int) -> int:
        """Invalidate a reconnected source immediately, then advance it on the worker."""
        with self._submission_lock:
            self._require_running()
            if source_id not in self._generations:
                raise ValueError(f"unknown MTMC source {source_id}")
            generation = self._generations[source_id] + 1
            self._generations[source_id] = generation
            self._id_map = MappingProxyType(
                {
                    key: global_id
                    for key, global_id in self._id_map.items()
                    if key[0] != source_id
                }
            )
            self._queue.put(_GenerationBump(source_id, generation))
            return generation

    def id_map_snapshot(self) -> Mapping[tuple[int, int], int]:
        """Return the currently published immutable source-local ID map."""
        return self._id_map

    def wait_until_idle(self) -> None:
        """Wait for queued work; intended for tests and orderly shutdown only."""
        self._queue.join()
        self._raise_worker_failure()

    def shutdown(self, *, json_path: str | Path | None = None) -> None:
        if self._closed:
            self._raise_worker_failure()
            if json_path is not None:
                self._write_json(json_path)
            return
        if self._started:
            assert self._thread is not None
            if self._failure is None and self._thread.is_alive():
                self._queue.put(_STOP)
                self._queue.join()
            self._thread.join()
        self._closed = True
        self._raise_worker_failure()
        if json_path is not None:
            self._write_json(json_path)

    def _run_worker(self) -> None:
        try:
            while True:
                work = self._queue.get()
                try:
                    if work is _STOP:
                        stats_before = self._fuser.stats()
                        self._fuser.flush_ready(force=True)
                        self._report_desync(stats_before, self._fuser.stats())
                        self._publish_snapshot()
                        return
                    if isinstance(work, _GenerationBump):
                        generation = self._fuser.bump_generation(work.source_id)
                        if generation != work.generation:
                            raise RuntimeError(
                                f"generation mismatch for source {work.source_id}: "
                                f"worker={generation}, submitted={work.generation}"
                            )
                        self._publish_snapshot()
                    else:
                        assert isinstance(work, _ObservationBatch)
                        self._fuser.observe(
                            work.timestamp_ns,
                            work.source_id,
                            work.observations,
                        )
                        stats_before = self._fuser.stats()
                        self._fuser.flush_ready()
                        self._report_desync(stats_before, self._fuser.stats())
                        self._publish_snapshot()
                finally:
                    self._queue.task_done()
        except Exception as exc:  # noqa: BLE001 - surface worker failures to caller
            self._failure = exc
            self._discard_queued_work()

    def _publish_snapshot(self) -> None:
        qualified = self._fuser.qualified_id_map()
        with self._submission_lock:
            generations = dict(self._generations)
        self._assignment_history.update(qualified)
        self._qualified_id_map = MappingProxyType(qualified)
        self._id_map = MappingProxyType(
            {
                (source_id, object_id): global_id
                for (source_id, generation, object_id), global_id in qualified.items()
                if generation == generations[source_id]
            }
        )
        self._stats = MappingProxyType(self._fuser.stats())

    def _write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "source_id": source_id,
                "generation": generation,
                "object_id": object_id,
                "global_id": global_id,
            }
            for (source_id, generation, object_id), global_id in sorted(
                self._assignment_history.items()
            )
        ]
        output_path.write_text(
            json.dumps({"id_map": rows, "stats": dict(self._stats)}, indent=2) + "\n"
        )

    def _report_desync(
        self,
        before: Mapping[str, int],
        after: Mapping[str, int],
    ) -> None:
        if self._on_desync is None:
            return
        incomplete = after["desync_buckets"] - before["desync_buckets"]
        if incomplete:
            self._on_desync(
                MappingProxyType(
                    {
                        "reason": "missing_sources",
                        "desync_buckets": incomplete,
                    }
                )
            )
        skew_refusals = after["skew_refusals"] - before["skew_refusals"]
        if skew_refusals:
            self._on_desync(
                MappingProxyType(
                    {
                        "reason": "timestamp_skew",
                        "skew_refusals": skew_refusals,
                    }
                )
            )

    def _discard_queued_work(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return
            self._queue.task_done()

    def _require_running(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("MTMC runtime is not running")
        self._raise_worker_failure()

    def _raise_worker_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError("MTMC worker failed") from self._failure


def frame_timestamp_ns(*, frame_num: int, ntp_timestamp: int, is_rtsp: bool) -> int:
    """Return the shared MTMC clock for a DeepStream frame."""
    if is_rtsp:
        timestamp_ns = int(ntp_timestamp)
        if timestamp_ns <= 0:
            raise ValueError("RTSP frame has no valid NTP timestamp")
        return timestamp_ns
    if frame_num < 0:
        raise ValueError("file frame number cannot be negative")
    return int(frame_num) * _FILE_FRAME_INTERVAL_NS


def project_person_detections(
    *,
    timestamp_ns: int,
    source_id: int,
    detections: Iterable[object],
    homography: Homography,
    embeddings: Mapping[int, np.ndarray] | None = None,
) -> tuple[GroundObservation, ...]:
    """Project reliable tracked-person foot points into the ground plane."""
    observations = []
    for detection in detections:
        if int(detection.class_id) != 0:
            continue
        left = float(detection.left)
        top = float(detection.top)
        width = float(detection.width)
        height = float(detection.height)
        if width <= 0.0 or height <= 0.0:
            continue
        if not is_foot_reliable(
            left,
            top,
            width,
            height,
            image_width=homography.image_width,
            image_height=homography.image_height,
        ):
            continue
        u, v = foot_point(left, top, width, height)
        pixel_sigma = foot_pixel_sigma(
            left,
            top,
            width,
            height,
            image_height=homography.image_height,
        )
        try:
            x, y = project_point(homography.matrix, u, v)
            sigma = projection_sigma(homography.matrix, u, v, pixel_sigma)
        except ValueError:
            continue
        if not np.isfinite((x, y, sigma)).all() or sigma <= 0.0:
            continue
        object_id = int(detection.object_id)
        embedding = None
        if embeddings is not None and object_id in embeddings:
            embedding = np.asarray(embeddings[object_id], dtype=np.float32).copy()
        observations.append(
            GroundObservation(
                timestamp_ns=int(timestamp_ns),
                source_id=int(source_id),
                object_id=object_id,
                x=x,
                y=y,
                sigma=sigma,
                embedding=embedding,
            )
        )
    return tuple(observations)


def with_global_ids(
    detections: Iterable[Detection],
    *,
    source_id: int,
    id_map: Mapping[tuple[int, int], int],
) -> list[Detection]:
    """Return CSV-ready detections while preserving local tracker object IDs."""
    return [
        replace(
            detection,
            source_id=source_id,
            global_id=id_map.get((source_id, int(detection.object_id)), -1),
        )
        for detection in detections
    ]


def load_homography_bindings(
    bindings: Iterable[str],
    *,
    expected_sources: Iterable[int],
    image_width: int,
    image_height: int,
) -> Mapping[int, Homography]:
    """Load and validate explicit ``SOURCE_ID=PATH`` homography bindings."""
    expected = frozenset(int(source_id) for source_id in expected_sources)
    loaded: dict[int, Homography] = {}
    for binding in bindings:
        source_text, separator, path = binding.partition("=")
        if not separator or not source_text or not path:
            raise ValueError(
                f"invalid homography binding {binding!r}; expected SOURCE_ID=PATH"
            )
        try:
            source_id = int(source_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid homography source ID {source_text!r} in {binding!r}"
            ) from exc
        if source_id in loaded:
            raise ValueError(f"duplicate homography binding for source {source_id}")
        if source_id not in expected:
            raise ValueError(f"homography bound to unknown source {source_id}")

        homography = load_homography(path)
        if homography.source_id != source_id:
            raise ValueError(
                f"homography {path!r} declares source {homography.source_id}, "
                f"but is bound to source {source_id}"
            )
        if (homography.image_width, homography.image_height) != (
            image_width,
            image_height,
        ):
            raise ValueError(
                f"homography for source {source_id} targets "
                f"{homography.image_width}x{homography.image_height}, expected "
                f"{image_width}x{image_height}"
            )
        if not np.isfinite(homography.matrix).all():
            raise ValueError(f"homography for source {source_id} contains non-finite values")
        if np.linalg.matrix_rank(homography.matrix) < 3:
            raise ValueError(f"homography for source {source_id} is singular")
        loaded[source_id] = homography

    missing = expected.difference(loaded)
    if missing:
        missing_text = ", ".join(str(source_id) for source_id in sorted(missing))
        raise ValueError(f"missing homography bindings for sources: {missing_text}")
    return MappingProxyType(loaded)
