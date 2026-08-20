import csv
from pathlib import Path

from pipelines.metadata_parser import Detection

_HEADER = ["frame_num", "object_id", "class_id", "class_label",
           "confidence", "left", "top", "width", "height",
           "source_id", "global_id", "generation", "association_bucket",
           "association_accepted"]


class CsvSink:
    def __init__(self, path: str | Path) -> None:
        self._file = open(path, "w", newline="")  # noqa: SIM115 - owned until close()
        self._writer = csv.writer(self._file)
        self._writer.writerow(_HEADER)

    def write(self, detections: list[Detection]) -> None:
        for d in detections:
            self._writer.writerow([
                d.frame_num, d.object_id, d.class_id, d.class_label,
                d.confidence, d.left, d.top, d.width, d.height,
                d.source_id, d.global_id, d.generation,
                "" if d.association_bucket is None else d.association_bucket,
                int(d.association_accepted),
            ])
        self._file.flush()

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
