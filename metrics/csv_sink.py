import csv
from pathlib import Path

from pipelines.metadata_parser import Detection

_HEADER = ["frame_num", "object_id", "class_id", "class_label",
           "confidence", "left", "top", "width", "height"]


class CsvSink:
    def __init__(self, path: str | Path) -> None:
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_HEADER)

    def write(self, detections: list[Detection]) -> None:
        for d in detections:
            self._writer.writerow([
                d.frame_num, d.object_id, d.class_id, d.class_label,
                d.confidence, d.left, d.top, d.width, d.height,
            ])

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
