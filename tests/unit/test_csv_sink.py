import csv

from metrics.csv_sink import CsvSink
from pipelines.metadata_parser import Detection

HEADER = ["frame_num", "object_id", "class_id", "class_label",
          "confidence", "left", "top", "width", "height"]


def _det(**kwargs) -> Detection:
    defaults = dict(frame_num=0, object_id=1, class_id=0, class_label="car",
                    confidence=0.9, left=10.0, top=20.0, width=100.0, height=50.0)
    defaults.update(kwargs)
    return Detection(**defaults)


def test_creates_file_with_header(tmp_path):
    path = tmp_path / "out.csv"
    sink = CsvSink(path)
    sink.close()
    rows = list(csv.reader(path.open()))
    assert rows[0] == HEADER


def test_write_appends_row(tmp_path):
    path = tmp_path / "out.csv"
    sink = CsvSink(path)
    sink.write([_det()])
    sink.close()
    rows = list(csv.reader(path.open()))
    assert len(rows) == 2  # header + 1 data row


def test_write_field_values(tmp_path):
    path = tmp_path / "out.csv"
    sink = CsvSink(path)
    sink.write([_det(frame_num=5, object_id=3, class_id=1, class_label="person",
                     confidence=0.75, left=1.0, top=2.0, width=3.0, height=4.0)])
    sink.close()
    rows = list(csv.DictReader(path.open()))
    r = rows[0]
    assert int(r["frame_num"]) == 5
    assert int(r["object_id"]) == 3
    assert r["class_label"] == "person"
    assert float(r["confidence"]) == 0.75


def test_write_flushes_without_explicit_close(tmp_path):
    path = tmp_path / "out.csv"
    sink = CsvSink(path)
    sink.write([_det()])
    # data must be on disk immediately — no close() called
    rows = list(csv.reader(path.open()))
    assert len(rows) == 2
    sink.close()


def test_flush_persists_data(tmp_path):
    path = tmp_path / "out.csv"
    sink = CsvSink(path)
    sink.write([_det()])
    sink.flush()
    rows = list(csv.reader(path.open()))
    assert len(rows) == 2


def test_write_multiple_detections(tmp_path):
    path = tmp_path / "out.csv"
    sink = CsvSink(path)
    sink.write([_det(object_id=1), _det(object_id=2)])
    sink.close()
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert {int(r["object_id"]) for r in rows} == {1, 2}
