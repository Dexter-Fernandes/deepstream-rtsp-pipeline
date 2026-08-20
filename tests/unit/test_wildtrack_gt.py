import json
from pathlib import Path

import pytest

from metrics.wildtrack_gt import (
    decode_position_id,
    find_wildtrack_pairs,
    frame_index_from_stem,
    load_wildtrack_gt,
    load_wildtrack_mtmc_gt,
    parse_wildtrack_annotation,
    to_mot_rows,
)


def _annotation(objects: list[dict]) -> dict:
    return {"description": "", "tags": [], "size": {"height": 1080, "width": 1920}, "objects": objects}


def _pedestrian(x1, y1, x2, y2) -> dict:
    return {
        "classTitle": "pedestrian",
        "geometryType": "rectangle",
        "points": {"exterior": [[x1, y1], [x2, y2]], "interior": []},
    }


def _with_identity(obj: dict, *, person_id: int, position_id: int) -> dict:
    obj["tags"] = [
        {"name": "person id", "value": person_id},
        {"name": "position id", "value": position_id},
    ]
    return obj


# ---------------------------------------------------------------------------
# parse_wildtrack_annotation
# ---------------------------------------------------------------------------


def test_parse_wildtrack_annotation_single_box():
    ann = _annotation([_pedestrian(1510, 139, 1561, 299)])
    boxes = parse_wildtrack_annotation(ann)
    assert boxes == [{"left": 1510.0, "top": 139.0, "width": 51.0, "height": 160.0}]


def test_parse_wildtrack_annotation_reversed_corners():
    # exterior corners aren't guaranteed min-first; box geometry must not depend on order
    ann = _annotation([_pedestrian(1561, 299, 1510, 139)])
    boxes = parse_wildtrack_annotation(ann)
    assert boxes == [{"left": 1510.0, "top": 139.0, "width": 51.0, "height": 160.0}]


def test_parse_wildtrack_annotation_skips_non_pedestrian():
    obj = _pedestrian(0, 0, 10, 10)
    obj["classTitle"] = "car"
    ann = _annotation([obj])
    assert parse_wildtrack_annotation(ann) == []


def test_parse_wildtrack_annotation_empty_objects():
    assert parse_wildtrack_annotation(_annotation([])) == []


def test_parse_wildtrack_annotation_multiple_boxes():
    ann = _annotation([_pedestrian(0, 0, 10, 20), _pedestrian(50, 50, 70, 90)])
    boxes = parse_wildtrack_annotation(ann)
    assert len(boxes) == 2
    assert boxes[1] == {"left": 50.0, "top": 50.0, "width": 20.0, "height": 40.0}


def test_parse_wildtrack_annotation_identity_is_opt_in():
    pedestrian = _with_identity(
        _pedestrian(1510, 139, 1561, 299),
        person_id=122,
        position_id=456826,
    )
    ann = _annotation([pedestrian])

    assert parse_wildtrack_annotation(ann) == [
        {"left": 1510.0, "top": 139.0, "width": 51.0, "height": 160.0}
    ]
    assert parse_wildtrack_annotation(ann, with_identity=True) == [
        {
            "left": 1510.0,
            "top": 139.0,
            "width": 51.0,
            "height": 160.0,
            "person_id": 122,
            "position_id": 456826,
        }
    ]


# ---------------------------------------------------------------------------
# load_wildtrack_gt
# ---------------------------------------------------------------------------


def test_load_wildtrack_gt_reads_file(tmp_path: Path):
    ann_path = tmp_path / "C1_00000000.png.json"
    ann_path.write_text(json.dumps(_annotation([_pedestrian(0, 0, 10, 10)])))
    boxes = load_wildtrack_gt(ann_path)
    assert boxes == [{"left": 0.0, "top": 0.0, "width": 10.0, "height": 10.0}]


def test_load_wildtrack_gt_can_include_identity(tmp_path: Path):
    ann_path = tmp_path / "C1_00000000.png.json"
    pedestrian = _with_identity(_pedestrian(0, 0, 10, 20), person_id=7, position_id=480)
    ann_path.write_text(json.dumps(_annotation([pedestrian])))

    boxes = load_wildtrack_gt(ann_path, with_identity=True)

    assert boxes[0]["person_id"] == 7
    assert boxes[0]["position_id"] == 480


def test_decode_position_id_maps_grid_cells_to_world_metres():
    assert decode_position_id(0) == pytest.approx((-3.0, -9.0))
    assert decode_position_id(481) == pytest.approx((-2.975, -8.975))


def test_frame_index_from_stem_preserves_the_two_fps_clip_mapping():
    assert frame_index_from_stem("C1_00000045") == 9


# ---------------------------------------------------------------------------
# find_wildtrack_pairs
# ---------------------------------------------------------------------------


def _make_labelled_ds(tmp_path: Path, cameras: dict[str, list[str]]) -> Path:
    root = tmp_path / "labelled_ds"
    for camera, names in cameras.items():
        img_dir = root / "images" / camera
        gt_dir = root / "gt_labels" / camera
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        for name in names:
            (img_dir / f"{name}.png").write_bytes(b"")
            (gt_dir / f"{name}.png.json").write_text(json.dumps(_annotation([])))
    return root


def test_find_wildtrack_pairs_all_cameras(tmp_path: Path):
    root = _make_labelled_ds(tmp_path, {"C1": ["a", "b"], "C2": ["c"]})
    pairs = find_wildtrack_pairs(root)
    assert len(pairs) == 3
    assert all(img.exists() and js.exists() for img, js in pairs)


def test_find_wildtrack_pairs_camera_filter(tmp_path: Path):
    root = _make_labelled_ds(tmp_path, {"C1": ["a"], "C2": ["b"], "C3": ["c"]})
    pairs = find_wildtrack_pairs(root, cameras=["C2"])
    assert len(pairs) == 1
    assert pairs[0][0].parent.name == "C2"


def test_find_wildtrack_pairs_skips_missing_annotation(tmp_path: Path):
    root = _make_labelled_ds(tmp_path, {"C1": ["a"]})
    (root / "images" / "C1" / "orphan.png").write_bytes(b"")
    pairs = find_wildtrack_pairs(root)
    assert len(pairs) == 1


def test_load_wildtrack_mtmc_gt_binds_camera_frame_identity_and_ground_position(tmp_path: Path):
    root = _make_labelled_ds(tmp_path, {"C1": ["C1_00000005"]})
    pedestrian = _with_identity(
        _pedestrian(10, 20, 30, 60),
        person_id=7,
        position_id=481,
    )
    annotation_path = root / "gt_labels" / "C1" / "C1_00000005.png.json"
    annotation_path.write_text(json.dumps(_annotation([pedestrian])))

    rows = load_wildtrack_mtmc_gt(root, ["C1"], {"C1": 4})

    assert rows == [
        {
            "frame_num": 1,
            "camera": "C1",
            "source_id": 4,
            "person_id": 7,
            "position_id": 481,
            "world_x": pytest.approx(-2.975),
            "world_y": pytest.approx(-8.975),
            "left": 10.0,
            "top": 20.0,
            "width": 20.0,
            "height": 40.0,
            "image_width": 1920,
            "image_height": 1080,
        }
    ]


def test_load_wildtrack_mtmc_gt_rejects_ambiguous_source_bindings(tmp_path: Path):
    root = _make_labelled_ds(
        tmp_path,
        {"C1": ["C1_00000000"], "C2": ["C2_00000000"]},
    )

    with pytest.raises(ValueError, match="unique source ID"):
        load_wildtrack_mtmc_gt(root, ["C1", "C2"], {"C1": 0, "C2": 0})


def test_to_mot_rows_converts_frames_and_identity_for_one_source():
    rows = [
        {
            "frame_num": 0,
            "source_id": 0,
            "person_id": 12,
            "left": 10.0,
            "top": 20.0,
            "width": 30.0,
            "height": 40.0,
        },
        {
            "frame_num": 0,
            "source_id": 1,
            "person_id": 99,
            "left": 1.0,
            "top": 2.0,
            "width": 3.0,
            "height": 4.0,
        },
    ]

    assert to_mot_rows(rows, source_id=0) == [
        {
            "frame": 1,
            "obj_id": 12,
            "left": 10.0,
            "top": 20.0,
            "width": 30.0,
            "height": 40.0,
        }
    ]
