"""
Ground-truth loader for the hand-labelled WildTrack pedestrian set.

Labels are Supervisely/DatasetNinja-format JSON, one file per image, at
<root>/gt_labels/<camera>/<image_name>.json, mirroring images under
<root>/images/<camera>/<image_name>. Boxes are axis-aligned rectangles given
as two corner points in native image pixel space.

Usage:
    from metrics.wildtrack_gt import find_wildtrack_pairs, load_wildtrack_gt

    pairs = find_wildtrack_pairs(Path(".../labelled_ds"), cameras=["C1", "C2"])
    for image_path, json_path in pairs:
        boxes = load_wildtrack_gt(json_path)
"""

import json
from pathlib import Path

_PEDESTRIAN_CLASS = "pedestrian"


def decode_position_id(
    position_id: int,
    *,
    grid_width: int = 480,
    origin_x: float = -3.0,
    origin_y: float = -9.0,
    cell_m: float = 0.025,
) -> tuple[float, float]:
    """Decode a WildTrack ground-grid cell into world coordinates in metres."""
    grid_x = int(position_id) % grid_width
    grid_y = int(position_id) // grid_width
    return origin_x + cell_m * grid_x, origin_y + cell_m * grid_y


def frame_index_from_stem(stem: str, *, step: int = 5) -> int:
    """Map a labelled image stem such as C1_00000045 to zero-based clip frame 9."""
    _, separator, annotation_index = stem.rpartition("_")
    if not separator or not annotation_index.isdigit():
        raise ValueError(f"invalid WildTrack image stem: {stem!r}")
    index = int(annotation_index)
    if index % step:
        raise ValueError(f"annotation index {index} is not divisible by step {step}")
    return index // step


def parse_wildtrack_annotation(annotation: dict, *, with_identity: bool = False) -> list[dict]:
    """Extract pedestrian boxes from one parsed Supervisely annotation dict.

    Returns list of {left, top, width, height} in native image pixel space.
    When with_identity is true, also returns the global person_id and ground
    grid position_id stored in the object's tags.
    """
    boxes = []
    for obj in annotation.get("objects", []):
        if obj.get("classTitle") != _PEDESTRIAN_CLASS:
            continue
        exterior = obj["points"]["exterior"]
        (x1, y1), (x2, y2) = exterior[0], exterior[1]
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        box = {
            "left": float(left),
            "top": float(top),
            "width": float(width),
            "height": float(height),
        }
        if with_identity:
            tags = {tag.get("name"): tag.get("value") for tag in obj.get("tags", [])}
            box["person_id"] = int(tags["person id"])
            box["position_id"] = int(tags["position id"])
        boxes.append(box)
    return boxes


def load_wildtrack_gt(json_path: Path, *, with_identity: bool = False) -> list[dict]:
    """Load and parse one Supervisely annotation file."""
    annotation = json.loads(Path(json_path).read_text())
    return parse_wildtrack_annotation(annotation, with_identity=with_identity)


def find_wildtrack_pairs(root: Path, cameras: list[str] | None = None) -> list[tuple[Path, Path]]:
    """Pair each labelled image with its annotation JSON, sorted by (camera, filename).

    root is the labelled_ds/ directory containing images/ and gt_labels/.
    cameras restricts to a subset (e.g. ["C1", "C2"]); None uses every camera present.
    """
    images_dir = Path(root) / "images"
    gt_dir = Path(root) / "gt_labels"

    camera_dirs = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    if cameras is not None:
        camera_dirs = [c for c in camera_dirs if c in cameras]

    pairs = []
    for camera in camera_dirs:
        for image_path in sorted((images_dir / camera).glob("*.png")):
            json_path = gt_dir / camera / f"{image_path.name}.json"
            if json_path.exists():
                pairs.append((image_path, json_path))
    return pairs


def load_wildtrack_mtmc_gt(
    root: Path,
    cameras: list[str],
    camera_to_source: dict[str, int],
) -> list[dict]:
    """Load identity-aware WildTrack rows with an explicit camera/source binding."""
    missing_bindings = set(cameras) - camera_to_source.keys()
    if missing_bindings:
        missing = ", ".join(sorted(missing_bindings))
        raise ValueError(f"missing source ID binding for camera(s): {missing}")
    source_ids = [camera_to_source[camera] for camera in cameras]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("each camera must have a unique source ID")

    rows = []
    for image_path, json_path in find_wildtrack_pairs(root, cameras=cameras):
        camera = image_path.parent.name
        frame_num = frame_index_from_stem(image_path.stem)
        annotation = json.loads(json_path.read_text())
        image_size = annotation["size"]
        for box in parse_wildtrack_annotation(annotation, with_identity=True):
            world_x, world_y = decode_position_id(box["position_id"])
            rows.append(
                {
                    "frame_num": frame_num,
                    "camera": camera,
                    "source_id": int(camera_to_source[camera]),
                    **box,
                    "world_x": world_x,
                    "world_y": world_y,
                    "image_width": int(image_size["width"]),
                    "image_height": int(image_size["height"]),
                }
            )
    return rows


def to_mot_rows(rows: list[dict], source_id: int | None = None) -> list[dict]:
    """Adapt identity-aware rows to the existing tracker evaluator's MOT row contract."""
    return [
        {
            "frame": int(row["frame_num"]) + 1,
            "obj_id": int(row["person_id"]),
            "left": float(row["left"]),
            "top": float(row["top"]),
            "width": float(row["width"]),
            "height": float(row["height"]),
        }
        for row in rows
        if source_id is None or int(row["source_id"]) == source_id
    ]
