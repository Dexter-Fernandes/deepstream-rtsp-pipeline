from dataclasses import dataclass, field

from pipelines.metadata_parser import Detection, parse_frame_meta

# ---------------------------------------------------------------------------
# Fake pyds-like objects (duck-typed stand-ins for the real NvDs structs)
# ---------------------------------------------------------------------------

@dataclass
class _Rect:
    left: float = 10.0
    top: float = 20.0
    width: float = 100.0
    height: float = 50.0


@dataclass
class _ObjMeta:
    object_id: int = 1
    class_id: int = 2
    obj_label: str = "person"
    confidence: float = 0.9
    rect_params: _Rect = field(default_factory=_Rect)


@dataclass
class _Node:
    data: object = None
    next: object = None  # next _Node or None


@dataclass
class _FrameMeta:
    frame_num: int = 0
    source_id: int = -1
    obj_meta_list: object = None  # _Node or None


@dataclass
class _BatchMeta:
    frame_meta_list: object = None  # _Node or None


# Cast helper that passes the fake objects straight through.
def _id(value):
    return value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_batch_returns_empty_list():
    batch = _BatchMeta(frame_meta_list=None)
    assert parse_frame_meta(batch, _cast_frame=_id, _cast_obj=_id) == []


def test_single_detection_frame_num():
    obj_node = _Node(data=_ObjMeta())
    frame = _FrameMeta(frame_num=42, obj_meta_list=obj_node)
    batch = _BatchMeta(frame_meta_list=_Node(data=frame))
    result = parse_frame_meta(batch, _cast_frame=_id, _cast_obj=_id)
    assert result[0].frame_num == 42


def test_single_detection_object_id():
    obj_node = _Node(data=_ObjMeta(object_id=7))
    frame = _FrameMeta(frame_num=0, obj_meta_list=obj_node)
    batch = _BatchMeta(frame_meta_list=_Node(data=frame))
    result = parse_frame_meta(batch, _cast_frame=_id, _cast_obj=_id)
    assert result[0].object_id == 7


def test_detection_identity_fields_are_backward_compatible_defaults():
    detection = Detection(
        frame_num=0,
        object_id=7,
        class_id=0,
        class_label="person",
        confidence=0.9,
        left=1.0,
        top=2.0,
        width=3.0,
        height=4.0,
    )

    assert detection.source_id == -1
    assert detection.global_id == -1
    assert detection.generation == 0


def test_frame_source_id_is_preserved_on_each_detection():
    obj_node = _Node(data=_ObjMeta(object_id=7))
    frame = _FrameMeta(frame_num=3, source_id=2, obj_meta_list=obj_node)
    batch = _BatchMeta(frame_meta_list=_Node(data=frame))

    result = parse_frame_meta(batch, _cast_frame=_id, _cast_obj=_id)

    assert result[0].source_id == 2


def test_single_detection_class_and_label():
    obj_node = _Node(data=_ObjMeta(class_id=0, obj_label="car"))
    frame = _FrameMeta(frame_num=0, obj_meta_list=obj_node)
    batch = _BatchMeta(frame_meta_list=_Node(data=frame))
    result = parse_frame_meta(batch, _cast_frame=_id, _cast_obj=_id)
    assert result[0].class_id == 0
    assert result[0].class_label == "car"


def test_single_detection_bbox():
    rect = _Rect(left=5.0, top=15.0, width=80.0, height=40.0)
    obj_node = _Node(data=_ObjMeta(rect_params=rect))
    frame = _FrameMeta(frame_num=0, obj_meta_list=obj_node)
    batch = _BatchMeta(frame_meta_list=_Node(data=frame))
    result = parse_frame_meta(batch, _cast_frame=_id, _cast_obj=_id)
    d = result[0]
    assert d.left == 5.0
    assert d.top == 15.0
    assert d.width == 80.0
    assert d.height == 40.0


def test_multiple_detections_in_frame():
    obj_a = _Node(data=_ObjMeta(object_id=1, obj_label="car"))
    obj_b = _Node(data=_ObjMeta(object_id=2, obj_label="person"))
    obj_a.next = obj_b
    frame = _FrameMeta(frame_num=0, obj_meta_list=obj_a)
    batch = _BatchMeta(frame_meta_list=_Node(data=frame))
    result = parse_frame_meta(batch, _cast_frame=_id, _cast_obj=_id)
    assert len(result) == 2
    assert {d.object_id for d in result} == {1, 2}
