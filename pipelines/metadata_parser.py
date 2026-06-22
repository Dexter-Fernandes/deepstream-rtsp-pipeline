from dataclasses import dataclass


@dataclass
class Detection:
    frame_num: int
    object_id: int
    class_id: int
    class_label: str
    confidence: float
    left: float
    top: float
    width: float
    height: float


def parse_frame_meta(
    batch_meta,
    *,
    _cast_frame=None,
    _cast_obj=None,
) -> list[Detection]:
    if _cast_frame is None:
        import pyds
        _cast_frame = pyds.NvDsFrameMeta.cast
    if _cast_obj is None:
        import pyds
        _cast_obj = pyds.NvDsObjectMeta.cast

    detections: list[Detection] = []
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = _cast_frame(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj = _cast_obj(l_obj.data)
            detections.append(Detection(
                frame_num=frame_meta.frame_num,
                object_id=int(obj.object_id),
                class_id=obj.class_id,
                class_label=obj.obj_label,
                confidence=obj.confidence,
                left=obj.rect_params.left,
                top=obj.rect_params.top,
                width=obj.rect_params.width,
                height=obj.rect_params.height,
            ))
            l_obj = l_obj.next
        l_frame = l_frame.next
    return detections
