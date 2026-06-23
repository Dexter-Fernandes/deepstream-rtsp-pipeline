import numpy as np


def parse_yolo26_output(
    tensor: np.ndarray,
    conf_threshold: float = 0.25,
) -> list[dict]:
    """
    Decode YOLO26n output tensor into detection dicts.

    Input shape: [1, 300, 6] or [300, 6].
    Each row: [x1, y1, x2, y2, confidence, class_id] in pixel space (0–640).
    Returns list of {left, top, width, height, confidence, class_id}.
    NMS is already applied by the model's one-to-one matching head.
    """
    if tensor.ndim == 3:
        tensor = tensor[0]
    results = []
    for row in tensor:
        conf = float(row[4])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        results.append({
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "confidence": conf,
            "class_id": int(row[5]),
        })
    return results
