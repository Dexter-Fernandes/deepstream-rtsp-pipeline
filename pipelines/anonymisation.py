import cv2
import numpy as np

from pipelines.metadata_parser import Detection


def blur_bboxes(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    h, w = frame.shape[:2]
    for det in detections:
        x1 = max(0, int(det.left))
        y1 = max(0, int(det.top))
        x2 = min(w, int(det.left + det.width))
        y2 = min(h, int(det.top + det.height))
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (51, 51), 0)
    return frame
