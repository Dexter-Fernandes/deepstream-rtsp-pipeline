# deepstream-rtsp-pipeline
Built a video analytics pipeline using NVIDIA DeepStream. It pulls from RTSP streams or a webcam, runs a custom YOLO model for detection, and tracks objects across frames in real time on consumer hardware.

## Privacy by Design

The pipeline applies Gaussian blur to every detected bounding-box region before frames reach the display sink. This is implemented as a GStreamer buffer probe on the `nvdsosd` sink pad in `pipelines/rtsp.py`, calling `blur_bboxes()` (`pipelines/anonymisation.py`) on the raw `NvBufSurface`-backed numpy array for each frame in the batch.

Blurring runs _before_ `nvdsosd` renders the overlay boxes, so the anonymised pixels are written back into the GPU surface and any downstream consumer (display or encode) sees the blurred content. No raw face or licence-plate data is written to the CSV metadata sink — only the bounding-box coordinates and class labels are persisted.

`blur_bboxes()` clips all coordinates to the frame boundary and skips zero-area regions, so out-of-range detections are handled safely without crashing the pipeline.
