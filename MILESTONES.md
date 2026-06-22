# Milestones & Implementation Steps

> **TDD convention:** every implementation step follows red→green. Write the test first, confirm it fails, then write the minimal code to pass. Unit tests live alongside the milestone they belong to — not saved for M3.4.

## M1 — Pipeline Plumbing

**Exit criteria:** live bounding boxes rendering from RTSP source (mediamtx/MOT17) using stock detection model.

### M1.1 — Environment Setup ✓
- [x] Pull NGC DeepStream container (upgraded to `nvcr.io/nvidia/deepstream:9.0-triton-multiarch`; requires driver ≥ 590.48)
- [x] Verify GPU passthrough: `docker run --gpus all --rm nvcr.io/nvidia/deepstream:7.1-triton-multiarch nvidia-smi`
- [x] Install `mediamtx` on host; confirm it starts and serves an RTSP path
- [x] Convert MOT17-04, MOT17-13, MOT17-02 image sequences to MP4 via ffmpeg
- [x] Configure mediamtx (`configs/mediamtx.yml`) to serve all three on `stream0/1/2`
- [x] Verify RTSP streams playable: `ffplay rtsp://localhost:8554/stream0`

### M1.2 — RTSP Pipeline ✓
- [x] Scaffold `pipelines/rtsp.py`
- [x] Chain: `rtspsrc → rtph264depay → nvv4l2decoder → nvstreammux → nvinfer → nvtracker → nvdsosd` (nvosd renamed to nvdsosd in DS 7.x)
- [x] Use ResNet-18 TrafficCamNet FP32 as placeholder `nvinfer` config (ResNet-10 Caffe model absent in DS 7.1; ONNX model used instead)
- [x] Test against `rtsp://localhost:8554/stream0` (MOT17-04) — pipeline runs stably
- [x] Add RTSP reconnect handling (`rtspsrc` `retry` and `timeout` properties)
- [x] Confirm pipeline launches without GST errors; pipeline running confirmed
- [x] Pre-build TensorRT FP32 engine via `trtexec`; cache in `models/engines/` (mounted at run time)
- [x] 9 unit tests passing (CPU, CI-safe): config defaults, arg parsing, source properties

### M1.4 — CSV Metadata Sink ✓
- [x] `pipelines/metadata_parser.py` — `Detection` dataclass + `parse_frame_meta()` (TDD, 20 tests passing)
- [x] `metrics/csv_sink.py` — `CsvSink` with `write()` + `flush()` (TDD, 20 tests passing)
- [x] Add pyds `GstBuffer` probe on `nvdsosd` sink pad wiring `parse_frame_meta` + `CsvSink`
- [x] Confirm CSV populated: `wc -l output.csv` → 28298 lines on live pipeline (stream0, MOT17-04)
- [x] Docker image updated to DS 9.0 (`nvcr.io/nvidia/deepstream:9.0-triton-multiarch`); pyds compiled from master branch; NVIDIA driver upgraded to 595

### M1.5 — Anonymisation Probe ✓
- [x] `pipelines/anonymisation.py` — `blur_bboxes()` (TDD, 6 tests passing)
- [x] Wire probe after `nvosd` calling `blur_bboxes` on detected bbox regions
- [ ] Confirm faces/plates are blurred in display output
- [x] Add "Privacy by Design" section to README

---

## M2 — Custom Model + C++ Decode Plugin

**Exit criteria:** YOLOv8n running through DeepStream at FP16 with stable ByteTrack IDs; C++ decode plugin built and benchmarked.

### M2.1 — YOLOv8n Export
- [ ] Install `ultralytics` in container
- [ ] Export `yolov8n.pt` → ONNX (no NMS in graph): `model.export(format='onnx', nms=False)`
- [ ] Write `models/convert.py`: ONNX → TensorRT FP32 baseline engine via `trtexec`
- [ ] Add FP16 conversion path to `convert.py`: `--fp16` flag
- [ ] Document INT8 exclusion in `convert.py` comments (no Tensor Cores on 1660Ti)
- [ ] Confirm both engines load in TensorRT without error

### M2.2 — nvinfer Custom Output Parser
- [ ] `models/output_parser.py` — `parse_yolov8_output()` anchor-free decode (TDD, 5 tests passing)
- [ ] Wire parser into `nvinfer` config (`parse-bbox-func-name`, `custom-lib-path`)
- [ ] Confirm detections appear correctly on MOT17-04 stream (visual check)

### M2.3 — C++ Decode Plugin (Part 1)
- [ ] Scaffold `plugins/yolov8_decode/` with `CMakeLists.txt`
- [ ] Implement `IPluginV2DynamicExt` skeleton: `getOutputDimensions`, `enqueue`, `serialize`
- [ ] Implement CUDA kernel for sigmoid + anchor-free box coordinate transform in `enqueue`
- [ ] Build plugin `.so`: `cmake .. && make` inside container

### M2.4 — C++ Decode Plugin (Part 2)
- [ ] Load plugin in `convert.py` via `trt.Runtime` / `ctypes.CDLL`
- [ ] Re-export TensorRT engine with plugin replacing CPU ONNX decode nodes
- [ ] Confirm detections match CPU-decode baseline (bbox coordinates identical within tolerance)
- [ ] Add TensorRT `IProfiler` instrumentation to isolate decode step latency

### M2.5 — Decode Plugin Comparison Report
- [ ] Run pipeline with FP32 CPU-decode engine; record latency, FPS, VRAM, decode step time
- [ ] Run pipeline with FP16 GPU-decode plugin engine; record same metrics
- [ ] Produce `metrics/decode_comparison.ipynb` with side-by-side table and commentary

---

## M3 — Tracker Comparison + Multi-Stream + Hardening

**Exit criteria:** tracker comparison report with full MOTA/HOTA/IDF1; 3-stream pipeline running; all tests passing; docs complete.

### M3.1 — Tracker Configs
- [ ] Write `configs/tracker_iou.yml` (IOU tracker config for DeepStream)
- [ ] Write `configs/tracker_nvdcf.yml` (NvDCF config)
- [ ] Write `configs/tracker_bytetrack.yml` (ByteTrack config)
- [ ] Add `--tracker` CLI flag to `pipelines/rtsp.py` to swap configs without code change
- [ ] Run each tracker on MOT17-04 for 60 seconds; save separate CSV per tracker

### M3.2 — Tracker Metrics (py-motmetrics)
- [ ] Install `py-motmetrics` in container
- [ ] Write `metrics/evaluate_tracker.py`: load MOT17-04 GT + pipeline CSV → compute MOTA, MOTP, HOTA, IDF1
- [ ] Add no-GT metrics: parse CSV for ID switches, track fragmentation, mean FPS, peak VRAM
- [ ] Produce `metrics/tracker_comparison.ipynb` with full comparison table

### M3.3 — Multi-Stream Pipeline
- [ ] Write `pipelines/multi_stream.py` — 3 concurrent `rtspsrc` bins into single `nvstreammux`
- [ ] Configure `nvstreammux` batch size, width, height for 3-stream load
- [ ] Serve MOT17-04 on `stream0`, MOT17-13 on `stream1`, MOT17-02 on `stream2`
- [ ] Run for 30 minutes; record per-stream FPS and aggregate VRAM
- [ ] Document throughput degradation curve (1→2→3 streams) in README

### M3.4 — GPU Tests + CI
- [ ] Write `tests/smoke/test_pipeline_smoke.py` — launch rtsp pipeline for 10 seconds, assert frames_processed > 0 and CSV non-empty (requires GPU, `pytest --gpu`)
- [ ] Write `tests/integration/test_motmetrics_integration.py` — run metrics on known MOT17-04 excerpt, assert HOTA within expected range (requires GPU, `pytest --gpu`)
- [ ] Wire GitHub Actions workflow for unit tests (CPU only, no GPU runner)

### M3.5 — Docs + README
- [ ] Write `docs/jetson-upgrade.md` — component diff table: x86 dGPU config → Jetson equivalent (nvargus, JetPack, INT8, unified memory, TDP modes)
- [ ] Write `docs/isp-and-camera-input.md` — ISP pipeline stages (demosaicing, AWB, gamma, lens distortion), nvargus on Jetson, how ISP misconfiguration degrades model accuracy
- [ ] Write `docs/system-design.md` — fleet-scale architecture: 1→5000 sensors, edge→cloud metadata path, sensor failure/reconnect, JetPack fleet upgrade strategy
- [ ] Complete README: architecture diagram, quickstart, tracker comparison table summary, decode plugin results, known gaps with explicit reasoning, Privacy by Design section
- [ ] Final 30-minute stability run on RTSP pipeline; confirm no crash/leak

---

## Stretch Goals (post M3)

- [ ] Grafana dashboard wired to CSV/InfluxDB output
- [ ] Full C++ pipeline refactor (using `plugins/` as the starting point)
- [ ] INT8 calibration pipeline (document calibration dataset requirements; defer execution to RTX/Jetson hardware)
- [ ] `docs/golang-integration.md` — note on where Go fits in VivaCity's stack (JD nice-to-have)
