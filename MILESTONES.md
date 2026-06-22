# Milestones & Implementation Steps

> **TDD convention:** every implementation step follows red→green. Write the test first, confirm it fails, then write the minimal code to pass. Unit tests live alongside the milestone they belong to — not saved for M3.4.

## M1 — Pipeline Plumbing

**Exit criteria:** detections written to CSV from all three RTSP streams concurrently via a single `nvstreammux`; blurred output re-streamed and visible via `ffplay`; FP32 baseline FPS recorded.

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
- [x] TensorRT FP32 engine built by nvinfer on first run; cached in `models/primary_detector/` (workspace volume mount — persists across container restarts)
- [x] 9 unit tests passing (CPU, CI-safe): config defaults, arg parsing, source properties

### M1.3 — (merged into M1.2)

### M1.4 — CSV Metadata Sink ✓
- [x] `pipelines/metadata_parser.py` — `Detection` dataclass + `parse_frame_meta()` (TDD, 20 tests passing)
- [x] `metrics/csv_sink.py` — `CsvSink` with `write()` + `flush()` (TDD, 20 tests passing)
- [x] Add pyds `GstBuffer` probe on `nvdsosd` sink pad wiring `parse_frame_meta` + `CsvSink`
- [x] Confirm CSV populated: `wc -l output.csv` → 28298 lines on live pipeline (stream0, MOT17-04)
- [x] Docker image updated to DS 9.0 (`nvcr.io/nvidia/deepstream:9.0-triton-multiarch`); pyds compiled from master branch; NVIDIA driver upgraded to 595

### M1.5 — Anonymisation Function ✓
- [x] `pipelines/anonymisation.py` — `blur_bboxes()` (TDD, 6 tests passing)
- [x] Add "Privacy by Design" section to README

### M1.6 — Stream Validation
- [x] Run pipeline against `stream0` (MOT17-04), `stream1` (MOT17-13), `stream2` (MOT17-02) in sequence; confirm CSV populated for each
- [x] Confirm clean EOS and pipeline teardown on stream end for each
- [ ] Record FP32 baseline: frames processed, mean FPS, peak VRAM (`nvidia-smi dmon`) on stream0 for 60 seconds

### M1.7 — Anonymisation Write-back + RTSP Re-stream
- [x] Add `nvvideoconvert` CPU-copy path or CUDA memcpy to map NVMM frame to host before blurring — `pipelines/frame_accessor.py` (`get_frame_rgba`/`write_frame_rgba`) with injectable `_get_surface` for unit testing (4 tests)
- [x] Wire `blur_bboxes` back into the `_probe` in `pipelines/rtsp.py` using the mapped frame
- [x] Write blurred frame back to NVMM surface via `write_frame_rgba` + `np.copyto`
- [x] Add `nvrtspoutsinkbin` to re-stream blurred output on a second mediamtx path (`--restream-uri` flag; 3 new tests)
- [x] Confirm blurred output visible via `ffplay rtsp://localhost:8556/stream0_out`

### M1.8 — Multi-Stream Pipeline
- [x] Write `pipelines/multi_stream.py` — 3 concurrent `rtspsrc` bins into a single `nvstreammux` (batch-size=3)
- [x] Per-source CSV output keyed by `source_id` from `NvDsFrameMeta`
- [x] Per-source restream via `nvrtspoutsinkbin` (ports 8556/8557/8558 for stream0/1/2)
- [x] Replace three-container docker-compose with single `pipeline` service running `multi_stream.py`
- [x] Unit tests: multi-stream config defaults, source URI list parsing (TDD, CPU-safe, 8 tests)
- [x] Run all three streams concurrently; confirm all three CSVs populated and clean EOS

---

## M2 — Custom Model + C++ Decode Plugin

**Exit criteria:** YOLO26n running through DeepStream at FP16 with stable ByteTrack IDs; C++ decode plugin built and benchmarked.

### M2.1 — YOLO26n Export ✓
- [x] Install `ultralytics` in container (`docker/Dockerfile`)
- [x] Export `yolo26n.pt` → ONNX (YOLO26 is NMS-free by default; no `nms=False` flag needed): `models/export_yolo26.py`
- [x] Write `models/convert.py`: ONNX → TensorRT FP32 baseline engine via `trtexec`
- [x] Add FP16 conversion path to `convert.py`: `--fp16` flag
- [x] Document INT8 exclusion in `convert.py` comments (no Tensor Cores on 1660Ti)
- [x] Container auto-init: `docker/init_models.py` — sequential export → FP32 → FP16 on first start, skip-if-exists on warm restart; wired via `ENTRYPOINT` in Dockerfile (21 new unit tests, 68 total)
- [ ] Confirm both engines load in TensorRT without error (pending successful `trtexec` run)

### M2.2 — nvinfer Custom Output Parser
- [ ] `models/output_parser.py` — `parse_yolo26_output()` anchor-free, DFL-free decode (TDD, 5 tests passing)
- [ ] Wire parser into `nvinfer` config (`parse-bbox-func-name`, `custom-lib-path`)
- [ ] Confirm detections appear correctly on MOT17-04 stream (visual check)

### M2.3 — C++ Decode Plugin (Part 1)
- [ ] Scaffold `plugins/yolo26_decode/` with `CMakeLists.txt`
- [ ] Implement `IPluginV2DynamicExt` skeleton: `getOutputDimensions`, `enqueue`, `serialize`
- [ ] Implement CUDA kernel for anchor-free box coordinate transform in `enqueue` (no sigmoid needed — YOLO26 head is DFL-free)
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

**Exit criteria:** tracker comparison report with full MOTA/HOTA/IDF1; multi-stream benchmarking complete; all tests passing; docs complete.

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

### M3.3 — Multi-Stream Benchmarking
- [ ] 30-minute stability run on all three streams; confirm no crash/memory leak
- [ ] Record per-stream FPS and aggregate VRAM during 3-stream load
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
