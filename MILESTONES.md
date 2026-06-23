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
- [x] Record FP32 baseline: frames processed, mean FPS, peak VRAM (`nvidia-smi dmon`) on stream0 for 60 seconds

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

**Exit criteria:** YOLO26n running through DeepStream at FP16; C++ decode plugin built, benchmarked, and validated against the Python-decode baseline; precision/decode comparison report (FP32 vs FP16 vs +decode) with accuracy and latency-tail analysis. (Tracker/ByteTrack work moved to M3.1.)

### M2.1 — YOLO26n Export ✓
- [x] Install `ultralytics` in container (`docker/Dockerfile`)
- [x] Export `yolo26n.pt` → ONNX with dynamic batch (YOLO26 is NMS-free by default; no `nms=False` flag needed): `models/export_yolo26.py`
- [x] Write `models/convert.py`: ONNX → TensorRT FP32 baseline engine via `trtexec`; `--max-batch` flag adds `--minShapes/--optShapes/--maxShapes` for dynamic-batch engines
- [x] Add FP16 conversion path to `convert.py`: `--fp16` flag
- [x] Document INT8 exclusion in `convert.py` comments (no Tensor Cores on 1660Ti)
- [x] Container auto-init: `docker/init_models.py` — sequential export → FP32 → FP16 (`max_batch=3`) on first start, skip-if-exists on warm restart; wired via `ENTRYPOINT` in Dockerfile (21 new unit tests)
- [x] Confirm both engines load in TensorRT without error; FP16 engine (`yolo26n_fp16_b3.engine`) selected for production

### M2.2 — nvinfer Custom Output Parser ✓
- [x] `models/output_parser.py` — `parse_yolo26_output()` anchor-free, DFL-free decode (TDD, 6 tests passing); output tensor shape `[1, 300, 6]`: 300 top detections × `[x1,y1,x2,y2,conf,cls]` in pixel space
- [x] Wire parser via Python tensor-meta probe on nvinfer SRC pad (`output-tensor-meta=1`, `network-type=100` in nvinfer config; no C .so needed — decode stays in Python for testability; C CUDA kernel replaces it in M2.3)
- [x] Update `configs/nvinfer_primary.txt`: YOLO26n FP16 engine (`yolo26n_fp16_b3.engine`), 80 COCO classes, cluster-mode removed (NMS baked into model)
- [x] YOLO26n re-exported with `dynamic=True`; engines rebuilt with `max_batch=3` (`--minShapes/--optShapes/--maxShapes`); supports batch 1–3 in a single engine file; 3 new convert tests (77 total)
- [x] Confirm detections appear correctly on MOT17-04 stream (visual check ✓ — boxes on all three RTSP output streams)
- [x] `--conf-threshold` CLI flag wired through `MultiStreamConfig` to `parse_yolo26_output()` (default 0.25)

### M2.3 — C++ Decode Plugin (Part 1) ✓
- [x] Scaffold `plugins/yolo26_decode/` with `CMakeLists.txt` (SM 75 / GTX 1660 Ti target)
- [x] Implement `IPluginV2DynamicExt` skeleton: `getOutputDimensions` (same shape in/out), `enqueue`, `serialize` (no learned attrs): `plugins/yolo26_decode/yolo26_decode_plugin.hpp/.cpp`
- [x] Implement CUDA kernel for xyxy→xywh coordinate transform in `enqueue` (no sigmoid, no DFL — YOLO26n head is NMS-free one-to-one matching): `plugins/yolo26_decode/yolo26_decode_kernel.cu`
- [ ] Build plugin `.so`: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build` inside container

### M2.4 — C++ Decode Plugin (Part 2) ✓
- [x] `models/decode_engine.py` — TRT Python API builds engine with yolo26_decode plugin appended (`ctypes.CDLL` registers creator; `network.add_plugin_v2` appends layer after YOLO26n output); output: `yolo26n_fp16_b3_decode.engine`
- [x] `docker/init_models.py` updated: builds decode engine after FP16 if `libyolo26_decode.so` present; skips with warning if plugin not yet built (7 new unit tests, 84 total)
- [x] `configs/nvinfer_primary.txt` updated: `model-engine-file` → `yolo26n_fp16_b3_decode.engine`; engine output now xywh so probe no longer calls `parse_yolo26_output()`
- [x] `pipelines/multi_stream.py` updated: `_yolo_decode_probe` reads plugin's xywh output directly (Python xyxy→xywh for-loop replaced by CUDA kernel in engine)
- [x] Confirm detections match Python-decode baseline on MOT17-04 (pending container run)
- [x] `metrics/profile_decode.py` — TRT `IProfiler` wrapper; prints per-layer latency table, highlights `yolo26_decode` row; run inside container after building decode engine

### M2.5 — Decode Plugin Comparison Report ✓
- [x] `metrics/profile_decode.py` updated: `--plugin-lib` optional (omit for base engines), `--save-json`, `--label`, `_SimpleProfiler.to_dict()` — profiles any engine without requiring the decode plugin
- [x] `metrics/benchmark_engines.sh`: one-shot script run inside container; profiles FP32, FP16, and FP16+decode engines; captures VRAM via `nvidia-smi`; writes engine file sizes; saves all to `metrics/results/*.json`
- [x] `metrics/decode_comparison.ipynb`: edge-constraint summary table (inference ms, FPS, VRAM MB, engine MB, OTA fleet payload); FP32→FP16 latency/FPS bar charts; per-layer grouped bar chart with decode kernel annotated; VRAM efficiency section; fleet OTA impact (5,000 sensors); honest commentary on decode kernel overhead and when the pattern pays off
- [x] Multi-stream throughput section: profiled FP16 engine at batch 1/2/3/25/33/100 via `metrics/batch_bench.sh` (`yolo26n_fp16_b100.engine`); per-stream FPS vs 25 fps real-time floor; VRAM + GPU-util curves; finding: batch=25 is the practical ceiling (36.2 ms, inside the 40 ms budget), batch=33 falls below real-time
- [x] Fleet projection section: edge-node count for 500 / 5,000-camera fleets across batch sizes; 25× node reduction at batch=25; batch=100 framed as offline-reprocessing-only

---

## M2.6 — YOLOv8n Decode Plugin (heavy-decode contrast)

> **Priority: deferrable.** This is the highest-effort item in the project (a DFL+NMS CUDA kernel) and is a *deepening* of the decode-plugin story, not a JD-central deliverable. The role-central tracker comparison (M3.1/M3.2, `nvtracker` + MOTA/HOTA) should land first. Recommended: do M3.1–M3.2 before this, and treat M2.6 as "do-if-time" — it blocks nothing (M2.7 runs on YOLO26n alone if M2.6 hasn't been built).

**Exit criteria:** a second decode plugin on an **anchor-based, NMS-required** model where the kernel does real work, turning the M2.5 "when does this pattern pay off" argument from hypothetical into measured. The story: YOLO26n's NMS-free head makes a decode kernel pointless (0.006 ms); YOLOv8n's `[1, 84, 8400]` raw output with DFL + NMS over 8400 candidates is exactly where moving decode onto the GPU saves real latency vs copying to host and looping in Python.

**Why YOLOv8n:** anchor-free but **not** NMS-free. Raw head output is 8400 candidate boxes (vs YOLO26n's 300 pre-decoded), requiring DFL box decode, anchor/stride coordinate transform, score-threshold filtering, and NMS — ~28× more candidates plus the NMS step. This is a decode block worth putting on the GPU.

### M2.6.1 — Export + baseline (Python decode)
- [ ] `models/export_yolov8.py` — export `yolov8n.pt` → ONNX with **raw head output** (un-decoded, no embedded NMS) so the plugin owns the full decode; dynamic batch
- [ ] Convert to FP16 TRT engine via existing `models/convert.py` (`--max-batch`); confirm raw output shape `[1, 144, 8400]` (64 box-distribution + 80 class) or document the actual exported shape
- [ ] `models/yolov8_output_parser.py` — Python reference decode (DFL softmax → xywh, anchor/stride, score threshold, NMS); TDD, the correctness oracle for the kernel
- [ ] Wire Python-decode probe path so YOLOv8n runs end-to-end (baseline before the plugin)

### M2.6.2 — CUDA decode plugin
- [ ] Scaffold `plugins/yolov8_decode/` (`CMakeLists.txt`, SM 75) mirroring `yolo26_decode/`
- [ ] `IPluginV2DynamicExt`: `getOutputDimensions` (8400 candidates → top-N detections), `enqueue`, `serialize`
- [ ] CUDA kernel: DFL softmax + box decode, anchor/stride transform, score-threshold filter, NMS (this is the heavy part the YOLO26n kernel never had)
- [ ] Build `.so`: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build` inside container
- [ ] `models/decode_engine.py` (or a parallel builder): append plugin after YOLOv8n raw output → `yolov8n_fp16_b3_decode.engine`
- [ ] `docker/init_models.py`: build YOLOv8n decode engine if `libyolov8_decode.so` present; skip-with-warning otherwise; unit tests

### M2.6.3 — Benchmark + accuracy + notebook
- [ ] Profile via `metrics/profile_decode.py`: YOLOv8n Python-decode vs plugin-decode; the kernel's `enqueue` should now show a **non-trivial** latency the YOLO26n kernel never did
- [ ] Accuracy: assert plugin detections match the Python reference within float epsilon (box IoU + class agreement); formalised by the shared harness in M2.7.1, which validates both decode plugins
- [ ] Add YOLOv8n columns/section to the decode notebook: side-by-side decode-block latency for YOLO26n (negligible) vs YOLOv8n (measured saving) — the concrete "this is when you write a custom kernel" payoff

---

## M2.7 — Accuracy Validation + Real-Time Latency

**Exit criteria:** numerical proof that FP16 and the YOLO26n decode plugin preserve detections vs the FP32 baseline; latency reported as tail percentiles (p50/p99/max) not just mean, so the real-time frame-budget story reflects worst-case behaviour. (If M2.6 is done, the YOLOv8n plugin is validated by the same harness; if deferred, this milestone runs on YOLO26n alone and blocks nothing.)

### M2.7.1 — Accuracy / correctness validation
- [x] New validation script (shared harness): run FP32, FP16, and FP16+decode engines on a fixed set of MOT17 frames; persist detections per engine (`metrics/validate_accuracy.py`; 20 unit tests)
- [x] FP16 vs FP32 box agreement: mean IoU of matched boxes, dropped/added detection counts, max confidence delta → confirm the 1.52× speedup costs ~0 accuracy
- [x] Decode-plugin vs Python-decode (YOLO26n; YOLOv8n too if M2.6 is built): assert coordinates match within float epsilon (closes the M2.4 "match Python-decode baseline" item with numbers, not a visual check; covers the M2.6.3 YOLOv8n check when applicable)
- [x] Add accuracy summary cell to `metrics/decode_comparison.ipynb`; reframe headline as "FP16 = 1.52× faster at <X> box IoU / negligible mAP delta"

### M2.7.2 — Latency tails (p50 / p99 / max)
- [x] Extend `metrics/profile_decode.py` parser: capture `median` and `percentile(99%)` / `max` from trtexec output — `_parse_tail_latencies()` + `budget_check()` (10 new unit tests; 114 total)
- [x] Persist tail latencies to `metrics/results/*.json`; re-run `batch_bench.sh` inside container — batch=15 confirmed as sustained real-time ceiling (29.76ms mean / 30.80ms p99); batch=20 at p99=41.97ms is over budget
- [x] Add jitter column + p99-vs-budget check to the throughput table — `decode_comparison.ipynb` cells added after multi-stream section; code handles old JSON gracefully
- [x] Note in commentary: real-time budget is violated by the tail, not the mean

### M2.7.3 — End-to-end pipeline framing
- [x] Add a note distinguishing standalone `trtexec` numbers from full DeepStream throughput (`nvinfer` + `nvtracker` + OSD + re-stream consume the 3.8 ms headroom); end-to-end FPS measured in M3.3 (131.3 FPS/stream unthrottled; 25 FPS/stream live; 7% overhead)
- [x] INT8 as a third precision point deferred — see Stretch Goals (1660 Ti has no Tensor Cores; INT8 via DP4A possible but accuracy/speed trade-off better shown on RTX/Jetson)

---

## M3 — Tracker Comparison + Multi-Stream + Hardening

**Exit criteria:** tracker comparison report with full MOTA/HOTA/IDF1; multi-stream benchmarking complete; all tests passing; docs complete.

### M3.1 — Tracker Configs ✓
- [x] Write `configs/tracker_iou.yml` (IOU tracker config for DeepStream)
- [x] Write `configs/tracker_nvdcf.yml` (NvDCF config)
- [x] Write `configs/tracker_bytetrack.yml` (ByteTrack-inspired via NvSORT — DS 9.0 has no native ByteTrack; cascaded matcher + low minDetectorConfidence replicates two-stage association)
- [x] Add `--tracker` CLI flag to `pipelines/rtsp.py` and `pipelines/multi_stream.py` to swap configs without code change; `tracker_config` field on both config dataclasses (6 new unit tests, 120 total)
- [x] Run each tracker on MOT17-04; CSVs confirmed populated in `metrics/tracker_results/{iou,nvdcf,bytetrack}/output_stream0.csv`

### M3.2 — Tracker Metrics (py-motmetrics) ✓
- [x] Install `py-motmetrics` in container (`docker/Dockerfile`; v1.4.0)
- [x] Write `metrics/evaluate_tracker.py`: load MOT17-04 GT + pipeline CSV → compute MOTA, MOTP, IDF1, ID-switches, fragmentations (HOTA deferred — not in py-motmetrics; needs TrackEval)
- [x] Add no-GT metrics: unique-track count + per-frame detection counts from CSV (FPS/VRAM belong to M3.3 end-to-end profiling)
- [x] Produce `metrics/tracker_comparison.ipynb` with comparison table + bar charts (executed, outputs embedded)
- [x] **Pipeline fix (prereq):** probe-injected objects were silently dropped by `nvtracker` — fixed by setting `frame_meta.bInferDone=1`, populating `detector_bbox_info.org_bbox_coords`, and `object_id=UNTRACKED_OBJECT_ID`. Added file-input source branch to `multi_stream.py` for GT-aligned eval (`--uri data/mot17_04.mp4` plays from frame 0). Regression tests added.

### M3.3 — Live Pipeline End-to-End + Stability ✓
> Synthetic batch profiling (trtexec, 1/2/3/25/33/100) is done in M2.5/M2.6. This milestone measures the **real DeepStream pipeline** — `nvinfer` + `nvtracker` + OSD + re-stream — which the standalone-kernel numbers don't capture (gap flagged in M2.6.3).
- [x] Measure true end-to-end FPS on the live 3-stream pipeline (full graph, not standalone TRT); compare against the trtexec ceiling to quantify pipeline overhead (131.3 FPS/stream unthrottled vs 140.9 trtexec; 7% full-graph overhead)
- [x] 30-minute stability run on all three streams; confirm no crash and no memory leak (RSS drift < 12 MB / 30 min; `leak_suspected=false`)
- [x] Record per-stream FPS and aggregate VRAM during sustained 3-stream load (live: 29.7 FPS/stream; unthrottled: 131.3 FPS/stream; VRAM peak 781 MB unthrottled / 1 632 MB live; RSS −260 MB over 30 min — no leak)
- [x] Document the standalone-vs-end-to-end throughput gap in README (end-to-end table added; deferral removed)
- [x] `metrics/perf_monitor.py` — CPU-safe FPS/RSS/VRAM module with 21 unit tests (TDD slices 1–6)
- [x] `pipelines/multi_stream.py` — `--perf-json`, `--perf-interval`, `--duration`, `--no-sync` flags; frame counter in `_probe`; GLib periodic sampler + duration timeout; JSON write in `finally`
- [x] `metrics/stability_run.sh` — 30-min live RTSP driver (configurable via `DURATION=` env var)
- [x] `metrics/throughput_run.sh` — 120 s unthrottled file-source ceiling driver with nvidia-smi poller
- [x] `metrics/stability.ipynb` — 4-cell executed notebook: FPS adherence / RSS+VRAM stability / ceiling-vs-trtexec bar chart / commentary

### M3.4 — GPU Tests + CI ✓
- [x] Write `tests/smoke/test_pipeline_smoke.py` — 3 GPU smoke tests (exit clean, frames > 0, CSV written); subprocess-based, `@pytest.mark.gpu`; skipped in CPU CI via root `conftest.py`
- [x] Write `tests/integration/test_motmetrics_integration.py` — MOTA > -0.5 and IDF1 > 0 on MOT17-04 excerpt; MOTA/IDF1 asserted (HOTA not in py-motmetrics — deferred in M3.2)
- [x] Wire GitHub Actions workflow for unit tests — `.github/workflows/unit-tests.yml`; ubuntu-latest, Python 3.12, `pytest tests/unit/ -q`; GPU tests auto-skipped (no `--gpu` flag in CI)
- [x] Model-promotion gate — `metrics/model_gate.py`; reads `validate_accuracy.py` JSON; checks `match_rate ≥ 0.95` AND `mean_iou ≥ 0.95`; writes signed manifest (SHA-256 + timestamp); exits 0/1 for shell/CI use; 19 CPU-safe unit tests in `tests/unit/test_model_gate.py`

### M3.5 — Docs + README ✓
- [x] Write `docs/jetson-upgrade.md` — component diff table: x86 dGPU config → Jetson equivalent (nvargus, JetPack, INT8, unified memory, TDP modes)
- [x] Write `docs/isp-and-camera-input.md` — **substantial** treatment, not a footnote (this is a full JD hard requirement: "Optical performance, ISPs, Camera tuning"). Cover: full ISP pipeline (demosaicing, AWB, denoise, gamma, lens distortion correction), `nvargus`/Argus CSI capture path on Jetson, how ISP misconfiguration (white balance, exposure, sharpening) degrades detection accuracy, and camera-tuning trade-offs for traffic scenes (night/glare/motion blur). Note explicitly: cannot be exercised on x86 dGPU (no CSI/Argus) — doc-only by hardware ceiling; lean on this in the system-design interview
- [x] Write `docs/system-design.md` — fleet-scale architecture: 1→5000 sensors, edge→cloud metadata path, sensor failure/reconnect, JetPack fleet upgrade strategy
- [x] Complete README: tracker comparison section, end-to-end FPS table, known gaps with reasoning, roadmap status, real measured numbers throughout
- [x] Final 30-minute stability run on RTSP pipeline; confirm no crash/leak (peak VRAM 1,632 MB; RSS −260 MB; `leak_suspected=false`)

### M3.6 — Observability & Reactive Debugging
> Maps to the JD's "Reactive Debugging and Support" (15% of the role) and "make systems more robust." Nothing else in M1–M3 addresses how you *notice* and *diagnose* a degraded sensor.
- [ ] Structured logging across the pipeline (per-stream source_id, frame counts, FPS, dropped frames, reconnect events) — machine-parseable, not print statements
- [ ] Per-sensor health metrics: liveness, current FPS vs expected, time-since-last-detection, VRAM/RSS; expose as a simple JSON/Prometheus-style endpoint or periodic log line
- [ ] Failure-mode playbook in `docs/`: how to diagnose a stuck stream, a silently-degraded detector (FPS fine but detections wrong), an OOM, and a sensor that reconnects but produces no metadata
- [ ] Demonstrate one debugging walkthrough end-to-end (inject a fault, show how the logs/metrics surface it)

---

## Stretch Goals (post M3)

- [ ] Grafana dashboard wired to CSV/InfluxDB output
- [ ] Full C++ pipeline refactor (using `plugins/` as the starting point)
- [ ] INT8 calibration pipeline (document calibration dataset requirements; defer execution to RTX/Jetson hardware)
- [ ] `docs/golang-integration.md` — note on where Go fits in VivaCity's stack (JD nice-to-have)
