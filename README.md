# deepstream-rtsp-pipeline

NVIDIA DeepStream pipeline running three concurrent RTSP streams through GPU-accelerated inference, object tracking, and per-source CSV metadata output — built on a GTX 1660Ti (6 GB VRAM) with a full TDD test suite.

## What this demonstrates

- Three RTSP sources batched through a single `nvstreammux`, demuxed back per-source for independent OSD and restream output
- YOLO26n FP16 running end-to-end: `.pt → ONNX (dynamic batch) → TRT FP16` via `trtexec`; Python tensor-meta probe decodes `[300, 6]` output and populates `NvDsObjectMeta` without a compiled C parser
- `IPluginV2DynamicExt` CUDA kernel appended to the TRT network via TRT Python API; converts xyxy→xywh on GPU inside TRT, replacing the Python coordinate-transform loop; `metrics/profile_decode.py` isolates per-layer latency via `IProfiler`
- 174 CPU-safe unit tests written before implementation (red→green); no GPU required for the test suite
- Gaussian blur applied to every detected bbox region before `nvdsosd` renders or any output leaves the pipeline
- FP32 vs FP16 vs FP16+decode-plugin compared on latency, VRAM, engine size, and fleet OTA cost; batch sweep 1–100 against a 25 fps real-time budget with fleet-sizing projections (`metrics/decode_comparison.ipynb`)
- Per-frame CSV sink; mediamtx-served MOT17 sequences as the RTSP source (MOT17-04 has ground truth for MOTA/HOTA/IDF1 tracker evaluation)
- NGC DeepStream 9.0 + pyds compiled from source; `docker compose up` handles model export and conversion on first run

---

## Pipeline architecture

```
mediamtx (RTSP server)
  ├─ stream0 (MOT17-04)  ──┐
  ├─ stream1 (MOT17-13)  ──┤
  └─ stream2 (MOT17-02)  ──┘

Per-source source bins (× 3):
  rtspsrc → rtph264depay → nvv4l2decoder → queue ──→ nvstreammux.sink_{i}

Shared inference chain (batched, N=3):
  nvstreammux → nvinfer (YOLO26n FP16 + yolo26_decode plugin, network-type=100, output-tensor-meta=1)
             ← [nvinfer SRC probe: reads xywh tensor → NvDsObjectMeta (80 COCO classes)]
             → nvtracker (NvMultiObjectTracker)
             → nvstreamdemux

Per-source output branches (× 3):
  demux.src_{i} → queue → nvvideoconvert (RGBA, unified mem)
               → nvdsosd ← [Python probe: blur + CSV write]
               → nvrtspoutsinkbin (ports 8556/8557/8558)
```

A single `nvdsosd` on the batched buffer only draws on source 0 — the per-branch placement is required and mirrors NVIDIA's `deepstream-demux-multi-in-multi-out` reference topology.

---

## Quick-start

**Prerequisites:** NVIDIA driver ≥ 590.48, `nvidia-container-toolkit`, `mediamtx` on host, MOT17-04/13/02 clips as MP4 in `data/`.

```bash
# Start RTSP source streams
mediamtx configs/mediamtx.yml &

# Build image (first run only — pyds compiled from source, ~5 min)
docker compose build

# Run pipeline
# First run: auto-exports YOLO26n → ONNX (dynamic batch) → FP32 + FP16 TRT engines (max_batch=3), then starts
# Warm restart: skips all model steps and launches immediately
docker compose up

# Verify output
wc -l output_stream{0,1,2}.csv
ffplay rtsp://localhost:8556/stream0_out   # YOLO26n boxes on stream0
ffplay rtsp://localhost:8557/stream1_out   # YOLO26n boxes on stream1
ffplay rtsp://localhost:8558/stream2_out   # YOLO26n boxes on stream2

# Tune detection confidence (default 0.25)
# Add --conf-threshold 0.5 to the compose command: or run directly:
# docker run ... ds-pipeline python3 pipelines/multi_stream.py --uri ... --conf-threshold 0.4
```

---

## Tests

```bash
pip install pytest
pytest tests/unit/ -v      # 174 tests, CPU-only, no GPU required
```

| Module | Tests | What they cover |
|--------|-------|-----------------|
| `metadata_parser` | 6 | `Detection` dataclass, `parse_frame_meta` with fake pyds structs |
| `csv_sink` | 6 | Header, field values, flush-on-write, multi-detection roundtrip |
| `anonymisation` | 6 | Blur applied, pixels outside bbox unchanged, out-of-bounds clip |
| `frame_accessor` | 4 | NVMM surface accessor with injectable `_get_surface` |
| `rtsp_pipeline` | 17 | Config defaults, arg parsing, source props, restream URI parsing |
| `multi_stream` | 30 | Multi-URI parsing, CSV path routing, port offset, `_make_nvinfer_config`; tracker flag; `bInferDone` / `detector_bbox_info` / file-URI regression tests; perf flag defaults + wiring |
| `convert` | 14 | `engine_path` naming, `build_trtexec_cmd` flags, dynamic-batch shape profile, `parse_args` |
| `export_yolo26` | 3 | `parse_args` for weights path and output-dir |
| `init_models` | 9 | Skip/run logic for all cold-start and warm-start combinations; decode-engine skip/build paths |
| `output_parser` | 6 | Threshold filtering, xyxy→xywh conversion, class_id extraction, batch-dim squeeze |
| `decode_engine` | 5 | `decode_engine_path` naming, `parse_args` defaults and flags |
| `validate_accuracy` | 20 | `box_iou`, greedy IoU matching, per-engine comparison, per-coord decode delta, `preprocess_frame` shape/dtype/range |
| `profile_decode` | 10 | `_parse_tail_latencies` (min/median/p99/max from trtexec output), `budget_check` (mean+p99 vs frame budget), `_SimpleProfiler.to_dict` tail fields |
| `evaluate_tracker` | 17 | GT loading (visibility filter), prediction loading (frame-offset), `MOTAccumulator` build, MOTA/MOTP/IDF1 compute, unique-track count |
| `perf_monitor` | 21 | `compute_interval_fps`, `PerfMonitor.record/summary` (FPS, VRAM, RSS, leak heuristic), `to_dict`/`write_json` round-trip, `sample_rss_mb`, `sample_vram_mb` |

GPU smoke tests (`pytest --gpu`) are planned for M3.4.

---

## Benchmark results

Standalone TRT engine timings on the GTX 1660 Ti (640×640, `trtexec`, 50 iterations). Full analysis and charts in `metrics/decode_comparison.ipynb`.

| Engine | Inference | Throughput | VRAM | Engine size |
|--------|-----------|-----------|------|-------------|
| FP32 base | 4.96 ms | 202 FPS | 357 MB | 11.0 MB |
| FP16 base | 3.26 ms | 307 FPS | 358 MB | 6.3 MB |
| FP16 + decode plugin | 3.35 ms | 298 FPS | 402 MB | 6.3 MB |

- **FP16 is 1.52× faster than FP32**, not the often-quoted 2×: the 1660 Ti (Turing, SM 75) has no Tensor Cores, so the gain comes from halved memory bandwidth, not faster compute.
- **FP16 saves no inference VRAM** (358 vs 357 MB). Weights are a small fraction of the runtime working set; activations dominate. The disk engine is 43% smaller, which matters for OTA fleet updates (≈23 GB saved per 5,000-sensor rollout), not for runtime headroom.
- **The decode plugin adds ~0.1 ms**, almost all of it kernel-launch overhead rather than compute. YOLO26n is NMS-free and emits only 300 pre-decoded boxes, so the kernel has little to do. M2.6 adds a YOLOv8n plugin (8400 candidates + DFL + NMS) to show where this pattern actually pays off.

**Accuracy validation** — measured across 1,050 frames of MOT17-04-SDP via `metrics/validate_accuracy.py`:

| Comparison | Matched boxes | Mean IoU | Match rate | Max conf delta |
|---|---|---|---|---|
| FP16 vs FP32 | 11,690 / 11,804 | **0.9937** | 99.0% | 0.44 |

The 1.52× speedup costs essentially no detection accuracy: 99% of FP32 boxes are matched at IoU > 0.5, mean overlap is 0.994, and the unmatched 1% are low-confidence borderline detections where FP16 rounding shifts a box just below the confidence threshold.

Decode plugin coordinate check (decode engine vs Python baseline, 11,693 matched pairs):

| Stat | Value |
|---|---|
| Mean coord delta | **0.041 px** |
| p99 coord delta | **0.5 px** |
| Max coord delta | 14.0 px (single outlier from different TRT graph fusions) |

Mean and p99 confirm the CUDA kernel is correct; the 14 px max is a single outlier where the two independently-compiled TRT engines chose different kernel fusions for the same backbone layer, not a kernel arithmetic error.

Multi-stream batch sweep (FP16, single `nvstreammux` batch):

- **batch=15 is the practical ceiling for live RTSP under sustained load** at 29.8 ms mean / 30.8 ms p99, leaving 9 ms of headroom inside the 40 ms / 25 fps budget. batch=20 (40.2 ms mean / 42.0 ms p99) exceeds the budget at the tail. The earlier cold-run figure of batch=25 / 36.2 ms reflects a warmed-up GPU — p99 measurements under sustained sequential load give the operationally accurate number.
- Consolidating to 15 streams/node cuts a 5,000-camera fleet from 5,000 nodes to 334, a 15× reduction.
- batch=100 (208 ms) is offline-reprocessing only.

**End-to-end pipeline FPS (M3.3)** — measured with the full DeepStream graph (nvinfer → nvtracker → nvdsosd → nvrtspoutsinkbin) on the same GTX 1660 Ti:

| Scenario | FPS / stream | Notes |
|---|---|---|
| `trtexec` batch=3 (bare TRT kernel) | **140.9** | Pure inference, no graph overhead |
| Unthrottled 3× file source | **131.3** | Full graph, `sync=false`; 7% overhead vs bare kernel |
| Live 3-stream RTSP (30 min, `-re` cap) | **29.7** | Exceeds 25 fps floor; 4.4× headroom vs ceiling |

Full-graph overhead vs the bare TRT kernel is **7 %** (131 vs 140.9 fps/stream) — the IOU tracker, OSD, and Python probe are cheap at this batch size; the main cost is GStreamer scheduling. The live pipeline sustains > 25 fps × 3 streams over 30 minutes with a peak VRAM of **1,632 MB** (IOU tracker) and an RSS that *decreased* by 260 MB over the run (DeepStream releasing initialisation caches) — definitively no memory leak. Full analysis and stability charts in `metrics/stability.ipynb`.

**Tracker comparison (M3.2)** — three `nvtracker` algorithms evaluated on MOT17-04 ground truth (47,557 GT boxes) via `py-motmetrics`. All three see the identical YOLO26n detection stream so differences isolate the tracker, not the detector. Full analysis in `metrics/tracker_comparison.ipynb`.

| Tracker | MOTA ↑ | IDF1 ↑ | ID-switches ↓ | Fragments ↓ | VRAM |
|---|---|---|---|---|---|
| IOU (baseline) | 0.118 | 0.127 | 252 | 637 | lowest |
| NvDCF | **0.138** | **0.257** | 70 | 641 | ~+200 MB (DCF feature maps) |
| ByteTrack / NvSORT | 0.104 | 0.187 | **37** | **188** | same as IOU |

- **NvDCF** wins on identity (IDF1 2× IOU) — the DCF appearance model re-acquires targets through brief occlusions common in crowded scenes. Recommended where track continuity matters (re-identification, counting across zones).
- **ByteTrack/NvSORT** has the fewest ID-switches and by far the fewest fragmentations — two-stage cascaded association recovers low-confidence detections, so tracks break far less. Best stability-per-compute trade-off with no appearance model.
- **IOU** is the fastest and cheapest baseline. No motion or appearance model; identities churn when boxes stop overlapping frame-to-frame.
- Low absolute MOTA (~0.10–0.14) is expected: YOLO26n-nano recovers only ~11k of 47,557 GT boxes, so MOTA is dominated by missed detections (recall), not tracker quality. The relative ranking is meaningful — all trackers see the same detection stream.

---

## Roadmap

**M1 — Pipeline Plumbing** ✓ *(complete)*
Three-stream concurrent pipeline; TrafficCamNet ResNet-18 FP32 placeholder; per-source CSV; anonymisation probe; RTSP restream; 47 unit tests.

**M2 — Custom Model + C++ Decode Plugin** ✓ *(mostly complete)*
YOLO26n FP16 runs end-to-end through DeepStream with a C++ TRT decode plugin. The `IPluginV2DynamicExt` CUDA kernel does the xyxy→xywh coordinate transform on GPU inside TRT; `models/decode_engine.py` builds the plugin-appended engine via TRT Python API. Precision comparison, multi-stream batch sweep (up to 100 streams), accuracy validation against a FP32 baseline, and latency-tail analysis (p99, jitter) are all complete. Deferred: M2.6 YOLOv8n heavy-decode plugin (lower priority than M3; would demonstrate where the DFL+NMS kernel pays off vs YOLO26n's ~0.1 ms overhead).

**M3 — Tracker Comparison + Hardening** *(in progress)*

- ✓ **M3.1** — Three tracker configs (IOU / NvDCF / ByteTrack); `--tracker` CLI flag; `probationAge` tuning; tracker CSVs in `metrics/tracker_results/`
- ✓ **M3.2** — MOTA/MOTP/IDF1 evaluation via `py-motmetrics` on MOT17-04 GT; `metrics/evaluate_tracker.py`; `metrics/tracker_comparison.ipynb` with summary table + bar charts; fixed `bInferDone` / `detector_bbox_info` probe bugs; file-input source branch for GT-aligned eval
- ✓ **M3.3** — Live end-to-end FPS (131 fps unthrottled / 29.7 fps live) + 30-min stability run; `metrics/perf_monitor.py` (21 CPU-safe tests); `--perf-json / --duration / --no-sync` flags; `metrics/stability.ipynb`
- ☐ **M3.4** — GPU smoke + integration tests; GitHub Actions for unit tests; model-promotion gate
- ✓ **M3.5** — `docs/jetson-upgrade.md`, `docs/isp-and-camera-input.md`, `docs/system-design.md`; README completeness pass
- ☐ **M3.6** — Observability: structured per-stream logging, per-sensor health metrics, failure-mode playbook

---

## Key design decisions

**`network-type=100` + Python tensor-meta probe for YOLO26n.** nvinfer's built-in bbox parsers expect anchor-based or NMS-post-processed output in a specific layout. YOLO26n's one-to-one matching head emits `[batch, 300, 6]` (end-to-end NMS baked in). Rather than compile a C `.so` custom parser, we use `network-type=100` (custom) with `output-tensor-meta=1`: nvinfer exposes the raw tensor in `NvDsInferTensorMeta` and a Python probe on the nvinfer SRC pad calls `parse_yolo26_output()` and populates `NvDsObjectMeta` directly. The decode logic stays in pure Python, is fully unit-testable without a GPU. M2.3+M2.4 replaced this with the `IPluginV2DynamicExt` CUDA kernel; the Python probe now only reads the already-decoded xywh tensor from `NvDsInferTensorMeta` and populates `NvDsObjectMeta`.

**Dynamic-batch ONNX export.** Exporting with `dynamic=True` makes the batch dimension flexible. `trtexec` is then called with `--minShapes=images:1x3x640x640 --optShapes=images:3x3x640x640 --maxShapes=images:3x3x640x640`, producing a single engine file (`yolo26n_fp16_b3.engine`) that nvinfer can use for any batch size in [1, 3] — both single-stream testing and 3-stream production use the same engine.

**C++ `IPluginV2DynamicExt` decode plugin — xyxy→xywh on GPU.** The M2.2 Python probe looped over 300 detections on CPU to convert xyxy → xywh. M2.3+M2.4 replace this with a CUDA kernel (`plugins/yolo26_decode/yolo26_decode_kernel.cu`) compiled as a TRT `IPluginV2DynamicExt` plugin. `models/decode_engine.py` uses the TRT Python API to parse the ONNX, unmark the YOLO output, append the plugin as a custom layer, and rebuild the engine — the resulting `yolo26n_fp16_b3_decode.engine` emits xywh from TRT directly. The probe reads the transformed coordinates with no Python for-loop. `metrics/profile_decode.py` uses TRT `IProfiler` to print per-layer latency and isolate the `yolo26_decode` kernel time.

**`_make_nvinfer_config` for TRT batch-size (legacy engines).** `nvinfer.set_property("batch-size", n)` overrides the config but does not trigger an engine rebuild — a cached batch-1 engine gives undefined behaviour at batch-3. The fix rewrites both `batch-size` and the engine file path in a temp config. For YOLO26n the engine already covers batch 1–3, so only `batch-size` is rewritten; the path is left unchanged.

**Per-branch `nvdsosd` after demux.** A single batched OSD only composites onto the first frame in the batch (source 0). Each branch gets its own `nvvideoconvert(unified) → nvdsosd` so boxes render correctly on every stream.

**`nvbuf-memory-type=3` on `nvvideoconvert`.** Default NVMM is device-only; `pyds.get_nvds_buf_surface` from a Python probe segfaults. CUDA unified memory (`type=3`) keeps the `NvBufSurface` CPU-accessible without an explicit `cudaMemcpy`.

**mediamtx over real IP cameras.** Provides a reproducible, loopable, committable source. MOT17-04 has free ground truth annotations enabling quantitative tracker evaluation in M3.

---

## Known gaps

| Gap | Reason | Mitigation |
|-----|--------|------------|
| Jetson / nvargus | No Jetson hardware available | `docs/jetson-upgrade.md` — component diff table: x86 dGPU → JetPack; nvargus CSI path; INT8 on Jetson; TDP modes |
| INT8 quantisation | GTX 1660Ti has no Tensor Cores; INT8 has no hardware speedup | Documented in `models/convert.py`; would enable on Jetson AGX Orin or RTX-class GPU |
| GPU smoke tests | Require GPU runner; written last to avoid slow CI | Planned M3.4 via `pytest --gpu` and `tests/smoke/` |
| Decode plugin shows little gain on YOLO26n | YOLO26n is NMS-free (300 pre-decoded boxes), so the kernel does ~0.006 ms of work; the accuracy comparison is between two separately-compiled TRT graphs, not a controlled kernel isolation | M2.6: YOLOv8n plugin (8,400 candidates + DFL + NMS) demonstrates where the pattern pays off |
| DeepSORT tracker | Re-ID model exceeds 6 GB VRAM ceiling | Documented in M3 tracker comparison rationale; ByteTrack recommended instead |

---

## Privacy by Design

The pipeline blurs every detected bounding-box region before frames reach any output sink. A GStreamer buffer probe on each per-source `nvdsosd_{i}` sink pad calls `blur_bboxes()` (`pipelines/anonymisation.py`) on the raw `NvBufSurface`-backed numpy array for each frame.

Blurring runs *before* `nvdsosd` renders the overlay boxes, so anonymised pixels are written back into the GPU surface and any downstream consumer (display or encode) sees the blurred content. The CSV metadata sink stores only bounding-box coordinates, class labels, object IDs, and confidence scores — no raw face or licence-plate pixel data.

`blur_bboxes()` clips coordinates to the frame boundary and skips zero-area regions, so out-of-range detections are handled safely without crashing the pipeline.
