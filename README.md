# deepstream-rtsp-pipeline

NVIDIA DeepStream pipeline running three concurrent RTSP streams through GPU-accelerated inference, object tracking, and per-source CSV metadata output — built on a GTX 1660Ti (6 GB VRAM) with a full TDD test suite.

## What this demonstrates

- **Multi-stream batching** — three RTSP sources muxed into a single `nvstreammux` (batch-size=3); per-source outputs demuxed back and rendered independently
- **Custom model integration** — YOLO26n FP16 running end-to-end: `.pt → ONNX (dynamic batch) → TRT FP16` via `trtexec`; Python tensor-meta probe decodes `[300, 6]` output and populates `NvDsObjectMeta` without a compiled C parser
- **TDD discipline** — 77 CPU-safe unit tests written before implementation (vertical red→green slices); no GPU required for the test suite
- **Privacy by design** — anonymisation blur probe wired before `nvdsosd`; detected bbox regions blurred on the NVMM surface before any output leaves the pipeline
- **Benchmarking pipeline** — per-frame CSV metadata sink; mediamtx RTSP source with MOT17 sequences (MOT17-04 has ground truth for MOTA/HOTA/IDF1 evaluation in M3)
- **Reproducible environment** — NGC DeepStream 9.0 container + pyds compiled from source; `docker compose up` auto-exports and converts the model on first run, then starts the pipeline

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
  nvstreammux → nvinfer (YOLO26n FP16, network-type=100, output-tensor-meta=1)
             ← [nvinfer SRC probe: tensor decode → NvDsObjectMeta (80 COCO classes)]
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
pytest tests/unit/ -v      # 77 tests, CPU-only, no GPU required
```

| Module | Tests | What they cover |
|--------|-------|-----------------|
| `metadata_parser` | 6 | `Detection` dataclass, `parse_frame_meta` with fake pyds structs |
| `csv_sink` | 6 | Header, field values, flush-on-write, multi-detection roundtrip |
| `anonymisation` | 6 | Blur applied, pixels outside bbox unchanged, out-of-bounds clip |
| `frame_accessor` | 4 | NVMM surface accessor with injectable `_get_surface` |
| `rtsp_pipeline` | 14 | Config defaults, arg parsing, source props, restream URI parsing |
| `multi_stream` | 11 | Multi-URI parsing, CSV path routing, port offset, `_make_nvinfer_config` batch-size + engine path rewrite |
| `convert` | 14 | `engine_path` naming, `build_trtexec_cmd` flags, dynamic-batch shape profile, `parse_args` |
| `export_yolo26` | 3 | `parse_args` for weights path and output-dir |
| `init_models` | 7 | Skip/run logic for all cold-start and warm-start combinations |
| `output_parser` | 6 | Threshold filtering, xyxy→xywh conversion, class_id extraction, batch-dim squeeze |

GPU smoke tests (`pytest --gpu`) are planned for M3.4.

---

## Roadmap

**M1 — Pipeline Plumbing** ✓ *(complete)*
Three-stream concurrent pipeline; TrafficCamNet ResNet-18 FP32 placeholder; per-source CSV; anonymisation probe; RTSP restream; 47 unit tests.

**M2 — Custom Model + C++ Decode Plugin** *(in progress — M2.1 + M2.2 complete)*
YOLO26n FP16 running end-to-end through DeepStream: `.pt → ONNX (dynamic batch) → TRT FP16 (batch 1–3)` via `trtexec`; container auto-init on first start; Python tensor-meta decode probe (`parse_yolo26_output()`, `network-type=100`, `output-tensor-meta=1`) creates `NvDsObjectMeta` without a C parser; boxes confirmed on all three RTSP output streams; `--conf-threshold` CLI flag; 77 unit tests total. Next: C++ `IPluginV2DynamicExt` CUDA decode plugin (`plugins/yolo26_decode/`), FP32 vs FP16+plugin latency comparison report.

**M3 — Tracker Comparison + Hardening** *(planned)*
Three-way tracker comparison (IOU → NvDCF → ByteTrack) with MOTA/HOTA/IDF1 on MOT17-04 ground truth; 30-minute stability run; GPU smoke + integration tests; `docs/jetson-upgrade.md`, `docs/isp-and-camera-input.md`, `docs/system-design.md`.

---

## Key design decisions

**`network-type=100` + Python tensor-meta probe for YOLO26n.** nvinfer's built-in bbox parsers expect anchor-based or NMS-post-processed output in a specific layout. YOLO26n's one-to-one matching head emits `[batch, 300, 6]` (end-to-end NMS baked in). Rather than compile a C `.so` custom parser, we use `network-type=100` (custom) with `output-tensor-meta=1`: nvinfer exposes the raw tensor in `NvDsInferTensorMeta` and a Python probe on the nvinfer SRC pad calls `parse_yolo26_output()` and populates `NvDsObjectMeta` directly. The decode logic stays in pure Python, is fully unit-testable without a GPU, and will be replaced by a C++ CUDA kernel in M2.3.

**Dynamic-batch ONNX export.** Exporting with `dynamic=True` makes the batch dimension flexible. `trtexec` is then called with `--minShapes=images:1x3x640x640 --optShapes=images:3x3x640x640 --maxShapes=images:3x3x640x640`, producing a single engine file (`yolo26n_fp16_b3.engine`) that nvinfer can use for any batch size in [1, 3] — both single-stream testing and 3-stream production use the same engine.

**`_make_nvinfer_config` for TRT batch-size (legacy engines).** `nvinfer.set_property("batch-size", n)` overrides the config but does not trigger an engine rebuild — a cached batch-1 engine gives undefined behaviour at batch-3. The fix rewrites both `batch-size` and the engine file path in a temp config. For YOLO26n the engine already covers batch 1–3, so only `batch-size` is rewritten; the path is left unchanged.

**Per-branch `nvdsosd` after demux.** A single batched OSD only composites onto the first frame in the batch (source 0). Each branch gets its own `nvvideoconvert(unified) → nvdsosd` so boxes render correctly on every stream.

**`nvbuf-memory-type=3` on `nvvideoconvert`.** Default NVMM is device-only; `pyds.get_nvds_buf_surface` from a Python probe segfaults. CUDA unified memory (`type=3`) keeps the `NvBufSurface` CPU-accessible without an explicit `cudaMemcpy`.

**mediamtx over real IP cameras.** Provides a reproducible, loopable, committable source. MOT17-04 has free ground truth annotations enabling quantitative tracker evaluation in M3.

---

## Known gaps

| Gap | Reason | Mitigation |
|-----|--------|------------|
| Jetson / nvargus | No Jetson hardware available | `docs/jetson-upgrade.md` (M3.5) — component diff table: x86 dGPU → JetPack |
| INT8 quantisation | GTX 1660Ti has no Tensor Cores; INT8 has no hardware speedup | Documented in `models/convert.py`; would enable on Jetson AGX Orin or RTX-class GPU |
| GPU smoke tests | Require GPU runner; written last to avoid slow CI | Planned M3.4 via `pytest --gpu` and `tests/smoke/` |
| FP32 vs FP16 latency comparison | Pipeline now runs FP16; FP32 baseline not yet benchmarked | M2.5: structured comparison once C++ decode plugin (M2.3–2.4) is in place |
| DeepSORT tracker | Re-ID model exceeds 6 GB VRAM ceiling | Documented in M3 tracker comparison rationale; ByteTrack recommended instead |

---

## Privacy by Design

The pipeline applies Gaussian blur to every detected bounding-box region before frames reach any output sink. This is implemented as a GStreamer buffer probe on each per-source `nvdsosd_{i}` sink pad (`pipelines/multi_stream.py`), calling `blur_bboxes()` (`pipelines/anonymisation.py`) on the raw `NvBufSurface`-backed numpy array for each frame.

Blurring runs *before* `nvdsosd` renders the overlay boxes, so anonymised pixels are written back into the GPU surface and any downstream consumer (display or encode) sees the blurred content. No raw face or licence-plate pixel data is written to the CSV metadata sink — only bounding-box coordinates, class labels, object IDs, and confidence scores are persisted.

`blur_bboxes()` clips all coordinates to the frame boundary and skips zero-area regions, so out-of-range detections are handled safely without crashing the pipeline.
