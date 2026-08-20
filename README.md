# deepstream-rtsp-pipeline

NVIDIA DeepStream pipeline running three concurrent RTSP streams through GPU-accelerated inference, object tracking, and per-source CSV metadata output — built on a GTX 1660Ti (6 GB VRAM) with a full TDD test suite.

## What this demonstrates

- Three RTSP sources batched through a single `nvstreammux`, demuxed back per-source for independent OSD and restream output
- YOLO26n FP16 running end-to-end: `.pt → ONNX (dynamic batch) → TRT FP16` via `trtexec`; Python tensor-meta probe decodes `[300, 6]` output and populates `NvDsObjectMeta` without a compiled C parser
- `IPluginV2DynamicExt` CUDA kernel appended to the TRT network via TRT Python API; converts xyxy→xywh on GPU inside TRT, replacing the Python coordinate-transform loop; `metrics/profile_decode.py` isolates per-layer latency via `IProfiler`
- 221 CPU-safe unit tests written before implementation (red→green); no GPU required for the test suite
- Gaussian blur applied to every detected bbox region before `nvdsosd` renders or any output leaves the pipeline
- FP32 vs FP16 vs FP16+decode-plugin compared on latency, VRAM, engine size, and fleet OTA cost; batch sweep 1–100 against a 25 fps real-time budget with fleet-sizing projections (`metrics/decode_comparison.ipynb`)
- Per-frame CSV sink; mediamtx-served Wildtrack Cam1–3 (5-min 1080p60 clips, TCP RTSP) as the live source; MOT17-04 played via a file-source branch for GT-aligned MOTA/IDF1 tracker evaluation
- Optional cross-camera MTMC identity fusion: uncertainty-aware ground-plane geometry, generation-qualified tracker IDs, an optional DeepStream ReID SGIE, and a final authoritative assignment map for evaluation
- Structured JSON-line logging (`pipelines/structured_log.py`) with per-stream `source_id` on every record; per-sensor health monitor (`metrics/health_monitor.py`) tracks liveness, rolling FPS vs expected, and time-since-last-detection, emitting a `health_tick` log line and `WARNING source_stalled` alerts via a GLib periodic callback
- NGC DeepStream 9.0 + pyds compiled from source; `docker compose up` handles model export and conversion on first run

---

## Pipeline architecture

```
mediamtx (RTSP server)
  ├─ stream0 (Wildtrack cam1)  ──┐
  ├─ stream1 (Wildtrack cam2)  ──┤
  └─ stream2 (Wildtrack cam3)  ──┘

Per-source source bins (× 3):
  rtspsrc → rtph264depay → nvv4l2decoder → queue ──→ nvstreammux.sink_{i}

Shared inference chain (batched, N=3):
  nvstreammux → nvinfer (YOLO26n FP16 + yolo26_decode plugin, network-type=100, output-tensor-meta=1)
             ← [nvinfer SRC probe: reads xywh tensor → NvDsObjectMeta (80 COCO classes)]
             → nvtracker (NvMultiObjectTracker)
             → optional nvinfer SGIE (ReIdentificationNet, sparse tensor metadata)
             ← [MTMC probe: atomic mux batch → ground plane → immutable global-ID map]
             → nvstreamdemux

Per-source output branches (× 3):
  demux.src_{i} → queue → nvvideoconvert (RGBA, unified mem)
               → nvdsosd ← [Python probe: blur + CSV write]
               → nvrtspoutsinkbin (ports 8556/8557/8558)
```

A single `nvdsosd` on the batched buffer only draws on source 0 — the per-branch placement is required and mirrors NVIDIA's `deepstream-demux-multi-in-multi-out` reference topology.

---

## Quick-start

**Prerequisites:** NVIDIA driver ≥ 590.48, `nvidia-container-toolkit`, `mediamtx` on host, Wildtrack cam1/cam2/cam3 5-min clips as MP4 in `data/` (live streaming); MOT17-04 as MP4 in `data/` for GT-aligned tracker evaluation (`metrics/evaluate_tracker.py`).

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

# Tune detection confidence (default 0.18 — F1-optimal on WildTrack)
# Add --conf-threshold 0.5 to the compose command: or run directly:
# docker run ... ds-pipeline python3 pipelines/multi_stream.py --uri ... --conf-threshold 0.4
```

### Cross-camera identity fusion

MTMC is opt-in, so the existing per-camera tracker baseline is unchanged. The deterministic file-input path below writes frame-bound `global_id`, reconnect `generation`, accepted batch identity, and acceptance status to each CSV, plus a whole-run authoritative map to `mtmc.json`. Legacy CSVs default to generation 0 and use aligned `frame_num` as their accepted batch identity:

```bash
mkdir -p /tmp/mtmc-run
docker compose run --rm -v /tmp/mtmc-run:/tmp/mtmc-run pipeline \
  python3 pipelines/multi_stream.py \
  --uri file:///workspace/data/wildtrack_c1_gt.mp4 \
  --uri file:///workspace/data/wildtrack_c2_gt.mp4 \
  --uri file:///workspace/data/wildtrack_c3_gt.mp4 \
  --output-dir /tmp/mtmc-run --no-sync --mtmc \
  --homography 0=configs/homography_C1.json \
  --homography 1=configs/homography_C2.json \
  --homography 2=configs/homography_C3.json \
  --mtmc-min-affinity 0.3 --mtmc-json /tmp/mtmc-run/mtmc.json

.venv/bin/python -m metrics.evaluate_mtmc \
  --wildtrack-root /path/to/WildTrack/labelled_ds \
  --pred /tmp/mtmc-run/output_stream{0,1,2}.csv \
  --camera 0=C1 1=C2 2=C3 \
  --homography 0=configs/homography_C1.json \
               1=configs/homography_C2.json \
               2=configs/homography_C3.json \
  --fuse-offline --final-map /tmp/mtmc-run/mtmc.json \
  --min-affinity 0.3
```

Add `--mtmc-appearance` to enable the sparse ReID SGIE. The bundled model is provisioned on demand; if it is unavailable, startup explicitly falls back to geometry-only fusion. For RTSP, DeepStream derives `NvDsFrameMeta.ntp_timestamp` from RTCP sender reports. Fusion waits until every source clock is ready, treats each complete mux batch atomically, and refuses incomplete batches or timestamp skew above the default 100 ms gate.

---

## Tests

```bash
pip install pytest
pytest tests/unit/ -v      # 378 tests, CPU-only, no GPU required
```

| Module | Tests | What they cover |
|--------|-------|-----------------|
| `metadata_parser` | 8 | `Detection` dataclass, identity defaults, source propagation, and fake pyds structs |
| `csv_sink` | 8 | Header, field values, identity generations, flush-on-write, and multi-detection roundtrip |
| `anonymisation` | 6 | Blur applied, pixels outside bbox unchanged, out-of-bounds clip |
| `frame_accessor` | 4 | NVMM surface accessor with injectable `_get_surface` |
| `rtsp_pipeline` | 17 | Config defaults, arg parsing, source props, restream URI parsing |
| `multi_stream` | 50 | Multi-URI parsing, CSV path routing, tracker/perf flags, file and RTSP clocks, RTCP clock acquisition/loss/recovery, frame-bound receipt handoff, ReID placement, reconnects, and OSD labels |
| `convert` | 14 | `engine_path` naming, `build_trtexec_cmd` flags, dynamic-batch shape profile, `parse_args` |
| `export_yolo26` | 3 | `parse_args` for weights path and output-dir |
| `init_models` | 9 | Skip/run logic for all cold-start and warm-start combinations; decode-engine skip/build paths |
| `output_parser` | 6 | Threshold filtering, xyxy→xywh conversion, class_id extraction, batch-dim squeeze |
| `decode_engine` | 5 | `decode_engine_path` naming, `parse_args` defaults and flags |
| `validate_accuracy` | 20 | `box_iou`, greedy IoU matching, per-engine comparison, per-coord decode delta, `preprocess_frame` shape/dtype/range |
| `profile_decode` | 10 | `_parse_tail_latencies` (min/median/p99/max from trtexec output), `budget_check` (mean+p99 vs frame budget), `_SimpleProfiler.to_dict` tail fields |
| `evaluate_tracker` | 17 | GT loading (visibility filter), prediction loading (frame-offset), `MOTAccumulator` build, MOTA/MOTP/IDF1 compute, unique-track count |
| `perf_monitor` | 21 | `compute_interval_fps`, `PerfMonitor.record/summary` (FPS, VRAM, RSS, leak heuristic), `to_dict`/`write_json` round-trip, `sample_rss_mb`, `sample_vram_mb` |
| `model_gate` | 19 | `match_rate ≥ 0.95` AND `mean_iou ≥ 0.95` gate checks, signed manifest (SHA-256 + timestamp), exit 0/1 for CI |
| `structured_log` | 8 | JSON formatter, required fields (`ts`/`logger`/`level`/`event`), optional `source_id`, arbitrary extra fields, idempotent `configure_pipeline_logging` |
| `health_monitor` | 12 | Per-source liveness window, `is_live` flag, rolling FPS, `fps_vs_expected` ratio, `time_since_last_detection_s`, system dict, never-seen-source defaults |
| WildTrack / ground plane / homography | 36 | GT identity loading, frame-aligned clip validation, foot-point reliability and uncertainty, projection Jacobians, held-out homography fitting, and calibration quality gates |
| MTMC / runtime / evaluator | 81 | Synchronised association, constrained clustering, reconnect generations, frame-bound runtime/CSV receipts, TTL/liveness eviction, authoritative history, rejected-evidence filtering, batch-parity agreement, and image/cross-camera/ground metrics |
| ReID | 13 | Tensor-contract validation, L2-normalised embedding extraction, SGIE configuration, sparse feature caching, and geometry-only fallback |

GPU smoke tests: `pytest --gpu` (`tests/smoke/` — requires GPU runner; auto-skipped in CI).

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

**Cross-camera identity fusion (M3.8)** — measured on the three 401-frame, GT-aligned WildTrack clips. The tracker and detector remain per-camera; only the identity layer changes.

| Identity mode | Pooled IDF1 ↑ | Cross-camera P / R / F1 ↑ | Ground MODA / MODP @ 0.5 m ↑ | FPS / source | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Per-camera IDs (no MTMC) | 0.197702 | 0 / 0 / 0 | -0.320235 / 0.556104 | 56.59 | 951 MB |
| Geometry MTMC, `min_affinity=0.3` | **0.244635** | **0.669591 / 0.281068 / 0.395937** | **-0.245535 / 0.557548** | **52.48** | **945 MB** |
| Geometry + sparse ReID, `w_app=0.5` | 0.241628 | 0.651049 / 0.286550 / **0.397948** | -0.254465 / 0.554020 | 37.79 | 1,085 MB |

Geometry is the selected default. Sparse ReID improves cross-camera F1 by only 0.002011 absolute while lowering pooled IDF1 and ground-plane scores, reducing throughput by 28%, and adding 140 MB peak VRAM. The optional path remains available for camera networks where appearance is more discriminative. The geometry online shutdown map agrees with offline fusion on **10,608 / 10,608 post-warm-up rows (100%)** after a one-to-one ID permutation.

The live RTSP path completed a 300-second, three-source appearance soak at 29.80 FPS/source and 1,173 MB peak VRAM. Accepted association batches had zero missing-source desyncs and zero skew refusals. DeepStream emitted eight isolated invalid-NTP mux attempts; each was rejected before fusion and recovered within 31.7 ms. A qualitative live ID (`1751`) was written across all three camera CSVs for 1,679 rows over a 28.0-second span. The exact commands, hashes, clock transitions, bounded-state proof, and cleanup audit are recorded in the M3.8 comparison artifact.

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
- ✓ **M3.4** — GPU smoke tests (`tests/smoke/`); motmetrics integration test; GitHub Actions unit-test workflow; model-promotion gate (`metrics/model_gate.py`, 19 CPU-safe tests) with SHA-256 signed manifest and CI exit 0/1
- ✓ **M3.5** — `docs/jetson-upgrade.md`, `docs/isp-and-camera-input.md`, `docs/system-design.md`; README completeness pass
- ✓ **M3.8** — WildTrack calibration and GT-aligned clips; CPU MTMC core and evaluator; online atomic mux-batch fusion; optional sparse ReID; 100% online/offline final-map agreement; 300-second RTSP soak; `metrics/mtmc_comparison.ipynb`
- *(in progress)* **M3.6** — Observability / reactive debugging:
  - ✓ Structured JSON logging (`pipelines/structured_log.py`) — `configure_pipeline_logging` / `get_pipeline_logger` / `log_event`; all pipeline `print()` calls replaced with levelled JSON-line records (DEBUG/INFO/WARNING/ERROR) to stderr; 8 CPU-safe tests
  - ✓ Per-sensor health metrics (`metrics/health_monitor.py`) — per-source liveness (configurable window), rolling FPS vs expected, time-since-last-detection; `_health_tick` GLib callback emits a `health_tick` JSON line every interval and a `WARNING source_stalled` for any dead stream; 12 CPU-safe tests
  - ✓ Failure-mode playbook (`docs/failure-mode-playbook.md`) — how to diagnose a stuck stream, a silently-degraded detector (FPS fine but detections wrong, using the real M3.2 `nvtracker` association bug as a worked example), an OOM, and a sensor that reconnects but produces no metadata; grounded in the actual `event` names/fields the pipeline emits
  - ☐ End-to-end debugging walkthrough — inject a fault, show how logs/metrics surface it

---

## Key design decisions

**`network-type=100` + Python tensor-meta probe for YOLO26n.** nvinfer's built-in bbox parsers expect anchor-based or NMS-post-processed output in a specific layout. YOLO26n's one-to-one matching head emits `[batch, 300, 6]` (end-to-end NMS baked in). Rather than compile a C `.so` custom parser, we use `network-type=100` (custom) with `output-tensor-meta=1`: nvinfer exposes the raw tensor in `NvDsInferTensorMeta` and a Python probe on the nvinfer SRC pad calls `parse_yolo26_output()` and populates `NvDsObjectMeta` directly. The decode logic stays in pure Python, is fully unit-testable without a GPU. M2.3+M2.4 replaced this with the `IPluginV2DynamicExt` CUDA kernel; the Python probe now only reads the already-decoded xywh tensor from `NvDsInferTensorMeta` and populates `NvDsObjectMeta`.

**Dynamic-batch ONNX export.** Exporting with `dynamic=True` makes the batch dimension flexible. `trtexec` is then called with `--minShapes=images:1x3x640x640 --optShapes=images:3x3x640x640 --maxShapes=images:3x3x640x640`, producing a single engine file (`yolo26n_fp16_b3.engine`) that nvinfer can use for any batch size in [1, 3] — both single-stream testing and 3-stream production use the same engine.

**C++ `IPluginV2DynamicExt` decode plugin — xyxy→xywh on GPU.** The M2.2 Python probe looped over 300 detections on CPU to convert xyxy → xywh. M2.3+M2.4 replace this with a CUDA kernel (`plugins/yolo26_decode/yolo26_decode_kernel.cu`) compiled as a TRT `IPluginV2DynamicExt` plugin. `models/decode_engine.py` uses the TRT Python API to parse the ONNX, unmark the YOLO output, append the plugin as a custom layer, and rebuild the engine — the resulting `yolo26n_fp16_b3_decode.engine` emits xywh from TRT directly. The probe reads the transformed coordinates with no Python for-loop. `metrics/profile_decode.py` uses TRT `IProfiler` to print per-layer latency and isolate the `yolo26_decode` kernel time.

**`_make_nvinfer_config` for TRT batch-size (legacy engines).** `nvinfer.set_property("batch-size", n)` overrides the config but does not trigger an engine rebuild — a cached batch-1 engine gives undefined behaviour at batch-3. The fix rewrites both `batch-size` and the engine file path in a temp config. For YOLO26n the engine already covers batch 1–3, so only `batch-size` is rewritten; the path is left unchanged.

**Per-branch `nvdsosd` after demux.** A single batched OSD only composites onto the first frame in the batch (source 0). Each branch gets its own `nvvideoconvert(unified) → nvdsosd` so boxes render correctly on every stream.

**`nvbuf-memory-type=3` on `nvvideoconvert`.** Default NVMM is device-only; `pyds.get_nvds_buf_surface` from a Python probe segfaults. CUDA unified memory (`type=3`) keeps the `NvBufSurface` CPU-accessible without an explicit `cudaMemcpy`.

**mediamtx over real IP cameras.** Provides a reproducible, loopable, committable source. Wildtrack Cam1–3 give realistic sustained multi-stream RTSP load for the live pipeline; Wildtrack does have per-frame pedestrian GT (JSON, sampled every 5th frame), but `metrics/evaluate_tracker.py` currently only parses MOT17's dense per-frame CSV layout, so MOT17-04 is used separately (via a file-source branch, bypassing RTSP) for quantitative tracker evaluation in M3.2.

**Global identity above `nvtracker`, not inside it.** `nvtracker` remains the per-camera motion/appearance tracker and its `object_id` is never overwritten. The MTMC layer qualifies that local ID with source and reconnect generation, projects reliable person foot points through per-camera homographies, and associates only complete mux batches whose true source timestamps pass the skew gate. Its published map is immutable and TTL-bounded for the live path; compact whole-run support/co-occurrence statistics produce the authoritative shutdown map without retaining every historical detection.

**Appearance is optional and sparse.** The bundled ReIdentificationNet SGIE runs in classifier scheduling mode with tensor metadata enabled and a reinference interval of 14. The MTMC core caches the last valid L2-normalised embedding per generation-qualified tracklet. Missing embeddings are neutral rather than falsely treated as a perfect match, and model-provisioning failure explicitly selects geometry-only fusion.

---

## Known gaps

| Gap | Reason | Mitigation |
|-----|--------|------------|
| Jetson / nvargus | No Jetson hardware available | `docs/jetson-upgrade.md` — component diff table: x86 dGPU → JetPack; nvargus CSI path; INT8 on Jetson; TDP modes |
| INT8 quantisation | GTX 1660Ti has no Tensor Cores; INT8 has no hardware speedup | Documented in `models/convert.py`; would enable on Jetson AGX Orin or RTX-class GPU |
| GPU smoke tests | Require GPU runner; written last to avoid slow CI | Planned M3.4 via `pytest --gpu` and `tests/smoke/` |
| Decode plugin shows little gain on YOLO26n | YOLO26n is NMS-free (300 pre-decoded boxes), so the kernel does ~0.006 ms of work; the accuracy comparison is between two separately-compiled TRT graphs, not a controlled kernel isolation | M2.6: YOLOv8n plugin (8,400 candidates + DFL + NMS) demonstrates where the pattern pays off |
| DeepSORT tracker | Re-ID model exceeds 6 GB VRAM ceiling | Documented in M3 tracker comparison rationale; ByteTrack recommended instead |
| RTSP NTP discontinuities | DeepStream can emit an isolated zero-NTP frame after clock acquisition | Fusion fails closed, advances liveness without evidence, rate-limits the transition, and resumes only after a valid timestamp |
| Whole-run MTMC history | The authoritative final map needs compact support/co-occurrence statistics that grow with unique tracklets | Live observations and published IDs remain TTL-bounded; rotate or finalise very long capture sessions |
| MTMC worker overload | The worker queue is unbounded; the measured 52.48 fps/source file run and 29.80 fps/source soak showed headroom and no accumulation, but higher-rate inputs are not admission-controlled | Add queue-depth telemetry, then reserve capacity before the submission lock and emit rejected frame receipts on saturation |

---

## Privacy by Design

The pipeline blurs every detected bounding-box region before frames reach any output sink. A GStreamer buffer probe on each per-source `nvdsosd_{i}` sink pad calls `blur_bboxes()` (`pipelines/anonymisation.py`) on the raw `NvBufSurface`-backed numpy array for each frame.

Blurring runs *before* `nvdsosd` renders the overlay boxes, so anonymised pixels are written back into the GPU surface and any downstream consumer (display or encode) sees the blurred content. The CSV metadata sink stores only bounding-box coordinates, class labels, object IDs, and confidence scores — no raw face or licence-plate pixel data.

`blur_bboxes()` clips coordinates to the frame boundary and skips zero-area regions, so out-of-range detections are handled safely without crashing the pipeline.
