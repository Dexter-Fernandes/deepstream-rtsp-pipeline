# System Design: Edge AI Traffic Sensor Fleet

Fleet-scale architecture for a 5,000-sensor deployment, using this pipeline as the single-node building block. Covers the edge-to-cloud metadata path, sensor failure and reconnect, fleet upgrade strategy, scaling bottlenecks, and cross-team interfaces.

This document is preparation for the VivaCity system design interview.

---

## System overview

```
                          ┌─────────────────────────────────┐
                          │         VivaCity Cloud           │
                          │  (aggregation, analytics, APIs)  │
                          └──────────────┬──────────────────┘
                                         │  MQTT / Kafka
                              (metadata only — no pixel data)
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                           │
    ┌─────────▼──────┐        ┌──────────▼──────┐        ┌──────────▼──────┐
    │  Edge node A   │        │  Edge node B    │  ...   │  Edge node N    │
    │  (Jetson/dGPU) │        │  (Jetson/dGPU)  │        │  (Jetson/dGPU)  │
    │  15 streams    │        │  15 streams      │        │  15 streams     │
    └─────────┬──────┘        └──────────┬──────┘        └──────────┬──────┘
              │                          │                           │
      ┌───────┼───────┐          (similar)                  (similar)
      │       │       │
    cam0   cam1 ... cam14     (RTSP, H.264/HEVC)
```

**Two latency tiers:**
1. **Real-time at edge** — inference + tracking + CSV output within one frame budget (40 ms / 25 fps). Decisions that require sub-second response (e.g. adaptive signal control) happen here.
2. **Aggregated analytics in cloud** — per-minute / per-hour counts, turning movements, dwell times. Latency tolerance: seconds to minutes. Traffic pattern changes are slow; cloud aggregation does not need to be real-time.

---

## Single-node architecture (this project)

```
3 RTSP sources (MOT17 clips at 25 fps via mediamtx)
  → nvstreammux (batch=3, 1920×1080)
  → nvinfer (YOLO26n FP16, 7.1 ms / batch)      ← 140 FPS/stream bare kernel
  → nvtracker (NvMultiObjectTracker, IOU/NvDCF/ByteTrack)
  → nvstreamdemux → per-source branch
      → nvvideoconvert → nvdsosd
      → [Python probe: blur anonymisation + CSV write]
      → nvrtspoutsinkbin (restream for monitoring)

Performance (GTX 1660 Ti, measured):
  - Live RTSP: 29.7 FPS/stream (exceeds 25 fps real-time floor)
  - Unthrottled: 131.3 FPS/stream (4.4× headroom vs live floor)
  - VRAM: 1,632 MB peak (3-stream, NvDCF tracker)
  - RSS: stable — decreased 260 MB over 30 min (no leak)
```

### Scaling to 15 streams per node

The batch sweep (`metrics/decode_comparison.ipynb`) established:
- **batch=15**: 29.8 ms mean / 30.8 ms p99 — within the 40 ms budget (9 ms headroom)
- **batch=20**: 40.2 ms mean / 42.0 ms p99 — exceeds budget at the tail

At batch=15 on this hardware:
- 5,000 cameras ÷ 15 streams/node = **334 edge nodes**
- Upgrade from batch=3 → 15 reduces node count 5×: lower hosting cost, fewer failure domains to manage

On Jetson AGX Orin with INT8: batch ceiling rises to 30–50 streams/node (2–3× fewer nodes).

---

## Edge-to-cloud metadata path

**What goes over the wire**: metadata only, never pixels.

```
Per-frame CSV (edge node, local):
  frame_num, object_id, class_id, class_label, confidence, left, top, width, height

Transformed to JSON (edge agent):
  {
    "sensor_id": "junction-42-cam-0",
    "ts": "2026-06-23T14:32:11.040Z",
    "frame": 12345,
    "detections": [
      {"id": 7, "class": "person", "conf": 0.83, "bbox": [120, 340, 45, 112]}
    ]
  }
```

**Transport**: MQTT (low-bandwidth, sensor-to-broker pattern) or Kafka (higher throughput, cloud-side consumption at scale). MQTT is preferred for edge nodes with intermittent connectivity; Kafka for high-volume aggregation at the cloud ingress.

**Bandwidth estimate** (per sensor, per stream):
- ~10 detections/frame × 25 fps × ~100 bytes/detection (JSON) = 25 KB/s per stream
- 15 streams/node × 25 KB/s = 375 KB/s per node
- 334 nodes × 375 KB/s = **125 MB/s aggregate** (well within typical cloud ingress capacity)

Compare to pixel streaming: 1920×1080 H.264 at 4 Mbps × 5,000 streams = 2.5 TB/s. Metadata-only is a **20,000× bandwidth reduction**.

**Privacy**: pixel data never leaves the edge node. The per-frame CSV and JSON contain only coordinates, class labels, object IDs, and confidence — no face or number plate imagery. This is the architectural enforcement of Privacy by Design.

---

## Sensor failure and reconnect

### RTSP reconnect (existing in this repo)

`rtspsrc` has `retry` and `timeout` properties set in `MultiStreamConfig`:
```python
source.set_property("retry", config.retry)          # default 3
source.set_property("timeout", config.timeout_us)   # default 5,000,000 µs
```

On a dropped RTSP stream, `rtspsrc` retries up to `retry` times with `timeout` µs between attempts. After exhausting retries, an EOS is sent to the pipeline. The current implementation then shuts down — an acceptable behaviour for evaluation runs, not for production.

### Production reconnect loop (not yet implemented)

For production, the pipeline should loop on reconnect rather than exit:

```
RTSP disconnect
  → rtspsrc emits EOS / error message on the bus
  → bus handler catches ERROR, logs sensor_id + error code
  → pipeline transitions to PAUSED, removes old source bin, waits N seconds
  → creates new rtspsrc with the same URI, re-links to nvstreammux.sink_i
  → transitions back to PLAYING
  → emits reconnect event to cloud (MQTT: {"sensor_id": ..., "event": "reconnect", "ts": ...})
```

A watchdog timer in the Python process detects a stuck pipeline (no frames in the `frame_counts` probe counter for > 5 seconds) and forces a reconnect even if the GStreamer bus did not emit an error.

### Dead-stream detection

The `frame_counts` dict in `run()` (added in M3.3) enables per-stream heartbeat monitoring:

```python
# In the GLib perf tick:
for src_id, count in frame_counts.items():
    if count == prev_counts[src_id]:  # no new frames
        elapsed_since_last = time.time() - last_frame_time[src_id]
        if elapsed_since_last > 5.0:
            log.warning(f"[stream {src_id}] no frames for {elapsed_since_last:.0f}s — reconnecting")
            reconnect(src_id)
```

### Cold-start vs warm-start

- **Cold start**: first container boot. `init_models.py` exports YOLO26n → ONNX → TRT engines. Takes ~3 minutes on GTX 1660 Ti. During this time, no inference runs. For a production deployment, engines are pre-built and shipped as part of the OTA update package.
- **Warm start**: engines exist. Container starts and begins inference in < 5 seconds. This is the normal operational case.

---

## Fleet JetPack upgrade strategy

A 5,000-sensor fleet upgrade (JetPack 5.x → 6.x) is a multi-week operation. Sequence:

### 1. Canary (1%)
- Select 50 representative sensors (mix of junction types, lighting conditions, geographic regions)
- Push new JetPack image + rebuilt engines + updated pipeline container
- Monitor for 48 hours: FPS stability, detection count distributions, ID-switch rates, crash rates
- Gate: no regression vs production baseline on any metric

### 2. Gradual rollout (1% → 10% → 50% → 100%)
- Each tier requires 24–48 hours of monitoring before proceeding
- Automated rollback trigger: if crash rate > 0.1% or mean FPS drops > 5% below target, halt and roll back the tier

### Engine rebuild gate (model-promotion gate — M3.4)
Before any engine is deployed to production, it passes through:
```
new model weights
  → trtexec build (aarch64, target device class)
  → accuracy regression check (validate_accuracy.py vs FP32 baseline):
      matched_rate ≥ 95.0%  AND  mean_iou ≥ 0.95
  → latency check (profile_decode.py):
      p99 ≤ 30 ms for batch=15
  → PASS: engine approved for fleet rollout
  → FAIL: block rollout, alert model team
```

### OTA binary size
- FP16 engine: 6.3 MB per device
- 5,000 devices: 31.5 GB total transfer
- FP32 alternative: 11.0 MB × 5,000 = 55 GB
- **FP16 saves 23.5 GB per fleet update** — significant at constrained cellular upload speeds on edge nodes

### Rollback procedure
- Each update ships with a rollback manifest (previous engine hash + container tag)
- A failed canary triggers automatic rollback to the previous manifest
- Rollback time: < 2 minutes (pull previous container layer, restart, engines already on device)

---

## Scaling bottlenecks and mitigations

| Bottleneck | Manifestation | Mitigation |
|---|---|---|
| VRAM ceiling | batch=20 exceeds budget; DeepSORT Re-ID model doesn't fit | Use FP16 (not FP32); use ByteTrack (no Re-ID); use Jetson AGX Orin for higher VRAM |
| Python probe CPU cost | CSV write + blur on CPU path is ~2–4 ms/frame; becomes bottleneck at batch > 20 | Move CSV write to a background thread + asyncio queue; blur to a CUDA kernel |
| `nvrtspoutsinkbin` encode | H.264 NVENC encode adds ~1–2 ms/stream; at batch=15 = 15–30 ms extra | Disable restream in production (use `fakesink`); only enable for monitoring/debugging |
| GStreamer scheduling | `nvstreammux` wait for slowest source delays the whole batch | Set `live-source=1` and `drop-pipeline-eos-on-eos` on nvstreammux for live RTSP |
| MQTT broker | Single broker saturates at ~10,000 concurrent connections | Use a broker cluster (EMQX, HiveMQ) with topic partitioning by geographic region |

---

## Privacy and data governance

| Layer | Mechanism |
|---|---|
| At capture | No raw frame storage; GStreamer buffer never written to disk |
| At inference | `blur_bboxes()` probe anonymises every detected region before OSD renders |
| At output | CSV contains only bbox coordinates, class label, object ID, confidence — no pixels |
| At transport | MQTT payload is metadata JSON; TLS 1.3 in transit |
| At cloud | No reconstruction possible: coordinates without pixels cannot recover faces or plates |

VivaCity's Privacy by Design commitment is enforced at the pipeline level, not just by policy. Even if the cloud ingestion system were compromised, the attacker would receive only bounding box coordinates and class labels — no imagery.

---

## Cross-team interface points

### CV team → Cloud team

**Metadata schema** (owned by CV, consumed by Cloud):
```json
{
  "schema_version": "1.2",
  "sensor_id": "string (globally unique, provisioned at install)",
  "ts_utc": "ISO 8601 with millisecond precision",
  "frame_seq": "uint64 (monotonically increasing per sensor, per session)",
  "detections": [
    {
      "track_id": "uint64",
      "class_id": "uint8 (COCO class index)",
      "class_label": "string",
      "confidence": "float32 [0,1]",
      "bbox_xywh": [float, float, float, float],
      "bbox_norm": [float, float, float, float]  // normalised [0,1] for resolution-agnostic analytics
    }
  ]
}
```

Breaking schema changes require a version bump and a migration window agreed with the cloud team. Non-breaking additions (new fields) are backwards-compatible and can be deployed independently.

### CV team → MLOps team

**Model promotion gate** (M3.4) is the handoff boundary:
- CV team trains or fine-tunes a model, runs the accuracy regression check, and produces a signed engine artifact
- MLOps team manages the fleet rollout schedule, canary selection, and rollback triggers
- The gate outputs a go/no-go decision with a metrics report — MLOps does not need to understand the model internals

### CV team → Hardware team

**Sensor spec requirements** driven by the pipeline:
- Minimum resolution: 1280×720 (lower resolution reduces recall on small/distant objects)
- Frame rate: ≥ 25 fps (pipeline budget is 40 ms; lower frame rates cause tracker fragmentation)
- RTSP H.264 or HEVC output (nvv4l2decoder supports both)
- ISP tunability: AWB lock, AE lock, exposure override (required for night-mode stability)
- IR illuminator compatibility: 850 nm or 940 nm pass filter

### CV team → Dashboard team

**Analytics endpoint** (Prometheus or InfluxDB line format):
```
pipeline_fps_per_stream{sensor_id="junction-42-cam-0", stream="0"} 25.1 1719148331040
pipeline_vram_mb{sensor_id="junction-42-cam-0"} 2780 1719148331040
pipeline_rss_mb{sensor_id="junction-42-cam-0"} 1652 1719148331040
pipeline_detections_per_frame{sensor_id="junction-42-cam-0", stream="0"} 8.3 1719148331040
```

This is the M3.6 observability work: structured per-stream metrics exposed as a scrape endpoint, consumed by Grafana dashboards for the oncall team.
