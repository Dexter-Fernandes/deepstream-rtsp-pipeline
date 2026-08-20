# Failure-Mode Playbook

How to diagnose the pipeline in production using only what it already emits: the structured JSON-line logs (`pipelines/structured_log.py`) and the periodic health/perf snapshots (`metrics/health_monitor.py`, `metrics/perf_monitor.py`). No new tooling, just a triage guide for the observability built in M3.6.1/M3.6.2.

The goal is to notice a degraded sensor, form a hypothesis from whatever signal is available, and confirm it, not just describe the happy path.

---

## Reading the logs

Every record is one JSON object per line on stderr, from the `pipeline.*` logger namespace. Grep by `event`:

| `event` | Level | Emitted by | Meaning |
|---|---|---|---|
| `pipeline_start` | INFO | `run()` | Pipeline entered PLAYING; lists `uris`, `output_dir` |
| `health_tick` | INFO | `_health_tick()`, every `max(5, perf_interval)`s | Per-source liveness/FPS/detection-staleness snapshot + system VRAM/RSS |
| `source_stalled` | WARNING | `_health_tick()` | A source's `is_live` flipped false: no frame in `liveness_window_s` (default 5s) |
| `stream_reconnect` | WARNING | source-bin pad callback | `rtspsrc` pad-link returned non-OK (usually a benign RTCP pad, but check `detail`) |
| `perf_tick` | INFO | `_perf_tick()`, only if `--perf-json` set | Aggregate FPS/VRAM/RSS sample |
| `plugin_missing` | WARNING | module import | `libyolo26_decode.so` not found; decode engine build will fail |
| `pipeline_error` | ERROR | GStreamer bus `ERROR` message | `msg` + `debug` from the failing element |
| `pipeline_eos` / `pipeline_stop` | INFO | bus `EOS` / SIGINT / `--duration` | Clean shutdown, with `reason` |

The `health_tick` payload is the primary diagnostic surface. Per source it carries `is_live`, `time_since_last_frame_s`, `time_since_last_detection_s`, `current_fps`, `fps_vs_expected`, plus a `system` block with `vram_mb`/`rss_mb`.

**Quick reference: which field moves for each failure**

| Failure mode | `is_live` | `current_fps` | `time_since_last_detection_s` | `vram_mb` / `rss_mb` |
|---|---|---|---|---|
| Stuck stream | → `false`, then `source_stalled` fires | drops to 0 | frozen (stops advancing) | unaffected |
| Silently-degraded detector | stays `true` | stays ≈ expected | grows unbounded while frames keep arriving | unaffected |
| OOM | frames stop; often no `pipeline_error` at all | drops to 0 | frozen | spikes toward capacity right before the gap, or RSS drifts upward over many ticks |
| Reconnect, no metadata | `true` again after a `stream_reconnect` line | recovers to ≈ expected | keeps growing even though `current_fps` recovered | unaffected |

---

## 1. Stuck stream

**Symptom:** a source stops producing frames. Downstream (restream, CSV) goes quiet for that source only; the other sources are unaffected.

**What the logs show:** `health_tick`'s `sources[i].is_live` flips to `false` once `time_since_last_frame_s` exceeds the 5s liveness window, immediately followed by a `WARNING source_stalled` line naming the `source_id` and the exact staleness duration.

**Likely causes:**
- Upstream RTSP source (mediamtx / ffmpeg publisher) died or was killed. Check mediamtx's own logs for that path (`runOnInit command stopped`, `conn closed`).
- Network partition, or the `rtspsrc` element entered an error state without posting a bus `ERROR` (some transport failures degrade silently rather than erroring).
- A single-source GStreamer link failure (`stream_reconnect` warning for that source with a non-benign `detail`) that never actually recovered.

**Confirm & fix:** check the mediamtx server log for the same `source_id`'s path around the timestamp `time_since_last_frame_s` implies the source died at. If mediamtx shows the publisher still running and healthy, the problem is on the consumer side (`rtspsrc` in this pipeline), so restart that source bin rather than the whole pipeline. `runOnInitRestart: yes` in `mediamtx.yml` already handles the publisher-side case automatically.

---

## 2. Silently-degraded detector (FPS fine, detections wrong)

**Symptom:** the stream looks perfectly healthy from the outside, frames keep flowing at the expected rate, but the CSV output for that source is empty or the object count has dropped to near-zero.

**What the logs show:** `is_live` stays `true` and `current_fps` / `fps_vs_expected` stay near 1.0 (frames are fine), but `time_since_last_detection_s` grows without bound. This is the dangerous case because nothing in the coarse "is it alive" check catches it; you need the detection-staleness field, not just liveness.

**Likely causes (grounded in this project's own history):**
- **Tracker association silently dropping every object.** This actually happened during M3.2: `nvtracker` associates on `detector_bbox_info.org_bbox_coords`, not `rect_params`, and `nvinfer` running in output-tensor-meta mode never sets `frame_meta.bInferDone`. If either is unset, the tracker drops every injected detection with no error: frames flow, CSV stays empty. Fixed in `pipelines/multi_stream.py`'s probe, but any future change to the tensor-meta probe that omits one of those fields reproduces this exact signature.
- Confidence threshold (`--conf-threshold`) set too high for the current scene.
- Wrong tracker config swapped in via `--tracker` pointing at a mismatched or malformed YAML.
- A model-promotion gate bypass (`metrics/model_gate.py`) let an engine through with degraded accuracy. Check `metrics/results/accuracy.json` and the gate's signed manifest for the currently-deployed engine's `match_rate`/`mean_iou`.

**Confirm & fix:** grep the affected source's CSV for a header-only file over the stall window, then diff the current probe code against the M3.2 fix above (`detector_bbox_info`, `bInferDone`, `UNTRACKED_OBJECT_ID`). That regression is exactly what `tests/unit/test_multi_stream.py::test_yolo_decode_probe_*` guards against, so a passing test suite with this symptom present points at config (threshold/tracker file), not code.

---

## 3. OOM

**Symptom:** the pipeline dies abruptly, often with no graceful shutdown log line at all.

**What the logs show, depending on severity:**
- **Recoverable / caught:** GStreamer posts a bus `ERROR`, logged as `pipeline_error` with the failing element's `msg`/`debug`. Look for CUDA/TensorRT allocation failures in `debug`.
- **Container/host OOM-killed:** the process receives `SIGKILL` before it can log or flush anything. The signature here is an absence: no `pipeline_error`, no `pipeline_eos`, no `pipeline_stop`. The log just stops mid-stream. Confirm with `docker inspect --format='{{.State.ExitCode}}' <container>` (137 = SIGKILL) or `dmesg | grep -i "killed process"` on the host. Don't waste time grepping the pipeline's own logs for a cause that was never written.

**Leading indicators before it happens:** `health_tick`'s `system.vram_mb` climbing toward the GPU's ceiling (6,144 MB on the dev GTX 1660 Ti) across successive ticks, or `system.rss_mb` drifting upward. That's the same slope `metrics/perf_monitor.py`'s `PerfMonitor.summary()` flags as `leak_suspected` in a `--perf-json` run.

**Likely causes:** batch size or stream count increased beyond what was validated in the M2.6 batch sweep; a leak in the Python probe (e.g. holding references to `NvBufSurface` past the probe's scope) showing up as RSS drift rather than VRAM; multiple pipeline processes sharing a GPU that individually fit but collectively don't.

**Confirm & fix:** rerun with `--perf-json` and check `leak_suspected` / the VRAM trend before the crash. If VRAM was climbing rather than flat, this is a batch/stream-count sizing problem, not a code leak. Compare against the sizing table in `docs/system-design.md`.

---

## 4. Sensor reconnects but produces no metadata

**Symptom:** distinct from case 1: the source does recover after a `stream_reconnect` warning (frames resume, `is_live` goes back to `true`, `current_fps` returns to normal), but detections never resume for that source.

**What the logs show:** a `WARNING stream_reconnect` line for the `source_id`, followed by `health_tick` showing `is_live: true` and `current_fps` back near expected, but `time_since_last_detection_s` continuing to grow past the reconnect point instead of resetting.

**Likely causes:** this is the reconnect-specific variant of failure mode 2: the transport layer recovered cleanly but per-source tracker/association state did not. `NvMultiObjectTracker` keeps internal state keyed by stream; a PTS/DTS discontinuity across the reconnect gap can cause `nvtracker` to treat the resumed frames as out-of-order and drop association silently, with no bus error. Treat this as a hypothesis to confirm, not a settled cause. Rule out mode 2's simpler causes (threshold, tracker config) before assuming the reconnect itself is the trigger.

**Confirm & fix:** compare `time_since_last_detection_s` against the `stream_reconnect` timestamp for the same `source_id`. If detections resumed within a tick or two of the reconnect, it was a transient blip. If the gap persists indefinitely, the source needs a hard restart (kill and let `runOnInitRestart` relaunch the publisher) rather than expecting the tracker to self-recover.

---

## Status

All four failure modes are covered. What's missing is a live, captured end-to-end walkthrough: injecting a real fault against the running 3-stream pipeline and showing the actual log lines it produces. That's tracked as a follow-up rather than done here. Everything above is derived directly from the current implementation but hasn't yet been exercised against a live fault injection.
