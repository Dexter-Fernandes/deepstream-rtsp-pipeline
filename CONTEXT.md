# Project Context

This file captures all design decisions, rationale, and constraints established during the initial planning session. Use it to orient any future session without re-deriving decisions already made.

---

## Purpose

Portfolio project targeting the VivaCity Edge AI (Computer Vision) Engineer role. Goal: build hands-on production experience with NVIDIA DeepStream and GStreamer pipelines to close the skill gap from Intel/OpenVINO background.

**Target JD:** VivaCity Edge AI (Computer Vision) Engineer, Greater London (Hybrid Internship)

---

## Hardware

- Acer Nitro 5, GTX 1660Ti, 6GB VRAM, Turing (compute capability 7.5)
- **Critical:** 1660Ti ships without Tensor Cores — FP16 works via packed math, INT8 has no hardware acceleration benefit
- No Jetson hardware available — documented as known gap, not faked

---

## All Design Decisions

### Language & Container
- **Python + pyds** bindings for the pipeline. C++ refactor is a future parallel branch.
- **NGC DeepStream container** (`nvcr.io/nvidia/deepstream:7.1-triton-multiarch`) via `nvidia-container-toolkit`. Chosen over bare-metal to avoid dependency hell and make the repo reproducible.

### RTSP Source
- **mediamtx** re-streaming MOT17 clips on localhost. Chosen over real IP cameras (reproducible, scriptable, committable config).
- Three streams on separate paths:
  - `stream0` → MOT17-04 (benchmarking: static camera, high density, 1050 frames, has ground truth)
  - `stream1` → MOT17-13 (visual demo: busy crossing, most VivaCity-relevant visually)
  - `stream2` → MOT17-02 (multi-stream third feed: static, high density, different scene)

### Model
- **YOLOv8n pretrained on COCO** — no fine-tuning. Portfolio value is in DeepStream integration, not training.
- COCO classes (person, car, truck, bicycle) naturally present in MOT17 sequences.

### TensorRT Conversion
- Explicit `models/convert.py` script (PyTorch → ONNX → TensorRT). Engines are gitignored (`models/engines/`).
- **EfficientNMS_TRT** plugin replaces default ONNX NMS.
- **FP32** = baseline reference. **FP16** = production config.
- **INT8** = documented-but-excluded. Reason: no Tensor Cores on 1660Ti, no hardware speedup. Would enable on Jetson AGX Orin or RTX-class GPU.

### Custom TensorRT C++ Decode Plugin (Level 3)
- YOLOv8 box decode (sigmoid + anchor-free coordinate transform) moved from CPU ONNX graph to GPU via C++ `IPluginV2DynamicExt` plugin in `plugins/yolov8_decode/`.
- This is also the seed of the future C++ refactor branch.
- Comparison report: CPU decode vs GPU plugin (end-to-end latency, decode step via `IProfiler`, FPS delta, VRAM delta).

### Tracking
- **Three-way comparison:** IOU → NvDCF → ByteTrack. Config-file swap only (`--tracker` CLI flag), no code change per tracker.
- **DeepSORT excluded:** requires re-ID model, adds VRAM pressure beyond 6GB ceiling. Documented with explicit reasoning.
- **ByteTrack** is the production recommendation: handles low-confidence detections (partially occluded vehicles) without a re-ID model.

### Tracker Comparison Metrics
- **No ground truth required:** FPS, VRAM (`nvidia-smi`), frame latency (pyds probe timestamps), ID switch count (CSV analysis), track fragmentation (CSV analysis)
- **Ground truth required (MOT17-04):** MOTA, MOTP, HOTA, IDF1 via `py-motmetrics`
- HOTA is the current primary standard (balances detection + association quality).
- MOT17-04 used for all benchmarking because it has free ground truth annotations.

### Privacy
- Anonymisation blur probe between `nvosd` and display sink — detected bbox regions blurred before any output leaves the pipeline.
- "Privacy by Design" section in README — mirrors VivaCity's stated product values.

### Output & Observability
- **Primary sink:** console + CSV (per-frame: timestamp, object ID, class, bbox, confidence, tracker, FPS, VRAM).
- **Grafana dashboard:** documented bonus add-on, not a deliverable.

### Testing
- `tests/unit/` — metadata parsing, CSV roundtrip, motmetrics computation. CPU, runs in GitHub Actions.
- `tests/smoke/` — 10-second pipeline smoke test, assert frames > 0 and CSV non-empty. Requires GPU (`pytest --gpu`).
- `tests/integration/` — motmetrics on MOT17-04 excerpt, assert HOTA within expected range. Requires GPU (`pytest --gpu`).
- **GitHub Actions:** CPU unit tests only. GPU tests are local-only with clear `pytest --gpu` instructions.

### Documentation
- `docs/jetson-upgrade.md` — component diff table: x86 dGPU → Jetson (nvargus, JetPack, INT8, unified memory, TDP modes). Addresses JD requirement on platform upgrades.
- `docs/isp-and-camera-input.md` — ISP pipeline stages, nvargus on Jetson, how ISP misconfiguration degrades model accuracy. Addresses JD hard requirement on camera input / optical / ISP awareness.
- `docs/system-design.md` — fleet-scale architecture narrative (1→5000 sensors, edge→cloud metadata, sensor reconnect, JetPack fleet upgrades). Preparation for VivaCity's 1.5-hour system design interview.

---

## Repo Structure

```
deepstream-rtsp-pipeline/
├── docker/          # Dockerfile, docker-compose.yml, mediamtx config
├── models/          # convert.py, .gitignore for engines/ and weights
├── configs/         # nvinfer config, tracker configs (iou, nvdcf, bytetrack)
├── pipelines/       # webcam.py, rtsp.py, multi_stream.py
├── plugins/         # C++ TensorRT decode plugin (IPluginV2DynamicExt + CMakeLists.txt)
├── metrics/         # CSV sink, tracker_comparison.ipynb, decode_comparison.ipynb, evaluate_tracker.py
├── data/            # download.sh for MOT17 clips (clips not committed)
├── tests/
│   ├── unit/
│   ├── smoke/
│   └── integration/
├── docs/
│   ├── jetson-upgrade.md
│   ├── isp-and-camera-input.md
│   └── system-design.md
├── CONTEXT.md       # This file
├── MILESTONES.md    # M1–M3 implementation steps
└── DeepStream_Pipeline_PRD.md
```

---

## JD Gap Analysis

| JD Requirement | How Addressed |
|---|---|
| DeepStream + GStreamer pipelines in production | Core pipeline build (M1–M3) |
| nvinfer, nvtracker, nvstreammux | Explicit pipeline elements throughout |
| Platform upgrades (JetPack) | `docs/jetson-upgrade.md` |
| Edge constraints (latency, compute, memory) | VRAM tracking, FPS benchmarking, FP16 tuning |
| Camera input incl. ISPs and tuning | `docs/isp-and-camera-input.md` |
| Custom layers or kernels | C++ `IPluginV2DynamicExt` decode plugin (M2.3–M2.4) |
| nvargus | Documented as Jetson-only gap with explicit reasoning |
| Privacy by design | Anonymisation blur probe + README section |
| System design interview | `docs/system-design.md` + tracker/decode comparison reports |

---

## Known Gaps (Explicitly Documented)

- **Jetson/nvargus:** no hardware available — addressed via `docs/jetson-upgrade.md` and `docs/isp-and-camera-input.md`
- **INT8:** hardware-limited on 1660Ti (no Tensor Cores) — noted in `convert.py` and README
- **DeepSORT:** excluded due to VRAM cost — noted in tracker comparison report
- **Golang:** JD nice-to-have, not in scope — stretch goal `docs/golang-integration.md`
- **GPU CI:** GPU tests are local-only — documented in README and test runner instructions

---

## Interview Narrative

"Built and validated a DeepStream pipeline against both local and networked camera sources, integrated a custom TensorRT-optimized YOLOv8 model with a C++ GPU decode plugin (replacing the default CPU ONNX path), ran a three-way tracker comparison (IOU/NvDCF/ByteTrack) with full MOTA/HOTA/IDF1 evaluation on MOT17 ground truth, and stress-tested multi-stream batching under 6GB VRAM constraints — applying the same optimization discipline I used previously on the Intel VPU stack. I also documented the Jetson upgrade path and ISP camera stack in detail to demonstrate platform awareness beyond my available hardware."

---

## Hiring Process (VivaCity)

1. 30-minute screening interview
2. 1.5-hour system design interview (work together with a VivaCity engineer) — `docs/system-design.md` is prep for this
3. 1.5-hour final round: 45-min technical experience + 45-min soft skills
