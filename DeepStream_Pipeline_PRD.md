# PRD: RTSP/Webcam Traffic Perception Pipeline (DeepStream)

## 1. Purpose

Close the core skill gap for the VivaCity Edge AI (Computer Vision) Engineer role: hands-on production experience with NVIDIA DeepStream and GStreamer pipelines. Build a portfolio project that mirrors VivaCity's actual stack (camera → GStreamer → DeepStream inference → tracking → output) closely enough to discuss in technical interviews with specifics, not theory.

## 2. Background

Current CV strength is edge inference optimization on Intel hardware (OpenVINO, TensorRT, FP16/INT8 quantization). Gap is DeepStream-specific plumbing: `nvinfer`, `nvtracker`, `nvstreammux`, multi-stream batching, and RTSP ingestion at the edge. JD requirement is explicit: "experience working with complex and custom NVIDIA DeepStream and GStreamer-based pipelines in production."

Hardware available: Acer Nitro 5, GTX 1660Ti (6GB VRAM, Turing, compute capability 7.5). Note: 1660Ti is Turing architecture but **ships without Tensor Cores** — supports FP16 via packed math but has no hardware INT8 acceleration.

## 3. Goals

- Build and run a working DeepStream pipeline end-to-end on local hardware
- Integrate a custom YOLOv8 model via TensorRT with a custom C++ decode plugin
- Demonstrate multi-stream handling under edge constraints (compute/memory/latency)
- Validate against both local (webcam) and networked (RTSP) camera sources
- Produce tracker and decode plugin comparison reports as interview artifacts
- Document Jetson upgrade path and ISP/camera input awareness to address JD requirements without physical Jetson access
- Implement privacy-by-design anonymisation consistent with VivaCity's product values

## 4. Non-Goals

- Jetson/JetPack hardware deployment (no device available — documented gap, not faked)
- Camera ISP tuning / nvargus implementation (requires Jetson camera stack — documented in `docs/isp-and-camera-input.md`)
- INT8 quantization (no Tensor Cores on 1660Ti — documented as hardware-limited; viable on Jetson AGX Orin or RTX-class GPU)
- DeepSORT tracker (requires re-ID model; adds VRAM pressure beyond 6GB ceiling — documented exclusion)
- Production-grade dashboarding (Grafana is a documented bonus, not a deliverable)
- Golang (nice-to-have on JD, not in scope)
- Full fleet-scale deployment (addressed in `docs/system-design.md` as architecture narrative)

## 5. Architecture

### Language
Python + pyds bindings. The `plugins/` directory (C++ TensorRT decode plugin) seeds a future full C++ refactor branch.

### Source Design

**RTSP via mediamtx**
- `rtspsrc` → `rtph264depay` → `nvv4l2decoder` → `nvstreammux` → `nvinfer` → `nvtracker` → `nvosd`
- MOT17 MP4s looped via mediamtx on localhost. Reproducible, loopable, no network variables.
- Path A (webcam via `v4l2src`) removed — mediamtx is working and covers all use cases.

### RTSP Source Strategy
`mediamtx` re-streams MOT17 clips on separate paths:
- `rtsp://localhost:8554/stream0` → MOT17-04 (benchmarking — static, high density, 1050 frames)
- `rtsp://localhost:8554/stream1` → MOT17-13 (visual demo — busy crossing)
- `rtsp://localhost:8554/stream2` → MOT17-02 (multi-stream third feed — high density, different scene)

MOT17-04 used for all quantitative benchmarking (ground truth available → full MOTA/HOTA/IDF1 suite).

### Model & Inference
- Model: `yolov8n` pretrained on COCO (persons, cars, trucks, bicycles cover the MOT17 and traffic use case)
- Export: PyTorch → ONNX → TensorRT via `models/convert.py`
- NMS: NVIDIA `EfficientNMS_TRT` plugin (replaces default ONNX NMS)
- Precision: FP16 (production config); FP32 (baseline reference)
- Engines: gitignored (`models/engines/`); always built locally via `convert.py`
- Custom decode plugin: YOLOv8 box decode (sigmoid + anchor-free coordinate transform) moved from CPU ONNX graph to GPU via C++ `IPluginV2DynamicExt` plugin in `plugins/`

### Tracking
Three-way comparison: IOU → NvDCF → ByteTrack (config-file swap, no code change per tracker)

### Privacy
Anonymisation blur probe inserted between `nvosd` and display sink — detected bounding box regions blurred before any output leaves the pipeline.

### Output & Observability
- Primary sink: console + CSV (per-frame: timestamp, object ID, class, bbox, tracker, FPS, VRAM)
- Grafana dashboard: documented bonus, not a Week 3 deliverable
- CI: GitHub Actions for CPU unit tests; GPU tests run locally via `pytest --gpu` inside NGC container

## 6. Repo Structure

```
deepstream-rtsp-pipeline/
├── docker/          # Dockerfile, docker-compose.yml
├── models/          # convert.py (PyTorch→ONNX→TRT), .gitignore for engines/weights
├── configs/         # nvinfer config, tracker configs (iou.yml, nvdcf.yml, bytetrack.yml), mediamtx.yml
├── pipelines/       # Python pipeline scripts (webcam.py, rtsp.py, multi_stream.py)
├── plugins/         # C++ TensorRT decode plugin (IPluginV2DynamicExt + CMakeLists.txt)
├── metrics/         # CSV sink, tracker comparison notebook, decode plugin comparison notebook
├── data/            # download.sh for MOT17 clips (clips not committed)
├── docs/
│   ├── jetson-upgrade.md      # x86 dGPU → Jetson component diff table
│   ├── isp-and-camera-input.md  # ISP pipeline, nvargus, tuning awareness
│   └── system-design.md       # Fleet-scale architecture narrative
└── tests/
    ├── unit/        # metadata parsing, CSV output, motmetrics computation
    ├── smoke/       # 10-second pipeline smoke test (GPU)
    └── integration/ # motmetrics integration test against MOT17 GT (GPU)
```

## 7. Comparison Reports

### Tracker Comparison (MOT17-04, ByteTrack vs NvDCF vs IOU)
| Metric | Requires GT | Tool |
|---|---|---|
| FPS / throughput | No | Frame counter probe |
| VRAM usage | No | `nvidia-smi` |
| Frame latency | No | pyds probe timestamps |
| ID switch count | Estimated | CSV trajectory analysis |
| Track fragmentation | Estimated | CSV trajectory analysis |
| MOTA | Yes | `py-motmetrics` |
| MOTP | Yes | `py-motmetrics` |
| HOTA | Yes | `py-motmetrics` |
| IDF1 | Yes | `py-motmetrics` |

### Decode Plugin Comparison (CPU ONNX decode vs C++ GPU plugin)
| Metric | Tool |
|---|---|
| End-to-end pipeline latency | pyds timestamps |
| Decode step latency (isolated) | TensorRT `IProfiler` |
| FPS delta | Frame counter |
| VRAM delta | `nvidia-smi` |

## 8. Success Metrics

- Pipeline runs stably for 30+ min continuous on RTSP source without crash/leak
- Custom YOLO model achieves 15+ FPS at FP16 on 1660Ti
- Multi-stream test runs 3 concurrent sources (MOT17-04/13/02) without dropped frames beyond acceptable threshold
- Tracker comparison report produced with full MOTA/HOTA/IDF1 suite on MOT17-04
- Decode plugin comparison shows measurable latency improvement with `IProfiler` evidence
- README documents architecture, privacy approach, known gaps (Jetson, INT8, DeepSORT, webcam path) with explicit reasoning

## 9. Risks

| Risk | Mitigation |
|---|---|
| RTSP network instability derails week 1 | Sequence Path A first; mediamtx on localhost removes real network variables |
| 6GB VRAM ceiling blocks multi-stream | Use yolov8n + FP16; measure per-stream VRAM and document ceiling |
| No Jetson access reads as a gap | `docs/jetson-upgrade.md` + `docs/isp-and-camera-input.md` address proactively |
| C++ plugin scope creep delays Week 3 | Plugin is Week 2 stretch goal; pipeline smoke test gates Week 3 entry |
| MOT17 GT format parsing complexity | `py-motmetrics` handles MOT format natively; low risk |

## 10. Interview Narrative

"Built and validated a DeepStream pipeline against both local and networked camera sources, integrated a custom TensorRT-optimized YOLOv8 model with a C++ GPU decode plugin (replacing the default CPU ONNX path), ran a three-way tracker comparison (IOU/NvDCF/ByteTrack) with full MOTA/HOTA/IDF1 evaluation on MOT17 ground truth, and stress-tested multi-stream batching under 6GB VRAM constraints — applying the same optimization discipline I used previously on the Intel VPU stack. I also documented the Jetson upgrade path and ISP camera stack in detail to demonstrate platform awareness beyond my available hardware."
