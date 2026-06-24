# Jetson Upgrade Guide

Upgrading this pipeline from an x86 dGPU (GTX 1660 Ti) to a Jetson edge device. Covers every pipeline element that changes, the memory model difference, INT8 on Jetson, TDP / power modes, and the engine rebuild requirement.

This project runs on x86 because that is the available hardware. No Jetson-specific code is faked — gaps are documented explicitly and addressed here.

---

## Component diff: x86 dGPU → Jetson

| Pipeline element | x86 (this repo) | Jetson equivalent | Notes |
|---|---|---|---|
| **Camera input** | `rtspsrc → rtph264depay` (IP camera or mediamtx) | `nvarguscamerasrc` (CSI) **or** keep `rtspsrc` (IP cam) | CSI sensors require `nvarguscamerasrc`; IP cameras keep the RTSP path unchanged |
| **Decode** | `nvv4l2decoder` | `nvv4l2decoder` | Same element; Jetson routes to the on-chip NVDEC block; no code change |
| **Inference** | `nvinfer` + FP16 engine (x86 aarch64) | `nvinfer` + FP16 or INT8 engine (aarch64) | Engine must be rebuilt on the target device — cross-compiled engines do not work |
| **Tracker** | `nvtracker` + NvMultiObjectTracker | `nvtracker` + NvMultiObjectTracker | Same binary; reduce `featureImgSizeLevel` to 1 on Jetson Nano / Orin Nano (tighter VRAM) |
| **OSD** | `nvdsosd` | `nvdsosd` | Unchanged |
| **Restream** | `nvrtspoutsinkbin` | `nvrtspoutsinkbin` | Unchanged |
| **Buffer memory** | `nvbuf-memory-type=3` (CUDA unified, dGPU) | `nvbuf-memory-type=0` (NVMM, Jetson) | See Memory model section below |
| **Python probe surface access** | `pyds.get_nvds_buf_surface` (works with unified mem) | `nvbuf_utils.buffer_map` / `buffer_unmap` | Explicit map/unmap required on Jetson; segfaults if omitted |
| **Plugin `.so`** | Compiled for x86-64 | Must recompile for aarch64 | `cmake -DCMAKE_TOOLCHAIN_FILE=...` or build on-device |

---

## Camera input: nvargus and the Argus CSI path

`nvarguscamerasrc` is Jetson-only. It exposes the full Argus ISP stack (AWB, AE, noise reduction, lens shading) directly to GStreamer. The pipeline element replaces the entire `rtspsrc → rtph264depay → nvv4l2decoder` source bin:

```
nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1
  → nvvideoconvert → nvstreammux.sink_0
```

Key Argus properties exposed via `nvarguscamerasrc`:
- `awblock` — lock AWB to a fixed point; useful when entering a known lighting condition (tunnel, night)
- `aelock` — lock autoexposure; prevents confidence swings when a headlight passes through the frame
- `exposuretimerange` — constrain shutter speed (e.g. `"5000000 5000000"` = fixed 5 ms); important for motion-blur control on fast vehicles
- `gainrange` — cap digital gain to reduce noise amplification at night
- `wbmode` — force a white-balance preset (`5` = daylight; `1` = auto)

For IP cameras (VivaCity's deployed sensors are network-connected), keep the `rtspsrc` path — only the CSI-attached sensors use `nvarguscamerasrc`. See `docs/isp-and-camera-input.md` for ISP tuning detail.

---

## Memory model: CUDA unified vs NVMM

| | x86 dGPU (this repo) | Jetson |
|---|---|---|
| `nvbuf-memory-type` | `3` (CUDA unified memory) | `0` (NVMM — NVIDIA memory-mapped) |
| CPU access from Python probe | Direct via `pyds.get_nvds_buf_surface` | Requires explicit `nvbuf_utils.buffer_map(fd, plane)` → pointer → `buffer_unmap` |
| Why | dGPU has discrete VRAM; unified memory is a PCIe-backed mapping | Jetson has shared DRAM; NVMM is the native zero-copy path; unified memory adds unnecessary overhead |

Code change required in `pipelines/multi_stream.py`:

```python
# x86 (current)
converter.set_property("nvbuf-memory-type", 3)

# Jetson
converter.set_property("nvbuf-memory-type", 0)

# And in the probe, replace:
surface = pyds.get_nvds_buf_surface(hash(gst_buffer), 0)
# with:
import nvbuf_utils
fd = <buffer fd from NvBufSurface>
ptr = nvbuf_utils.buffer_map(fd, 0)
# ... operate on ptr as numpy array ...
nvbuf_utils.buffer_unmap(fd, 0)
```

---

## TensorRT engine rebuild

Engines are architecture-specific. An engine built on x86 will not load on Jetson (and vice versa). The existing `models/convert.py` pipeline runs unchanged on Jetson — the ONNX model is portable:

```bash
# On the Jetson device:
python3 models/convert.py --weights models/yolo26n.pt --batch 3 --fp16
# Produces: models/engines/yolo26n_fp16_b3.engine (aarch64)
```

The decode plugin `.so` must also be recompiled:
```bash
cd plugins/yolo26_decode
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Produces: libyolo26_decode.so (aarch64)
```

Engine filenames are unchanged; the `nvinfer` config and `init_models.py` cold-start logic work as-is.

---

## INT8 on Jetson

The GTX 1660 Ti has no Tensor Cores — INT8 quantisation has no hardware speedup on this device (INT8 uses DP4A which SM75 supports, but without dedicated tensor cores throughput gain is marginal and accuracy loss is not justified).

On Jetson AGX Orin or Jetson Orin NX (Ampere architecture, dedicated INT8 Tensor Cores), INT8 gives a genuine 2–4× throughput improvement over FP16.

### Wiring INT8 into convert.py

```python
# In models/convert.py, add --int8 flag:
if args.int8:
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = EntropyCalibrator(
        data_dir="data/calibration/",   # ~500 representative frames
        cache_file="models/int8_cache.bin",
        batch_size=1,
    )
    config.int8_calibrator = calibrator
```

Calibration dataset requirement: 200–1,000 representative frames covering day/night/weather conditions. Using only daytime frames will degrade night accuracy. The existing MOT17-04 clip is insufficient — it is a single scene, a single lighting condition.

### DLA (Deep Learning Accelerator)

Jetson AGX Orin has two DLA cores. DLA offloads convolution layers from the GPU, freeing CUDA cores for tracker + OSD. Not all layers are DLA-compatible (attention blocks in YOLO26n's transformer head are GPU-only). A mixed GPU+DLA engine can be built via TRT:

```python
config.default_device_type = trt.DeviceType.DLA
config.DLA_core = 0
config.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # GPU handles unsupported layers
```

---

## JetPack version mapping

| JetPack | DeepStream | CUDA | TensorRT | Python bindings |
|---|---|---|---|---|
| 5.1.x | 6.3 | 11.4 | 8.5 | pyds 1.1.8 |
| 6.0 | 7.0 | 12.2 | 10.0 | pyds 1.1.11 |
| 6.1 (current) | 7.1 | 12.6 | 10.3 | pyds 1.1.11 |

Breaking changes to expect when upgrading JetPack 5.x → 6.x:

- **pyds API changes**: `NvDsObjectMeta` field layout changed in DS 7.x; `object_id` is now `uint64_t` (was `int64_t` in some builds). The `UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF` constant was added to the Python API in DS 7.0 — earlier builds require the numeric literal (as in this repo).
- **CUDA 11→12**: `cudaMallocManaged` flags changed; unified memory behaviour differs on Hopper+ vs Turing. Jetson Orin is Ampere — test unified memory probes explicitly.
- **TensorRT 8→10**: Plugin API changed (`IPluginV2DynamicExt` → `IPluginV3`). The `yolo26_decode` plugin will need updating — see [TRT migration guide](https://docs.nvidia.com/deeplearning/tensorrt/migration-guide/).
- **GStreamer version**: JetPack 6 ships GStreamer 1.20; JetPack 5 shipped 1.16. `nvstreammux` batch semantics for `live-source=1` changed in DS 7.0.

---

## TDP and power modes

Jetson power modes directly affect sustained inference throughput. Set via `nvpmodel`:

| Device | Mode | TDP | Typical inference FPS (YOLO26n, 3-stream) |
|---|---|---|---|
| Jetson AGX Orin 64GB | `MAXN` | 60 W | ~200+ fps/stream (INT8 on Tensor Cores) |
| Jetson AGX Orin 64GB | `10W` | 10 W | ~40 fps/stream |
| Jetson Orin NX 16GB | `MAXN` | 25 W | ~120 fps/stream |
| Jetson Orin Nano 8GB | `MAXN` | 15 W | ~50 fps/stream |

For production traffic sensors:
```bash
# Set max-performance mode (after boot):
sudo nvpmodel -m 0          # MAXN
sudo jetson_clocks          # lock clocks to max frequency

# Verify:
sudo jetson_clocks --show
```

`jetson_clocks` prevents thermal-induced clock scaling from causing FPS variability mid-session. Without it, a warm sensor housing can trigger throttling that drops FPS below the 25 fps real-time floor partway through a shift.

VRAM budget: Jetson AGX Orin 64GB has 64 GB shared DRAM (CPU+GPU); Jetson Orin NX 16GB has 16 GB. The 3-stream pipeline at FP16 uses ~1.6 GB peak VRAM (from this project's measured 30-min stability run); INT8 with 3 streams fits comfortably on Orin NX 16GB with headroom for the OS and edge runtime. The 6 GB constraint in this project is specific to the GTX 1660 Ti discrete GPU.
