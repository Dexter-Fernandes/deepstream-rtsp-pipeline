# Engineering Debugging Case Studies

This document records integration failures that occurred while building the
pipeline. Each case is written as symptom → diagnosis → root cause → fix →
verification so that the repository shows the engineering work behind the
final architecture, not only the finished result.

## 1. Custom detections disappeared at `nvtracker`

### Context

YOLO26n runs through `nvinfer` with `network-type=100` and
`output-tensor-meta=1`. DeepStream therefore exposes the raw output tensor but
does not create detector object metadata. A source-pad probe reads the
`[300, 6]` tensor and adds one `NvDsObjectMeta` for every accepted detection.

### Symptom

The tensor probe contained valid boxes, but after the buffer passed through
`nvtracker`, no tracked objects appeared. Tracker CSV files were empty even
though inference had succeeded.

### Diagnosis

The failure boundary was between the custom metadata adapter and
`NvMultiObjectTracker`: detections existed before the tracker and disappeared
inside it. Inspecting the metadata contract exposed three fields that a normal
`nvinfer` detector path fills automatically but a tensor-output probe must fill
itself.

### Root cause

1. `network-type=100` did not mark the frame as having completed inference, so
   `frame_meta.bInferDone` remained false.
2. The probe populated `rect_params`, which OSD can draw, but the tracker
   associates detections using
   `detector_bbox_info.org_bbox_coords`.
3. New objects needed the untracked sentinel value; leaving `object_id` at its
   default made their tracking state ambiguous.

### Fix

The probe now explicitly establishes the detector-to-tracker contract:

```python
frame_meta.bInferDone = 1

obj_meta.object_id = 0xFFFFFFFFFFFFFFFF  # UNTRACKED_OBJECT_ID

bbox = obj_meta.detector_bbox_info.org_bbox_coords
bbox.left = left
bbox.top = top
bbox.width = width
bbox.height = height

rect = obj_meta.rect_params
rect.left = left
rect.top = top
rect.width = width
rect.height = height
```

See the implementation in
[`pipelines/multi_stream.py`](../pipelines/multi_stream.py).

### Verification

- The same detection stream now produces populated tracker CSVs and supports
  the quantitative IOU/NvDCF/NvSORT comparison in
  [`metrics/tracker_comparison.ipynb`](../metrics/tracker_comparison.ipynb).
- Regression tests assert that all three required metadata fields remain in
  the adapter:
  `test_yolo_decode_probe_marks_frame_inferred`,
  `test_yolo_decode_probe_sets_detector_bbox_info`, and
  `test_yolo_decode_probe_sets_untracked_object_id`.
- The GPU smoke test checks that an end-to-end run writes detection rows, not
  only a CSV header.

### Lesson

An object that is drawable is not necessarily trackable. When custom inference
code bypasses a framework's normal parser, the adapter must reproduce the full
downstream metadata contract—not only the fields visible on screen.

---

## 2. Overlays appeared only on the first camera

### Context

The three camera feeds are combined into one batched buffer by `nvstreammux`.
Inference and tracking operate on that batch before it is separated again by
`nvstreamdemux`.

### Symptom

With one `nvdsosd` in the shared batched chain, bounding boxes rendered on
source 0 but not on the other output streams.

### Diagnosis

Inference metadata was present for all sources, so the problem was not the
detector or tracker. The failure occurred in the rendering topology: the OSD
element was being asked to composite a batched surface before the outputs had
been split into source-specific buffers.

### Root cause

In this graph, a single pre-demux `nvdsosd` rendered onto only the first frame
surface in the batch. Rendering was therefore placed at the wrong ownership
boundary.

### Fix

Only the batch-friendly operations remain in the shared chain:

```text
nvstreammux → nvinfer → nvtracker → nvstreamdemux
```

Every demultiplexed source now has its own output branch:

```text
demux.src_i → queue_i → nvvideoconvert_i → nvdsosd_i → sink_i
```

The metadata, anonymisation, and CSV probe also runs on each branch's OSD sink
pad. This makes the buffer's surface index deterministic while preserving
`source_id` for output routing.

See [`build_pipeline()`](../pipelines/multi_stream.py) for the graph
construction.

### Verification

- Visual validation confirmed boxes on all three RTSP outputs.
- Each branch writes a separate `output_stream{i}.csv`.
- The 30-minute three-stream run maintained 29.7 FPS per stream through the
  complete graph.

### Lesson

Batching changes buffer ownership and indexing. Shared compute belongs before
the demux; source-specific rendering and side effects belong after it.

---

## 3. The anonymisation probe segfaulted while reading GPU frames

### Context

Anonymisation uses OpenCV from a Python pad probe. The probe needs a writable
NumPy view of the RGBA `NvBufSurface` so it can blur detected regions before
OSD and re-streaming.

### Symptom

Calling `pyds.get_nvds_buf_surface` against the default converted surface could
segfault on the dGPU path.

### Diagnosis

The bounding boxes and OpenCV operation were valid. The crash depended on the
memory backing the video surface, which isolated the problem to host access
rather than image-processing logic.

### Root cause

The default NVMM allocation was device-only. A Python/NumPy probe cannot safely
dereference that memory as a host-accessible array.

### Fix

The per-source `nvvideoconvert` elements now request CUDA unified memory:

```python
converter.set_property("nvbuf-memory-type", 3)
```

The probe runs after conversion to RGBA and before OSD. It edits the mapped
surface in place, so downstream consumers receive the anonymised frame without
an additional frame copy.

### Verification

- [`tests/unit/test_frame_accessor.py`](../tests/unit/test_frame_accessor.py)
  tests the surface-access boundary through an injected accessor.
- [`tests/unit/test_anonymisation.py`](../tests/unit/test_anonymisation.py)
  verifies clipping, zero-area handling, blur write-back, and preservation of
  pixels outside the selected boxes.
- GPU smoke tests exercise the real DeepStream graph and require frames and
  CSV detections to be produced before the run is accepted.

### Lesson

GPU video pipelines fail at memory boundaries as often as at model boundaries.
Pixel format, allocator type, and probe placement must be treated as explicit
parts of the interface.

---

## 4. A cached TensorRT engine did not match the stream batch

### Context

The original detector configuration was built for one stream. The
multi-stream graph changed `nvstreammux` and `nvinfer` to batch three frames at
a time.

### Symptom

Changing the `nvinfer` batch-size property while reusing a cached batch-1
TensorRT engine produced incompatible behaviour at batch 3.

### Diagnosis

The GStreamer element accepted the new property, but that property change did
not rebuild the already-cached TensorRT engine. The runtime configuration and
engine profile had diverged.

### Root cause

TensorRT batch support is encoded when the engine is built. A filename pointing
to a batch-1 engine cannot be made batch-3-compatible by changing only a
GStreamer property.

### Fix

Two safeguards were added:

1. YOLO26n is exported with a dynamic batch dimension and built with profiles
   covering batch 1 through 3.
2. `_make_nvinfer_config()` creates a temporary config for the requested batch.
   For legacy engine names it rewrites both `batch-size` and the `_bN` engine
   suffix, forcing `nvinfer` to select or build a compatible artifact.

### Verification

The unit suite checks that:

- batch 1 keeps the original configuration;
- batch 3 rewrites `batch-size=3`;
- a legacy `_b1_...engine` path becomes `_b3_...engine`;
- model conversion emits the expected dynamic shape profile and batch-specific
  filename.

The production engine then completed both the live three-stream test and the
batch sweep used for the p99 latency budget.

### Lesson

Runtime settings do not override an engine's compiled shape contract. Engine
identity, optimisation profiles, and serving configuration should be versioned
and validated together.

---

## Evidence index

| Evidence | Location |
|---|---|
| Pipeline topology and integration fixes | [`pipelines/multi_stream.py`](../pipelines/multi_stream.py) |
| Metadata-contract regression tests | [`tests/unit/test_multi_stream.py`](../tests/unit/test_multi_stream.py) |
| Surface-access and anonymisation tests | [`tests/unit/test_frame_accessor.py`](../tests/unit/test_frame_accessor.py), [`tests/unit/test_anonymisation.py`](../tests/unit/test_anonymisation.py) |
| End-to-end GPU checks | [`tests/smoke/test_pipeline_smoke.py`](../tests/smoke/test_pipeline_smoke.py) |
| Tracker metrics | [`metrics/tracker_comparison.ipynb`](../metrics/tracker_comparison.ipynb) |
| Stability and throughput | [`metrics/stability.ipynb`](../metrics/stability.ipynb) |
| Precision and decode comparison | [`metrics/decode_comparison.ipynb`](../metrics/decode_comparison.ipynb) |
