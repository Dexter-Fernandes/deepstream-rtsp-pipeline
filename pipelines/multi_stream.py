import argparse
import ctypes
import logging
import signal
from dataclasses import dataclass, field
from pathlib import Path

from pipelines.structured_log import configure_pipeline_logging, get_pipeline_logger, log_event

configure_pipeline_logging()
_log = get_pipeline_logger("multi_stream")

_PLUGIN_LIB = Path("/opt/ds_plugins/libyolo26_decode.so")
if _PLUGIN_LIB.exists():
    ctypes.CDLL(str(_PLUGIN_LIB), ctypes.RTLD_GLOBAL)
    log_event(_log, logging.INFO, event="plugin_loaded", plugin=str(_PLUGIN_LIB))
else:
    log_event(_log, logging.WARNING, event="plugin_missing", plugin=str(_PLUGIN_LIB),
              detail="decode engine will fail if plugin is required")


@dataclass
class MultiStreamConfig:
    uris: list[str] = field(default_factory=list)
    nvinfer_config: str = "configs/nvinfer_primary.txt"
    retry: int = 3
    timeout_us: int = 5_000_000
    mux_width: int = 1920
    mux_height: int = 1080
    output_dir: str = "."
    restream_base_port: int | None = None
    anonymise: bool = False
    conf_threshold: float = 0.25
    tracker_config: str = "configs/tracker_nvdcf.yml"
    perf_json: str | None = None
    perf_interval: float = 5.0
    duration: int | None = None
    no_sync: bool = False


def parse_args(argv: list[str] | None = None) -> MultiStreamConfig:
    parser = argparse.ArgumentParser(description="DeepStream multi-stream RTSP pipeline")
    parser.add_argument("--uri", action="append", dest="uris", default=[], metavar="URI", help="RTSP source URI (repeat for multiple)")
    parser.add_argument("--nvinfer-config", default="configs/nvinfer_primary.txt", dest="nvinfer_config")
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--output-dir", default=".", dest="output_dir", help="Directory for per-source CSV files")
    parser.add_argument("--restream-base-port", type=int, default=None, dest="restream_base_port", help="Base port for nvrtspoutsinkbin (stream0=base, stream1=base+1, ...)")
    parser.add_argument("--anonymise", action="store_true", dest="anonymise", help="Enable blur anonymisation")
    parser.add_argument("--conf-threshold", type=float, default=0.25, dest="conf_threshold", help="Detection confidence threshold (default: 0.25)")
    parser.add_argument(
        "--tracker",
        default="configs/tracker_nvdcf.yml",
        dest="tracker_config",
        help="Path to nvtracker YAML config (tracker_iou.yml / tracker_nvdcf.yml / tracker_bytetrack.yml)",
    )
    parser.add_argument("--perf-json", default=None, dest="perf_json", metavar="PATH", help="Write perf JSON to PATH (enables monitoring)")
    parser.add_argument("--perf-interval", type=float, default=5.0, dest="perf_interval", metavar="SECONDS", help="Perf sampling interval in seconds (default: 5.0)")
    parser.add_argument("--duration", type=int, default=None, dest="duration", metavar="SECONDS", help="Auto-stop after SECONDS (for unattended runs)")
    parser.add_argument("--no-sync", action="store_true", dest="no_sync", help="Disable sink sync (unthrottled throughput ceiling measurement)")
    args = parser.parse_args(argv)
    return MultiStreamConfig(
        uris=args.uris,
        nvinfer_config=args.nvinfer_config,
        retry=args.retry,
        output_dir=args.output_dir,
        restream_base_port=args.restream_base_port,
        anonymise=args.anonymise,
        conf_threshold=args.conf_threshold,
        tracker_config=args.tracker_config,
        perf_json=args.perf_json,
        perf_interval=args.perf_interval,
        duration=args.duration,
        no_sync=args.no_sync,
    )


def _output_csv_path(output_dir: str, source_id: int) -> str:
    return str(Path(output_dir) / f"output_stream{source_id}.csv")


def _restream_port(base_port: int, source_id: int) -> int:
    return base_port + source_id


def _make_nvinfer_config(base_config: str, n: int) -> str:
    """Return a nvinfer config path with batch-size set to n.

    Writes a temp file so the TensorRT engine is built (or cached) for the
    correct batch-size without modifying the original single-stream config.
    """
    if n == 1:
        return base_config

    import re

    with open(base_config) as f:
        content = f.read()

    content = re.sub(r'batch-size\s*=\s*\d+', f'batch-size={n}', content)
    # Rename the engine file path so nvinfer builds a fresh engine for this
    # batch-size rather than trying to reuse the cached batch-1 engine.
    content = re.sub(
        r'(model-engine-file\s*=\s*.+?)_b\d+(_gpu\d+_\w+\.engine)',
        rf'\g<1>_b{n}\g<2>',
        content,
    )

    out_path = f'/tmp/nvinfer_b{n}.txt'
    with open(out_path, 'w') as f:
        f.write(content)
    return out_path


def _is_file_uri(uri: str) -> bool:
    """A source is treated as a local file unless it's an rtsp:// URI."""
    return not uri.startswith("rtsp://")


def _make_file_source_bin(pipeline, config: MultiStreamConfig, idx: int):
    """Build filesrc→qtdemux→h264parse→decoder→queue for one MP4 file.

    Used for GT-aligned evaluation: a file source plays from frame 0 and emits
    EOS after exactly one pass, so prediction frame N maps to GT frame N+1.
    """
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    path = config.uris[idx]
    if path.startswith("file://"):
        path = path[len("file://"):]

    source = Gst.ElementFactory.make("filesrc", f"source_{idx}")
    demux = Gst.ElementFactory.make("qtdemux", f"qtdemux_{idx}")
    parser = Gst.ElementFactory.make("h264parse", f"h264parse_{idx}")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder_{idx}")
    queue = Gst.ElementFactory.make("queue", f"queue_{idx}")

    for name, el in [
        (f"filesrc_{idx}", source), (f"qtdemux_{idx}", demux),
        (f"h264parse_{idx}", parser), (f"nvv4l2decoder_{idx}", decoder),
        (f"queue_{idx}", queue),
    ]:
        if not el:
            raise RuntimeError(f"Could not create GStreamer element: {name}")
        pipeline.add(el)

    source.set_property("location", path)

    source.link(demux)
    parser.link(decoder)
    decoder.link(queue)

    def _on_pad_added(_demux, new_pad, _parser=parser, _idx=idx):
        sink_pad = _parser.get_static_pad("sink")
        if sink_pad.is_linked():
            return
        caps = new_pad.get_current_caps() or new_pad.query_caps(None)
        name = caps.to_string() if caps else ""
        if not name.startswith("video"):
            return  # skip audio/other tracks
        new_pad.link(sink_pad)

    demux.connect("pad-added", _on_pad_added)
    return queue


def _make_source_bin(pipeline, config: MultiStreamConfig, idx: int):
    """Build a source bin and return its queue element.

    Dispatches to a file branch for local MP4s (GT-aligned eval) or an rtsp
    branch for live streams.
    """
    if _is_file_uri(config.uris[idx]):
        return _make_file_source_bin(pipeline, config, idx)

    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    source = Gst.ElementFactory.make("rtspsrc", f"source_{idx}")
    depay = Gst.ElementFactory.make("rtph264depay", f"depay_{idx}")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder_{idx}")
    queue = Gst.ElementFactory.make("queue", f"queue_{idx}")

    for name, el in [
        (f"rtspsrc_{idx}", source), (f"rtph264depay_{idx}", depay),
        (f"nvv4l2decoder_{idx}", decoder), (f"queue_{idx}", queue),
    ]:
        if not el:
            raise RuntimeError(f"Could not create GStreamer element: {name}")
        pipeline.add(el)

    source.set_property("location", config.uris[idx])
    source.set_property("protocols", 4)
    source.set_property("retry", config.retry)
    source.set_property("timeout", config.timeout_us)

    depay.link(decoder)
    decoder.link(queue)

    def _on_pad_added(src, new_pad, _depay=depay, _idx=idx):
        sink_pad = _depay.get_static_pad("sink")
        if sink_pad.is_linked():
            return
        ret = new_pad.link(sink_pad)
        if ret not in (Gst.PadLinkReturn.OK, Gst.PadLinkReturn.WAS_LINKED):
            log_event(_log, logging.WARNING, source_id=_idx, event="stream_reconnect",
                      detail=f"pad link returned {ret} — likely RTCP, ignoring")

    source.connect("pad-added", _on_pad_added)
    return queue


def build_pipeline(config: MultiStreamConfig):
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # noqa: F401

    Gst.init(None)

    if not config.uris:
        raise ValueError("MultiStreamConfig.uris must contain at least one URI")

    pipeline = Gst.Pipeline()
    n = len(config.uris)

    # ── Shared inference chain (operates on the full batch) ──────────────────
    # nvstreammux collects one frame from each source into a batch of N before
    # pushing downstream; nvinfer and nvtracker then operate on all N at once.
    # Conversion + OSD are NOT here — a single nvdsosd on a batched buffer only
    # draws on the first frame (source 0). They live per-branch after the demux.
    mux = Gst.ElementFactory.make("nvstreammux", "mux")
    nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")

    # ── Demux: split batched stream back to per-source buffers ───────────────
    demux = Gst.ElementFactory.make("nvstreamdemux", "demux")

    for name, el in [
        ("nvstreammux", mux), ("nvinfer", nvinfer), ("nvtracker", tracker),
        ("nvstreamdemux", demux),
    ]:
        if not el:
            raise RuntimeError(f"Could not create GStreamer element: {name}")
        pipeline.add(el)

    mux.set_property("width", config.mux_width)
    mux.set_property("height", config.mux_height)
    mux.set_property("batch-size", n)
    mux.set_property("batched-push-timeout", 4_000_000)

    nvinfer.set_property("config-file-path", _make_nvinfer_config(config.nvinfer_config, n))

    tracker.set_property(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    )
    tracker.set_property("ll-config-file", config.tracker_config)

    # ── Source bins → mux ────────────────────────────────────────────────────
    # Each bin is: rtspsrc → rtph264depay → nvv4l2decoder → queue
    # The queue's src pad is linked to mux sink_{i} so all sources feed one mux.
    for i in range(n):
        queue = _make_source_bin(pipeline, config, i)
        mux_sink = mux.request_pad_simple(f"sink_{i}")
        queue_src = queue.get_static_pad("src")
        queue_src.link(mux_sink)

    # ── Shared chain: mux → inference → tracker → demux ──────────────────────
    # Everything up to the demux runs on the batched buffer. Conversion + OSD
    # are deliberately downstream of the demux (see below).
    mux.link(nvinfer)
    nvinfer.link(tracker)
    tracker.link(demux)

    # ── Per-source output branches ────────────────────────────────────────────
    # Each branch replicates the proven single-stream tail:
    #   demux.src_{i} → queue_out_{i} → nvvideoconvert_{i} → caps_rgba_{i}
    #                 → nvdsosd_{i} → sink_{i}
    # A per-branch nvdsosd is required: a single nvdsosd on the batched buffer
    # only draws on source 0. nvvideoconvert uses CUDA unified memory so the
    # optional anonymise probe can read/write the RGBA surface.
    # Linking: pad-level for demux→queue (queue always has a static "sink" pad);
    # element-level for the rest so GStreamer auto-negotiates the
    # nvrtspoutsinkbin ghost pad name.
    for i in range(n):
        queue_out = Gst.ElementFactory.make("queue", f"queue_out_{i}")
        converter = Gst.ElementFactory.make("nvvideoconvert", f"converter_{i}")
        caps_rgba = Gst.ElementFactory.make("capsfilter", f"caps_rgba_{i}")
        osd = Gst.ElementFactory.make("nvdsosd", f"osd_{i}")

        for name, el in [
            (f"queue_out_{i}", queue_out), (f"converter_{i}", converter),
            (f"caps_rgba_{i}", caps_rgba), (f"osd_{i}", osd),
        ]:
            if not el:
                raise RuntimeError(f"Could not create GStreamer element: {name}")
            pipeline.add(el)

        caps_rgba.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        )
        # nvbuf-mem-cuda-unified: required for pyds.get_nvds_buf_surface on dGPU
        converter.set_property("nvbuf-memory-type", 3)

        if config.restream_base_port is not None:
            sink = Gst.ElementFactory.make("nvrtspoutsinkbin", f"sink_{i}")
            if not sink:
                raise RuntimeError(f"Could not create nvrtspoutsinkbin for source {i}")
            pipeline.add(sink)
            sink.set_property("rtsp-port", _restream_port(config.restream_base_port, i))
            sink.set_property("rtsp-mount-point", f"/stream{i}_out")
        else:
            sink = Gst.ElementFactory.make("fakesink", f"sink_{i}")
            if not sink:
                raise RuntimeError(f"Could not create fakesink for source {i}")
            pipeline.add(sink)
            if config.no_sync:
                sink.set_property("sync", False)

        demux_src = demux.request_pad_simple(f"src_{i}")
        demux_src.link(queue_out.get_static_pad("sink"))
        queue_out.link(converter)
        converter.link(caps_rgba)
        caps_rgba.link(osd)
        osd.link(sink)

    return pipeline


def _parse_frame_detections(frame_meta):
    """Extract detections from a single NvDsFrameMeta (one source's frame)."""
    import pyds
    from pipelines.metadata_parser import Detection

    detections = []
    obj_list = frame_meta.obj_meta_list
    while obj_list is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(obj_list.data)
        except StopIteration:
            break
        rect = obj_meta.rect_params
        detections.append(Detection(
            frame_num=frame_meta.frame_num,
            object_id=obj_meta.object_id,
            class_id=obj_meta.class_id,
            class_label=obj_meta.obj_label,
            confidence=obj_meta.confidence,
            left=rect.left,
            top=rect.top,
            width=rect.width,
            height=rect.height,
        ))
        try:
            obj_list = obj_list.next
        except StopIteration:
            break
    return detections


def run(config: MultiStreamConfig) -> None:
    import ctypes
    import time
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    import pyds

    import numpy as np
    import cv2
    from metrics.csv_sink import CsvSink
    from metrics.health_monitor import HealthMonitor
    from metrics.perf_monitor import PerfMonitor, sample_rss_mb, sample_vram_mb
    from pipelines.anonymisation import blur_bboxes

    pipeline = build_pipeline(config)
    loop = GLib.MainLoop()

    csv_sinks = {
        i: CsvSink(_output_csv_path(config.output_dir, i))
        for i in range(len(config.uris))
    }

    n = len(config.uris)
    frame_counts = {i: 0 for i in range(n)}
    health_monitor = HealthMonitor(num_sources=n, expected_fps=25.0)

    # Decode probe on the nvinfer SRC pad — fires once per batched buffer before
    # the tracker. Reads NvDsInferTensorMeta (exposed because output-tensor-meta=1
    # in nvinfer_primary.txt) and creates NvDsObjectMeta for each YOLO26n detection.
    # network-type=100 means nvinfer itself won't create any object metas.
    #
    # The engine now includes the yolo26_decode TRT plugin (M2.3+M2.4): output is
    # already in xywh pixel-space so no coordinate transform is needed here.
    _NET_W, _NET_H = 640, 640
    _scale_x = config.mux_width / _NET_W
    _scale_y = config.mux_height / _NET_H
    _N_DETS, _N_ATTRS = 300, 6

    def _yolo_decode_probe(pad, info, _user_data):
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        frame_meta_list = batch_meta.frame_meta_list
        while frame_meta_list is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
            except StopIteration:
                break

            # nvinfer runs in output-tensor-meta mode and never marks the frame
            # as inferred, so nvtracker would skip it and drop every object.
            # We decode the tensor and inject objects here, so flag it ourselves.
            frame_meta.bInferDone = 1

            user_meta_list = frame_meta.frame_user_meta_list
            while user_meta_list is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    tensor = np.ctypeslib.as_array(ptr, shape=(_N_DETS * _N_ATTRS,)).reshape(
                        _N_DETS, _N_ATTRS
                    ).copy()

                    # Plugin output is [left, top, w, h, conf, cls] — no transform needed.
                    for i in range(_N_DETS):
                        conf = float(tensor[i, 4])
                        if conf < config.conf_threshold:
                            continue
                        obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                        obj_meta.object_id = 0xFFFFFFFFFFFFFFFF  # UNTRACKED_OBJECT_ID — nvtracker assigns a fresh ID
                        obj_meta.unique_component_id = tensor_meta.unique_id
                        obj_meta.confidence = conf
                        obj_meta.class_id = int(tensor[i, 5])
                        left = float(tensor[i, 0]) * _scale_x
                        top = float(tensor[i, 1]) * _scale_y
                        width = float(tensor[i, 2]) * _scale_x
                        height = float(tensor[i, 3]) * _scale_y
                        # nvtracker associates on detector_bbox_info.org_bbox_coords,
                        # not rect_params — must populate both or it drops the object.
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
                        rect.border_width = 3
                        rect.border_color.red = 0.0
                        rect.border_color.green = 1.0
                        rect.border_color.blue = 0.0
                        rect.border_color.alpha = 1.0
                        pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

                try:
                    user_meta_list = user_meta_list.next
                except StopIteration:
                    break

            try:
                frame_meta_list = frame_meta_list.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    nvinfer_src = pipeline.get_by_name("nvinfer").get_static_pad("src")
    nvinfer_src.add_probe(Gst.PadProbeType.BUFFER, _yolo_decode_probe, 0)

    # One probe per per-branch nvdsosd sink pad — fires before that branch's OSD
    # draws. After the demux each buffer carries a single source's frame, so the
    # surface batch-index is always 0; source_id from the frame meta still tells
    # us which stream it is, so CSV routing is unchanged.
    def _probe(pad, info, _user_data):
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        # Single-source buffer post-demux, but walk the list defensively
        frame_meta_list = batch_meta.frame_meta_list
        while frame_meta_list is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
            except StopIteration:
                break

            source_id = frame_meta.source_id
            frame_counts[source_id] = frame_counts.get(source_id, 0) + 1
            detections = _parse_frame_detections(frame_meta)
            health_monitor.record_frame(source_id, t=time.monotonic(), has_detection=bool(detections))

            # Optional: blur bboxes in-place on the NVMM surface before OSD draws
            if config.anonymise and detections:
                surface = pyds.get_nvds_buf_surface(hash(gst_buffer), 0)
                if surface is not None:
                    frame_view = np.array(surface, copy=False)
                    frame_bgr = cv2.cvtColor(frame_view[:, :, :3], cv2.COLOR_RGB2BGR)
                    blurred_bgr = blur_bboxes(frame_bgr, detections)
                    frame_view[:, :, :3] = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)

            # Write this frame's detections to its source's CSV
            if source_id in csv_sinks:
                csv_sinks[source_id].write(detections)

            try:
                frame_meta_list = frame_meta_list.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    for i in range(len(config.uris)):
        osd_sink_pad = pipeline.get_by_name(f"osd_{i}").get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, _probe, 0)

    # Optional perf monitoring — enabled only when --perf-json is set
    perf_monitor = None
    if config.perf_json:
        _perf_start = time.time()
        perf_monitor = PerfMonitor(num_sources=n, start_t=0.0)
        _prev_counts = {i: 0 for i in range(n)}

        def _perf_tick():
            now = time.time()
            dt = now - _perf_start
            vram = sample_vram_mb()
            rss = sample_rss_mb()
            counts = dict(frame_counts)
            perf_monitor.record(t=now - _perf_start, frame_counts=counts, vram_mb=vram, rss_mb=rss)
            fps_per = sum(counts.values()) / n / dt if dt > 0 and n > 0 else 0.0
            log_event(_log, logging.INFO, event="perf_tick", t_s=round(dt),
                      fps_per_stream=round(fps_per, 1), vram_mb=round(vram), rss_mb=round(rss))
            return True  # keep firing

        GLib.timeout_add_seconds(int(config.perf_interval), _perf_tick)

    # Health monitoring — always-on, fires every health_interval_s seconds
    _health_interval_s = max(5, int(config.perf_interval))

    def _health_tick():
        snap = health_monitor.snapshot(
            t_now=time.monotonic(),
            vram_mb=sample_vram_mb(),
            rss_mb=sample_rss_mb(),
        )
        log_event(_log, logging.INFO, event="health_tick",
                  sources=snap["sources"], system=snap.get("system"))
        for src in snap["sources"]:
            if not src["is_live"]:
                log_event(_log, logging.WARNING, source_id=src["source_id"],
                          event="source_stalled",
                          time_since_last_frame_s=src["time_since_last_frame_s"])
        return True  # keep firing

    GLib.timeout_add_seconds(_health_interval_s, _health_tick)

    if config.duration:
        def _on_duration_timeout():
            log_event(_log, logging.INFO, event="pipeline_stop",
                      reason="duration_elapsed", duration_s=config.duration)
            loop.quit()
            return False

        GLib.timeout_add_seconds(config.duration, _on_duration_timeout)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_message(_, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            log_event(_log, logging.INFO, event="pipeline_eos")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            log_event(_log, logging.ERROR, event="pipeline_error", msg=err.message, debug=debug)
            loop.quit()

    bus.connect("message", _on_message)

    def _on_sigint(_sig, _frame):
        log_event(_log, logging.INFO, event="pipeline_stop", reason="sigint")
        loop.quit()

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("Failed to set pipeline to PLAYING")

    log_event(_log, logging.INFO, event="pipeline_start",
              uris=config.uris, output_dir=config.output_dir)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        for sink in csv_sinks.values():
            sink.close()
        if perf_monitor is not None and config.perf_json:
            # Flush a final sample so short runs (EOS before first tick) have data
            now = time.time()
            perf_monitor.record(
                t=now - _perf_start,
                frame_counts=dict(frame_counts),
                vram_mb=sample_vram_mb(),
                rss_mb=sample_rss_mb(),
            )
            perf_monitor.write_json(config.perf_json)
            perf_monitor.print_summary()


if __name__ == "__main__":
    run(parse_args())
