import argparse
import signal
import sys
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    uri: str = "rtsp://localhost:8554/stream0"
    nvinfer_config: str = "configs/nvinfer_primary.txt"
    retry: int = 3
    timeout_us: int = 5_000_000
    mux_width: int = 1920
    mux_height: int = 1080
    batch_size: int = 1
    output_csv: str = "output.csv"
    restream_uri: str | None = None
    anonymise: bool = False
    tracker_config: str = "configs/tracker_iou.yml"


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="DeepStream RTSP pipeline")
    parser.add_argument(
        "--uri",
        default="rtsp://localhost:8554/stream0",
        help="RTSP source URI",
    )
    parser.add_argument(
        "--nvinfer-config",
        default="configs/nvinfer_primary.txt",
        dest="nvinfer_config",
        help="Path to nvinfer config file",
    )
    parser.add_argument("--retry", type=int, default=3, help="rtspsrc retry count")
    parser.add_argument("--output", default="output.csv", dest="output_csv", help="CSV output path")
    parser.add_argument("--restream-uri", default=None, dest="restream_uri", help="RTSP URI to re-stream blurred output")
    parser.add_argument("--anonymise", action="store_true", dest="anonymise", help="Enable blur anonymisation")
    parser.add_argument(
        "--tracker",
        default="configs/tracker_iou.yml",
        dest="tracker_config",
        help="Path to nvtracker YAML config (tracker_iou.yml / tracker_nvdcf.yml / tracker_bytetrack.yml)",
    )
    args = parser.parse_args(argv)
    return PipelineConfig(
        uri=args.uri,
        nvinfer_config=args.nvinfer_config,
        retry=args.retry,
        output_csv=args.output_csv,
        restream_uri=args.restream_uri,
        anonymise=args.anonymise,
        tracker_config=args.tracker_config,
    )


def _restream_sink_props(uri: str) -> dict:
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    return {
        "rtsp-port": parsed.port or 8554,
        "rtsp-mount-point": parsed.path or "/ds-test",
    }


def _source_props(config: PipelineConfig) -> dict:
    return {
        "location": config.uri,
        "protocols": 4,  # GST_RTSP_LOWER_TRANS_TCP — avoids UDP flakiness on localhost
        "retry": config.retry,
        "timeout": config.timeout_us,
    }


def build_pipeline(config: PipelineConfig):
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # noqa: F401

    Gst.init(None)

    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make("rtspsrc", "source")
    depay = Gst.ElementFactory.make("rtph264depay", "depay")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
    queue = Gst.ElementFactory.make("queue", "queue")
    mux = Gst.ElementFactory.make("nvstreammux", "mux")
    nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
    caps_rgba = Gst.ElementFactory.make("capsfilter", "caps_rgba")
    osd = Gst.ElementFactory.make("nvdsosd", "osd")
    if config.restream_uri:
        sink = Gst.ElementFactory.make("nvrtspoutsinkbin", "sink")
    else:
        sink = Gst.ElementFactory.make("fakesink", "sink")

    sink_element_name = "nvrtspoutsinkbin" if config.restream_uri else "fakesink"
    for name, el in [
        ("rtspsrc", source), ("rtph264depay", depay), ("nvv4l2decoder", decoder),
        ("queue", queue), ("nvstreammux", mux), ("nvinfer", nvinfer),
        ("nvtracker", tracker), ("nvvideoconvert", converter),
        ("capsfilter", caps_rgba), ("nvdsosd", osd), (sink_element_name, sink),
    ]:
        if not el:
            raise RuntimeError(f"Could not create GStreamer element: {name}")
        pipeline.add(el)

    for prop, val in _source_props(config).items():
        source.set_property(prop, val)

    mux.set_property("width", config.mux_width)
    mux.set_property("height", config.mux_height)
    mux.set_property("batch-size", config.batch_size)
    mux.set_property("batched-push-timeout", 4_000_000)

    nvinfer.set_property("config-file-path", config.nvinfer_config)

    if config.restream_uri:
        for prop, val in _restream_sink_props(config.restream_uri).items():
            sink.set_property(prop, val)

    tracker.set_property(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    )
    tracker.set_property("ll-config-file", config.tracker_config)

    # rtspsrc has dynamic src pads; link the rest of the chain statically
    depay.link(decoder)
    decoder.link(queue)

    mux_sink = mux.request_pad_simple("sink_0")
    queue_src = queue.get_static_pad("src")
    queue_src.link(mux_sink)

    caps_rgba.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )
    # nvbuf-mem-cuda-unified: NVMM surface accessible from CPU without segfault.
    # This is required for pyds.get_nvds_buf_surface in a Python probe on dGPU.
    converter.set_property("nvbuf-memory-type", 3)

    mux.link(nvinfer)
    nvinfer.link(tracker)
    tracker.link(converter)
    converter.link(caps_rgba)
    caps_rgba.link(osd)
    osd.link(sink)

    # rtspsrc emits pad-added when the stream negotiates; wire depay then
    def _on_pad_added(src, new_pad):
        sink_pad = depay.get_static_pad("sink")
        if sink_pad.is_linked():
            return
        # caps aren't negotiated yet at pad-added time; link and let GStreamer
        # reject incompatible pads (e.g. RTCP) with a non-OK return code
        ret = new_pad.link(sink_pad)
        if ret not in (Gst.PadLinkReturn.OK, Gst.PadLinkReturn.WAS_LINKED):
            print(f"[rtsp] pad link returned {ret} — likely RTCP pad, ignoring", file=sys.stderr)

    source.connect("pad-added", _on_pad_added)

    return pipeline


def run(config: PipelineConfig) -> None:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    import pyds

    import numpy as np
    import cv2
    from pipelines.metadata_parser import parse_frame_meta
    from metrics.csv_sink import CsvSink
    from pipelines.anonymisation import blur_bboxes

    pipeline = build_pipeline(config)
    loop = GLib.MainLoop()

    csv_sink = CsvSink(config.output_csv)

    # Probe on osd sink pad. converter uses nvbuf-mem-cuda-unified so the
    # NVMM surface is CPU-accessible; pyds.get_nvds_buf_surface works safely.
    osd_sink_pad = pipeline.get_by_name("osd").get_static_pad("sink")

    def _probe(pad, info, _user_data):
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK
        detections = parse_frame_meta(batch_meta)

        if config.anonymise and detections:
            surface = pyds.get_nvds_buf_surface(hash(gst_buffer), 0)
            if surface is not None:
                frame_view = np.array(surface, copy=False)
                frame_bgr = cv2.cvtColor(frame_view[:, :, :3], cv2.COLOR_RGB2BGR)
                blurred_bgr = blur_bboxes(frame_bgr, detections)
                frame_view[:, :, :3] = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)

        csv_sink.write(detections)
        return Gst.PadProbeReturn.OK

    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, _probe, 0)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_message(_, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("EOS received — stopping pipeline")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"Pipeline error: {err.message} ({debug})", file=sys.stderr)
            loop.quit()

    bus.connect("message", _on_message)

    def _on_sigint(_sig, _frame):
        print("\nInterrupted — stopping pipeline")
        loop.quit()

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("Failed to set pipeline to PLAYING")

    print(f"Pipeline running — source: {config.uri}  output: {config.output_csv}")
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        csv_sink.close()


if __name__ == "__main__":
    run(parse_args())
