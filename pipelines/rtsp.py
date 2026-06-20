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
    args = parser.parse_args(argv)
    return PipelineConfig(uri=args.uri, nvinfer_config=args.nvinfer_config, retry=args.retry)


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
    osd = Gst.ElementFactory.make("nvdsosd", "osd")
    sink = Gst.ElementFactory.make("fakesink", "sink")

    for name, el in [
        ("rtspsrc", source), ("rtph264depay", depay), ("nvv4l2decoder", decoder),
        ("queue", queue), ("nvstreammux", mux), ("nvinfer", nvinfer),
        ("nvtracker", tracker), ("nvdsosd", osd), ("fakesink", sink),
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

    tracker.set_property(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    )

    # rtspsrc has dynamic src pads; link the rest of the chain statically
    depay.link(decoder)
    decoder.link(queue)

    mux_sink = mux.request_pad_simple("sink_0")
    queue_src = queue.get_static_pad("src")
    queue_src.link(mux_sink)

    mux.link(nvinfer)
    nvinfer.link(tracker)
    tracker.link(osd)
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

    pipeline = build_pipeline(config)
    loop = GLib.MainLoop()

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

    print(f"Pipeline running — source: {config.uri}")
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    run(parse_args())
