#!/usr/bin/env bash
# Record the three anonymised, annotated output streams for the README hero demo.
#
# Runs the pipeline (detector + tracker + cross-camera fusion + blur + OSD) with
# --record-dir, so each source branch is encoded straight to MP4 inside the
# graph. The three files are frame-aligned by construction: they carry the
# source PTS of the same mux batches, with no RTSP client attach skew and
# nothing dropped by a stalled reader.
#
# Blur + MTMC at 3x1080p60 is far SLOWER than real time on a GTX 1660 Ti: about
# 6 fps per stream, against a 60 fps source. SOURCE_MODE=file (default) is what
# that leaves usable — the file-source branch with sink sync off processes and
# records every frame however long it takes, so ~300 s of wall clock yields
# ~29 s of footage. SOURCE_MODE=rtsp cannot absorb the same backlog: rtspsrc
# backpressure stalls the sources into EOS mid-run, so it is kept only for a
# qualitative look at the live path, not for producing the asset.
#
# Captures land outside the repository by default: the WildTrack sources are not
# redistributable, and only the short blurred excerpt built by
# tools/make_readme_demo.sh is ever committed.
#
# Usage:
#   bash tools/capture_demo_streams.sh
#   RUN_SECONDS=300 OUT_DIR=/tmp/demo bash tools/capture_demo_streams.sh
#   SOURCE_MODE=rtsp bash tools/capture_demo_streams.sh   # needs mediamtx

set -euo pipefail

REPO_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO_ROOT"

RUN_SECONDS=${RUN_SECONDS:-240}   # wall-clock budget for the pipeline run
OUT_DIR=${OUT_DIR:-/tmp/readme-demo}
SOURCE_MODE=${SOURCE_MODE:-file}  # file | rtsp
RTSP_BASE=${RTSP_BASE:-rtsp://localhost:8554}
CONF_THRESHOLD=${CONF_THRESHOLD:-0.18}
# The hero tiles are ~1/6 of source width, so the overlay is drawn oversized at
# 1080p to stay legible once three cameras are scaled to README width.
OSD_FONT_SIZE=${OSD_FONT_SIZE:-48}
MIN_AFFINITY=${MIN_AFFINITY:-0.3}
COMPOSE=${COMPOSE:-docker compose}

port_open() {
    (exec 3<>"/dev/tcp/localhost/$1") 2>/dev/null && exec 3<&- && exec 3>&-
}

case "$SOURCE_MODE" in
    file)
        SOURCES=(
            "file:///workspace/data/c1_5min.mp4"
            "file:///workspace/data/c2_5min.mp4"
            "file:///workspace/data/c3_5min.mp4"
        )
        SYNC_FLAGS=(--no-sync)
        ;;
    rtsp)
        if ! port_open 8554; then
            echo "[capture] ERROR: nothing listening on 8554." >&2
            echo "[capture] Start the source server first: mediamtx configs/mediamtx.yml &" >&2
            exit 1
        fi
        SOURCES=("$RTSP_BASE/stream0" "$RTSP_BASE/stream1" "$RTSP_BASE/stream2")
        SYNC_FLAGS=()
        ;;
    *)
        echo "[capture] ERROR: SOURCE_MODE must be 'file' or 'rtsp', got '$SOURCE_MODE'" >&2
        exit 1
        ;;
esac

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/cam1.mp4 "$OUT_DIR"/cam2.mp4 "$OUT_DIR"/cam3.mp4

echo "[capture] mode=$SOURCE_MODE budget=${RUN_SECONDS}s -> $OUT_DIR"

# shellcheck disable=SC2086  # COMPOSE is an intentional multi-word command
$COMPOSE run --rm -T -v "$OUT_DIR:$OUT_DIR" pipeline \
    python3 pipelines/multi_stream.py \
    --uri "${SOURCES[0]}" \
    --uri "${SOURCES[1]}" \
    --uri "${SOURCES[2]}" \
    --record-dir "$OUT_DIR" \
    --conf-threshold "$CONF_THRESHOLD" \
    --anonymise \
    --osd-labels \
    --osd-font-size "$OSD_FONT_SIZE" \
    --mtmc \
    --homography 0=configs/homography_C1.json \
    --homography 1=configs/homography_C2.json \
    --homography 2=configs/homography_C3.json \
    --mtmc-min-affinity "$MIN_AFFINITY" \
    --mtmc-json "$OUT_DIR/mtmc.json" \
    --output-dir "$OUT_DIR" \
    --duration "$RUN_SECONDS" \
    "${SYNC_FLAGS[@]+"${SYNC_FLAGS[@]}"}"

echo "[capture] recorded:"
for index in 1 2 3; do
    target="$OUT_DIR/cam${index}.mp4"
    if [ ! -s "$target" ]; then
        echo "[capture] ERROR: $target is missing or empty" >&2
        exit 1
    fi
    seconds=$(ffprobe -v error -select_streams v:0 \
        -show_entries format=duration -of csv=p=0 "$target")
    printf '[capture]   %s  %s s\n' "$target" "$seconds"
done

echo "[capture] next: bash tools/make_readme_demo.sh (use START1/2/3 to pick a section)"
