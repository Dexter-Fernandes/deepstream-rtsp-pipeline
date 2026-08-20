#!/usr/bin/env bash
# Build the README hero demo from the three captures produced by
# tools/capture_demo_streams.sh: trim, label, tile side by side, and export a
# GitHub-sized looping GIF plus a higher-quality MP4.
#
# Pure ffmpeg — no GPU and no container, so this half of the workflow is unit
# tested (tests/unit/test_make_readme_demo.py) with a fake ffmpeg on PATH.
#
# Usage:
#   bash tools/make_readme_demo.sh
#   DURATION=10 FPS=12 TILE_WIDTH=400 bash tools/make_readme_demo.sh
#   START1=2 START2=2.4 START3=2.1 bash tools/make_readme_demo.sh   # nudge tiles

set -euo pipefail

REPO_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)

IN_DIR=${IN_DIR:-/tmp/readme-demo}
OUT_DIR=${OUT_DIR:-$REPO_ROOT/docs/assets}
DURATION=${DURATION:-12}
FPS=${FPS:-10}                 # GIF frame rate
MP4_FPS=${MP4_FPS:-25}
# 3 x 294 = 882 px, so GitHub renders the GIF 1:1 in the README column instead
# of resampling it — the overlay text survives only at native size.
TILE_WIDTH=${TILE_WIDTH:-294}
MAX_COLORS=${MAX_COLORS:-64}   # 128 is visually identical at this size, ~40% bigger
MAX_MB=${MAX_MB:-10}
FONT=${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}
# Skip the first seconds of the capture: MTMC needs a few frames of co-occurrence
# before global IDs settle, and an unfused opening frame misrepresents the run.
START1=${START1:-10}
START2=${START2:-10}
START3=${START3:-10}

GIF="$OUT_DIR/pipeline-demo.gif"
MP4="$OUT_DIR/pipeline-demo.mp4"
PALETTE=$(mktemp --suffix=.png)
GIF_TMP=""
MP4_TMP=""
trap 'rm -f "$PALETTE" "$GIF_TMP" "$MP4_TMP"' EXIT

for index in 1 2 3; do
    capture="$IN_DIR/cam${index}.mp4"
    if [ ! -f "$capture" ]; then
        echo "[demo] ERROR: missing capture $capture" >&2
        echo "[demo] Record it first: bash tools/capture_demo_streams.sh" >&2
        exit 1
    fi
done

if [ ! -f "$FONT" ]; then
    echo "[demo] ERROR: font not found: $FONT" >&2
    echo "[demo] Install DejaVu fonts or set FONT=/path/to/a.ttf" >&2
    exit 1
fi

if [ $((TILE_WIDTH % 2)) -ne 0 ]; then
    echo "[demo] ERROR: TILE_WIDTH must be even (yuv420p needs even dimensions)" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# Render to temporaries alongside the real outputs and publish both only once
# the GIF is inside budget, so a rejected run neither leaves an oversized file
# to stage by accident nor replaces half of an already-published pair. Same
# directory as the outputs, so publishing is a rename, not a copy.
GIF_TMP=$(mktemp -p "$OUT_DIR" --suffix=.gif .pipeline-demo.XXXXXX)
MP4_TMP=$(mktemp -p "$OUT_DIR" --suffix=.mp4 .pipeline-demo.XXXXXX)

FONT_SIZE=$((TILE_WIDTH / 22))

# Per-tile: resample, downscale, then burn the camera label. The label is drawn
# after scaling so its size is independent of the source resolution.
tile_graph() {
    local fps=$1
    local graph=""
    for index in 0 1 2; do
        graph+="[${index}:v]fps=${fps},scale=${TILE_WIDTH}:-2,"
        graph+="drawtext=fontfile=${FONT}:text='CAM 0$((index + 1))':"
        graph+="x=8:y=8:fontsize=${FONT_SIZE}:fontcolor=white:"
        graph+="box=1:boxcolor=black@0.5:boxborderw=4[t${index}];"
    done
    graph+="[t0][t1][t2]hstack=inputs=3"
    printf '%s' "$graph"
}

inputs=(
    -ss "$START1" -t "$DURATION" -i "$IN_DIR/cam1.mp4"
    -ss "$START2" -t "$DURATION" -i "$IN_DIR/cam2.mp4"
    -ss "$START3" -t "$DURATION" -i "$IN_DIR/cam3.mp4"
)

echo "[demo] tiling 3x${TILE_WIDTH}px, ${DURATION}s @ ${FPS} fps ..."

# Pass 1: one palette for the whole tiled clip. stats_mode=diff biases the
# palette toward moving regions — the people and boxes, not the static ground.
ffmpeg -hide_banner -loglevel error -y \
    "${inputs[@]}" \
    -filter_complex "$(tile_graph "$FPS")[tiled];[tiled]palettegen=max_colors=${MAX_COLORS}:stats_mode=diff[pal]" \
    -map "[pal]" "$PALETTE"

# Pass 2: bayer dithering compresses far better than the default error diffusion,
# which matters without gifsicle available to post-optimise.
ffmpeg -hide_banner -loglevel error -y \
    "${inputs[@]}" -i "$PALETTE" \
    -filter_complex "$(tile_graph "$FPS")[tiled];[tiled][3:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[out]" \
    -map "[out]" -loop 0 "$GIF_TMP"

echo "[demo] encoding MP4 at ${MP4_FPS} fps ..."
ffmpeg -hide_banner -loglevel error -y \
    "${inputs[@]}" \
    -filter_complex "$(tile_graph "$MP4_FPS")[out]" \
    -map "[out]" \
    -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -movflags +faststart \
    "$MP4_TMP"

gif_bytes=$(stat -c %s "$GIF_TMP")
mp4_bytes=$(stat -c %s "$MP4_TMP")
budget=$((MAX_MB * 1024 * 1024))

if [ "$gif_bytes" -gt "$budget" ]; then
    echo "[demo] ERROR: GIF is $((gif_bytes / 1024 / 1024)) MB, over the ${MAX_MB} MB budget" >&2
    echo "[demo] Nothing published; $GIF is unchanged." >&2
    echo "[demo] Shorten DURATION, lower FPS/TILE_WIDTH, or reduce MAX_COLORS." >&2
    exit 1
fi

mv "$GIF_TMP" "$GIF"
mv "$MP4_TMP" "$MP4"
GIF_TMP=""
MP4_TMP=""
# mktemp creates 0600 and the mode survives the rename; published assets have
# to be readable by whatever serves or copies them.
chmod 644 "$GIF" "$MP4"

printf '[demo] %s: %s MB\n' "$GIF" "$((gif_bytes / 1024 / 1024))"
printf '[demo] %s: %s MB\n' "$MP4" "$((mp4_bytes / 1024 / 1024))"

echo "[demo] Done."
