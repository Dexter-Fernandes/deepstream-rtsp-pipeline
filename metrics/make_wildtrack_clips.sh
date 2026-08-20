#!/usr/bin/env bash
# Encode the labelled WildTrack PNG sequences into GT-aligned MP4 clips.
#
# The 401 labelled frames per camera are stems C<n>_00000000 .. C<n>_00002000 in
# steps of 5 (annotation index unit is 1/10 s, so the labelled cadence is 2 fps).
# Encoding them at -framerate 2 makes clip frame N correspond to annotation stem
# index N*5 *by construction* — no offset search, no alignment risk.
#
# This is deliberately NOT the same as data/c<n>_5min.mp4: those are the first
# five minutes of the raw 59.94 fps captures, and the mapping from labelled PNG
# to raw video frame is unknown. Those clips stay the live/RTSP demo path, where
# no GT-aligned metrics are claimed.
#
# All three clips get identical frame counts and identical timestamps, which is
# what makes cross-camera fusion (metrics/evaluate_mtmc.py) exact.
#
# Usage:
#   bash metrics/make_wildtrack_clips.sh
#   WILDTRACK_ROOT=/path/to/labelled_ds bash metrics/make_wildtrack_clips.sh

set -euo pipefail

WILDTRACK_ROOT=${WILDTRACK_ROOT:-/media/dexter/PortableSSD/Datasets/WildTrack/labelled_ds}
OUT_DIR=${OUT_DIR:-data}
CAMERAS=${CAMERAS:-"C1 C2 C3"}
CRF=${CRF:-12}

if [ ! -d "$WILDTRACK_ROOT/images" ]; then
    echo "[make_clips] ERROR: no images/ under $WILDTRACK_ROOT" >&2
    echo "[make_clips] Set WILDTRACK_ROOT to the labelled_ds directory." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

for camera in $CAMERAS; do
    src_dir="$WILDTRACK_ROOT/images/$camera"
    lower=$(echo "$camera" | tr '[:upper:]' '[:lower:]')
    out="$OUT_DIR/wildtrack_${lower}_gt.mp4"

    if [ ! -d "$src_dir" ]; then
        echo "[make_clips] ERROR: missing $src_dir" >&2
        exit 1
    fi

    n_png=$(find "$src_dir" -name "${camera}_*.png" | wc -l)
    echo "[make_clips] $camera: $n_png PNGs -> $out"

    if [ "$n_png" -ne 401 ]; then
        echo "[make_clips] ERROR: $camera has $n_png labelled PNGs, expected 401" >&2
        exit 1
    fi

    for frame_index in $(seq 0 400); do
        printf -v annotation_stem '%08d' "$((frame_index * 5))"
        expected_png="$src_dir/${camera}_${annotation_stem}.png"
        if [ ! -f "$expected_png" ]; then
            echo "[make_clips] ERROR: expected labelled frame $expected_png" >&2
            exit 1
        fi
    done

    # -g 1 forces all-intra: decode is deterministic and there are no GOP-boundary
    # surprises when the pipeline seeks or the tracker is compared run-to-run.
    ffmpeg -hide_banner -loglevel error -y \
        -framerate 2 -pattern_type glob -i "$src_dir/${camera}_*.png" \
        -c:v libx264 -crf "$CRF" -g 1 -pix_fmt yuv420p \
        "$out"

    n_frames=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "$out")
    echo "[make_clips] $camera: wrote $n_frames frames"

    if [ "$n_frames" -ne 401 ]; then
        echo "[make_clips] ERROR: $out has $n_frames frames, expected 401" >&2
        exit 1
    fi
done

echo "[make_clips] Done. Clip frame N <-> annotation stem index N*5."
