import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "tools" / "make_readme_demo.sh"

# Fake ffmpeg: append the argv to the log, then create the output (last arg).
# GIF_BYTES lets a test drive the size-budget check without producing real media.
FAKE_FFMPEG = """#!/usr/bin/env bash
printf "%s\\n" "$*" >> "$FFMPEG_LOG"
out="${@: -1}"
if [[ "$out" == *.gif ]]; then
    head -c "${GIF_BYTES:-1024}" /dev/zero > "$out"
else
    head -c 1024 /dev/zero > "$out"
fi
"""


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _run_script(tmp_path: Path, *, captures=("cam1", "cam2", "cam3"), font=True, **env_overrides):
    in_dir = tmp_path / "capture"
    in_dir.mkdir()
    for name in captures:
        (in_dir / f"{name}.mp4").touch()

    font_path = tmp_path / "Font.ttf"
    if font:
        font_path.touch()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "ffmpeg", FAKE_FFMPEG)

    env = os.environ.copy()
    env.update(
        {
            "IN_DIR": str(in_dir),
            "OUT_DIR": str(tmp_path / "assets"),
            "FONT": str(font_path),
            "FFMPEG_LOG": str(tmp_path / "ffmpeg.log"),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    env.update({key: str(value) for key, value in env_overrides.items()})

    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _ffmpeg_log(tmp_path: Path) -> str:
    return (tmp_path / "ffmpeg.log").read_text()


def test_demo_script_tiles_three_labelled_cameras(tmp_path: Path):
    result = _run_script(tmp_path, DURATION=12, FPS=10, TILE_WIDTH=360)

    assert result.returncode == 0, result.stderr
    log = _ffmpeg_log(tmp_path)
    assert "hstack=inputs=3" in log
    assert "CAM 01" in log and "CAM 02" in log and "CAM 03" in log
    assert "scale=360:-2" in log
    assert "fps=10" in log
    assert "-t 12" in log


def test_demo_script_builds_the_gif_in_two_palette_passes(tmp_path: Path):
    result = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    log = _ffmpeg_log(tmp_path)
    assert "palettegen" in log
    assert "paletteuse" in log
    assert "-loop 0" in log
    # palettegen must run before paletteuse consumes the palette
    assert log.index("palettegen") < log.index("paletteuse")


def test_demo_script_writes_both_gif_and_mp4(tmp_path: Path):
    result = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assets = tmp_path / "assets"
    assert (assets / "pipeline-demo.gif").exists()
    assert (assets / "pipeline-demo.mp4").exists()
    assert "libx264" in _ffmpeg_log(tmp_path)

    # Published assets are world-readable: they are rendered via mktemp, which
    # creates 0600, and the mode survives the rename.
    for name in ("pipeline-demo.gif", "pipeline-demo.mp4"):
        assert (assets / name).stat().st_mode & 0o044 == 0o044


def test_demo_script_applies_per_tile_start_offsets(tmp_path: Path):
    result = _run_script(tmp_path, START1=3, START2="3.5", START3=4)

    assert result.returncode == 0, result.stderr
    log = _ffmpeg_log(tmp_path)
    assert "-ss 3 " in log
    assert "-ss 3.5 " in log
    assert "-ss 4 " in log


def test_demo_script_fails_when_a_capture_is_missing(tmp_path: Path):
    result = _run_script(tmp_path, captures=("cam1", "cam2"))

    assert result.returncode != 0
    assert "cam3.mp4" in result.stderr


def test_demo_script_fails_when_the_font_is_missing(tmp_path: Path):
    result = _run_script(tmp_path, font=False)

    assert result.returncode != 0
    assert "font" in result.stderr.lower()


def test_demo_script_rejects_a_gif_over_the_readme_size_budget(tmp_path: Path):
    result = _run_script(tmp_path, MAX_MB=1, GIF_BYTES=2 * 1024 * 1024)

    assert result.returncode != 0
    assert "1 MB budget" in result.stderr

    # A rejected run must publish nothing: no oversized GIF to stage by
    # accident, and no half-updated pair of assets either.
    assets = tmp_path / "assets"
    assert list(assets.glob("*.gif")) == []
    assert list(assets.glob("*.mp4")) == []


def test_demo_script_keeps_the_previous_assets_when_a_run_is_rejected(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "pipeline-demo.gif").write_text("published")
    (assets / "pipeline-demo.mp4").write_text("published")

    result = _run_script(tmp_path, MAX_MB=1, GIF_BYTES=2 * 1024 * 1024)

    assert result.returncode != 0
    assert (assets / "pipeline-demo.gif").read_text() == "published"
    assert (assets / "pipeline-demo.mp4").read_text() == "published"
