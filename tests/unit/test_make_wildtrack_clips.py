import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "metrics" / "make_wildtrack_clips.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _run_script(tmp_path: Path, stems: list[int], *, probed_frames: int = 401):
    root = tmp_path / "labelled_ds"
    source = root / "images" / "C1"
    source.mkdir(parents=True)
    for stem in stems:
        (source / f"C1_{stem:08d}.png").touch()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "ffmpeg",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" > "$FFMPEG_LOG"\ntouch "${@: -1}"\n',
    )
    _write_executable(fake_bin / "ffprobe", f"#!/usr/bin/env bash\necho {probed_frames}\n")

    env = os.environ.copy()
    env.update(
        {
            "WILDTRACK_ROOT": str(root),
            "OUT_DIR": str(tmp_path / "clips"),
            "CAMERAS": "C1",
            "FFMPEG_LOG": str(tmp_path / "ffmpeg.log"),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clip_script_rejects_a_broken_frame_to_stem_sequence(tmp_path: Path):
    stems = [index * 5 for index in range(401)]
    stems[-1] = 9_999_999

    result = _run_script(tmp_path, stems)

    assert result.returncode != 0
    assert "expected labelled frame" in result.stderr


def test_clip_script_encodes_the_labelled_sequence_at_two_fps_all_intra(tmp_path: Path):
    result = _run_script(tmp_path, [index * 5 for index in range(401)])

    assert result.returncode == 0, result.stderr
    ffmpeg_args = (tmp_path / "ffmpeg.log").read_text()
    assert "-framerate 2" in ffmpeg_args
    assert "-pattern_type glob" in ffmpeg_args
    assert "C1_*.png" in ffmpeg_args
    assert "-g 1" in ffmpeg_args
    assert (tmp_path / "clips" / "wildtrack_c1_gt.mp4").exists()


def test_clip_script_rejects_an_encoded_clip_without_401_frames(tmp_path: Path):
    result = _run_script(
        tmp_path,
        [index * 5 for index in range(401)],
        probed_frames=400,
    )

    assert result.returncode != 0
    assert "has 400 frames, expected 401" in result.stderr
