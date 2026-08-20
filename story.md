# Project Story

## Current state

- M1 and M2 are complete apart from the deferred heavy-decode work and one container build check.
- M3.1–M3.5 are complete; M3.6 and M3.7 retain open observability/profiling work.
- The pipeline tracks each camera independently and does not yet provide a cross-camera identity.

## Latest milestone

- M3.8 has been planned in `MILESTONES.md`: geometry-first cross-camera person association on WildTrack, with optional ReID appearance evidence and RTSP-safe clocks/reconnect handling.

## Next step

- Start M3.8.0 by generating deterministic GT-aligned WildTrack clips, then implement the identity and ground-plane utilities before fitting homographies.

## Known risks or blockers

- WildTrack data must be available locally to generate clips and fit calibration.
- GPU hardware is required for the baseline and online pipeline work, but the core association and offline evaluation can be completed on CPU first.
- ReID model availability in DeepStream 9.0 is not yet verified and must not block the geometry baseline.

## Safe next commands

```bash
bash metrics/make_wildtrack_clips.sh
pytest tests/unit -q
ruff check .
```
