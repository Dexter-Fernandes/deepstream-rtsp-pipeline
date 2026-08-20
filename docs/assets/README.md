# README assets

## `pipeline-demo.gif` / `pipeline-demo.mp4`

Three WildTrack cameras processed concurrently by `pipelines/multi_stream.py`:
YOLO26n FP16 detection, `NvMultiObjectTracker` per camera, cross-camera identity
fusion, and Gaussian blur anonymisation — all drawn by `nvdsosd` inside the
pipeline. Each label reads `ID <tracker id> | G<global id>`; the same `G` number
appearing in two tiles is the same person fused across cameras.

12 s excerpt, 3 × 294 px, 10 fps.

### Provenance

Source footage: the **WildTrack** multi-camera pedestrian dataset, cameras C1–C3
(Chavdarova et al., *WILDTRACK: A Multi-camera HD Dataset for Dense Unscripted
Pedestrian Detection*, CVPR 2018; EPFL CVLab).

**Licence caveat, stated plainly:** the EPFL dataset page publishes download
links and technical documentation but **no licence or redistribution terms** —
only the accompanying WILDTRACK toolkit carries an explicit licence (GPLv3). The
excerpt here is therefore kept short, heavily downscaled, and anonymised, and the
raw source clips are never committed (`data/*.mp4` is git-ignored). If the
dataset authors object to this excerpt, delete `pipeline-demo.gif` /
`pipeline-demo.mp4` and regenerate locally with the commands below.

**Anonymisation is detector-bound.** `--anonymise` blurs every *detected* person
before `nvdsosd` draws and before any frame leaves the pipeline, but people the
detector misses — mostly the dense far-field crowd — are not blurred. At 294 px
per camera a pedestrian is a few pixels tall, so the downscale itself does most
of the remaining work; this is a limitation of the asset, not a claim about the
pipeline.

No FPS or throughput figure is burned into the footage. Measured performance
belongs in the README tables, where it is traceable to `metrics/results/`.

### Regenerating

```bash
# 1. Record the three annotated output streams (needs the GPU + built image).
#    File-source mode by default: every frame is processed and recorded,
#    however long that takes. ~300 s of wall clock yields ~29 s of footage.
bash tools/capture_demo_streams.sh

# 2. Tile, trim, label, and export the GIF + MP4 into this directory.
bash tools/make_readme_demo.sh
```

Both scripts are env-configurable; see the header comments. `START1/2/3` pick the
section of the capture to use, and the tiling script fails rather than committing
a GIF over the `MAX_MB` budget.
