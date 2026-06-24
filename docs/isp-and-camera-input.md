# ISP and Camera Input

End-to-end treatment of the camera-to-inference path: ISP pipeline stages, the nvargus/Argus CSI capture path on Jetson, how ISP misconfiguration degrades detection accuracy, and camera tuning trade-offs specific to traffic scenes.

This cannot be exercised on x86/dGPU hardware (no CSI or Argus stack). The content below is based on documented Jetson behaviour, ISP theory, and traffic-scene constraints.

---

## ISP pipeline stages

A raw Bayer sensor frame passes through roughly eight ISP stages before a usable image reaches the inference engine. Each stage can be misconfigured in ways that degrade detection.

### 1. Black-level correction

Subtracts the sensor's dark-current offset from every pixel. If miscalibrated (wrong pedestal value for the operating temperature), shadow regions clip to zero and low-contrast objects (a cyclist in a dark jersey against a kerb) lose the gradient information the detector relies on.

### 2. Lens shading correction (LSC / vignette)

Flat-field correction for the optical falloff toward frame edges. Without it, corner brightness drops 20–40% on typical lenses — corners appear darker, and detections near frame edges have systematically lower confidence scores. Traffic sensors mounted wide-angle (≥ 120° FOV) are especially vulnerable. Calibrate per lens + sensor pair; the calibration table is temperature-sensitive on long deployments.

### 3. Demosaicing (debayering)

Reconstructs full-colour pixels from the Bayer mosaic. Simple bilinear interpolation introduces colour fringing on high-contrast edges (e.g. a white lane marking against dark tarmac), which sharpening then amplifies. Better algorithms (AHD, MLAA) reduce this at higher compute cost. On Jetson, the ISP hardware block handles demosaicing; the algorithm is fixed per-sensor and configured via the sensor DTB + ISP tuning file.

### 4. Auto White Balance (AWB)

Estimates and removes the colour cast from the illuminant. **This is the highest-risk stage for traffic scenes at night.** Low-pressure sodium streetlights emit near-monochromatic orange (589 nm); AWB interprets this as extreme orange cast and applies a heavy blue gain to compensate. The resulting image has suppressed red/orange channels — a red traffic light or brake light loses saturation, and a red vehicle may shift enough to fall outside the detector's training distribution.

Fix: lock AWB (`awblock=true` in `nvarguscamerasrc`) during night hours, or use a fixed white-balance preset (`wbmode=5` daylight, `wbmode=3` fluorescent) calibrated for the predominant streetlight type in the deployment area.

### 5. Colour Correction Matrix (CCM)

A 3×3 matrix that maps sensor RGB to sRGB or another output colour space. The CCM is tuned for the primary illuminant in the calibration conditions. If the deployment environment has a different dominant illuminant (e.g. LED vs sodium vs mixed), the CCM will produce a systematic colour error. For traffic scenes this usually matters less than AWB (YOLO26n is illuminant-agnostic at inference time), but it affects downstream colour-based analytics (vehicle colour classification, red-light running).

### 6. Gamma / tone mapping

Maps the linear sensor response to a perceptual (gamma-encoded) output. The choice of gamma curve determines how much of the dynamic range is allocated to shadows vs highlights. For traffic:
- **Too dark a midtone curve**: vehicles in shade (under bridges, in tunnels) lose contrast; bboxes fragment or disappear.
- **Too aggressive highlight rolloff**: headlights and reflective number plates clip to white blobs; the detector sees a saturated rectangle rather than a vehicle front.

HDR sensors with two-exposure fusion (short + long) give the best result but add latency (two frames merged → one output) and complicate the inference frame timing.

### 7. Noise reduction

Two types:
- **Spatial (single-frame)**: bilateral or NLM filtering. Over-aggressive spatial NR smooths out fine texture and suppresses the detector's edge-based features. Pedestrian limb boundaries and tyre treads — both useful features for person/vehicle distinction — are the first to disappear.
- **Temporal (multi-frame)**: blends current frame with a motion-compensated previous frame. Aggressive temporal NR on fast-moving objects (vehicle at 50 mph ≈ 37 px/frame at 25fps with a 50mm-equivalent lens) produces motion blur — the bbox trailing edge appears smeared, making width estimation unreliable.

For traffic sensors, prefer light spatial NR over temporal NR. The pipeline does not benefit from reduced pixel noise as much as it suffers from blurred object boundaries.

### 8. Sharpening

Edge enhancement applied as the final ISP stage. Sharpening at the wrong frequency:
- **Over-sharpening**: creates ringing artefacts on high-contrast edges (lane markings, vehicle silhouettes). The detector may interpret each ringing lobe as a separate object boundary, causing false-positive bbox fragmentation — one vehicle appears as two or three overlapping detections.
- **Under-sharpening**: at long range, pedestrians and cyclists become soft blobs below the detector's feature resolution threshold.

For YOLO26n at 640×640 input, moderate sharpening tuned to the 10–30 px spatial frequency range (corresponding to the smallest objects at operating distance) gives the best recall without fragmentation.

---

## nvargus and the Argus CSI capture path on Jetson

On Jetson, `nvarguscamerasrc` exposes the full Argus ISP pipeline to GStreamer. The capture flow:

```
CSI sensor → kernel CSI driver (V4L2 subdev) → Tegra ISP block → Argus daemon
  → IImageConsumer → NvBufSurface (NVMM) → nvarguscamerasrc GStreamer element
```

`nvarguscamerasrc` is the bridge between the Argus daemon and GStreamer. It outputs frames directly into NVMM memory — zero-copy into the DeepStream pipeline.

**ISP tuning file**: Argus reads a sensor-specific tuning XML file (provided by the sensor vendor or written by the camera team) that sets AWB priors, CCM tables, noise reduction coefficients, tone-map curves, and lens shading tables. For production sensors deployed at scale, maintaining and version-controlling ISP tuning files per hardware revision is a significant part of the camera team's work.

**Key `nvarguscamerasrc` properties for production use:**

```bash
# Lock AWB after warmup to prevent mid-session color drift:
nvarguscamerasrc sensor-id=0 awblock=true

# Fixed 5ms shutter (200 Hz equivalent) — prevents motion blur at 25 fps:
nvarguscamerasrc sensor-id=0 exposuretimerange="5000000 5000000"

# Cap analog gain to 8× to limit noise amplification at night:
nvarguscamerasrc sensor-id=0 gainrange="1 8"

# Force daylight white balance (avoids sodium streetlight AWB failure):
nvarguscamerasrc sensor-id=0 wbmode=5 awblock=true
```

**Why x86 does not have this path**: `nvarguscamerasrc` is a Jetson-only element provided by the JetPack tegra-multimedia-api package. x86 DeepStream has no Argus daemon, no CSI kernel driver, and no ISP block. IP cameras on x86 use `rtspsrc`, which receives a pre-ISP'd RTSP stream from the camera's internal ISP — the ISP tuning happens inside the camera hardware, not in the edge node pipeline.

---

## How ISP misconfiguration degrades detection

Concrete failure modes, each with the ISP root cause and the detection symptom:

### Night AWB failure
**Cause**: AWB not locked; sodium streetlights trigger heavy blue gain compensation.
**Symptom**: Red/orange channels suppressed. Brake lights, traffic signals, and red/orange vehicles shift outside the training-distribution colour range. Confidence scores on these objects drop 15–30%; below-threshold detections increase. In MOT17 terms: missed detections rise, MOTA drops.

### Blown-out highlights from headlights
**Cause**: AE optimising for ambient illumination rather than headlight dynamic range; no HDR.
**Symptom**: Oncoming vehicle fronts clip to white saturated regions. The detector sees a white rectangle with no texture — no front grille, no headlight geometry. Confidence for `car` class drops; the object is missed or detected as a lower-confidence `truck` due to the blob shape.

### Temporal NR motion blur on fast vehicles
**Cause**: Aggressive temporal noise reduction with short motion-compensation window.
**Symptom**: Vehicle bbox trailing edge is blurred over 5–15 px. Width estimates are 10–20% too large. Track association using IoU degrades because the blurred bbox overlaps adjacent vehicles. ID-switches increase on multi-lane roads at peak hour.

### Over-sharpened edge ringing → bbox fragmentation
**Cause**: Sharpening radius tuned for large-object clarity; ringing at lane marking edges.
**Symptom**: A single vehicle near a lane marking produces two or three overlapping detections: one on the vehicle body, one or two on the ringing lobes at the marking boundary. IOU tracker generates spurious tracks; NvDCF mitigates this via appearance model but still shows elevated ID-switches.

### Incorrect lens shading correction
**Cause**: LSC table calibrated for a different lens or at a different temperature.
**Symptom**: Corner brightness drops 20–30%. Pedestrians and cyclists at frame edges have systematically lower confidence. With `conf_threshold=0.25`, edge objects are still detected; at `conf_threshold=0.35` they drop out. This appears as a position-dependent recall bias that worsens at the same threshold across all weather conditions.

---

## Camera tuning for traffic scenes

### Night / low-light

1. **Use analog gain before digital gain.** Analog gain (ISO) amplifies signal before ADC quantisation — less noise. Digital gain amplifies the already-quantised signal, amplifying quantisation noise. Set `gainrange="1 8"` (analog) and disable digital gain. For Jetson sensors this is done in the ISP tuning file.
2. **Consider fixed exposure.** AE hunting (continuously adjusting) during a vehicle approach-and-pass cycle causes brightness swings that affect detection confidence frame-to-frame. For a fixed-mount sensor with a known minimum illuminance, a fixed 5–8 ms shutter at ISO 6400 often gives more consistent detections than auto-exposure.
3. **Raise the near-infrared cut-off.** Many traffic sensors add IR illuminators for night use. Ensure the lens / filter stack passes 850 nm or 940 nm IR — a standard daylight cut filter at 650 nm will block the illuminator's output.

### Glare and headlights

1. **Polarising filter**: a circular polariser in front of the lens reduces specular reflections from wet tarmac and direct headlight glare by 3–4 stops. It reduces overall transmission by ~1.5 stops. Usable only on fixed-mount sensors where the polariser angle can be set once at install.
2. **HDR mode**: most Jetson-compatible sensors support two-exposure capture. Short exposure (0.5 ms) preserves headlight detail; long exposure (8 ms) preserves shadow detail. The ISP fuses both. Latency penalty: one extra frame period per fusion cycle. If the pipeline is running at 25 fps and the fusion adds one frame latency, the effective detection latency increases from 40 ms to 80 ms — relevant for red-light running applications.
3. **AE target**: set the AE target tone to 30–40% of full scale (not the default 50%) in scenes with frequent bright light sources. This reduces blown-out highlights at the cost of slightly darker shadows.

### Motion blur

The rule of thumb for blur-free detection: **shutter speed ≥ object speed in pixels per frame / 2**.

At 25 fps, a vehicle travelling at 50 mph (22 m/s) at 20 m distance with a 70° FOV lens covering 7 m at that distance:
- Speed at sensor: 22 m/s ÷ 7 m × frame_width px ≈ 22/7 × 1920 = 6034 px/s = 241 px/frame
- Max shutter for 1-pixel blur: 1/(241 × 2) s ≈ 2 ms

A 2 ms shutter requires either a fast lens (f/1.4) or high ISO or supplemental IR illumination for night use. This is the physical constraint that drives the night-time camera design.

DeepStream pipeline implication: **blur is not recoverable downstream**. Deblurring filters in the GStreamer probe would add > 10 ms per frame and still reduce edge sharpness. The correct fix is camera-side.

### Lens selection

| Parameter | Consideration for traffic |
|---|---|
| Focal length | Longer = higher resolution at distance (better for far-field vehicle counting); shorter = wider FOV (fewer sensors to cover a junction) |
| Aperture | Larger aperture (lower f/#) = more light at night; shorter depth of field (objects outside focal plane appear soft) |
| Depth of field | For a fixed-mount sensor, a deep depth of field (small aperture, wide angle) ensures all lanes at all distances are in focus |
| Distortion | Wide-angle lenses have barrel distortion — lane boundaries bow inward. Correct in ISP or post if lane-level accuracy is required for junction analytics |
| IR cut filter | Remove or replace with a dual-band filter for IR illuminator compatibility |

---

## Practical pipeline integration

**Where camera tuning affects inference** in the DeepStream graph:

```
nvarguscamerasrc (ISP: AWB, AE, NR, sharpening applied here)
  → nvv4l2decoder (decodes H.264/HEVC from IP cameras — ISP already done camera-side)
  → nvstreammux
  → nvinfer   ← detection quality is a direct function of ISP quality
  → nvtracker ← ID-switch rate is partly driven by bbox stability (blur / fragmentation → instability)
  → nvdsosd ← [blur probe operates on post-ISP frames]
```

For IP cameras over RTSP, the ISP is inside the camera (Hikvision, Axis, Dahua cameras all have embedded ISPs). Tuning happens via the camera's web UI or ONVIF API — not in the DeepStream pipeline. For CSI-connected cameras on Jetson, `nvarguscamerasrc` properties provide runtime control.

**Metrics to track for ISP drift detection**:
- Per-class confidence distribution over time (a shift in the mean suggests illuminant change or AWB failure)
- Bbox aspect ratio distribution (sudden increase in wide bboxes signals motion blur)
- Detection count per frame vs time-of-day (a drop at dusk without a corresponding streetlight-on event signals AWB failure)
- ID-switch rate per hour (elevation at night relative to day suggests temporal NR causing bbox instability)

These can be derived from the per-frame CSV output already produced by this pipeline and are the foundation of the M3.6 observability work.
