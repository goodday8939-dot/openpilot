# eGPU YOLO experiment

`carrot-egpu-yolo` is a separate feature experiment based on `carrot-cinque-terre`
at `90696ca69ae9a2325cb901cffb335ff45b95c0a7`. The Cinque Terre driving model and
its NAS manifest are unchanged. This experiment is intentionally not enabled
on the three maintained model variants.

## Execution and image coordinates

After all three driving publications, modeld may run YOLOv8n on the newest
`img_q` frame, already resident on the same USB AMD device. Its packed YUV420
input is four luma parity planes plus U and V. The adapter reconstructs luma,
upsamples chroma and uses the camera renderer's full-range conversion matrix,
then bilinearly resizes to 320 x 160 RGB on the GPU. The original 80 COCO classes
are preserved (a traffic-light class does not classify its signal color).

There is no second image transfer or preview stream. The GPU reduces output to
six values per anchor: xywh, winning confidence, class ID. CPU class-aware NMS
is capped at 200 candidates and 40 detections. Boxes are mapped through the
same per-frame model-to-camera warp into four normalized camera corners.

`carrotYolo` is an observational, logged service. It contains frame/camera IDs,
capture and driving publication timestamps, timing and admission counters,
and detections. It has no control consumers. Carrot Web draws these on its
existing Drive camera surface, following the presented-frame channel and
`CarrotVisionStageTransform`. Live-only results expire after 350 ms, are never
drawn on replay/wide-road video, and use recent history to avoid future-frame
boxes on a delayed live video. Browsers without video/cereal timestamp mapping
use frame IDs when available, otherwise a bounded latest-result fallback.

## Admission

Twenty consecutive camera frames establish cadence. The next arrival deadline
and web result expiry use the camera's CLOCK_BOOTTIME domain, including suspend.
The deadline
uses the earliest observed SOF-to-receive offset over 120 frames, capped by
receive time plus 50 ms. A late receive does not grant another free 50 ms.
Frame gaps, timestamp jumps, and dropped frames restart settling.

YOLO runs at most five times per second. Admission reserves 1.4 times the
largest measured full execution time plus a 5 ms guard. A completion exceeding
the deadline minus guard disables YOLO for that modeld session. This protects
subsequent submissions; it does not preempt an already submitted GPU job.
The actual eGPU must be measured before this is considered validated for use.

## Preparation

On a workstation, use `openpilot/tools/egpu_yolo/export_model.py` with the
official YOLOv8n `.pt`. The tool exports a static ONNX and checks three input
outputs against CPU PyTorch. Publish verified `big_driving_supercombo.onnx` and `manifest.json`
under `\\DS1821P\openpilot\models\carrot-egpu-yolo`.
The NAS endpoint only serves its established ONNX filename. In this separate
directory that file contains YOLO, and is installed as `/data/egpu_yolo/model.onnx`.

With ignition off, install the branch and download without opening the GPU:

```sh
cd /data/openpilot
python -m openpilot.selfdrive.modeld.egpu_yolo_prepare download
```

After the owner turns ignition on for commissioning, stop modeld using the
vehicle's manager maintenance procedure so the GPU is exclusively available.
Then run:

```sh
python -m openpilot.selfdrive.modeld.egpu_yolo_prepare compile
```

Compilation keeps the driving weights resident, records 30 completed YOLO
runs including readback/NMS, and verifies the serialized JIT round trip.
`/data/egpu_yolo/yolo.pkl` activates the runtime on the next modeld start.
The source fingerprint and driving queue shape must match. Loading and warmup
happen at initialization, never in an idle time slot. Runtime errors disable
the optional detector; driving inference keeps its existing fallback path.
For A/B commissioning use `EGPU_YOLO=0` in the modeld launch environment.

## Device commissioning still required

Compare baseline and enabled runs with the same Cinque Terre model. Record
camera-to-driving-publication latency and publication intervals (p50/p99/max),
frameDropPerc, YOLO runtime/admission/overrun counters, USB faults, memory use,
and temperature over a sustained run. `modelExecutionTime` alone misses a
delayed start of the next inference. Verify boxes on the actual camera view
in portrait, landscape and fullscreen. Compilation/real eGPU tests require
ignition and are only performed after the owner has been notified.

No user setting or vehicle control behavior is added. The display explanations
belong to the localized web UI; public user guides are not changed.
