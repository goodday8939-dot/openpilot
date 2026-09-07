# eGPU YOLO experiment

`carrot-egpu-yolo` is a separate feature experiment based on `carrot-cinque-terre`
at `90696ca69ae9a2325cb901cffb335ff45b95c0a7`. The Cinque Terre driving model and
its NAS manifest are unchanged. This experiment is intentionally not enabled
on the three maintained model variants.

## Internal GPU worker (commissioning)

The shared-eGPU execution path described below is retired. Its 3,732nd run
took 24.887 ms with only 10.152 ms remaining, and latched `overrun`. The earlier
short stationary measurements did not establish long-running isolation.
`modeld` now has no YOLO inference call or `carrotYolo` publisher. The original
USB helper modules remain available for reproducing the recorded experiment.

`qcom_yolod` owns the observational publisher in a separate process. It uses
the internal QCOM GPU, ordinary scheduling on CPU 4, and an interval of at
least 250 ms. It requests QCOM context priority 15 (lower priority than the
default 8); driver scheduling still requires measurement. Its input is an
owned copy of the latest road-camera NV12 frame,
not the eGPU's driving input. GPU preprocessing letterboxes the full camera
view into 512 x 256 RGB. For a 1344 x 760 camera the content is 453 x 256 with
horizontal padding. The inverse letterbox transform maps boxes to the original
camera. COCO classes, confidence 0.35, bounded NMS, and web expiry are unchanged.

The manager gate requires onroad, `UsbGpuActive`, no `UsbGpuLoading`, existing
`DisableDM` value 1 or 2, a compiled `yolo_qcom.pkl`, and an explicit local
`/data/egpu_yolo/qcom_enabled` commissioning marker. The worker additionally
requires fresh driving/manager/device messages and confirms both DM processes
are stopped before submitting work. No DM setting is changed by YOLO.
The test vehicle already had `DisableDM=2`, with both DM processes stopped.

Mode changes revoke new submissions and suppress late results. They do not
preempt an in-flight GPU kernel. QCOM still serves the driving image warp and
UI, and must take over driving inference if the eGPU fails. Therefore process
separation is not a guarantee of timing isolation. Compilation and live A/B
tests are required before creating the commissioning marker. A runtime above
40 ms or an execution error writes `qcom_fault`, preventing manager restarts
from repeatedly resubmitting failing optional work.

`qcom_yolo_prepare.py` compiles from a saved NV12 frame while offroad, with
camera, driving, DM and YOLO processes stopped. It verifies the existing ONNX checksum, tests
serialization, and writes a separate QCOM artifact/report. It does not enable
the worker. The NAS model and the eGPU driving artifacts are unchanged.
QCOM needs branch-join materialization for YOLO: its compiler rejected a fused
convolution/Concat kernel. This override is confined to the YOLO OnnxRunner;
the global ONNX operator table and driving compiler are unchanged.
Additional image-convolution kernels failed after splitting joins, so the
current preparation uses `IMAGE=0`. The dedicated compiler also assigns legal
workgroups of at most 64 threads to kernels without an explicit local size,
instead of exhaustively benchmarking hundreds of local sizes per kernel.
The compiler override is process-local and is not installed in modeld.

### Internal GPU measurements and camera recovery (2026-09-07)

The 512 x 256 YOLOv8n FP32 buffer implementation completed 60 standalone runs:
223.67 ms p50, 238.54 ms p95, 242.97 ms p99 and 246.67 ms maximum. Serialization
and reload passed. An independent NumPy NV12 conversion and ONNX Runtime check
on the same saved image found maximum RGB error 4.49e-6 and maximum box/score
error 0.00301; classes agreed for the three anchors above confidence 0.35.
This is one-image numerical validation, not detection-accuracy validation.

A separate profile measured median submission 1.34 ms, GPU wait 221.59 ms,
readback 0.76 ms and NMS 0.59 ms. One captured graph contained 172 kernels
spanning 220.53 ms. Individual preprocessing and convolution timings were not
retained, so this profile does not establish how much NV12 conversion costs.
Input is already NV12, but the existing network still receives RGB after GPU
preprocessing; it is not a network trained directly on YUV. These measurements
describe this compiler path, not an upper bound on internal-GPU performance.

A subsequent FP16 buffer compilation was interrupted after a reported camera
frame-rate warning. Its maintenance log contains repeated camera request skips
and encoder dequeue timeouts. No completed FP16 latency or correctness result
exists, and the FP16 prototype is not part of the deployed runner. The exact
CPU/GPU/driver contribution to the camera stalls was not isolated. Compilation
with live cameras must not be repeated; the preparation tool now rejects it.

After terminating the compiler and restoring the normal manager, a 20-second
read-only sample measured approximately 20 Hz for all three camera services
and the driving model, with no displayed alert or camera error event. The last
model sample reported zero frame drops and 37.07 ms execution. This recovery
sample does not validate running YOLO alongside driving.

The compiled FP32 artifact is retained for offline investigation, but
`qcom_enabled` remains absent and the historical shared-eGPU artifact is retired.
The requested ceiling is now 40 ms per complete YOLO frame. The measured
artifact fails that ceiling and must not be commissioned. The runtime limit
stops subsequent work after a slow completion; it cannot cancel an in-flight
kernel. Passing 40 ms alone would still require a camera/driving latency check.

## Historical shared-eGPU execution and image coordinates

After all three driving publications, modeld may run YOLOv8n on the newest
`img_q` frame, already resident on the same USB AMD device. Its packed YUV420
input is four luma parity planes plus U and V. The adapter reconstructs luma,
upsamples chroma and uses the camera renderer's full-range conversion matrix,
and keeps the native 512 x 256 driving YUV frame size for RGB inference on the GPU.
This preserves 2.56 times the pixels of the initial 320 x 160 detector; the
existing full-resolution web camera stream is unchanged. The original 80 COCO classes
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
The deadline uses the median observed SOF-to-complete-input-pair offset over
120 frames (after waiting for the synchronized extra camera), capped by
receive time plus 50 ms. A late receive does not grant another free 50 ms.
Frame gaps, timestamp jumps, and dropped frames restart settling.
An isolated early delivery therefore does not remove later frames' idle slots.
Before admission, a nonblocking probe checks the actual main-camera queue. If
another frame is waiting, YOLO is skipped and the frame plus its metadata are
preserved for the next driving iteration. The prediction remains best effort:
camera delivery jitter can still make a future frame arrive earlier than the
predicted deadline, so primary latency must be measured alongside YOLO timing.

YOLO runs at most five times per second. Admission reserves 1.2 times the
largest measured full execution time plus a 1 ms guard. A completion exceeding
the deadline minus guard disables YOLO for that modeld session. This protects
subsequent submissions; it does not preempt an already submitted GPU job.
The actual eGPU must be measured before this is considered validated for use.

YOLO alone opts into a compute-only USB AMD graph. It retains the device
timeline dependency and completion signal, but needs no multi-queue host
kickoff/reset. The normal graph factory is restored before driving inference.
Its result copy is submitted with a GPU timeline wait while compute completes;
the blocking USB read still finishes before decoding and publishing results.

Stationary commissioning on 2026-09-07, using YOLOv8n COCO at 320x160, measured
about 1.49 ms of GPU kernels. The original full path measured 7.74 ms p50 /
8.95 ms max; the YOLO-only serial graph and overlapped copy submission measured
4.64 ms p50 / 5.34 ms max (60 runs, driving weights resident). Three different
random YUV inputs produced exactly equal outputs before and after optimization.
These are isolated execution measurements, not a validated live detection rate.
The original conservative live scheduler admitted zero runs in a 40-second
sample. Live admission, primary latency and overlay checks remain necessary.
With USB optimization at 320x160, a later 40-second live sample admitted four
runs (5.31 ms p50 / 5.44 ms max), produced five detections and had no frame
drops or estimated-deadline overruns. Primary inference measured 35.90 ms p50 /
37.94 ms p99. A separate timestamp comparison measured about 2.63 ms from
inference completion to all driving publications; that is outside the reported
modelExecutionTime. The earliest-arrival scheduler often reported zero budget
despite a nominal 11 ms remainder before input-preparation overhead. An independent
camera subscriber measured a 56.43 ms median SOF-to-pair-ready offset, with a
41.72 ms minimum, motivating the median phase plus actual pending-frame check.
These 320x160 measurements do not validate the enlarged detector or new admission
policy. The result service now includes input-ready, inference-start/end,
publication/deadline timestamps and required time to distinguish each phase.

A first native-512x256 live sample measured 0.44 ms input preparation, 35.80 ms
inference and 2.64 ms postprocessing/publication (matched-frame medians), leaving
about 11.13 ms of a nominal 50 ms period. YOLO measured 5.66 ms p50 / 6.40 ms max
and admitted 28 runs in 60 seconds, with no drops or estimated-deadline overruns.
The initial 40% plus 2 ms reservation grew to 11.28 ms, slightly exceeding that
typical remainder. The revised 20% plus 1 ms reservation retains measured worst
execution, pending-camera priority and disable-on-overrun; it needs its own
live latency comparison before claiming a higher sustained detection rate.

With the revised reservation, a 60.06-second stationary sample recorded 223
YOLO executions (about 3.8 Hz over the 58-second settled window), 5.27 ms p50 /
5.62 ms p99 / 5.73 ms max, and zero frame drops or estimated-deadline overruns.
Primary inference across 1,162 frames measured 36.15 ms p50 / 38.02 ms p99 /
42.85 ms max. Matched YOLO frames measured 0.41 ms preparation, 2.65 ms
postprocessing/publication and 39.48 ms total ready-to-publication medians,
leaving about 10.52 ms nominal idle time; admission required 8.52 ms.
The scene had changed and this sample contained no COCO detections; the earlier
native-resolution sample produced 28 car detections, and the web SVG geometry
and car label were inspected against the existing camera image.

This is not a controlled same-scene driving A/B. Model-event timestamp latency
from camera EOF measured 85.38 ms p50 / 100.44 ms p99 / 113.04 ms max, compared
with an earlier baseline's 80.83 / 95.54 / 98.62 ms. Those event timestamps are
set immediately after inference, before filling and sending the publications.
Publication intervals measured 50.06 ms p50 / 66.57 ms p99 / 84.07 ms max.
Zero dropped frames does not establish zero latency impact, and the predicted
deadline cannot preempt a running GPU job when camera delivery arrives early.

## Preparation

On a workstation, use `openpilot/tools/egpu_yolo/export_model.py` with the
official YOLOv8n `.pt`. The tool exports a static ONNX and checks three input
outputs against CPU PyTorch. Publish verified `big_driving_supercombo.onnx` and `manifest.json`
under `\\DS1821P\openpilot\models\carrot-egpu-yolo-512x256`.
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
runs including readback/NMS with modeld's CPU 7 / FIFO 54 scheduling, and
verifies the serialized JIT round trip.
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
