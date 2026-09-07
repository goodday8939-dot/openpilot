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
100 ms or an execution error writes `qcom_fault`, preventing manager restarts
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
spanning 220.53 ms. A subsequent isolated profile measured 52.15 ms median
for the original Tensor gather/stack NV12 preprocessing alone.
Input is already NV12, but the existing network still receives RGB after GPU
preprocessing; it is not a network trained directly on YUV. These measurements
describe this compiler path, not an upper bound on internal-GPU performance.

A subsequent FP16 buffer compilation was interrupted after a reported camera
frame-rate warning. Its maintenance log contains repeated camera request skips
and encoder dequeue timeouts. That interrupted trial produced no FP16 result;
later saved-frame trials completed only after the cameras stopped. The exact
CPU/GPU/driver contribution to the camera stalls was not isolated. Compilation
with live cameras must not be repeated; the preparation tool now rejects it.

After terminating the compiler and restoring the normal manager, a 20-second
read-only sample measured approximately 20 Hz for all three camera services
and the driving model, with no displayed alert or camera error event. The last
model sample reported zero frame drops and 37.07 ms execution. This recovery
sample does not validate running YOLO alongside driving.

The owner subsequently made 40 ms an optimization target rather than a hard
acceptance ceiling, authorized some numerical precision loss, and requested
that 512 x 256 resolution be preserved. The worker's 100 ms overrun guard stops
subsequent submissions; it cannot cancel an in-flight kernel. The commissioning
marker remains absent pending a camera/driving latency comparison. Neither
40 ms nor 100 ms establishes isolation from the driving camera pipeline.

### Fixed-resolution optimization and quantization

All following trials use the same saved 1344 x 760 NV12 frame, with cameras,
driving inference and DM stopped. A separate supervisor checks fresh manager
and device state every 0.5 seconds and terminates optional work if onroad or
camera activity returns. No live-camera compilation is used. Timings cover
resident input through preprocessing, inference, readback and NMS; they exclude
the initial camera snapshot and tensor upload and are not live-camera timings.

The adapter now reads NV12 pixels directly in one GPU kernel. Its standalone
preprocessing median was 9.17 ms including dispatch/readback, versus 52.15 ms
before; the kernel itself took about 4.2-4.6 ms. Independent full-frame NumPy
conversion agreed within 4.49e-6. Each convolution result is materialized to
avoid recomputing shared branches. Unit ONNX stride/dilation tuples are
normalized to scalar 1 so tinygrad actually selects its Winograd convolution.
These overrides belong only to this YOLO runner; global driving operators
are unchanged.

| Candidate, all 512 x 256 | Median ms | Maximum ms | Outcome |
| --- | ---: | ---: | --- |
| Original FP32 buffer, 60 runs | 223.67 | 246.67 | Reference |
| Direct NV12 + materialized FP32 Conv, 30 runs | 145.21 | 148.74 | Numerically validated |
| Same with real FP32 Winograd, 30 runs | 84.12 | 87.19 | Validated Qualcomm-compiler fallback candidate |
| FP32 Winograd + Mesa IR3, 30 runs | 66.64 | 69.89 | Selected; numerically validated |
| FP16 storage / FP32 Conv, 30 runs | 125.16 | 127.85 | Low-confidence box drift; slower |
| Mesa IR3 FP16 image Conv, 30 runs | 88.34 | 92.05 | Slight detection drift; slower |
| Static INT8 Conv, 30 runs | 228.97 | 231.94 | Slower; not selected |

The 84.12 ms candidate passed serialization/reload and comparison with the
original ONNX Runtime FP32 network: maximum box/score error 0.00533, maximum
box error 0.0000611 pixels among anchors above confidence 0.35, and matching
classes. Workgroup variations gave roughly 82-85 ms with bit-exact outputs;
that small difference is not treated as a robust additional speedup.
The selected IR3 FP32 Winograd candidate additionally passed the same checks,
with maximum box/score error 0.00481 and relevant-box error 0.0000611 pixels.

INT8 was calibrated with 64 shuffled COCO128 images (seed 42), using ONNX
Runtime 1.20.1 static QOperator quantization, unsigned activations, signed
per-channel weights, MinMax calibration, and Conv-only quantization. All 64
calibration samples were accumulated. The model shrank from 12,709,299 to
3,382,556 bytes without changing input resolution. On the other 64 images,
at confidence 0.35 and IoU 0.5, true detections changed from 153 to 152 of
425 labels, false detections from 23 to 24, and detected people from 50 to 48
of 89 labels. This small sanity set is not a driving-quality or full-COCO mAP
evaluation. On the saved vehicle frame, INT8 lost the FP32 anchors above 0.35.
The GPU INT8 implementation also differed from quantized ONNX Runtime outputs
(maximum score difference 0.0485); it is not a numerically accepted candidate.

This tinygrad QLinearConv path expands arithmetic to int32 and does not use a
packed INT8 dot-product kernel. A direct compiler probe rejected
`cl_qcom_dot_product8` and `qcom_dot8_acc`; this describes the available compiler,
not a claim that all runtimes on this hardware lack INT8 acceleration. Merely
changing the model dtype therefore does not promise a faster inference.
The selected IR3 compiler/library is confined to the optional YOLO process
and `/data/egpu_yolo/lib`; system drivers and modeld are not changed. Preparation
and worker startup select `DEV=QCOM:IR3` before importing tinygrad and point
`MESA_PATH` at the verified local library. Older QCOM compiler artifacts are
rejected by the backend and adapter fingerprint checks and must be rebuilt.

The pinned dependency is the arm64 Linux Mesa 25.2.7 library referenced by the
repository's tinygrad Mesa binding:
`https://github.com/sirhcm/tinymesa/releases/download/v1/libtinymesa-mesa-25.2.7-linux-arm64.so`.
Place it at `/data/egpu_yolo/lib/libtinymesa.so` while preparing the experiment;
there is no onroad download or system library installation. Preparation and
worker startup verify SHA-256
`9436d1bd3da1c4394523a3debef4df9451b967c264f5d48bf9aeee03430c1364`.
Missing or mismatched libraries disable the optional worker. Offroad compilation
preserves the available CPU affinity when power management has CPUs 4-7 offline.

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
