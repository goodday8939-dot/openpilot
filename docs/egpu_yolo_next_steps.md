# YOLO continuation plan

Continue on the owner-requested `carrot-egpu-yolo2`, branched from
`carrot-egpu-yolo` at `1310ed43fe`. This remains a separate feature experiment
based on `carrot-cinque-terre`. Do not activate these features on the three
maintained model branches. The owner approved the following five stages.
The present outputs are observational; automatic vehicle control is not
validated or enabled by this plan.

## Current baseline

The follow-up has completed the initial 640 x 384 saved-input eGPU measurement
and independent numerical check. See [eGPU timing](egpu_yolo2_timing.md).
Its 5,000 resident-input runs took 5.45 ms p50 / 6.24 ms p99 / 7.39 ms maximum,
but uploading the original padded NV12 frame at 5 Hz took the total to
31.73 / 33.67 / 33.95 ms. The original input upload alone took 25.99 ms median.
This does not establish admission alongside live driving. The owner then
prioritized reusing the already-resident driving image for minimum latency,
explicitly accepting bicycle misses. Additional full-camera upload and the
higher-resolution input are no longer the immediate implementation target.
The same-frame 512 x 256 comparison took 3.64 ms resident-input median and
29.74 ms with original-buffer upload. Both trials passed serialization and
independent ONNX checks. Each has 5,000 resident runs and 300 uploads at 5 Hz;
neither measures live driving concurrency or a controlled thermal A/B.

The last internal-GPU continuous-display attempt had already stopped before
the follow-up: 421 results, then a 150.47 ms model-publication gap with frame-ID
delta 3 on the non-conflating observer. Its root cause remains unisolated.
There is currently no YOLO display worker or automatic activation marker.

See `egpu_yolo_experiment.md` for measurements and numerical checks.
The internal-GPU artifact uses YOLOv8n COCO FP32 at 512 x 256, confidence
0.35, direct NV12 preprocessing, Winograd, pinned Mesa IR3 and nine cooperative
batches. Saved input takes about 69 ms; live publication measured 120 ms
median at 3.85 Hz. Its 180-second trial published 688 results without reported
driving frame drops, but increased driving tail latency. The existing 350 ms
display expiry can leave gaps between boxes.

The owner subsequently requested continuous stationary display instead of a
three-minute trial. Device-local `qcom_display_coordinator.py` performs the
same guarded camera-stop/prewarm/normal-restore sequence, keeps the warmed
worker alive, and runs `qcom_display_observer.py`. This is a manual session,
not reboot-persistent activation. Fresh Park, controls disabled, active eGPU,
existing DM-disabled configuration and healthy camera/model services are
required. The observer stops the worker on changed conditions, camera/model
gaps above 120 ms, primary execution above 60 ms, frame drops above 1%, or
missing YOLO results for 10 seconds. It retains bounded current status rather
than accumulating an unlimited in-memory trial report. Inspect
`/data/egpu_yolo/qcom_display_status.json` and
`/data/egpu_yolo/qcom_display_coordinator.log` before any new GPU work.

Do not start a second worker or compile while cameras are live. The current
warm worker holds a GPU artifact in memory. Stop its supervisor cleanly before
new maintenance. Its PID file is `/data/egpu_yolo/qcom_display_coordinator.pid`;
verify the process command before signaling it. Automatic `qcom_enabled`
activation remains absent, and cold `qcom_yolod` launches are rejected.

Continuous-display retries at 4 Hz and 2 Hz stopped after 175 and 94 runs
respectively on a sampled model-publication gap above 120 ms. A subsequent
30-second baseline received all 600 publications with a 77.57 ms maximum gap.
The observer now checks a non-conflating model subscription, preserving the
120 ms limit, to distinguish actual publication gaps from missed intermediate
messages in a sampled view. The original outlier cause is not established.
The display retry uses a 500 ms interval without changing input resolution.

### Bicycle miss: resolution evidence

A saved current road frame visibly contains parked bicycles. In the actual
512 x 256 input, the bicycle candidate occupies approximately 38 x 22 pixels.
Independent ONNX Runtime inference on matching NV12 preprocessing produced a
maximum bicycle score of 0.0276, below the 0.35 display threshold. A plant class
won at that anchor; lowering the display threshold is not a reliable fix.

The same YOLOv8n weights were also tested on the saved RGB image using standard
Ultralytics CPU preprocessing. These are input dimensions, width x height:

| Input | Best bicycle confidence |
| --- | ---: |
| 512 x 256 | 0.0215 |
| 640 x 384 | 0.5764 |
| 896 x 512 | 0.8532 |
| 1344 x 768 | 0.8927 |
| Manually selected bicycle-region crop, 384 x 256 | 0.8861 |

The 640 x 384 result also contained a motorcycle classification at 0.4696
over the same bicycles. This is one-scene evidence, not a general accuracy
benchmark. The crop was selected after looking at the image; it does not
validate automatic region selection. CPU preprocessing differs slightly from
the live NV12 kernel, explaining the different low-resolution scores. The later
640 x 384 eGPU benchmark uses matching full-camera NV12 preprocessing and
does not use the historical driving-warp input.
The workstation now has a static 640 x 384 ONNX candidate (12,756,343 bytes),
SHA-256 `e02d75ddde2a1792e0275af57e3c23dd8dc307aa64a6c2f86c8f1889618d7f76`.
Its three random-input comparisons against PyTorch passed with maximum
absolute errors below 0.00087. The verified file was copied into a separate
vehicle benchmark directory. Its prospective NAS URL is still unpublished,
and neither live runtime has been switched to the 640 x 384 artifact.

## 1. Stabilize eGPU latency without shrinking resolution

The latest instruction is to reuse the native 512 x 256 driving `img_q` and
minimize execution time, even if bicycle detection fails. Do not add a second
full-camera USB transfer to recover those detections. A fresh same-frame CPU
reconstruction using current driving calibration scored the bicycle 0.841 in
the driving warp, 0.024 in full-view 512 x 256 and 0.638 in full-view 640 x 384.
The narrower driving field of view retains more object pixels in this scene;
these are CPU/ONNX comparisons, not reads of the live eGPU queue. Reusing the
queue needs current-buffer ownership, timing guards and its own sustained
primary-latency comparison before reactivating the retired execution path.

Use 512 x 256 as the timing baseline, and evaluate 640 x 384 or a justified
region strategy as an accuracy candidate given the bicycle miss. Do not reduce
resolution further. The historical shared-eGPU path measured 5.27 ms
median and 5.73 ms maximum in a short trial, but later overran at 24.887 ms.
It is currently retired. Reproduce and separate GPU kernels, queue waits,
USB readback, host scheduling, NMS and full camera-to-publication age. Its
already-resident driving-warp input differs from the internal GPU's full-camera
letterbox; do not treat equal tensor dimensions as equal field of view.

An 8 ms target is not an enforceable interruption of an in-flight kernel.
Evaluate bounded submissions and current primary-frame admission, then compare
baseline/enabled sustained tail latency under the same scene and thermal state.
Keep measured dropped-frame, timestamp and fault evidence. Reduce frequency
before reducing image resolution. Region crops require their own timing budget.

## 2. Track vehicles and associate radar objects for Web display

First establish which front/corner tracks this vehicle actually publishes;
schema fields and parser support do not establish valid live data. Calibrate
camera/radar geometry and align capture timestamps before association. Maintain
track IDs, assignment uncertainty and unmatched visual objects. Use raw tracks
where available rather than only already-selected leads, and avoid circular
validation against existing vision-fused lead outputs.

Display ID, class/confidence, matched radar ID, range, relative speed and match
quality. Log identity switches, association errors, misses and result age.
No longitudinal or lateral control consumer is introduced at this stage.

## 3. Validate cut-in detection

Combine temporal vehicle tracks, radar lateral motion and the ego path/lane
geometry. Distinguish adjacent vehicles, approaching boundary crossings and
vehicles already occupying the path. Compare with existing `leadsCutIn` and
`leadCutInRisk` without replacing them. Curves, ego lane changes, occlusion and
track handoff need separate examples. Score false alarms and missed cut-ins,
as well as advance warning time, using independently reviewed footage.

## 4. Extend to pedestrians and lateral hazards

Retain visual detections that have no radar match. COCO is not a general
unknown-obstacle detector. Confirm available BSM/corner data before adding it.
Forward/wide cameras do not cover the entire side/rear region; an unobserved
region must not be classified as clear. Measure small-object, occlusion and
night performance. Full-camera and wide-camera mappings must be explicit.

## 5. Add dedicated traffic-light and stop-line perception

COCO has a traffic-light location class, not red/green/arrow state or a stop-line
class. Evaluate dedicated training and original-resolution crops for signal
state, stop-line geometry, and association to the ego lane/turn. Stop signs are
not stop lines. Require labeled local examples and temporal validation before
considering any stopping behavior. Prepare and test model changes as experiment
artifacts without overwriting the driving model or its NAS manifest.

## Validation and handoff

The baseline code passed 72 focused YOLO tests and Ruff. Later changes require
their own relevant checks. Browser visual inspection was unavailable when the
UI tool could not verify its URL; Web API delivery was verified. Recheck actual
boxes when a browser surface is available. Stationary timing tests are not
moving-vehicle control validation. Carry forward measured limitations rather
than reporting the five stages complete after a prototype or successful demo.
