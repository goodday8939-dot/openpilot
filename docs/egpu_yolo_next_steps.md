# YOLO continuation plan

Continue on `carrot-egpu-yolo`, the explicitly separate feature experiment
based on `carrot-cinque-terre`. Do not activate these features on the three
maintained model branches. The owner approved the following five stages.
The present outputs are observational; automatic vehicle control is not
validated or enabled by this plan.

## Current baseline

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

## 1. Stabilize eGPU latency without shrinking resolution

Retain 512 x 256 initially. The historical shared-eGPU path measured 5.27 ms
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
