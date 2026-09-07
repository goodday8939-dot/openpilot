# eGPU YOLO timing and driving-buffer reuse, 2026-09-07

`carrot-egpu-yolo2` continues the separate YOLO feature experiment from
`carrot-egpu-yolo` commit `1310ed43fe`, as requested by the owner. It is not a
new driving-model variant. The Cinque Terre model selection and NAS manifest
remain unchanged. The three maintained model branches do not gain this feature.

## Initial state and measurement conditions

The previous internal-GPU display supervisor had stopped after 421 results.
Its non-conflating observer recorded a 150.47 ms driving-publication gap and
frame-ID delta 3. The supervisor, observer and worker were all gone; the exact
cause of the gap was not established. A fresh 20-second check received about
20 Hz from the three cameras and driving model, with no displayed alert.

Both saved-input trials use the same 1344 x 760 bicycle frame. NV12 layout is
stride 1408, padded luma height 768 and 2,428,928 buffer bytes. The full camera
view is letterboxed on the eGPU. At 640 x 384 the content is 640 x 362 with
11 pixels of padding above and below. At 512 x 256 the content is 453 x 256,
with 29 pixels on the left and 30 on the right. These inputs differ from the
historical driving-camera warp, so comparisons with its timing also change
the input path and field of view.

An exclusive maintenance coordinator holds the existing maintenance lock,
requires fresh Park, standstill and disabled controls, stops the normal manager,
and runs a maintenance manager with camera, inference, control and UI processes
blocked. Fresh Park and stopped-process state are checked every 0.5 seconds;
failure terminates the benchmark and invokes normal recovery. Compilation never
runs with live cameras. Driving weights are loaded and kept in eGPU memory,
but driving inference is idle during these measurements.

The benchmark uses the USB AMD LLVM backend, `GMMU=0`, `FLOAT16=1`, `TC_OPT=2`,
`JIT_BATCH_SIZE=0`, and the existing YOLO-only serial graph and overlapped output
copy. Timed execution uses CPU 7 / FIFO 54. The shared NV12 conversion is reused
without installing the QCOM compiler or convolution overrides. Threshold 0.35
and bounded class-aware NMS are unchanged.

## 640 x 384 results

The candidate is 12,756,343 bytes with SHA-256
`e02d75ddde2a1792e0275af57e3c23dd8dc307aa64a6c2f86c8f1889618d7f76`.
Its static input is `[1, 3, 384, 640]`; compact output is 120,960 bytes.

| Uninstrumented measurement | Runs | p50 ms | p95 ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident NV12 through preprocessing, inference, readback and NMS | 5,000 | 5.450 | 6.064 | 6.237 | 7.395 |
| Original NV12 upload at 5 Hz | 300 | 25.989 | 27.402 | 28.000 | 28.273 |
| Inference and readback after that upload | 300 | 4.929 | 5.051 | 5.075 | 5.121 |
| NMS after that upload | 300 | 0.764 | 0.791 | 0.795 | 0.823 |
| Upload through NMS at 5 Hz | 300 | 31.731 | 33.081 | 33.674 | 33.953 |

The first compilation/execution call took 50.964 seconds and the capture call
5.076 seconds. The following two warmup calls took 5.494 and 5.547 ms. These
setup costs are outside the steady-state timing and require stopped cameras.

A separate 120-run diagnostic inserts an explicit GPU synchronization before
readback. Its p50 phases were 1.690 ms host submission, 2.982 ms GPU completion
wait, 1.886 ms USB readback and 0.653 ms NMS. Synchronization removes normal
readback overlap, so those values must not be summed to explain the faster
overlapped path. The wait includes host completion observation, not just kernels.

The last graph in a separate instrumented six-run profile contained 70 kernels.
Kernel sum and span were both 2.808 ms; the NV12 letterbox kernel took 12.88 us
and the longest kernel 177.92 us. This is one profiled graph, not a kernel-tail
distribution. The custom-kernel timing is independent of the 5,000-run host
distribution. Serialization/reload matched within `rtol=atol=1e-3`, and no
device error was reported.

## Matched-input 512 x 256 baseline

The existing 512 x 256 ONNX (SHA-256
`6331e082c495063318fdb154b9803a6ccd83af2b190077b32ea7a98c8519e616`)
was compiled separately with the same full-camera conversion, saved frame,
resident driving weights and timing procedure. This is a sequential
matched-input comparison, not a controlled thermal comparison.

| Uninstrumented measurement | Runs | p50 ms | p95 ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident NV12 through readback and NMS | 5,000 | 3.638 | 4.236 | 4.363 | 5.394 |
| Original NV12 upload at 5 Hz | 300 | 25.703 | 26.168 | 26.365 | 26.545 |
| Upload through NMS at 5 Hz | 300 | 29.745 | 30.275 | 30.489 | 30.564 |

The separate 70-kernel profile spanned 1.687 ms. Serialization passed and the
device reported no error. The independent validator measured maximum RGB error
0.000011086, maximum box error 0.006256 pixels and maximum winning-score error
0.000000954. No anchor reached 0.35; the ONNX reference's best bicycle score
was 0.027582. There are no relevant high-confidence classes to compare in this
baseline. The two extractions therefore also have different NMS workloads:
no baseline detections versus two detections at 640 x 384.

Both resolutions spend about 26 ms in the same original-buffer upload phase.
The higher-resolution resident benchmark costs another 1.81 ms median, while
the original-input path is about 1.99 ms slower overall. These observed
differences motivate fixing the transfer path before shrinking model input.

## Numerical and bicycle check

The workstation validator reconstructs RGB with independent NumPy bilinear
sampling and executes the checksum-matched original ONNX through ONNX Runtime.
For 640 x 384 it measured:

- RGB maximum absolute error: 0.000011951.
- Maximum box error over all anchors: 0.007538 pixels.
- Maximum winning-score error: 0.000006765.
- Six anchors at or above 0.35 in either result: matching classes; maximum
  relevant box error 0.000061 pixels.
- Maximum bicycle score: 0.575065 reference, 0.575062 eGPU.

The bicycle clears the unchanged display threshold in this saved scene. An
overlapping motorcycle detection at 0.481420 also remains. This is numerical
agreement and one-scene evidence, not general detection-accuracy validation
or an on-screen Web rendering check.

## Native driving-buffer reuse

The owner's latest instruction prioritizes the existing eGPU driving buffer
and minimum latency, accepting bicycle misses. The original-buffer upload
measurements are a comparison, not the selected input path. The current device
uses QCOM for the camera warp and AMD for its input queues: raw NV12 stays on
the internal GPU, while `img_q` and wide-camera `big_img_q` are both native
512 x 256 packed YUV. The latter is not a higher-resolution image.

One fresh road frame was captured with contemporaneous calibration and the
driving warp reconstructed on the CPU with the same nearest-sample YUV mapping.
The ONNX bicycle score was 0.840759 for this driving warp, 0.024321 for full-view
512 x 256 and 0.637529 for full-view 640 x 384. Its narrower field of view kept
the bicycle larger. This is CPU reconstruction of the current input geometry,
not a read of the GPU-resident tensor; do not infer live-queue validation from it.

The new adapter passes the complete `img_q` allocation directly to one color
conversion kernel. The kernel indexes the newest history entry and reconstructs
RGB at native 512 x 256; it does not upload another camera image or resize the
driving warp. Only compact detections return to the CPU. An interleaved prototype
comparison measured the original Tensor adapter and direct-kernel variants:

| Prototype, 3,000 calls each | p50 ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: |
| Original adapter, FP32 compact output | 4.039 | 4.823 | 6.040 |
| Native kernel, FP32 compact output | 2.427 | 2.892 | 3.098 |
| Native kernel, FP16 compact output | 2.471 | 2.742 | 13.807 |

FP16 halves the output from 64,512 to 32,256 bytes, but its host NMS was slower
and contained a 12.008 ms outlier. Garbage collection was already disabled;
the outlier's cause is not established. Selecting by p99 alone initially chose
that variant, even though admission reserves measured maximum cost. Selection
now compares the maximum first, then p99, and FP32 is the live candidate.

These prototype timings are fixed-input evidence. The first live attempt exposed
a stale-input bug: passing a realized `queue[-1]` slice to the custom kernel kept
the capture-time address in the HCQ graph. Changing frame IDs still produced the
saved frame's bit-identical bicycle box and score. That session was stopped and
its results excluded from live validation. The correction indexes history inside
the kernel while passing the whole queue as a direct JIT input parameter.
The benchmark now checks a different image in a new allocation, mutation of the
original allocation, restoration of the original pixels, and changed-input
replay after serialization. An artifact is installed atomically only after these
checks pass; the runtime rejects artifacts lacking this validation flag.

With direct whole-queue binding, a fresh 5,000-call FP32 measurement passed all
four changed-input checks and produced the following steady-state costs. This
replaces the lower prototype numbers for the reusable artifact:

| Verified native queue phase | p50 ms | p95 ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: |
| Host submission | 1.750 | 2.373 | 2.512 | 2.798 |
| GPU completion and overlapped readback | 1.763 | 1.843 | 2.027 | 2.386 |
| NMS | 0.462 | 0.537 | 0.573 | 0.903 |
| Total | 3.984 | 4.622 | 4.806 | 5.418 |

Driving weights were resident but inference was idle; no new image upload is
inside the timed region. Independent NumPy/ONNX Runtime validation measured
maximum box error 0.000763 pixels and score error 0.000000715, with all ten
relevant anchors retaining their classes. The saved packed-frame bicycle score
was 0.8407596 versus the reference's 0.8407594. Device error was absent and the
serialized artifact accepted both original and different input allocations.

The manually leased runtime is part of the driving owner, so no second USB GPU
process is launched. It executes only after `modelV2`, `drivingModelData` and
`cameraOdometry` are published. Fresh Park/standstill, disabled controls, inactive
actuation flags, a primary run below 60 ms, a stable camera cadence and no pending
next frame are required. Admission reserves the measured maximum multiplied by
1.2 plus 1 ms. The initial comparison capped attempts at 5 Hz; the later
per-frame experiment removes that interval for this native-buffer runtime only.
GPU submission, readback and nonblocking
CPU delivery are included in the overrun latch. An overrun or exception prevents
further optional submissions.

Activation is a device-local session lease, expiring within five seconds. A
prepared but disabled lease allows compiled startup replay and emits `paused`
status; an enabled fresh lease permits admission. Missing or expired leases do
not run YOLO. The 20 Hz model loop configures its state subscription at 20 Hz,
including its 100 Hz control/state inputs, so the freshness check uses the actual
sampling cadence. No new setting or vehicle-control consumer is introduced.

The first correctly bound live trial produced different bicycle scores, 0.3844
and 0.6063, but its second result took 29.878 ms: 2.778 ms submission, 2.190 ms
GPU completion/readback, and 24.621 ms CPU postprocessing. Admission latched an
overrun and the supervisor removed the lease. The 30-second baseline had 600
primary messages and zero frame drops; the short enabled sample also reported
zero drops. This is a failed sustained trial, not a successful latency result.
The exact cause of the CPU stall is not established.

CPU postprocessing now runs in a separate process on CPU 4 with ordinary
scheduling. A private bounded nonblocking Unix socket carries only the compact
64 KiB output and its originating frame/transform; no camera image is uploaded
again. The GPU owner proceeds to the next primary frame immediately after this
handoff. A full socket skips optional delivery, and the CPU worker drains to the
newest queued result and rejects anything older than 250 ms. It owns the
`carrotYolo` publisher, performs NMS/projection and rechecks the session lease.
It neither imports tinygrad nor opens a GPU device. Parent socket closure ends
the worker. `executionTime` measures the result pipeline through CPU completion;
`submitTime` and `readbackTime` isolate the work retained on the primary owner.

## Completed stationary comparison with the initial 5 Hz cap

The CPU-separated implementation at `169aca698a` completed a 30-second disabled
baseline, 180 seconds enabled, and a 30-second disabled recovery in the same
stationary scene. The GPU/CPU workers stayed loaded throughout these phases.
All primary and camera publications were inspected with non-conflating sockets.
There were 703 YOLO results, 3.91 Hz, with zero admission overruns and zero
reported driving frame drops. Of those results, 688 contained 690 distinct
detections; 15 had no detection above threshold. This verifies
live input changes, not general object-detection accuracy.

| Live YOLO phase, 703 results | p50 ms | p95 ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: |
| Submission and GPU completion/readback, measured together | 4.840 | 5.060 | 5.240 | 5.593 |
| CPU postprocessing on CPU 4 | 0.756 | 5.305 | 8.196 | 13.183 |
| Full result pipeline through CPU completion | 9.281 | 15.260 | 19.692 | 25.702 |
| Camera EOF through result publication | 92.952 | 101.855 | 105.840 | 110.435 |

The result pipeline includes nonblocking delivery and worker scheduling as well
as compute. It does not hold the driving owner while CPU postprocessing runs.
Submission and readback medians individually were 2.658 and 2.171 ms. The runtime
reservation increased from 7.501 to 8.089 ms during the trial, retaining its 1.2
margin and 1 ms guard. CPU postprocessing can still be delayed, but its 13.183 ms
maximum did not create an admission overrun on the primary owner.

| Driving model | Frames | p50 ms | p99 ms | Maximum ms | Drop percentage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disabled baseline, 30 s | 600 | 36.467 | 38.338 | 39.036 | 0 |
| Enabled, 180 s | 3,601 | 36.183 | 38.232 | 40.912 | 0 |
| Disabled recovery, 30 s | 600 | 36.175 | 38.214 | 40.078 | 0 |

Each camera delivered 3,600 or 3,601 frames in the enabled window; the one-frame
count difference is at the window boundary. The slight median/p99 differences
do not establish an improvement in driving inference: phase lengths differ and
this is a single sequential scene, not a randomized thermal comparison. USB
disconnect/fault injection and moving-vehicle validation have not been performed.
The maximum primary publication gap was 85.428 ms enabled, versus 74.082 ms in
baseline and 71.374 ms in recovery; all three phases had maximum primary frame-ID
delta one. The enabled cameras' maximum capture gaps were below 68.665 ms.
Device thermal status stayed green; the enabled phase ended with its hottest
reported CPU sensor at 57.9 C. These are device sensors, not external eGPU
temperature measurements.

After recovery, the manual supervisor resumed continuous stationary display.
The Web API returned a fresh actual detection. This verifies data delivery to
the Web backend, not visual rendering in a browser. The CPU worker was observed
under the modeld PID on CPU 4 with ordinary scheduling and no USB/DRI/KGSL device
descriptors. Only modeld owns the USB GPU. Lease/state/timing guards remain active;
this is not automatic activation on the next vehicle restart.

Evidence is retained in the local experiment cache as `live-reuse-final` with
the complete per-frame samples, phase report and analysis. The two excluded
attempts remain in `live-reuse-stale` and `live-reuse-bound`. The native adapter,
lease/admission, nonblocking delivery, CPU decoder, USB helpers and existing QCOM
checks pass 113 focused tests. Ruff passes for the changed code; two pre-existing
modeld findings (`ISC002`, `F841`) are excluded without modifying unrelated code.

## Per-frame admission follow-up

At the owner's request, `df63d1669b` removes the 200 ms interval from the
native-buffer runtime. It considers YOLO after every completed primary frame.
The older optional runtime retains its default interval. Camera prefetch,
deadline prediction, worst-observed cost multiplied by 1.2 plus 1 ms, fresh
Park/disabled state and the latched overrun stop remain unchanged. A regression
test covers consecutive-frame admission and rejection for pending cameras,
insufficient budget and overruns; the focused suite passes 114 tests.

The source-fingerprint change required rebuilding the artifact under exclusive
stationary maintenance. The new 5,000-run saved-input total measured 3.966 ms
p50 / 4.777 ms p99 / 5.431 ms maximum. Changed allocations, in-place pixels and
serialized rebinding passed again. Independent ONNX validation retained maximum
box error 0.000763 pixels and score error 0.000000715. Evidence is in
`results-reuse-every-frame`; the earlier 5 Hz measurements remain separate.

A 35 ms driving inference plus about 5 ms retained YOLO GPU work can fit inside
a 50 ms camera period. That arithmetic omits work outside driving inference and
camera-delivery jitter. Admission therefore considers the remaining deadline
budget, not just the displayed inference time. Removing the interval permits
consecutive frames but does not promise a YOLO result on all 20 frames/second.

The repeated stationary 30/180/30-second comparison passed with 2,091 results
in 180.012 seconds, **11.62 Hz**, versus the earlier 3.91 Hz. All 3,600 enabled
primary frames were observed, with zero reported drops and zero YOLO overruns.
Of the results, 2,060 contained 2,061 distinct detections and 31 were empty.
The reserved time grew from 7.517 to 8.223 ms with live observations; frames
without enough remaining time or with the next camera already pending still
skip optional work. Skipped reasons are not recorded for every frame, so this
trial does not quantify each rejection reason separately.

| Per-frame trial measurement | p50 ms | p95 ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: |
| YOLO submission + GPU completion/readback | 4.802 | 5.037 | 5.172 | 5.687 |
| Full YOLO result pipeline, including CPU worker | 8.730 | 16.064 | 21.180 | 28.697 |
| CPU postprocessing, outside primary owner | 0.787 | 5.230 | 9.957 | 17.375 |
| Driving inference, baseline | 36.685 | 37.796 | 38.446 | 38.816 |
| Driving inference, enabled | 36.279 | 37.489 | 38.228 | 43.382 |
| Driving inference, recovery | 36.240 | 37.527 | 38.210 | 41.709 |

Baseline and recovery each received 600 primary and 600 messages per camera.
The enabled camera counts were 3,600 or 3,601 at the window boundaries. Maximum
primary publication gaps were 75.453 / 81.340 / 67.571 ms for baseline / enabled
/ recovery; all maximum primary frame-ID deltas were one. Enabled camera gaps
were below 67.736 ms. Camera EOF to YOLO publication measured 94.206 ms p50 /
108.195 ms p99 / 118.297 ms maximum. Thermal status stayed green and the enabled
phase's final hottest CPU reading was 58.2 C. These remain sequential stationary
measurements, not moving-vehicle or USB-fault validation.

Complete samples and analysis are retained in `live-reuse-every-frame`. The
supervisor then resumed continuous stationary display with per-frame admission.
The CPU output worker was again observed on CPU 4 with ordinary scheduling and
no GPU device descriptors. The badge's milliseconds describe one result's
pipeline latency, whereas its count is cumulative completed GPU runs; neither
number alone specifies the result rate.

The Web overlay keeps a previous valid result for at most 350 ms when frames
are skipped. A new valid result with no detections replaces its boxes immediately;
confidence threshold crossings can therefore flicker without any skipped
inference. Expiry, camera-time mismatch and fetch failures can also hide boxes.
The browser schedules its next fetch 150 ms after a response, so it does not
display every result at the measured 11.62 Hz. These code paths identify possible
causes, not a synchronized diagnosis of the owner's observed screen.
The fixed input shape keeps neural-network work independent of how many objects
are visible. CPU candidate filtering/NMS can cost more with more candidates;
the decoder caps this at 200 candidates and 40 final detections. Dense-scene
timing was not measured in this stationary comparison.

## Replay and transport optimization follow-up

After the per-frame trial, the owner requested increasing the remaining frame
budget through optimization. A separate stationary maintenance comparison at
`9fd1ec2a0e` measured 5,000 replays each with FP32 and FP16 output. The replay
profiler runs afterward; its instrumented timings are excluded from latency
samples. Both variants captured one graph, with no JIT input-buffer copy.
Evidence is retained in `results-profile-before`.

The original FP32 submission plus completion/readback measured 3.554 ms p50 /
4.270 ms p99 / 4.920 ms maximum. FP16 transport measured 3.420 / 4.097 / 4.358 ms,
but direct half-precision CPU decoding made the full pipeline slower at the
median (4.058 versus 4.024 ms) and included an 18.313 ms total outlier. This does
not establish the cause of that CPU delay or validate live FP16 activation.

A proposed batched indirect-command write was withdrawn before device deployment.
The existing compute-ring cache already skips identical words. Unconditional
bulk writes would bypass that cache and could leave its view inconsistent with
the shared hardware ring. The final runtime keeps the existing command writes,
ordering, timeline waits and readback dependencies.

Instead, `84d738b11a` caches the arguments of an already validated YOLO TinyJit
replay while both the captured graph and exact immutable input UOp are unchanged.
New allocations/views go through the normal JIT validation. This caches input
metadata, not image pixels or inference results. CPU tests cover in-place pixel
changes, new allocations, invalid views, reset and serialization; the cache is
not serialized. JIT-disabled and nested-capture paths retain normal behavior.
This is local to the YOLO helper; the driving TinyJit path is unchanged.

The CPU worker also accepts compact FP16 transport and converts it to FP32 before
filtering/NMS, outside the primary owner. The benchmark uses the same conversion.
FP16 reduces the 2,688-anchor packet from 64,512 to 32,256 bytes without removing
candidates. It quantizes coordinates/scores, so independent numerical checks
still apply. No extra camera-image upload or resize is introduced.

The repeated maintenance benchmark selected native FP16 after 5,000 samples
per variant, using the same maximum-total-then-p99 criterion. Complete evidence
is in `results-replay-optimized`.

| Saved-input measurement | p50 ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: |
| Original FP32 GPU submission + completion/readback | 3.554 | 4.270 | 4.920 |
| Cached replay, FP32 GPU submission + completion/readback | 3.263 | 3.918 | 4.208 |
| Cached replay, FP16 GPU submission + completion/readback | 3.126 | 3.788 | 4.039 |
| Cached replay, FP16 total including CPU conversion/NMS | 3.582 | 4.253 | 4.541 |

Changed allocations, in-place image mutation and serialized replay accepted the
new inputs. Independent ONNX validation measured maximum FP16 box error 0.12497
pixels and score error 0.000216; all ten relevant anchor classes agreed. The saved
bicycle score was 0.8408203 versus the reference's 0.8407594. Both variants still
used one graph with no JIT input copy. The focused suite passes 117 tests and
Ruff passes all changed Python files.

### Live optimized comparison

The same guarded 30/180/30-second procedure passed at `84d738b11a` with native
FP16 transport. The enabled window produced 2,037 results in 180.014 seconds,
11.32 Hz, and observed 3,600 messages from the driving model and each camera.
There were zero YOLO overruns and zero reported driving drops. This improved
per-run cost, but did **not** demonstrate a result-rate increase over the earlier
11.62 Hz trial. These are different sequential stationary windows, not a
controlled interleaved comparison of camera-delivery jitter or thermal state.

| Live measurement | Earlier FP32 p50 / p99 / max ms | Optimized p50 / p99 / max ms |
| --- | ---: | ---: |
| Submission + GPU completion/readback | 4.802 / 5.172 / 5.687 | 4.349 / 4.699 / 4.985 |
| Full result pipeline | 8.730 / 21.180 / 28.697 | 7.941 / 22.694 / 70.179 |
| CPU postprocessing, outside primary | 0.787 / 9.957 / 17.375 | 0.780 / 8.297 / 59.008 |

For admitted frames, the median deadline budget before YOLO was 9.513 ms.
Subtracting submission and readback left 5.201 ms median versus 4.778 ms earlier;
this excludes the short nonblocking output handoff and is not a hard future
bound. The reservation grew from 6.450 to 7.384 ms, versus 8.223 ms at the end of
the earlier trial. Median headroom above the full reservation was 2.133 ms versus
1.455 ms earlier. Guard multipliers and deadlines were not relaxed.

One CPU postprocessing wall-time sample was 59.008 ms (result latency 70.179 ms).
Its GPU submission/readback cost was only 4.368 ms. The corresponding driving
frame and its neighbors had zero drops and about 49 ms publication gaps. Twelve
results exceeded 30 ms overall; several involved delivery/scheduling wait rather
than NMS. The exact cause of these CPU delays is not isolated, so this change
must not be described as improving the full result pipeline's worst latency.

Driving execution p50/p99/max was 36.418/37.989/38.515 ms baseline,
36.263/38.129/42.690 ms enabled, and 36.048/37.597/38.076 ms recovery. Maximum
primary publication gaps were 108.238/77.510/70.544 ms, with frame-ID delta one
throughout. Baseline counts ranged from 598 to 600 (599 primary); its road-camera
maximum capture gap was 97.146 ms. Enabled camera gaps were below 68.929 ms.
Recovery received 600 messages on each primary/camera stream. Camera EOF to
YOLO publication measured 92.345/110.605/152.951 ms p50/p99/max. Thermal status
stayed green; the enabled phase's final hottest CPU reading was 55.6 C.

Of 2,037 results, 2,007 contained 2,012 distinct detections and 30 were empty.
There were 1,074 consecutive-frame result pairs. The CPU worker again ran on
CPU 4 with ordinary scheduling and no GPU device descriptors. The supervisor
resumed continuous stationary display. Complete samples and analysis remain in
`live-reuse-optimized`; previous reports are retained separately.

### Owner-disabled ONNX lane/BSD service

After this trial, the owner reported disabling ONNX lane/BSD recognition and
observing smoother YOLO results. The setting is `ShareData`; its observed raw
value was `0`, modification time 1788782491.012, and managerState confirmed
`xiaoge_data` stopped. The completed 60-second display window after that change
contained 839 YOLO results and 1,200 primary frames (approximately 14.0 Hz over
the 59.951-second first-to-last primary span), with zero drops or overruns.

YOLO retained GPU cost measured 4.254/4.573/4.671 ms p50/p99/max, full result
latency 6.469/13.176/15.519 ms, and CPU postprocessing 0.715/4.992/8.503 ms.
Driving inference measured 35.977/37.514/41.085 ms. Median remaining budget after
YOLO GPU work was 5.669 ms. Evidence is retained as
`live_reuse_sharedata_off_samples.json` and `live_reuse_sharedata_off_summary.json`
inside `live-reuse-optimized`.

The separate lane/BSD service runs OpenCV DNN on CPU, with vision threads pinned
to cores 0-3 and two OpenCV threads; the YOLO output worker is on CPU 4. Reduced
background CPU/memory activity is a plausible explanation for the improvement,
not proof of direct contention on CPU 4 or the eGPU. The earlier phase did not
record this setting continuously, and these are unequal sequential windows,
not repeated controlled ON/OFF trials. Do not attribute the earlier CPU outlier
to this service as an established cause. No setting was changed by the agent.

## Recovery and validation limits

The first recovery hit an operational issue: `restart.sh` begins with `git pull`,
and the new device branch did not yet have an upstream. The normal manager had
already been stopped, so that failure prevented restoration. Recovery then
verified no existing manager or GPU owner, ran the normal restart procedure
without the Git update step, and kept the exact measured checkout. A completed
20-second recovery sample received approximately 20 Hz from all three cameras
and modelV2; the final model sample reported zero drops and 35.62 ms execution.
The camera-only early startup sample was not treated as full recovery.
Subsequent maintenance uses this local recovery procedure without depending
on Git tracking or network availability. Always verify actual streams afterward.

The original full-camera path's bottleneck is moving the padded buffer. The USB
allocator uses a 256 KiB staging buffer, so that input requires ten chunks;
the measured upload phase includes host/USB transactions and staging waits,
not merely wire transfer time. Simply counting 2.808 ms of kernels would omit
most of the required work.

The full-camera representation is now a secondary candidate. Fixed-input
timings do not establish sustained same-scene baseline/enabled tail latency,
camera-to-publication age, USB fault behavior or thermal behavior alongside
driving. In-flight GPU work cannot be preempted by the admission estimate;
measured maximum cost is not a hard future bound.
Radar association and later perception stages remain downstream work.

## Reproduction and retained evidence

`openpilot/tools/egpu_yolo/benchmark_saved.py` takes a directory containing the
verified `model.onnx` and `manifest.json`, and a saved NV12 frame. Run it only
under an exclusive maintenance coordinator that continuously guards Park and
stopped cameras/controls and restores the normal manager on exit:

```sh
python -m openpilot.tools.egpu_yolo.benchmark_saved \
  --directory /data/egpu_yolo/egpu2-640 \
  --input-nv12 /data/egpu_yolo/bicycle_sample.npy \
  --samples 5000 --paced-samples 300 --interval 0.2 --stationary-maintenance
```

The result directory contains `benchmark_report.json` with all timing samples,
`benchmark_validation.npz` and `benchmark.pkl`. The artifact name cannot activate
either YOLO runtime. The model is a direct-copy benchmark candidate; its NAS URL
is not yet published, and the existing 512 x 256 download/production restrictions
remain unchanged.

On the workstation, validate the copied report/output against the same frame:

```sh
python -m openpilot.tools.egpu_yolo.validate_saved \
  --directory <copied-result-directory> --model <checksum-matched-model.onnx> \
  --input-nv12 <same-saved-frame.npy>
```

The validator checks model/input hashes before comparing values and retains
`numerical_validation.json`. Its comparison thresholds are fixed in code.
Camera frames and raw validation arrays remain in the local experiment cache;
they are not repository assets. No setting, public user guide or vehicle-control
consumer is changed by this benchmark work.

The native-queue benchmark uses a saved `(6,128,256)` uint8 packed frame and the
unchanged 512 x 256 ONNX/manifest under the same exclusive maintenance guard:

```sh
python -m openpilot.tools.egpu_yolo.benchmark_reuse \
  --directory /data/egpu_yolo/egpu2-reuse \
  --input-packed /data/egpu_yolo/reuse_current_packed.npy \
  --variants native_fp32 --samples 5000 --stationary-maintenance
```

It retains `reuse_report.json`, validation arrays and the compiled
`yolo_reuse.pkl`. On the workstation, `openpilot.tools.egpu_yolo.validate_reuse`
takes `--directory`, `--input-packed` and `--model`; it checks source hashes,
reconstructs packed YUV independently with NumPy and compares all boxes/scores
and relevant classes against ONNX Runtime.
