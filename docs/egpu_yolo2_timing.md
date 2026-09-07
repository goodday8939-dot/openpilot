# Full-camera eGPU timing, 2026-09-07

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

## Recovery and next work

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

The present bottleneck is moving the original padded camera buffer. The USB
allocator uses a 256 KiB staging buffer, so that input requires ten chunks;
the measured upload phase includes host/USB transactions and staging waits,
not merely wire transfer time. Simply counting 2.808 ms of kernels would omit
most of the required work.

Next reuse the already-resident native driving image and measure preprocessing,
GPU submission, output transfer and NMS, following the owner's latency priority.
The full-camera representation is now a secondary candidate. Do not restore the
historical in-modeld execution path based only on this saved-input result.
Current-primary-frame admission, sustained same-scene
baseline/enabled tail latency, camera-to-publication age, USB fault behavior and
thermal state remain to be measured. Neither trial has concurrent driving
inference, a controlled thermal A/B, live acquisition or publication timing.
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
