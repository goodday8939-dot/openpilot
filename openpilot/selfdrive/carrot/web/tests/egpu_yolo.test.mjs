import test from "node:test";
import assert from "node:assert/strict";
import { detectionLabel, radarStatusLabel, projectCameraBox, selectDetectionFrame } from "../src/features/drive/contents/vision/egpu_yolo.js";

test("visual IDs and provisional radar labels never invent unmatched distance", () => {
  const detection = { trackId: 7, label: "bicycle", confidence: .8 };
  assert.equal(detectionLabel(detection), "#7 bicycle 80%");
  const radar = { state: "candidate", source: "frontRadar", trackId: 12, dRel: 20, vRel: -2, ageSeconds: .05 };
  assert.equal(detectionLabel({ ...detection, radar }), "#7 bicycle 80% · R?12 20.0m -2.0m/s");
  assert.equal(detectionLabel({ ...detection, radar: { ...radar, source: "corner235" } }), "#7 bicycle 80%");
  assert.equal(detectionLabel({ ...detection, radar: { ...radar, ageSeconds: .2 } }), "#7 bicycle 80%");
  assert.equal(radarStatusLabel({ state: "no_front_points", sources: { corner235: 1 } }, (_, fallback) => fallback), "Radar: corner only");
});

test("camera projection follows the same viewport crop and scale as live video", () => {
  const stage = { videoWidth: 1000, videoHeight: 500, scale: .5, tx: -100, ty: 20 };
  const box = { cameraPoints: [.2, .1, .4, .1, .4, .5, .2, .5] };
  assert.deepEqual(projectCameraBox(box, stage), [[0,45],[100,45],[100,145],[0,145]]);
  assert.equal(projectCameraBox({ cameraPoints: [NaN] }, stage), null);
});

test("live overlay rejects future, old, wrong-camera, and replay detections", () => {
  const frame = { camera: "road", timestampEof: 1e9, frameId: 10, ageAtReceive: .01, receivedAt: 100 };
  const presented = { source: "live", cameraTimestampEof: 1.1e9 };
  assert.equal(selectDetectionFrame([frame], presented, 110), frame);
  assert.equal(selectDetectionFrame([frame], presented, 600), null);
  assert.equal(selectDetectionFrame([frame], { ...presented, source: "replay" }, 110), null);
  assert.equal(selectDetectionFrame([frame], { ...presented, cameraTimestampEof: .9e9 }, 110), null);
  assert.equal(selectDetectionFrame([{ ...frame, camera: "wideRoad" }], presented, 110), null);
  assert.equal(selectDetectionFrame([frame], { ...presented, cameraTimestampEof: null, frameId: 18 }, 110), null);
});
