import test from "node:test";
import assert from "node:assert/strict";
import { projectCameraBox, selectDetectionFrame } from "../src/features/drive/contents/vision/egpu_yolo.js";

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
