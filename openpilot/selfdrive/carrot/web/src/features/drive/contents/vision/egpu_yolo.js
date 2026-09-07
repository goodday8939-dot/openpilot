// Experimental read-only detections on the existing live camera surface.
// Reuse its presented-frame notifications and published viewport transform.
const MAX_AGE_SECONDS = 0.35;
const SVG_NS = "http://www.w3.org/2000/svg";

export function detectionLabel(detection) {
  const identity = Number.isSafeInteger(detection.trackId) ? `#${detection.trackId} ` : "";
  const base = `${identity}${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
  const radar = detection.radar;
  if (radar?.state !== "candidate" || radar.source !== "frontRadar"
    || ![radar.dRel, radar.vRel, radar.ageSeconds].every(Number.isFinite) || Math.abs(radar.ageSeconds) > .12) return base;
  return `${base} · R?${radar.trackId} ${radar.dRel.toFixed(1)}m ${radar.vRel >= 0 ? "+" : ""}${radar.vRel.toFixed(1)}m/s`;
}

export function radarStatusLabel(status, tr) {
  if (!status) return "";
  if (status.state === "no_front_points") {
    return Object.keys(status.sources || {}).some(source => source.startsWith("corner"))
      ? tr("radar_corner_only", "Radar: corner only") : tr("radar_no_front", "Radar: no front objects");
  }
  return status.state === "candidate_only" ? tr("radar_candidate", "Radar: candidates only")
    : tr("radar_unavailable", "Radar: unavailable");
}

export function selectDetectionFrame(history, presented, now) {
  if (presented?.source !== "live") return null;
  const timestamp = presented.cameraTimestampEof;
  return [...history].reverse().find(frame => {
    const age = frame.ageAtReceive + (now - frame.receivedAt) / 1000;
    if (frame.camera !== "road" || !Number.isFinite(age) || age < 0 || age > MAX_AGE_SECONDS) return false;
    if (Number.isFinite(timestamp) && timestamp > 0) {
      const delta = (timestamp - frame.timestampEof) / 1e9;
      return delta >= -0.01 && delta <= MAX_AGE_SECONDS;
    }
    if (Number.isFinite(presented.frameId)) {
      const delta = presented.frameId - frame.frameId;
      return delta >= 0 && delta <= 7;
    }
    return true; // Unmapped browser: hold latest only within the receipt-age limit.
  }) || null;
}

export function projectCameraBox(detection, stage) {
  const points = detection.cameraPoints;
  if (!Array.isArray(points) || points.length !== 8 || !points.every(Number.isFinite)
    || !stage || ![stage.videoWidth, stage.videoHeight, stage.scale, stage.tx, stage.ty].every(Number.isFinite)
    || stage.videoWidth <= 0 || stage.videoHeight <= 0 || stage.scale <= 0) return null;
  return points.reduce((result, value, i) => {
    if (i % 2 === 0) result.push([value * stage.videoWidth * stage.scale + stage.tx,
      points[i + 1] * stage.videoHeight * stage.scale + stage.ty]);
    return result;
  }, []);
}

export function installEgpuYoloOverlay(target = globalThis) {
  const document = target.document;
  const host = document?.getElementById("carrotStage");
  const channel = target.DriveVisionPresentedFrames;
  if (!host || !channel || target.CarrotEgpuYolo) return;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.id = "carrotYoloOverlay";
  svg.setAttribute("aria-label", "YOLO detections");
  svg.style.cssText = "position:absolute;pointer-events:none;z-index:7;overflow:hidden;display:none";
  host.appendChild(svg);
  const badge = document.createElement("div");
  badge.id = "carrotYoloStatus";
  badge.style.cssText = "position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:8;pointer-events:none;background:#10251ed9;color:#a5ffda;border:1px solid #4d9074;border-radius:12px;padding:4px 10px;font:12px system-ui;display:none";
  host.appendChild(badge);
  const history = [];
  let lastPayload = null;
  let signature = "";
  let disposed = false;
  let timer;
  const visible = () => !document.hidden && host.getClientRects().length > 0;

  function clear() {
    svg.style.display = "none";
    signature = "";
  }

  function draw(presented) {
    if (!visible() || presented?.source !== "live") { clear(); badge.style.display = "none"; return; }
    const state = lastPayload?.status || {};
    const recovering = lastPayload?.supervisor?.stage === "cooldown";
    const stopped = recovering || lastPayload?.supervisor?.stage === "stopped"
      || ["error", "overrun", "stopped", "dm_active", "egpu_wait", "offroad"].includes(state.state);
    const tr = (key, fallback) => target.getUIText?.(`egpu_yolo_${key}`, fallback) || fallback;
    badge.textContent = ["run", "no_budget", "camera_pending"].includes(state.state)
      ? `YOLO · ${((state.executionTime || 0) * 1000).toFixed(1)} ms · ${state.runs || 0}`
      : `YOLO · ${tr(state.state || "waiting", "Waiting")}`;
    if (recovering) badge.textContent = `YOLO · ${tr("recovering", "Waiting for stable timing")}`;
    badge.style.display = lastPayload ? "block" : "none";
    const frame = stopped ? null : selectDetectionFrame(history, presented, target.performance.now());
    if (frame?.radarStatus) badge.textContent += ` · ${radarStatusLabel(frame.radarStatus, tr)}`;
    const stage = target.CarrotVisionStageTransform;
    if (!frame || !stage || !(stage.stageWidth > 0 && stage.stageHeight > 0)) { clear(); return; }
    const nextSignature = [frame.timestampEof, stage.scale, stage.tx, stage.ty, stage.stageWidth, stage.stageHeight,
      stage.viewportLeft, stage.viewportTop, stage.videoWidth, stage.videoHeight].join(":");
    if (signature === nextSignature) return;
    svg.style.left = `${stage.viewportLeft || 0}px`;
    svg.style.top = `${stage.viewportTop || 0}px`;
    svg.style.width = `${stage.stageWidth}px`;
    svg.style.height = `${stage.stageHeight}px`;
    svg.setAttribute("viewBox", `0 0 ${stage.stageWidth} ${stage.stageHeight}`);
    const nodes = [];
    for (const detection of (frame.detections || []).slice(0, 40)) {
      const points = projectCameraBox(detection, stage);
      if (!points || !Number.isFinite(detection.confidence)) continue;
      const polygon = document.createElementNS(SVG_NS, "polygon");
      polygon.setAttribute("points", points.map(p => p.join(",")).join(" "));
      polygon.setAttribute("fill", "none");
      polygon.setAttribute("stroke", "#73f0b7");
      polygon.setAttribute("stroke-width", "2");
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", String(Math.max(3, Math.min(stage.stageWidth - 110, points[0][0]))));
      label.setAttribute("y", String(Math.max(16, Math.min(stage.stageHeight - 4, points[0][1] - 5))));
      label.setAttribute("fill", "#a5ffda");
      label.setAttribute("stroke", "#07110c");
      label.setAttribute("stroke-width", "3");
      label.setAttribute("paint-order", "stroke");
      label.setAttribute("font-size", "13");
      label.setAttribute("font-family", "system-ui");
      label.textContent = detectionLabel(detection);
      nodes.push(polygon, label);
    }
    svg.replaceChildren(...nodes);
    svg.style.display = "block";
    signature = nextSignature;
  }

  async function refresh() {
    try {
      if (visible()) {
        const response = await target.fetch("/api/egpu/yolo", { cache: "no-store", signal: AbortSignal.timeout(1500) });
        if (!response.ok) throw new Error(String(response.status));
        lastPayload = await response.json();
        const frame = lastPayload.frame;
        if (frame && !history.some(f => f.timestampEof === frame.timestampEof)) {
          history.push({ ...frame, receivedAt: target.performance.now(), ageAtReceive: Number(lastPayload.ageSeconds) });
          if (history.length > 8) history.shift();
        }
        if (lastPayload.stale || ["error", "overrun"].includes(lastPayload.status?.state)) clear();
      } else { clear(); badge.style.display = "none"; }
    } catch (_) {
      history.length = 0;
      clear();
      badge.style.display = "none";
    } finally {
      if (!disposed) timer = target.setTimeout(refresh, visible() ? 150 : 1000);
    }
  }
  const unsubscribe = channel.subscribe(draw);
  const hide = () => { clear(); badge.style.display = "none"; };
  document.addEventListener("visibilitychange", hide);
  target.addEventListener("carrot:visionchange", hide);
  target.CarrotEgpuYolo = Object.freeze({
    status: () => lastPayload,
    destroy() {
      disposed = true; target.clearTimeout(timer); unsubscribe();
      document.removeEventListener("visibilitychange", hide);
      target.removeEventListener("carrot:visionchange", hide);
      svg.remove(); badge.remove();
    },
  });
  void refresh();
}
