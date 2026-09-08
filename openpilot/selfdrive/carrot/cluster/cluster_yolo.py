"""Read-only, expiring YOLO summary for the yolo2 experimental cluster."""
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class YoloDisplay:
    state: str
    runs: int = 0
    execution_ms: float | None = None
    objects: tuple[tuple[str, int, int], ...] = ()
    more: int = 0


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def build_yolo_display(message, *, enabled, valid, transport_age, image_age):
    if message is None:
        return YoloDisplay("waiting") if enabled else None
    runs = max(0, int(_get(message, "runs", 0)))
    if not enabled:
        return YoloDisplay("off", runs)
    if not math.isfinite(transport_age) or not 0 <= transport_age <= 2.0:
        return YoloDisplay("stale", runs)
    state = str(_get(message, "state", "waiting"))
    # Paused/error messages are deliberately invalid as detection data, but
    # their current status must remain visible and clear previous objects.
    if state in ("paused", "error", "overrun"):
        return YoloDisplay(state, runs)
    if state != "run":
        return YoloDisplay("waiting", runs)
    if not valid:
        return YoloDisplay("invalid", runs)
    if not math.isfinite(image_age) or not 0 <= image_age <= .35:
        return YoloDisplay("stale", runs)
    seconds = float(_get(message, "executionTime", 0))
    execution_ms = seconds*1000 if math.isfinite(seconds) and seconds > 0 else None
    grouped = {}
    for detection in list(_get(message, "detections", ()))[:40]:
        score = float(_get(detection, "confidence", 0))
        if not math.isfinite(score) or not 0 <= score <= 1:
            continue
        label = str(_get(detection, "label", "object"))[:32]
        count, best = grouped.get(label, (0, 0))
        grouped[label] = (count+1, max(best, round(score*100)))
    groups = sorted(((label, count, best) for label, (count, best) in grouped.items()), key=lambda x: (-x[2], x[0]))
    return YoloDisplay("run", runs, execution_ms, tuple(groups[:3]), sum(g[1] for g in groups[3:]))


_KO_LABELS = {"person": "사람", "bicycle": "자전거", "car": "자동차", "motorcycle": "오토바이",
              "bus": "버스", "truck": "트럭", "traffic light": "신호등", "stop sign": "정지표지",
              "chair": "의자", "dog": "개", "cat": "고양이", "potted plant": "화분"}


def yolo_text(display, language):
    korean = language == "ko"
    states = {"run": ("실행", "RUN"), "paused": ("일시정지", "PAUSED"), "error": ("오류", "ERROR"),
              "overrun": ("시간초과", "OVERRUN"), "waiting": ("준비대기", "WAITING"),
              "stale": ("갱신대기", "STALE"), "invalid": ("수신오류", "INVALID"), "off": ("꺼짐", "OFF")}
    title = f"YOLO {states.get(display.state, states['waiting'])[0 if korean else 1]}"
    if display.execution_ms is not None:
        title += f" {display.execution_ms:.1f}ms"
    title += f" · #{display.runs}"
    if display.state != "run":
        return title, "검출 표시 대기" if korean else "Detections unavailable"
    details = [f"{_KO_LABELS.get(label, label) if korean else label} {count} ({best}%)"
               for label, count, best in display.objects]
    if display.more:
        details.append(f"+{display.more}")
    return title, " · ".join(details) if details else ("검출 없음" if korean else "No detections")
