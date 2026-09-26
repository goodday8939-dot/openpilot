"""Display-only Xiaoge state; unknown or unevaluated sides are never shown clear."""

from dataclasses import dataclass
import json
import math

from openpilot.selfdrive.carrot.xiaoge.xiaoge_vision import (
  XIAOGE_BLINDSPOT_TIMEOUT_NS, XIAOGE_LANE_TIMEOUT_NS, XiaogeVisionResult, parse_xiaoge_vision_payload,
)


XIAOGE_OBJECT_TIMEOUT_NS = 1_500_000_000


@dataclass(frozen=True)
class VisionObject:
  side: str
  track_id: int
  classification: str
  confidence: float
  bbox: tuple[float, float, float, float]
  received_nanos: int


@dataclass(frozen=True)
class VisionDisplayPacket:
  result: XiaogeVisionResult
  blindspot_side: str = ""
  latency_ms: float | None = None
  objects: tuple[VisionObject, ...] = ()


@dataclass(frozen=True)
class VisionDisplayState:
  state: str = "waiting"
  left_lane: int = -1
  right_lane: int = -1
  clear_side: str = ""
  latency_ms: float | None = None


def parse_vision_display_packet(payload: bytes) -> VisionDisplayPacket:
  result = parse_xiaoge_vision_payload(payload)
  data = json.loads(payload)
  side = data["blindspot"].get("side", "")
  side = side if side in ("left", "right") else ""
  latency = data["lane"].get("latencyMs")
  if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
    latency = None

  parsed_objects = []
  objects = data.get("objects")

  if isinstance(objects, dict):
    for object_side in ("left", "right"):
      raw_items = objects.get(object_side, [])
      received = objects.get(
        f"{object_side}UpdatedMonoTimeNanos",
        0,
      )

      if isinstance(received, bool) or not isinstance(received, int):
        received = 0

      if not isinstance(raw_items, list):
        continue

      for item in raw_items[:12]:
        if not isinstance(item, dict):
          continue

        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
          continue

        try:
          coords = tuple(float(v) for v in bbox)
        except (TypeError, ValueError):
          continue

        if not all(math.isfinite(v) for v in coords):
          continue

        confidence = item.get("confidence", 0.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
          confidence = 0.0

        track_id = item.get("trackId", -1)
        if isinstance(track_id, bool) or not isinstance(track_id, int):
          track_id = -1

        classification = item.get("classification", "UNKNOWN")
        if not isinstance(classification, str):
          classification = "UNKNOWN"

        parsed_objects.append(
          VisionObject(
            side=object_side,
            track_id=track_id,
            classification=classification,
            confidence=float(confidence),
            bbox=coords,
            received_nanos=received,
          )
        )

  return VisionDisplayPacket(
    result,
    side,
    latency,
    tuple(parsed_objects),
  )


def vision_display_state(packet: VisionDisplayPacket | None, now_nanos: int) -> VisionDisplayState:
  if packet is None:
    return VisionDisplayState()
  result = packet.result
  lane_age = now_nanos - result.lane_received_nanos
  lane_fresh = result.lane_valid and result.lane_received_nanos > 0 and 0 <= lane_age <= XIAOGE_LANE_TIMEOUT_NS
  blindspot_age = now_nanos - result.blindspot_received_nanos
  blindspot_fresh = (result.blindspot_valid and result.blindspot_received_nanos > 0 and
                    0 <= blindspot_age <= XIAOGE_BLINDSPOT_TIMEOUT_NS)
  side_detected = result.left_blindspot if packet.blindspot_side == "left" else result.right_blindspot
  return VisionDisplayState(
    state="running" if lane_fresh else "stale",
    left_lane=result.left_lane if lane_fresh else -1,
    right_lane=result.right_lane if lane_fresh else -1,
    clear_side=packet.blindspot_side if blindspot_fresh and not side_detected else "",
    latency_ms=packet.latency_ms if lane_fresh else None,
  )
