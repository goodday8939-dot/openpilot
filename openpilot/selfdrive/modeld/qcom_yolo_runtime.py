"""Cooperative execution for the fixed internal-GPU YOLO experiment only."""
from functools import lru_cache
import math
import time

_admission_check = None


class YoloInterrupted(RuntimeError):
  """The current frame lost admission between GPU batches."""


def partition_costs(costs, budget_ms=8.):
  if not math.isfinite(budget_ms) or budget_ms <= 0:
    raise ValueError('positive finite batch budget required')
  groups, group, total = [], [], 0.
  for index, cost in enumerate(costs):
    if not math.isfinite(cost) or cost < 0:
      raise ValueError('finite nonnegative kernel timings required')
    if group and total + cost > budget_ms:
      groups.append(group)
      group, total = [], 0.
    group.append(index)
    total += cost
  if group:
    groups.append(group)
  return groups


@lru_cache(maxsize=1)
def cooperative_graph_type():
  from tinygrad.runtime.graph.hcq import HCQGraph

  class CooperativeQcomGraph(HCQGraph):
    single_queue = True

    def __call__(self, *args, **kwargs):
      if _admission_check is not None and not _admission_check():
        raise YoloInterrupted('optional frame admission changed')
      result = super().__call__(*args, **kwargs)
      self.devices[0].synchronize()
      time.sleep(.0002)
      return result

  return CooperativeQcomGraph


def configure_runtime(admission_check=None):
  """Call only in the separate YOLO process, after selecting QCOM:IR3."""
  from tinygrad.device import Device
  global _admission_check
  _admission_check = admission_check
  Device['QCOM'].graph = cooperative_graph_type()


def is_prepared(run):
  from tinygrad.engine.realize import get_graph_runtime
  from tinygrad.uop.ops import Ops
  if run.captured is None:
    return False
  for call in run.captured.linear.src:
    ast = call.src[0]
    if ast.op != Ops.CUSTOM_FUNCTION or ast.arg != 'graph':
      return False
    runtime = get_graph_runtime(ast)
    if not isinstance(runtime, cooperative_graph_type()) or runtime.kickoff_value < 1:
      return False
  return bool(run.captured.linear.src)


def repartition(run, kernel_profile, budget_ms=8.):
  from tinygrad import TinyJit
  from tinygrad.engine.jit import CapturedJit, create_graph_call
  from tinygrad.uop.ops import Ops
  captured = run.captured
  if captured is None:
    raise ValueError('captured YOLO graph required')
  calls = []
  for call in captured.linear.src:
    if call.src[0].op == Ops.CUSTOM_FUNCTION and call.src[0].arg == 'graph':
      calls.extend(call.src[0].src[0].src)
    else:
      calls.append(call)
  if len(calls) != len(kernel_profile) or not all(call.op == Ops.CALL and call.src[0].op == Ops.PROGRAM for call in calls):
    raise ValueError('kernel profile does not match the compute-only graph')
  if any(call.src[0].arg.function_name != timing['name'] for call, timing in zip(calls, kernel_profile, strict=True)):
    raise ValueError('kernel profile order/name mismatch')
  groups = partition_costs([timing['ms'] for timing in kernel_profile], budget_ms)
  linear = captured.linear.replace(src=tuple(create_graph_call([calls[index] for index in group]) for group in groups))
  return TinyJit(None, CapturedJit(captured.ret, linear, captured.expected_names, captured.expected_input_info))
