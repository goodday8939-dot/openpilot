import numpy as np
import pytest

from openpilot.selfdrive.modeld.qcom_yolo_model import camera_transform, letterbox_geometry, local_size_for, nv12_to_rgb
from openpilot.selfdrive.modeld.qcom_yolod import configured, permit_reason


def test_backend_rejects_missing_or_modified_library_before_selecting_gpu(tmp_path, monkeypatch):
  import os
  from openpilot.selfdrive.modeld.qcom_yolo_model import configure_environment
  monkeypatch.setenv('DEV', 'CPU:X86')
  with pytest.raises(ValueError, match='verified'):
    configure_environment(tmp_path)
  (tmp_path / 'lib').mkdir()
  (tmp_path / 'lib' / 'libtinymesa.so').write_bytes(b'wrong library')
  with pytest.raises(ValueError, match='verified'):
    configure_environment(tmp_path)
  assert os.environ['DEV'] == 'CPU:X86'


def test_backend_pins_verified_library_without_changing_system_search_path(tmp_path, monkeypatch):
  import hashlib
  import os
  import openpilot.selfdrive.modeld.qcom_yolo_model as adapter
  (tmp_path / 'lib').mkdir()
  library = tmp_path / 'lib' / 'libtinymesa.so'
  library.write_bytes(b'test library')
  monkeypatch.setattr(adapter, 'MESA_SHA256', hashlib.sha256(library.read_bytes()).hexdigest())
  for key in ('DEV', 'WARP_DEV', 'MESA_PATH', 'IMAGE', 'FLOAT16', 'NOLOCALS', 'JIT_BATCH_SIZE', 'OPENPILOT_HACKS', 'QCOM_PRIORITY'):
    monkeypatch.setenv(key, os.environ.get(key, ''))
  monkeypatch.setenv('LD_LIBRARY_PATH', '/original')
  adapter.configure_environment(tmp_path)
  assert os.environ['DEV'] == 'QCOM:IR3'
  assert os.environ['MESA_PATH'] == str(library)
  assert os.environ['LD_LIBRARY_PATH'] == '/original'


@pytest.mark.parametrize('global_size', [(131072, 1, 1), (16, 8, 16), (3, 7, 11), (1, 1, 1), (12, 32, 64)])
def test_workgroups_fit_hardware_budget_and_divide_global_size(global_size):
  local = local_size_for(global_size)
  assert 1 <= np.prod(local) <= 64
  assert all(g % l == 0 for g, l in zip(global_size, local, strict=True))


def test_compiler_matcher_is_compatible_with_tinygrad_and_leaves_cpu_alone(monkeypatch):
  from tinygrad import Tensor
  from tinygrad.engine import realize
  from openpilot.selfdrive.modeld.qcom_yolo_model import configure_compiler
  monkeypatch.setattr(realize, 'pm_optimize_local_size', realize.pm_optimize_local_size)
  configure_compiler()
  np.testing.assert_array_equal((Tensor([1, 2, 3]) + 7).numpy(), [8, 9, 10])


@pytest.mark.parametrize('change,expected', [
  ({'onroad': False}, 'offroad'), ({'egpu_active': False}, 'egpu_wait'), ({'loading': True}, 'egpu_wait'),
  ({'dm_disabled': False}, 'dm_active'), ({'dm_running': True}, 'dm_active'),
  ({'model_alive': False}, 'warming'), ({'manager_alive': False}, 'warming'), ({}, 'run'),
])
def test_permit_requires_live_egpu_and_stopped_dm(change, expected):
  state = {'onroad': True, 'egpu_active': True, 'loading': False, 'dm_disabled': True, 'dm_running': False,
           'model_alive': True, 'manager_alive': True}
  assert permit_reason(**(state | change)) == expected


def test_internal_worker_stays_disabled_until_commissioned_and_fault_latches(tmp_path):
  class Params:
    values = {'UsbGpuActive': True, 'UsbGpuLoading': False, 'DisableDM': 2}
    def get_bool(self, key): return bool(self.values[key])
    def get_int(self, key): return int(self.values[key])
  params = Params()
  (tmp_path / 'yolo_qcom.pkl').touch()
  assert not configured(params, tmp_path)
  (tmp_path / 'qcom_enabled').touch()
  assert configured(params, tmp_path)
  for key, value in [('UsbGpuActive', False), ('UsbGpuLoading', True), ('DisableDM', 0), ('DisableDM', 7)]:
    original = params.values[key]
    params.values[key] = value
    assert not configured(params, tmp_path)
    params.values[key] = original
  (tmp_path / 'qcom_fault').touch()
  assert not configured(params, tmp_path)


def test_letterbox_retains_complete_camera_and_inverts_content_bounds():
  width, height, left, top = letterbox_geometry((1344, 760), (512, 256))
  assert (width, height, left, top) == (453, 256, 29, 0)
  points = np.array([[left, top, 1], [left + width, top + height, 1]])
  actual = points @ camera_transform((1344, 760), (512, 256)).T
  np.testing.assert_allclose(actual, [[0, 0, 1], [1344, 760, 1]], atol=1e-4)
  with pytest.raises(ValueError):
    letterbox_geometry((0, 760), (512, 256))


def test_nv12_padding_and_uv_offsets_do_not_contaminate_image():
  from tinygrad import Tensor
  # Image 4x4; padded stride 8 and luma height 6. Padding is deliberately bright.
  layout = (4, 4, 8, 6, 2, 64)
  frame = np.full(64, 255, dtype=np.uint8)
  frame[:48].reshape(6, 8)[:4, :4] = 64
  frame[48:].reshape(2, 8)[:, :4:2] = 128
  frame[48:].reshape(2, 8)[:, 1:4:2] = 128
  actual = nv12_to_rgb(Tensor(frame), layout, (8, 4)).numpy()[0]
  np.testing.assert_allclose(actual[:, :, :2], 114 / 255, atol=1e-6)
  np.testing.assert_allclose(actual[:, :, 6:], 114 / 255, atol=1e-6)
  y, uv = 64 / 255, 128 / 255 - .5
  expected = np.array([y + 1.402 * uv, y - .344 * uv - .714 * uv, y + 1.772 * uv])
  np.testing.assert_allclose(actual[:, :, 2:6], np.broadcast_to(expected[:, None, None], (3, 4, 4)), atol=1e-6)


def test_nv12_luma_interpolation_at_pixel_centers():
  from tinygrad import Tensor
  frame = np.concatenate((np.tile([0, 64, 128, 192], 4), np.full(8, 128))).astype(np.uint8)
  actual = nv12_to_rgb(Tensor(frame), (4, 4, 4, 4, 2, 24), (2, 2)).numpy()[0]
  expected_y = np.array([32, 160]) / 255
  np.testing.assert_allclose(actual[0], np.tile(expected_y + 1.402 * (128 / 255 - .5), (2, 1)), atol=1e-6)


def test_runner_compacts_class_scores_after_nv12_conversion():
  from types import SimpleNamespace
  from tinygrad import Tensor, dtypes
  from openpilot.selfdrive.modeld.qcom_yolo_model import make_runner

  class Runner:
    graph_inputs = {'images': SimpleNamespace(shape=(1, 3, 4, 4), dtype=dtypes.float32)}
    onnx_ops = {'Concat': lambda *xs, axis: Tensor.cat(*xs, dim=axis)}

    def __call__(self, inputs):
      box = Tensor([2, 2, 1, 1]).reshape(1, 4, 1)
      scores = Tensor([.1, .8]).reshape(1, 2, 1)
      return {'output': self.onnx_ops['Concat'](box, scores, axis=1) + inputs['images'].mean() * 0}

  raw = make_runner(Runner(), (4, 4, 4, 4, 2, 24), (4, 4))(Tensor(np.full(24, 128, dtype=np.uint8))).numpy()
  np.testing.assert_allclose(raw.flatten(), [2, 2, 1, 1, .8, 1], atol=1e-6)


@pytest.mark.parametrize('name', ['camerad', 'modeld', 'dmonitoringmodeld', 'dmonitoringd', 'qcom_yolod'])
def test_compilation_rejects_live_camera_and_inference_processes(name):
  from types import SimpleNamespace
  from openpilot.selfdrive.modeld.qcom_yolo_prepare import check_maintenance_state
  with pytest.raises(RuntimeError, match='stop camera'):
    check_maintenance_state([SimpleNamespace(name=name, running=True)], onroad=False, driver_preview=False)
  check_maintenance_state([SimpleNamespace(name=name, running=False)], onroad=False, driver_preview=False)


@pytest.mark.parametrize('onroad,preview', [(True, False), (False, True)])
def test_compilation_rejects_onroad_and_driver_preview_even_without_live_processes(onroad, preview):
  from openpilot.selfdrive.modeld.qcom_yolo_prepare import check_maintenance_state
  with pytest.raises(RuntimeError):
    check_maintenance_state([], onroad=onroad, driver_preview=preview)


def test_stationary_maintenance_requires_explicit_mode_and_park():
  from openpilot.selfdrive.modeld.qcom_yolo_prepare import check_maintenance_state
  for mode, parked in ((False, True), (True, False)):
    with pytest.raises(RuntimeError):
      check_maintenance_state([], onroad=True, driver_preview=False, stationary_maintenance=mode, parked=parked)
  check_maintenance_state([], onroad=True, driver_preview=False, stationary_maintenance=True, parked=True)


@pytest.mark.parametrize('name', ['camerad', 'modeld', 'dmonitoringmodeld', 'dmonitoringd', 'qcom_yolod',
                                  'controlsd', 'selfdrived', 'joystickd', 'maneuversd', 'lateral_maneuversd'])
def test_stationary_maintenance_never_allows_live_camera_inference_or_controls(name):
  from types import SimpleNamespace
  from openpilot.selfdrive.modeld.qcom_yolo_prepare import check_maintenance_state
  with pytest.raises(RuntimeError, match='stop camera'):
    check_maintenance_state([SimpleNamespace(name=name, running=True)], onroad=True, driver_preview=False,
                            stationary_maintenance=True, parked=True)


def test_runner_uses_winograd_for_unit_onnx_strides_without_changing_global_ops(monkeypatch):
  from types import SimpleNamespace
  from tinygrad import Tensor, dtypes
  from tinygrad.helpers import Context
  from tinygrad.nn.onnx import onnx_ops
  from openpilot.selfdrive.modeld.qcom_yolo_model import make_runner

  original_conv = onnx_ops['Conv']
  original_winograd = Tensor._conv2d_winograd
  calls = []

  def observed_winograd(self, *args, **kwargs):
    calls.append(self.shape)
    return original_winograd(self, *args, **kwargs)

  monkeypatch.setattr(Tensor, '_conv2d_winograd', observed_winograd)
  runner = SimpleNamespace(graph_inputs={'images': SimpleNamespace(shape=(1, 3, 4, 4), dtype=dtypes.float32)}, onnx_ops=onnx_ops)
  make_runner(runner, (4, 4, 4, 4, 2, 24), (4, 4))
  rng = np.random.default_rng(8)
  x = Tensor(rng.normal(size=(1, 8, 8, 8)).astype(np.float32))
  w = Tensor(rng.normal(size=(8, 8, 3, 3)).astype(np.float32))
  with Context(WINO=0):
    expected = original_conv(x, w, strides=(1, 1), pads=(1, 1, 1, 1)).numpy()
    actual = runner.onnx_ops['Conv'](x, w, strides=(1, 1), dilations=(1, 1), pads=(1, 1, 1, 1)).numpy()
  np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=1e-4)
  assert calls == [(1, 8, 8, 8)]
  assert onnx_ops['Conv'] is original_conv
