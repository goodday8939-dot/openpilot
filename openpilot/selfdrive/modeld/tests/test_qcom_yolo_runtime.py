import pytest

from openpilot.selfdrive.modeld.qcom_yolo_runtime import partition_costs


def test_partition_preserves_every_kernel_in_order_and_bounds_batch_cost():
  costs = [.8, .5, 6.3, 3.2, .1, .2, 9., 0., 1., 7., .3]
  groups = partition_costs(costs)
  assert [index for group in groups for index in group] == list(range(len(costs)))
  assert all(len(group) == 1 or sum(costs[index] for index in group) <= 8. for group in groups)
  assert [6] in groups  # A single expensive kernel cannot be subdivided here.


@pytest.mark.parametrize('costs,budget', [([float('nan')], 8.), ([-1.], 8.), ([1.], 0.), ([1.], float('inf'))])
def test_partition_rejects_invalid_profile_or_budget(costs, budget):
  with pytest.raises(ValueError):
    partition_costs(costs, budget)
