import sequence_layers.mlx as sl
from sequence_layers.specs import test_utils as spec


class SequenceLayerTest(spec.SequenceLayerTest):
  """Base class for MLX SequenceLayer tests."""

  sl = sl
