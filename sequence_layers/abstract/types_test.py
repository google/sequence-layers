import unittest

import numpy as np

from sequence_layers.abstract import types


class SemanticDTypeTest(unittest.TestCase):

  def test_semantic_dtype_matches_itself(self):
    # Tests that the un-registered instance matches itself
    self.assertTrue(types.FLOAT32.matches(types.FLOAT32))
    
    # And doesn't match other instances
    self.assertFalse(types.FLOAT32.matches(types.FLOAT16))
    
  def test_semantic_dtype_matches_registered(self):
    # Given a generic mock type
    class MockFloat32:
        pass
    mock_type = MockFloat32()
    
    # It rejects it natively
    self.assertFalse(types.FLOAT32.matches(mock_type))
    
    # Once registered, it matches exactly
    types.FLOAT32.register_backend_type(mock_type)
    self.assertTrue(types.FLOAT32.matches(mock_type))
    
  def test_semantic_dtype_matches_type_classes(self):
    # Backends often use type classes (np.float32) as values.
    # Set membership natively handles this perfectly checking equality.
    types.INT32.register_backend_type(np.int32)
    self.assertTrue(types.INT32.matches(np.int32))
    self.assertFalse(types.INT32.matches(np.float32))

if __name__ == '__main__':
  unittest.main()
