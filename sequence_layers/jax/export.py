# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for exporting JAX models to TensorFlow and tf.lite."""

from collections.abc import Mapping
import dataclasses
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import orbax.export
from sequence_layers.jax import types
from sequence_layers.jax import typing
import tensorflow as tf


_PATH_SEP = '/'


class TFSequence(NamedTuple):
  values: tf.Tensor
  mask: tf.Tensor


class TFMaskedSequence(TFSequence):
  pass


def _to_tf_sequence(a: types.Sequence | jax.Array) -> TFSequence | jax.Array:
  if isinstance(a, types.Sequence):
    return TFSequence(a.values, a.mask)
  return a


def _from_tf_sequence(a: TFSequence | jax.Array) -> types.Sequence | jax.Array:
  if isinstance(a, TFMaskedSequence):
    return types.MaskedSequence(a.values, a.mask)
  elif isinstance(a, TFSequence):
    return types.Sequence(a.values, a.mask)
  return a


def _tree_to_tf_sequence(tree: typing.AnyPyTree) -> typing.AnyPyTree:
  return jax.tree.map(
      _to_tf_sequence, tree, is_leaf=lambda a: isinstance(a, types.Sequence)
  )


def _tree_from_tf_sequence(tree: typing.AnyPyTree) -> typing.AnyPyTree:
  return jax.tree.map(
      _from_tf_sequence, tree, is_leaf=lambda a: isinstance(a, TFSequence)
  )


def _to_tf_dtype(jax_dtype):
  if jax_dtype == jnp.bfloat16:
    return tf.bfloat16
  elif jax_dtype == jax.dtypes.float0:
    return tf.float32
  else:
    return tf.dtypes.as_dtype(jax_dtype)


def _tf_spec_to_polymorphic_shape(tree: typing.AnyPyTree) -> typing.PyTree[str]:
  """Converts a PyTree of tensors to a string representation of their shape."""
  return tf.nest.map_structure(
      lambda spec: _tf_shape_to_polymorphic_shape(spec.shape), tree
  )


def _tf_shape_to_polymorphic_shape(shape: tf.TensorShape) -> str:
  """Converts a TensorShape to a string representation of its shape.

  Unknown dimensions are assigned a dimension letter according to their index.
  This is useful for expressing the common constraint that batch and time
  dimension should match.

  Args:
    shape: TensorShape to convert.

  Returns:
    A string matching jax2tf's polymorphic shape spec:
    https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#shape-polymorphic-conversion
  """
  dims = []
  for i, d in enumerate(shape.as_list()):
    if d is None:
      if i > 25:
        raise NotImplementedError('too many unknown dimensions')
      dims.append(chr(ord('a') + i))
    else:
      dims.append(str(d))
  return f'({", ".join(dims)})'


def _jax_key_path_to_str(key_path) -> str:
  """Converts a jax tree_util.KeyPath to string."""
  match key_path:
    case jax.tree_util.SequenceKey():
      return str(key_path.idx)
    case jax.tree_util.DictKey():
      return str(key_path.key)
    case jax.tree_util.GetAttrKey():
      return key_path.name
    case jax.tree_util.FlattenedIndexKey():
      return str(key_path.key)
    case _:
      return str(key_path)


def _get_shape(
    input_shape: types.ShapeLike, unknown_batch: bool, unknown_time: bool
) -> tuple[int | None, ...]:
  shape = list(input_shape)
  if unknown_batch:
    shape[0] = None
  if unknown_time:
    shape[1] = None
  return tuple(shape)


def _input_keys_and_leaves(tree):
  leaves = jax.tree.leaves_with_path(
      tree, is_leaf=lambda a: isinstance(a, TFSequence)
  )
  for path, v in leaves:
    k = _PATH_SEP.join(_jax_key_path_to_str(p) for p in path)
    yield k, v


def _tree_to_flat_dict(tree):
  input_dict = {}
  for k, v in _input_keys_and_leaves(tree):
    if isinstance(v, TFSequence):
      input_dict[f'{k}{_PATH_SEP}values'] = v.values
      input_dict[f'{k}{_PATH_SEP}mask'] = v.mask
    else:
      input_dict[k] = v
  return input_dict


def _result_dict_to_tree(result_dict, tree):
  """Map keys in result_dict back to entries in tree without assuming dict flattening order was preserved."""
  leaves, treedef = jax.tree.flatten_with_path(tree)
  key_to_index = {}

  for i, (path, v) in enumerate(leaves):
    k = _PATH_SEP.join(_jax_key_path_to_str(p) for p in path)
    key_to_index[k] = i

  leaves = [None] * len(leaves)
  for k, v in result_dict.items():
    leaves[key_to_index[k]] = v

  return jax.tree.unflatten(treedef, leaves)


def _tree_to_input_signature(
    tree: typing.PyTree[jax.Array],
    unknown_batch: bool = True,
    unknown_time: bool = False,
) -> typing.PyTree[tf.TensorSpec]:
  """Returns a PyTree of tf.TensorSpec for the provided PyTree."""
  leaves, treedef = jax.tree.flatten_with_path(
      tree, is_leaf=lambda a: isinstance(a, TFSequence)
  )
  spec_leaves = []

  for path, v in leaves:
    k = _PATH_SEP.join(_jax_key_path_to_str(p) for p in path)
    if isinstance(v, TFSequence):
      spec_leaves.append(
          TFSequence(
              tf.TensorSpec(
                  _get_shape(v.values.shape, unknown_batch, unknown_time),
                  _to_tf_dtype(v.values.dtype),
                  name=f'{k}{_PATH_SEP}values',
              ),
              tf.TensorSpec(
                  _get_shape(v.mask.shape, unknown_batch, unknown_time),
                  _to_tf_dtype(v.mask.dtype),
                  name=f'{k}{_PATH_SEP}mask',
              ),
          )
      )
    else:
      spec_leaves.append(
          tf.TensorSpec(
              _get_shape(v.shape, unknown_batch, False),
              _to_tf_dtype(v.dtype),
              name=k,
          )
      )

  return jax.tree.unflatten(treedef, spec_leaves)


@dataclasses.dataclass(frozen=True)
class Signature:
  """A signature to export to a TensorFlow saved model."""

  # The function to be exported. Must take a tree of parameters as its first
  # argument and a set of keyword arguments matching input_kwargs.
  fn: Callable[..., Any]
  # A set of keyword arguments to use as the input arguments to the function.
  # The values of the keyword arguments can be arbitrary PyTrees of arrays. The
  # argument names for each array in the tree will represent the path of the
  # array in the tree separated by slashes.
  input_kwargs: Mapping[str, Any]
  # Whether to allow an unknown batch dimension for array and sequence inputs.
  unknown_batch: bool = False
  # Whether to allow an unknown time dimension for sequence inputs.
  unknown_time: bool = False
  # An optional TensorFlow pre-processing function.
  tf_preprocessor: Callable[..., Any] | None = None
  # An optional TensorFlow post-processing function.
  tf_postprocessor: Callable[..., Any] | None = None


def export_to_tf_saved_model(
    params: typing.PyTree[jax.Array],
    signatures: Mapping[str, Signature],
    export_dir: str,
) -> None:
  """Exports the provided JAX function to a TensorFlow saved model.

  Args:
    params: The parameters to the function.
    signatures: A list of signatures to include in the exported saved model.
    export_dir: The directory to export the saved model to.
  """

  serving_configs = []
  functions = {}
  polymorphic_shapes = {}

  for key, signature in signatures.items():
    input_kwargs = _tree_to_tf_sequence(signature.input_kwargs)
    input_signature = _tree_to_input_signature(
        input_kwargs,
        unknown_batch=signature.unknown_batch,
        unknown_time=signature.unknown_time,
    )
    input_polymorphic_shape = _tf_spec_to_polymorphic_shape(input_signature)

    def fn_wrapper(params, args, fn=signature.fn):
      args = _tree_from_tf_sequence(args)
      result = fn(params, **args)
      result = _tree_to_tf_sequence(result)
      return _tree_to_flat_dict(result)

    functions[key] = fn_wrapper
    polymorphic_shapes[key] = input_polymorphic_shape

    serving_configs.append(
        orbax.export.ServingConfig(
            key,
            input_signature=[input_signature],
            tf_preprocessor=signature.tf_preprocessor,
            tf_postprocessor=signature.tf_postprocessor,
            method_key=key,
        )
    )

  jax_module = orbax.export.JaxModule(
      params, functions, input_polymorphic_shape=polymorphic_shapes
  )

  manager = orbax.export.ExportManager(jax_module, serving_configs)
  manager.save(export_dir)


def tflite_convert(
    saved_model_path: str,
    allow_custom_ops: bool = False,
    use_flex: bool = False,
    signature_keys: list[str] | None = None,
    tags: list[str] | None = None,
) -> bytes:
  """Converts the provided TesnorFlow saved model to a tf.lite model.

  Args:
    saved_model_path: The path to the saved model to convert.
    allow_custom_ops: Whether to allow custom ops.
    use_flex: Enable tf.lite flex ops. Requires linking
      //third_party/tensorflow/lite/delegates/flex:delegate
    signature_keys: The signature keys to convert.
    tags: The tags to convert.

  Returns:
    The serialized tf.lite model.
  """
  converter = tf.lite.TFLiteConverter.from_saved_model(
      saved_model_path, signature_keys=signature_keys, tags=tags
  )

  if allow_custom_ops:
    converter.allow_custom_ops = True
  if use_flex:
    converter.target_spec.supported_ops.add(tf.lite.OpsSet.SELECT_TF_OPS)

  return converter.convert()
