# Sequence Layers

Note: This is not an officially supported Google product

## Overview

A library for sequence modeling in TensorFlow 2, enabling easy creation of
sequence models that can be executed both layer-by-layer (e.g. teacher forced
training) and step-by-step (e.g. autoregressive sampling).

A key feature of the library is that layers support streaming (step-by-step)
operation. To achieve this, every layer has a notion of state when and a `step`
function in addition to the typical layer-wise processing feature found in other
libraries like Keras. When layers support a `step` method, their `layer` method
produces identical results for the same sequence of input blocks enabling easy
switching between step-wise and layer-wise processing depending on the use case.

## Goals

Increased development velocity for both research and production applications of
sequence modeling.

*   Support for layer-by-layer and step-by-step processing in a single
    implementation.
*   Declarative API.
*   Composable, thin abstractions.
*   Easy mix-and-match of popular sequence modeling paradigms (convolutional,
    recurrent, attention architectures).
*   A quick path to deployment with tf.lite support for every layer.
*   Tracking of invalid timesteps (those computed from padding).

<!-- TODO(dthkao): ## Code Samples -->

## Protocol Buffer API

An optional protobuf API allows you to specify SequenceLayers from proto.

See `sequence_layers/proto.proto`. The proto API is
intended to closely match the Python API.

### Custom SequenceLayers and the proto API.

Registering your own custom SequenceLayers for use in the protocol buffer API is
possible via the use of protocol buffer extensions.

Simply define a custom proto message for your SequenceLayer:

```proto
// custom_proto.proto
import "third_party/py/sequence_layers/proto.proto";

message CustomLayer {
  optional float param = 1;
  optional string name = 2;
}

extend sequence_layers.SequenceLayer {
  optional CustomLayer custom_layer = 344129823;
}
```

Then register a factory function to tell `build_sequence_layer` how to
instantiate the layer from configuration:

```python
from sequence_layers import proto as slp

@slp.register_factory(custom_proto_pb2.CustomLayer)
def _build_custom_layer(spec: custom_proto_pb2.CustomLayer) -> CustomLayer:
  return CustomLayer(spec.param, spec.name)
```
