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
