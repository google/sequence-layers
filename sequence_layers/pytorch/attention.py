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
"""Attention layers for PyTorch."""

import dataclasses
import fractions
import math
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceLayer,
    Emitting,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
    Emits,
)
from sequence_layers.pytorch.dense import Dense


__all__ = [
    'DotProductSelfAttention',
    'SelfAttentionEmits',
    'AttentionInputProjection',
    'CombinedQueryKeyValueProjection',
    'SeparateQueryKeyValueProjection',
    'QueryAndKeyValueProjection',
    'QueryAndSharedKeyValueProjection',
]


# =============================================================================
# Attention Emits
# =============================================================================

@dataclasses.dataclass
class SelfAttentionEmits:
    """Emits for self-attention layers."""
    probabilities: Sequence


# =============================================================================
# Input Projection Configurations
# =============================================================================

@dataclasses.dataclass
class CombinedQueryKeyValueProjection:
    """Combined projection for query, key, and value."""
    use_bias: bool = True
    share_kv_projection: bool = False


@dataclasses.dataclass
class SeparateQueryKeyValueProjection:
    """Separate projections for query, key, and value."""
    use_bias: bool = True


@dataclasses.dataclass
class QueryAndKeyValueProjection:
    """Query projection separate from combined key-value projection."""
    use_bias: bool = True


@dataclasses.dataclass
class QueryAndSharedKeyValueProjection:
    """Query projection separate from shared key-value projection."""
    use_bias: bool = True


QueryKeyValueProjectionConfig = Union[
    CombinedQueryKeyValueProjection,
    SeparateQueryKeyValueProjection,
    QueryAndKeyValueProjection,
    QueryAndSharedKeyValueProjection,
]


# =============================================================================
# Input Projection Helper
# =============================================================================

class AttentionInputProjection(nn.Module):
    """Helper for attention input projections."""
    
    def __init__(
        self,
        input_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        units_per_head: int,
        projection_config: QueryKeyValueProjectionConfig,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.units_per_head = units_per_head
        self.projection_config = projection_config
        
        self._setup_projections()
    
    def _setup_projections(self):
        """Setup projection layers based on configuration."""
        if isinstance(self.projection_config, CombinedQueryKeyValueProjection):
            if self.projection_config.share_kv_projection:
                # Query projection + shared KV projection
                self.q_proj = Dense(
                    features=self.num_query_heads * self.units_per_head,
                    use_bias=self.projection_config.use_bias,
                )
                
                self.shared_kv_proj = Dense(
                    features=self.num_kv_heads * self.units_per_head,
                    use_bias=self.projection_config.use_bias,
                )
            else:
                # Combined QKV projection
                total_heads = self.num_query_heads + 2 * self.num_kv_heads
                self.qkv_proj = Dense(
                    features=total_heads * self.units_per_head,
                    use_bias=self.projection_config.use_bias,
                )
                
        elif isinstance(self.projection_config, SeparateQueryKeyValueProjection):
            self.q_proj = Dense(
                features=self.num_query_heads * self.units_per_head,
                use_bias=self.projection_config.use_bias,
            )
            
            self.k_proj = Dense(
                features=self.num_kv_heads * self.units_per_head,
                use_bias=self.projection_config.use_bias,
            )
            
            self.v_proj = Dense(
                features=self.num_kv_heads * self.units_per_head,
                use_bias=self.projection_config.use_bias,
            )
            
        elif isinstance(self.projection_config, QueryAndKeyValueProjection):
            self.q_proj = Dense(
                features=self.num_query_heads * self.units_per_head,
                use_bias=self.projection_config.use_bias,
            )
            
            self.kv_proj = Dense(
                features=2 * self.num_kv_heads * self.units_per_head,
                use_bias=self.projection_config.use_bias,
            )
            
        elif isinstance(self.projection_config, QueryAndSharedKeyValueProjection):
            self.q_proj = Dense(
                features=self.num_query_heads * self.units_per_head,
                use_bias=self.projection_config.use_bias,
            )
            
            self.shared_kv_proj = Dense(
                features=self.num_kv_heads * self.units_per_head,
                use_bias=self.projection_config.use_bias,
            )
    
    def get_qkv(self, x: Sequence) -> Tuple[Sequence, Sequence, Sequence]:
        """Get query, key, value projections."""
        if isinstance(self.projection_config, CombinedQueryKeyValueProjection):
            if self.projection_config.share_kv_projection:
                queries = self.q_proj.layer(x)
                shared_kv = self.shared_kv_proj.layer(x)
                
                # Reshape to [batch, time, num_heads, units_per_head]
                queries_reshaped = Sequence(
                    queries.values.reshape(
                        queries.values.shape[0], queries.values.shape[1], 
                        self.num_query_heads, self.units_per_head
                    ),
                    queries.mask
                )
                shared_kv_reshaped = Sequence(
                    shared_kv.values.reshape(
                        shared_kv.values.shape[0], shared_kv.values.shape[1],
                        self.num_kv_heads, self.units_per_head
                    ),
                    shared_kv.mask
                )
                
                return queries_reshaped, shared_kv_reshaped, shared_kv_reshaped
            else:
                qkv = self.qkv_proj.layer(x)
                
                # Reshape to [batch, time, total_heads, units_per_head]
                qkv_reshaped = qkv.values.reshape(
                    qkv.values.shape[0], qkv.values.shape[1], 
                    self.num_query_heads + 2 * self.num_kv_heads, self.units_per_head
                )
                
                # Split into Q, K, V
                q_end = self.num_query_heads
                k_end = q_end + self.num_kv_heads
                queries = Sequence(qkv_reshaped[:, :, :q_end], qkv.mask)
                keys = Sequence(qkv_reshaped[:, :, q_end:k_end], qkv.mask)
                values = Sequence(qkv_reshaped[:, :, k_end:], qkv.mask)
                return queries, keys, values
                
        elif isinstance(self.projection_config, SeparateQueryKeyValueProjection):
            queries = self.q_proj.layer(x)
            keys = self.k_proj.layer(x)
            values = self.v_proj.layer(x)
            
            # Reshape to [batch, time, num_heads, units_per_head]
            queries_reshaped = Sequence(
                queries.values.reshape(
                    queries.values.shape[0], queries.values.shape[1], 
                    self.num_query_heads, self.units_per_head
                ),
                queries.mask
            )
            keys_reshaped = Sequence(
                keys.values.reshape(
                    keys.values.shape[0], keys.values.shape[1], 
                    self.num_kv_heads, self.units_per_head
                ),
                keys.mask
            )
            values_reshaped = Sequence(
                values.values.reshape(
                    values.values.shape[0], values.values.shape[1], 
                    self.num_kv_heads, self.units_per_head
                ),
                values.mask
            )
            
            return queries_reshaped, keys_reshaped, values_reshaped
            
        elif isinstance(self.projection_config, QueryAndKeyValueProjection):
            queries = self.q_proj.layer(x)
            kv = self.kv_proj.layer(x)
            
            # Reshape query to [batch, time, num_query_heads, units_per_head]
            queries_reshaped = Sequence(
                queries.values.reshape(
                    queries.values.shape[0], queries.values.shape[1], 
                    self.num_query_heads, self.units_per_head
                ),
                queries.mask
            )
            
            # Reshape KV to [batch, time, 2*num_kv_heads, units_per_head] and split
            kv_reshaped = kv.values.reshape(
                kv.values.shape[0], kv.values.shape[1], 
                2 * self.num_kv_heads, self.units_per_head
            )
            
            # Split KV into K and V
            k_end = self.num_kv_heads
            keys = Sequence(kv_reshaped[:, :, :k_end], kv.mask)
            values = Sequence(kv_reshaped[:, :, k_end:], kv.mask)
            
            return queries_reshaped, keys, values
            
        elif isinstance(self.projection_config, QueryAndSharedKeyValueProjection):
            queries = self.q_proj.layer(x)
            shared_kv = self.shared_kv_proj.layer(x)
            
            # Reshape to [batch, time, num_heads, units_per_head]
            queries_reshaped = Sequence(
                queries.values.reshape(
                    queries.values.shape[0], queries.values.shape[1], 
                    self.num_query_heads, self.units_per_head
                ),
                queries.mask
            )
            shared_kv_reshaped = Sequence(
                shared_kv.values.reshape(
                    shared_kv.values.shape[0], shared_kv.values.shape[1],
                    self.num_kv_heads, self.units_per_head
                ),
                shared_kv.mask
            )
            
            return queries_reshaped, shared_kv_reshaped, shared_kv_reshaped
        else:
            raise ValueError(f"Unknown projection config: {self.projection_config}")


# =============================================================================
# Core Attention Functions
# =============================================================================

def _scale_query(
    queries: Tensor,
    per_dim_scale: Optional[Tensor] = None,
    query_scale: Optional[float] = None,
) -> Tensor:
    """Scale queries for attention computation."""
    if per_dim_scale is not None:
        queries = queries * (1.0 + per_dim_scale)
    
    if query_scale is not None:
        queries = queries * query_scale
    else:
        # Default scaling by 1/sqrt(units_per_head)
        units_per_head = queries.shape[-1]
        queries = queries / math.sqrt(units_per_head)
    
    return queries


def _mask_attention_logits(
    logits: Tensor,
    mask: Optional[Tensor] = None,
    invalid_logit_value: float = -1e9,
) -> Tensor:
    """Mask attention logits."""
    if mask is not None:
        # mask is [batch, 1, 1, key_time] or broadcastable
        return torch.where(mask, logits, invalid_logit_value)
    return logits


def _soft_cap_attention_logits(logits: Tensor, soft_cap: float) -> Tensor:
    """Apply soft cap to attention logits."""
    return soft_cap * torch.tanh(logits / soft_cap)


def _self_attention_causal_mask(
    query_time: int,
    key_time: int,
    max_past_horizon: int,
    max_future_horizon: int,
    device: torch.device,
) -> Optional[Tensor]:
    """Create causal mask for self-attention."""
    if max_past_horizon == -1 and max_future_horizon == -1:
        return None  # No masking needed
    
    # Create position matrices
    query_positions = torch.arange(query_time, device=device).unsqueeze(1)  # [query_time, 1]
    key_positions = torch.arange(key_time, device=device).unsqueeze(0)      # [1, key_time]
    
    # Compute relative positions (key_pos - query_pos)
    relative_positions = key_positions - query_positions  # [query_time, key_time]
    
    # Apply past horizon mask (can look back max_past_horizon steps)
    if max_past_horizon >= 0:
        past_mask = relative_positions >= -max_past_horizon
    else:
        past_mask = torch.ones_like(relative_positions, dtype=torch.bool)
    
    # Apply future horizon mask (can look forward max_future_horizon steps)
    if max_future_horizon >= 0:
        future_mask = relative_positions <= max_future_horizon
    else:
        future_mask = torch.ones_like(relative_positions, dtype=torch.bool)
    
    # Combine masks
    causal_mask = past_mask & future_mask
    
    # Add batch and head dimensions: [1, 1, query_time, key_time]
    return causal_mask.unsqueeze(0).unsqueeze(0)


def dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor] = None,
    causal_mask: Optional[Tensor] = None,
    dropout: Optional[nn.Module] = None,
    training: bool = False,
    per_dim_scale: Optional[Tensor] = None,
    query_scale: Optional[float] = None,
    attention_logits_soft_cap: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute dot-product attention.
    
    Args:
        queries: [batch, query_time, num_heads, units_per_head]
        keys: [batch, key_time, num_kv_heads, units_per_head]
        values: [batch, key_time, num_kv_heads, units_per_head]
        mask: [batch, key_time] or [batch, 1, 1, key_time]
        causal_mask: [1, 1, query_time, key_time] or None
        dropout: Dropout module for attention probabilities
        training: Whether in training mode
        per_dim_scale: [units_per_head] per-dimension scale factor
        query_scale: Scalar query scale factor
        attention_logits_soft_cap: Soft cap for attention logits
        
    Returns:
        context_vectors: [batch, query_time, num_heads, units_per_head]
        probabilities: [batch, query_time, num_heads, key_time]
    """
    batch_size, query_time, num_heads, units_per_head = queries.shape
    key_time = keys.shape[1]
    num_kv_heads = keys.shape[2]
    
    # Scale queries
    queries = _scale_query(queries, per_dim_scale, query_scale)
    
    # Handle grouped query attention (GQA)
    if num_heads != num_kv_heads:
        # Repeat keys and values for each query head group
        group_size = num_heads // num_kv_heads
        keys = keys.repeat_interleave(group_size, dim=2)
        values = values.repeat_interleave(group_size, dim=2)
    
    # Reshape for batch matrix multiplication
    # queries: [batch, query_time, num_heads, units_per_head] -> [batch, num_heads, query_time, units_per_head]
    # keys: [batch, key_time, num_heads, units_per_head] -> [batch, num_heads, key_time, units_per_head]
    queries = queries.transpose(1, 2)  # [batch, num_heads, query_time, units_per_head]
    keys = keys.transpose(1, 2)        # [batch, num_heads, key_time, units_per_head]
    values = values.transpose(1, 2)    # [batch, num_heads, key_time, units_per_head]
    
    # Compute attention logits: [batch, num_heads, query_time, key_time]
    logits = torch.matmul(queries, keys.transpose(-2, -1))
    
    # Apply soft cap if specified
    if attention_logits_soft_cap is not None:
        logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)
    
    # Apply causal mask
    if causal_mask is not None:
        # causal_mask: [1, 1, query_time, key_time]
        # logits: [batch, num_heads, query_time, key_time]
        # Need to expand causal_mask to match logits dimensions
        causal_mask_expanded = causal_mask.expand(batch_size, num_heads, query_time, key_time)
        logits = logits.masked_fill(~causal_mask_expanded, -1e9)
    
    # Apply key mask
    if mask is not None:
        if mask.dim() == 2:  # [batch, key_time]
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, key_time]
        # Expand mask to match logits dimensions
        mask_expanded = mask.expand(batch_size, num_heads, query_time, key_time)
        logits = logits.masked_fill(~mask_expanded, -1e9)
    
    # Compute attention probabilities
    probabilities = F.softmax(logits, dim=-1)
    
    # Apply dropout
    if dropout is not None and training:
        probabilities = dropout(probabilities)
    
    # Compute context vectors: [batch, num_heads, query_time, units_per_head]
    context_vectors = torch.matmul(probabilities, values)
    
    # Transpose back to original format: [batch, query_time, num_heads, units_per_head]
    context_vectors = context_vectors.transpose(1, 2)
    probabilities = probabilities.transpose(1, 2)  # [batch, query_time, num_heads, key_time]
    
    return context_vectors, probabilities


# =============================================================================
# DotProductSelfAttention Layer
# =============================================================================

class DotProductSelfAttention(Emitting):
    """A multi-headed dot-product self attention layer."""
    
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        units_per_head: int,
        max_past_horizon: int = -1,
        max_future_horizon: int = 0,
        num_kv_heads: Optional[int] = None,
        input_projection: Optional[QueryKeyValueProjectionConfig] = None,
        attention_probabilities_dropout_rate: float = 0.0,
        broadcast_dropout_across_queries: bool = False,
        per_dim_scale: bool = False,
        query_scale: Optional[float] = None,
        attention_logits_soft_cap: Optional[float] = None,
        zero_fully_masked: bool = False,
        query_network: Optional[SequenceLayer] = None,
        key_network: Optional[SequenceLayer] = None,
        value_network: Optional[SequenceLayer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        
        self.input_size = input_size
        self.num_heads = num_heads
        self.units_per_head = units_per_head
        self.max_past_horizon = max_past_horizon
        self.max_future_horizon = max_future_horizon
        self.num_kv_heads = num_kv_heads or num_heads
        self.attention_probabilities_dropout_rate = attention_probabilities_dropout_rate
        self.broadcast_dropout_across_queries = broadcast_dropout_across_queries
        self.query_scale = query_scale
        self.attention_logits_soft_cap = attention_logits_soft_cap
        self.zero_fully_masked = zero_fully_masked
        
        # Validate configuration
        if self.max_past_horizon < -1:
            raise ValueError(f"max_past_horizon must be >= -1, got {self.max_past_horizon}")
        if self.max_future_horizon < -1:
            raise ValueError(f"max_future_horizon must be >= -1, got {self.max_future_horizon}")
        if self.max_past_horizon == 0 and self.max_future_horizon == 0:
            raise ValueError("Both max_past_horizon and max_future_horizon cannot be 0")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        
        # Setup input projection
        if input_projection is None:
            input_projection = CombinedQueryKeyValueProjection()
        self.input_projection_config = input_projection
        
        self.input_projection = AttentionInputProjection(
            input_size=input_size,
            num_query_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            units_per_head=units_per_head,
            projection_config=input_projection,
        )
        
        # Setup per-dimension scale
        if per_dim_scale:
            self.per_dim_scale = nn.Parameter(torch.zeros(units_per_head))
        else:
            self.per_dim_scale = None
        
        # Setup dropout
        self.dropout = nn.Dropout(attention_probabilities_dropout_rate) if attention_probabilities_dropout_rate > 0 else None
        
        # Setup optional networks
        self.query_network = query_network
        self.key_network = key_network
        self.value_network = value_network
        
        # Validate network properties
        for network, name in [(query_network, "query"), (key_network, "key"), (value_network, "value")]:
            if network is not None:
                if not hasattr(network, 'output_ratio') or network.output_ratio != fractions.Fraction(1, 1):
                    raise ValueError(f"{name}_network must have output_ratio of 1")
                if not hasattr(network, 'block_size') or network.block_size != 1:
                    raise ValueError(f"{name}_network must have block_size of 1")
    
    def supports_step(self) -> bool:
        """Check if layer supports step-wise execution."""
        base_supports = (
            self.max_past_horizon >= 0 and
            self.max_future_horizon >= 0
        )
        
        # Check networks
        if self.query_network and not self.query_network.supports_step():
            return False
        if self.key_network and not self.key_network.supports_step():
            return False
        if self.value_network and not self.value_network.supports_step():
            return False
        
        return base_supports
    
    @property
    def block_size(self) -> int:
        return 1
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(1, 1)
    
    @property
    def input_latency(self) -> int:
        return max(0, self.max_future_horizon) if self.max_future_horizon >= 0 else 0
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 1:
            raise ValueError(f"DotProductSelfAttention requires rank 3 input, got {input_shape}")
        return (self.num_heads, self.units_per_head)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def get_initial_state(self, batch_size: int, input_spec: ChannelSpec, 
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for streaming execution."""
        # Get device from input_projection parameters, or use CPU as default
        try:
            device = next(self.input_projection.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        
        dtype = input_spec.dtype
        
        # Calculate initial buffer size based on horizons
        # For KV buffers, we start with empty buffers that grow as needed
        # For query delay buffer, we pre-allocate the future horizon size
        
        # Initialize empty KV buffers (they will grow as needed)
        kv_buffer_keys = torch.zeros(
            batch_size, 0, self.num_kv_heads, self.units_per_head,
            device=device, dtype=dtype
        )
        kv_buffer_values = torch.zeros(
            batch_size, 0, self.num_kv_heads, self.units_per_head,
            device=device, dtype=dtype
        )
        kv_buffer_mask = torch.zeros(
            batch_size, 0, device=device, dtype=torch.bool
        )
        
        # Initialize query delay buffer if needed
        if self.max_future_horizon > 0:
            query_delay_buffer = Sequence(
                torch.zeros(
                    batch_size, self.max_future_horizon, self.num_heads, self.units_per_head,
                    device=device, dtype=dtype
                ),
                torch.zeros(batch_size, self.max_future_horizon, device=device, dtype=torch.bool)
            )
        else:
            query_delay_buffer = None
        
        # Initialize timestep counter (scalar, not per-batch)
        time_step = torch.tensor(0, device=device, dtype=torch.int32)
        
        # Initialize network states
        query_network_state = None
        key_network_state = None
        value_network_state = None
        
        query_spec = ChannelSpec((self.num_heads, self.units_per_head), dtype)
        kv_spec = ChannelSpec((self.num_kv_heads, self.units_per_head), dtype)
        
        if self.query_network:
            query_network_state = self.query_network.get_initial_state(
                batch_size, query_spec, training, constants
            )
        
        if self.key_network:
            key_network_state = self.key_network.get_initial_state(
                batch_size, kv_spec, training, constants
            )
        
        if self.value_network:
            value_network_state = self.value_network.get_initial_state(
                batch_size, kv_spec, training, constants
            )
        
        return {
            'kv_buffer_keys': kv_buffer_keys,
            'kv_buffer_values': kv_buffer_values,
            'kv_buffer_mask': kv_buffer_mask,
            'query_delay_buffer': query_delay_buffer,
            'time_step': time_step,
            'query_network_state': query_network_state,
            'key_network_state': key_network_state,
            'value_network_state': value_network_state,
        }
    
    def step_with_emits(self, x: Sequence, state: State, training: bool = False,
                        constants: Optional[Constants] = None) -> Tuple[Sequence, State, Emits]:
        """Step-wise execution with emits."""
        if not self.supports_step():
            raise ValueError(f"{self.__class__.__name__} does not support step-wise execution")
        
        batch_size, x_time = x.shape[:2]
        
        # Get projections
        queries, keys, values = self.input_projection.get_qkv(x)
        
        # Unpack state
        kv_buffer_keys = state['kv_buffer_keys']
        kv_buffer_values = state['kv_buffer_values']
        kv_buffer_mask = state['kv_buffer_mask']
        query_delay_buffer = state['query_delay_buffer']
        time_step = state['time_step']
        query_network_state = state['query_network_state']
        key_network_state = state['key_network_state']
        value_network_state = state['value_network_state']
        
        # Apply networks
        if self.query_network:
            queries, query_network_state = self.query_network.step(
                queries, query_network_state, training, constants
            )
        
        if self.key_network:
            keys, key_network_state = self.key_network.step(
                keys, key_network_state, training, constants
            )
        
        if self.value_network:
            values, value_network_state = self.value_network.step(
                values, value_network_state, training, constants
            )
        
        # Mask invalid values
        queries = queries.mask_invalid()
        keys = keys.mask_invalid()
        values = values.mask_invalid()
        
        # Combine key and value masks with input mask
        combined_mask = keys.mask & values.mask & x.mask
        
        # Update KV buffers by appending new keys/values
        kv_buffer_keys = torch.cat([kv_buffer_keys, keys.values], dim=1)
        kv_buffer_values = torch.cat([kv_buffer_values, values.values], dim=1)
        kv_buffer_mask = torch.cat([kv_buffer_mask, combined_mask], dim=1)
        
        # Current absolute time position
        current_time_step = time_step.item()
        
        # Store the original input mask for output mask computation
        original_input_mask = x.mask
        
        # Handle query delay buffer for future horizon
        if query_delay_buffer is not None:
            # Add current queries to delay buffer
            query_delay_buffer = query_delay_buffer.concatenate(queries)
            
            # Extract queries that should be processed now (accounting for future horizon)
            if query_delay_buffer.shape[1] >= self.max_future_horizon:
                # Extract the oldest queries (those that have waited long enough)
                queries = Sequence(
                    query_delay_buffer.values[:, :x_time],
                    query_delay_buffer.mask[:, :x_time]
                )
                
                # Keep remaining queries in buffer
                query_delay_buffer = Sequence(
                    query_delay_buffer.values[:, x_time:],
                    query_delay_buffer.mask[:, x_time:]
                )
                
                # The queries we're processing are from an earlier time step
                query_time_step = current_time_step - self.max_future_horizon
                
                # For future horizon, the output mask should be the mask of the queries
                # that were delayed, which corresponds to the current input mask
                output_mask = queries.mask
            else:
                # Not enough queries in buffer yet, return empty output
                empty_output = Sequence(
                    torch.zeros(batch_size, 0, self.num_heads, self.units_per_head, 
                               dtype=x.dtype, device=x.device),
                    torch.zeros(batch_size, 0, dtype=torch.bool, device=x.device)
                )
                empty_emits = SelfAttentionEmits(probabilities=empty_output)
                
                new_state = {
                    'kv_buffer_keys': kv_buffer_keys,
                    'kv_buffer_values': kv_buffer_values,
                    'kv_buffer_mask': kv_buffer_mask,
                    'query_delay_buffer': query_delay_buffer,
                    'time_step': time_step + x_time,
                    'query_network_state': query_network_state,
                    'key_network_state': key_network_state,
                    'value_network_state': value_network_state,
                }
                
                return empty_output, new_state, empty_emits
        else:
            # No query delay, process current queries
            query_time_step = current_time_step
            # For no future horizon, use the original input mask
            output_mask = original_input_mask
        
        # Create causal mask with correct time positions
        # For step-wise execution, we need to consider the absolute positions
        query_positions = torch.arange(query_time_step, query_time_step + queries.shape[1], 
                                     device=queries.values.device).unsqueeze(1)  # [query_time, 1]
        key_positions = torch.arange(current_time_step + x_time - kv_buffer_keys.shape[1], 
                                   current_time_step + x_time, 
                                   device=queries.values.device).unsqueeze(0)  # [1, kv_buffer_time]
        
        # Compute relative positions (key_pos - query_pos)
        relative_positions = key_positions - query_positions  # [query_time, kv_buffer_time]
        
        # Apply horizons to create causal mask
        if self.max_past_horizon == -1 and self.max_future_horizon == -1:
            causal_mask = None
        else:
            # Apply past horizon mask (can look back max_past_horizon steps)
            if self.max_past_horizon >= 0:
                past_mask = relative_positions >= -self.max_past_horizon
            else:
                past_mask = torch.ones_like(relative_positions, dtype=torch.bool)
            
            # Apply future horizon mask (can look forward max_future_horizon steps)
            if self.max_future_horizon >= 0:
                future_mask = relative_positions <= self.max_future_horizon
            else:
                future_mask = torch.ones_like(relative_positions, dtype=torch.bool)
            
            # Combine masks
            causal_mask = past_mask & future_mask
            
            # Add batch and head dimensions: [1, 1, query_time, key_time]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply attention
        context_vectors, probabilities = dot_product_attention(
            queries.values, kv_buffer_keys, kv_buffer_values,
            mask=kv_buffer_mask,
            causal_mask=causal_mask,
            dropout=self.dropout,
            training=training,
            per_dim_scale=self.per_dim_scale,
            query_scale=self.query_scale,
            attention_logits_soft_cap=self.attention_logits_soft_cap,
        )
        
        # Trim KV buffer to maintain proper size
        # Keep past_horizon + future_horizon + 1 timesteps
        max_buffer_size = max(1, self.max_past_horizon if self.max_past_horizon > 0 else 0) + \
                         max(0, self.max_future_horizon if self.max_future_horizon > 0 else 0) + 1
        
        if kv_buffer_keys.shape[1] > max_buffer_size:
            kv_buffer_keys = kv_buffer_keys[:, -max_buffer_size:]
            kv_buffer_values = kv_buffer_values[:, -max_buffer_size:]
            kv_buffer_mask = kv_buffer_mask[:, -max_buffer_size:]
        
        # Update state
        new_state = {
            'kv_buffer_keys': kv_buffer_keys,
            'kv_buffer_values': kv_buffer_values,
            'kv_buffer_mask': kv_buffer_mask,
            'query_delay_buffer': query_delay_buffer,
            'time_step': time_step + x_time,
            'query_network_state': query_network_state,
            'key_network_state': key_network_state,
            'value_network_state': value_network_state,
        }
        
        # Create output and emits
        output = Sequence(context_vectors, output_mask)
        emits = SelfAttentionEmits(probabilities=Sequence(probabilities, output_mask))
        
        # Ensure masked timesteps have zero values
        output = output.mask_invalid()
        
        return output, new_state, emits
    
    def layer_with_emits(self, x: Sequence, training: bool = False,
                        initial_state: Optional[State] = None,
                        constants: Optional[Constants] = None) -> Tuple[Sequence, Emits]:
        """Layer-wise execution with emits."""
        batch_size, seq_len = x.shape[:2]
        
        # Get projections
        queries, keys, values = self.input_projection.get_qkv(x)
        
        # Apply networks
        if self.query_network:
            queries = self.query_network.layer(queries, training, None, constants)
        
        if self.key_network:
            keys = self.key_network.layer(keys, training, None, constants)
        
        if self.value_network:
            values = self.value_network.layer(values, training, None, constants)
        
        # Mask invalid values in all projections
        queries = queries.mask_invalid()
        keys = keys.mask_invalid()
        values = values.mask_invalid()
        
        # Create causal mask
        causal_mask = _self_attention_causal_mask(
            seq_len, seq_len,
            self.max_past_horizon, self.max_future_horizon,
            queries.values.device
        )
        
        # Apply attention
        context_vectors, probabilities = dot_product_attention(
            queries.values, keys.values, values.values,
            mask=keys.mask,
            causal_mask=causal_mask,
            dropout=self.dropout,
            training=training,
            per_dim_scale=self.per_dim_scale,
            query_scale=self.query_scale,
            attention_logits_soft_cap=self.attention_logits_soft_cap,
        )
        
        # Create output and emits
        output = Sequence(context_vectors, queries.mask)
        # Ensure that invalid query timesteps are masked out in the output
        output = output.mask_invalid()
        
        # For future horizon, mask out the first max_future_horizon timesteps
        # since they can't be computed without future context
        if self.max_future_horizon > 0:
            output_mask = output.mask.clone()
            output_mask[:, :self.max_future_horizon] = False
            output = Sequence(output.values, output_mask)
            # Re-apply masking after updating the mask
            output = output.mask_invalid()
        
        emits = SelfAttentionEmits(probabilities=Sequence(probabilities, output.mask))
        
        return output, emits 