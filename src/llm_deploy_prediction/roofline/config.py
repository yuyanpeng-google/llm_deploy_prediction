'''Configuration classes and loaders for roofline analysis.

This module defines the dataclasses used to represent model configurations,
hardware specifications, and sharding strategies, as well as functions
to load them from JSON files.
'''

from dataclasses import dataclass
import json
from typing import Any, Dict

@dataclass
class ModelConfig:
    '''Configuration for the LLM model parameters.
    
    Attributes:
        hidden_size: Dimension of the hidden layers.
        num_q_heads: Number of query heads in attention.
        num_kv_heads: Number of key/value heads in attention (for GQA).
        attn_head_dim: Dimension of each attention head.
        num_layers: Number of transformer layers.
        vocab_size: Size of the vocabulary.
        num_experts: Total number of experts in MoE layer.
        num_activated_experts: Number of activated experts per token.
        intermediate_size: Intermediate dimension in MoE experts.
        bytes_per_param: Bytes per parameter (default 2 for FP16/BF16).
        attn_op_precision: Precision for attention operations ('fp8' or 'bf16').
        kv_cache_precision: Precision for KV cache storage ('fp8' or 'bf16').
    '''
    
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    attn_head_dim: int
    num_layers: int
    vocab_size: int
    num_experts: int
    num_activated_experts: int
    intermediate_size: int
    bytes_per_param: int = 2
    attn_op_precision: str = 'fp8'
    kv_cache_precision: str = 'fp8'

@dataclass
class HardwareSpec:
    '''Configuration for the hardware specifications.
    
    Attributes:
        peak_bf16_flops: Peak BF16 compute performance in TFLOPs/s.
        peak_fp8_flops: Peak FP8 compute performance in TFLOPs/s.
        hbm_bandwidth: Peak HBM bandwidth in GB/s.
        ici_ar_ag_bandwidth: Unidirectional ICI bandwidth for All-Reduce and All-Gather in GB/s.
        ici_a2a_bandwidth: Unidirectional ICI bandwidth for All-to-All in GB/s.
        hbm_capacity: HBM capacity per chip in GB.
        num_chips: Number of chips associated with this spec.
    '''
    
    peak_bf16_flops: float
    peak_fp8_flops: float
    hbm_bandwidth: float
    ici_ar_ag_bandwidth: float
    ici_a2a_bandwidth: float
    hbm_capacity: float = 192.0
    num_chips: int = 4

@dataclass
class ShardingStrategy:
    '''Configuration for sharding strategy.
    
    Attributes:
        num_chips: Total number of chips used.
        attn_tp_degree: TP degree for attention.
        attn_dp_degree: DP degree for attention.
        moe_tp_degree: TP degree for MoE.
        moe_ep_degree: EP degree for MoE.
    '''
    
    num_chips: int
    attn_tp_degree: int = 1
    attn_dp_degree: int = 1
    moe_tp_degree: int = 1
    moe_ep_degree: int = 1

def load_model_config(file_path: str) -> ModelConfig:
    '''Loads ModelConfig from a JSON file.

    Args:
        file_path: Path to the JSON config file.

    Returns:
        An instance of ModelConfig.
    '''
    with open(file_path, 'r') as f:
        data = json.load(f)
    return ModelConfig(**data)

def load_hardware_spec(file_path: str) -> HardwareSpec:
    '''Loads HardwareSpec from a JSON file.

    Args:
        file_path: Path to the JSON config file.

    Returns:
        An instance of HardwareSpec.
    '''
    with open(file_path, 'r') as f:
        data = json.load(f)
    return HardwareSpec(**data)
