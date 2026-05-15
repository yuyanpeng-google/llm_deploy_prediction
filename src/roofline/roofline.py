'''Roofline model for calculating theoretical performance of LLM deployment.

This script calculates FLOPs, memory access, and latency for LLM models,
focusing on per-layer calculation for prefill and decode phases.
'''

from dataclasses import dataclass
from typing import Dict, Any
import json
import argparse

@dataclass
class ModelConfig:
    '''Configuration for the LLM model parameters.'''
    
    hidden_size: int
    '''Dimension of the hidden layers.'''
    
    num_q_heads: int
    '''Number of query heads in attention.'''
    
    num_kv_heads: int
    '''Number of key/value heads in attention (for GQA).'''
    
    attn_head_dim: int
    '''Dimension of each attention head.'''
    
    num_layers: int
    '''Number of transformer layers.'''
    
    vocab_size: int
    '''Size of the vocabulary.'''
    
    num_experts: int
    '''Total number of experts in MoE layer.'''
    
    num_activated_experts: int
    '''Number of activated experts per token.'''
    
    intermediate_size: int
    '''Intermediate dimension in MoE experts.'''
    
    bytes_per_param: int = 2
    '''Bytes per parameter (default 2 for FP16/BF16).'''
    
    attn_op_precision: str = 'fp8'
    '''Precision for attention operations ('fp8' or 'bf16').'''
    
    kv_cache_precision: str = 'fp8'
    '''Precision for KV cache storage ('fp8' or 'bf16').'''

@dataclass
class HardwareSpec:
    '''Configuration for the hardware specifications.'''
    
    peak_bf16_flops: float
    '''Peak BF16 compute performance in TFLOPs/s.'''
    
    peak_fp8_flops: float
    '''Peak FP8 compute performance in TFLOPs/s.'''
    
    hbm_bandwidth: float
    '''Peak HBM bandwidth in GB/s.'''

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

def calculate_attention_flops(config: ModelConfig, seq_len: int, batch_size: int, is_prefill: bool) -> Dict[str, float]:
    '''Calculates the FLOPs for the attention mechanism in a single layer.

    Args:
        config: Model configuration.
        seq_len: Current sequence length.
        batch_size: Batch size.
        is_prefill: True if prefill phase, False if decode phase.

    Returns:
        Dictionary with 'fp8_flops' and 'bf16_flops'.
    '''
    B = batch_size
    S = seq_len
    H = config.hidden_size
    N_kv = config.num_kv_heads
    D = config.attn_head_dim
    
    kv_hidden = N_kv * D
    q_hidden = config.num_q_heads * config.attn_head_dim
    
    if is_prefill:
        # Projections
        q_flops = 2 * B * S * H * q_hidden
        k_flops = 2 * B * S * H * kv_hidden
        v_flops = 2 * B * S * H * kv_hidden
        # Score QK^T
        score_flops = 2 * B * config.num_q_heads * S * S * config.attn_head_dim
        # Value Score * V
        value_flops = 2 * B * config.num_q_heads * S * S * config.attn_head_dim
        # Output projection
        out_flops = 2 * B * S * q_hidden * H
    else:
        # Decode (S is current length, but we only process 1 new token)
        # Projections for 1 new token
        q_flops = 2 * B * 1 * H * q_hidden
        k_flops = 2 * B * 1 * H * kv_hidden
        v_flops = 2 * B * 1 * H * kv_hidden
        # Score QK^T (Q is 1 token, K is S tokens)
        score_flops = 2 * B * config.num_q_heads * 1 * S * config.attn_head_dim
        # Value Score * V
        value_flops = 2 * B * config.num_q_heads * 1 * S * config.attn_head_dim
        # Output projection
        out_flops = 2 * B * 1 * q_hidden * H
        
    proj_flops = q_flops + k_flops + v_flops + out_flops
    attn_op_flops = score_flops + value_flops
    
    if config.attn_op_precision == 'bf16':
        if config.bytes_per_param == 1:
            fp8_flops = float(proj_flops)
            bf16_flops = float(attn_op_flops)
        elif config.bytes_per_param == 2:
            fp8_flops = 0.0
            bf16_flops = float(proj_flops + attn_op_flops)
        else:
            raise ValueError(f"Unsupported bytes_per_param: {config.bytes_per_param}")
    elif config.attn_op_precision == 'fp8':
        if config.bytes_per_param == 1:
            fp8_flops = float(proj_flops + attn_op_flops)
            bf16_flops = 0.0
        elif config.bytes_per_param == 2:
            fp8_flops = float(attn_op_flops)
            bf16_flops = float(proj_flops)
        else:
            raise ValueError(f"Unsupported bytes_per_param: {config.bytes_per_param}")
    else:
        raise ValueError(f"Unsupported attn_op_precision: {config.attn_op_precision}")
            
    return {'fp8_flops': fp8_flops, 'bf16_flops': bf16_flops}

def calculate_moe_flops(config: ModelConfig, seq_len: int, batch_size: int, is_prefill: bool) -> Dict[str, float]:
    '''Calculates the FLOPs for the MoE layer in a single layer.

    Args:
        config: Model configuration.
        seq_len: Current sequence length.
        batch_size: Batch size.
        is_prefill: True if prefill phase, False if decode phase.

    Returns:
        Dictionary with 'fp8_flops' and 'bf16_flops'.
    '''
    B = batch_size
    S = seq_len if is_prefill else 1  # In decode, we only process 1 token per step
    H = config.hidden_size
    E = config.num_experts
    K = config.num_activated_experts
    I = config.intermediate_size
    
    # Gating FLOPs (small, but included)
    gate_flops = 2 * B * S * H * E
    
    # Experts FLOPs
    # Each activated expert does gate_up (4HI) and down (2IH) = 6HI flops per token
    expert_flops = B * S * K * 6 * H * I
    
    flops = float(gate_flops + expert_flops)
    
    if config.bytes_per_param == 1:
        return {'fp8_flops': flops, 'bf16_flops': 0.0}
    elif config.bytes_per_param == 2:
        return {'fp8_flops': 0.0, 'bf16_flops': flops}
    else:
        raise ValueError(f"Unsupported bytes_per_param: {config.bytes_per_param}")

def calculate_memory_access(config: ModelConfig, seq_len: int, batch_size: int, is_prefill: bool) -> float:
    '''Calculates the memory access in bytes for a single layer.

    Focuses on weights and KV cache.

    Args:
        config: Model configuration.
        seq_len: Current sequence length.
        batch_size: Batch size.
        is_prefill: True if prefill phase, False if decode phase.

    Returns:
        Total bytes accessed in one layer.
    '''
    B = batch_size
    S = seq_len
    H = config.hidden_size
    N_kv = config.num_kv_heads
    D = config.attn_head_dim
    E = config.num_experts
    I = config.intermediate_size
    bytes_per_param = config.bytes_per_param
    
    kv_hidden = N_kv * D
    q_hidden = config.num_q_heads * config.attn_head_dim
    
    # Weights size in bytes
    attn_weights = (2 * H * q_hidden + 2 * H * kv_hidden) * bytes_per_param
    moe_weights = (H * E + 3 * E * H * I) * bytes_per_param
    total_weights = attn_weights + moe_weights
    
    if config.kv_cache_precision == 'fp8':
        bytes_per_kv_param = 1
    elif config.kv_cache_precision == 'bf16':
        bytes_per_kv_param = 2
    else:
        raise ValueError(f"Unsupported kv_cache_precision: {config.kv_cache_precision}")
    
    if is_prefill:
        # Read weights (once per layer)
        # Write KV cache for S tokens
        kv_write = 2 * B * S * kv_hidden * bytes_per_kv_param
        
        return float(total_weights + kv_write)
    else:
        # Decode
        # Read weights (for every token step)
        # Read KV cache for S tokens (past + current)
        kv_read = 2 * B * S * kv_hidden * bytes_per_kv_param
        # Write new KV cache for 1 token
        kv_write = 2 * B * 1 * kv_hidden * bytes_per_kv_param
        
        return float(total_weights + kv_read + kv_write)

def calculate_kv_cache_size(config: ModelConfig, seq_len: int, batch_size: int) -> float:
    '''Calculates the total memory capacity needed for the KV cache.

    Args:
        config: Model configuration.
        seq_len: Sequence length.
        batch_size: Batch size.

    Returns:
        Total bytes needed for KV cache across all layers.
    '''
    if config.kv_cache_precision == 'fp8':
        bytes_per_kv_param = 1
    elif config.kv_cache_precision == 'bf16':
        bytes_per_kv_param = 2
    else:
        raise ValueError(f"Unsupported kv_cache_precision: {config.kv_cache_precision}")
        
    kv_hidden = config.num_kv_heads * config.attn_head_dim
    # Factor of 2 for K and V
    return float(2 * batch_size * seq_len * kv_hidden * bytes_per_kv_param * config.num_layers)


def calculate_roofline(config: ModelConfig, hardware: HardwareSpec, seq_len: int, batch_size: int, is_prefill: bool) -> Dict[str, Any]:
    '''Calculates the roofline performance and latency.

    Args:
        config: Model configuration.
        hardware: Hardware specifications.
        seq_len: Current sequence length.
        batch_size: Batch size.
        is_prefill: True if prefill phase, False if decode phase.

    Returns:
        Dictionary containing calculated metrics.
    '''
    attn_flops = calculate_attention_flops(config, seq_len, batch_size, is_prefill)
    moe_flops = calculate_moe_flops(config, seq_len, batch_size, is_prefill)
    
    fp8_flops = (attn_flops['fp8_flops'] + moe_flops['fp8_flops']) * config.num_layers
    bf16_flops = (attn_flops['bf16_flops'] + moe_flops['bf16_flops']) * config.num_layers
    
    total_flops = fp8_flops + bf16_flops
    mem_access = calculate_memory_access(config, seq_len, batch_size, is_prefill) * config.num_layers
    
    # Convert to GB
    gb_access = mem_access / 1e9
    
    compute_latency = (fp8_flops / 1e12 / hardware.peak_fp8_flops) + (bf16_flops / 1e12 / hardware.peak_bf16_flops)
    memory_latency = gb_access / hardware.hbm_bandwidth  # seconds
    
    roofline_latency = max(compute_latency, memory_latency)
    
    kv_cache_size = calculate_kv_cache_size(config, seq_len, batch_size)
    
    return {
        'flops': total_flops,
        'mem_access_bytes': mem_access,
        'compute_latency_ms': compute_latency * 1000,
        'memory_latency_ms': memory_latency * 1000,
        'roofline_latency_ms': roofline_latency * 1000,
        'bound_by': 'compute' if compute_latency > memory_latency else 'memory',
        'kv_cache_size_gb': kv_cache_size / 1e9
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate roofline for LLM deployment.')
    parser.add_argument('--model_config', type=str, help='Path to model config JSON file.')
    parser.add_argument('--hardware_spec', type=str, help='Path to hardware spec JSON file.')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length.')
    parser.add_argument('--prefill_batch_size', type=int, default=1, help='Batch size for prefill phase.')
    parser.add_argument('--decode_batch_size', type=int, default=1, help='Batch size for decode phase.')
    
    args = parser.parse_args()
    
    prefill_batch = args.prefill_batch_size
    decode_batch = args.decode_batch_size
    
    if args.model_config:
        model_cfg = load_model_config(args.model_config)
    else:
        print("Warning: No model config file provided. Using dummy values.")
        model_cfg = ModelConfig(
            hidden_size=6144,
            num_q_heads=96,
            num_kv_heads=8,
            attn_head_dim=128,
            num_layers=62,
            vocab_size=151936,
            num_experts=160,
            num_activated_experts=8,
            intermediate_size=2560,
            bytes_per_param=1,
            attn_op_precision='bf16',
            kv_cache_precision='fp8'
        )
        
    if args.hardware_spec:
        hw_spec = load_hardware_spec(args.hardware_spec)
    else:
        print("Warning: No hardware spec file provided. Using dummy values.")
        hw_spec = HardwareSpec(
            peak_bf16_flops=2307.0,
            peak_fp8_flops=4614.0,
            hbm_bandwidth=7380.0
        )
        
    print(f"--- Prefill Phase (Seq Len {args.seq_len}, Batch {prefill_batch}) ---")
    prefill_res = calculate_roofline(model_cfg, hw_spec, seq_len=args.seq_len, batch_size=prefill_batch, is_prefill=True)
    for k, v in prefill_res.items():
        print(f'{k}: {v}')
        
    print(f"\n--- Decode Phase (Seq Len {args.seq_len}, Batch {decode_batch}, 1 step) ---")
    decode_res = calculate_roofline(model_cfg, hw_spec, seq_len=args.seq_len, batch_size=decode_batch, is_prefill=False)
    for k, v in decode_res.items():
        print(f'{k}: {v}')
