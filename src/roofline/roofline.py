'''Roofline model for calculating theoretical performance of LLM deployment.

This script calculates FLOPs, memory access, and latency for LLM models,
focusing on per-layer calculation for prefill and decode phases.
'''

from dataclasses import dataclass
from typing import Dict, Any, List
import json
import argparse
import os
import glob

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
    
    ici_ar_ag_bandwidth: float
    '''Unidirectional ICI bandwidth for All-Reduce and All-Gather in GB/s.'''
    
    ici_a2a_bandwidth: float
    '''Unidirectional ICI bandwidth for All-to-All in GB/s.'''
    
    hbm_capacity: float = 192.0
    '''HBM capacity per chip in GB.'''
    
    num_chips: int = 4
    '''Number of chips associated with this spec.'''

@dataclass
class ShardingStrategy:
    '''Configuration for sharding strategy.'''
    
    num_chips: int
    '''Total number of chips used.'''
    
    attn_tp_degree: int = 1
    '''TP degree for attention.'''
    
    attn_dp_degree: int = 1
    '''DP degree for attention.'''
    
    moe_tp_degree: int = 1
    '''TP degree for MoE.'''
    
    moe_ep_degree: int = 1
    '''EP degree for MoE.'''

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

def calculate_attention_flops(config: ModelConfig, seq_len: int, global_batch_size: int, is_prefill: bool) -> Dict[str, float]:
    '''Calculates the FLOPs for the attention mechanism in a single layer.

    Args:
        config: Model configuration.
        seq_len: Current sequence length.
        global_batch_size: Global batch size.
        is_prefill: True if prefill phase, False if decode phase.

    Returns:
        Dictionary with 'fp8_flops' and 'bf16_flops'.
    '''
    B = global_batch_size
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

def calculate_moe_flops(config: ModelConfig, seq_len: int, global_batch_size: int, is_prefill: bool) -> Dict[str, float]:
    '''Calculates the FLOPs for the MoE layer in a single layer.

    Args:
        config: Model configuration.
        seq_len: Current sequence length.
        global_batch_size: Global batch size.
        is_prefill: True if prefill phase, False if decode phase.

    Returns:
        Dictionary with 'fp8_flops' and 'bf16_flops'.
    '''
    B = global_batch_size
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
        fp8_flops = flops
        bf16_flops = 0.0
    elif config.bytes_per_param == 2:
        fp8_flops = 0.0
        bf16_flops = flops
    else:
        raise ValueError(f"Unsupported bytes_per_param: {config.bytes_per_param}")
        
    return {'fp8_flops': fp8_flops, 'bf16_flops': bf16_flops}

def calculate_memory_access(config: ModelConfig, seq_len: int, global_batch_size: int, is_prefill: bool, strategy: ShardingStrategy = None) -> float:
    '''Calculates the memory access in bytes for a single layer.

    Focuses on weights and KV cache.

    Args:
        config: Model configuration.
        seq_len: Current sequence length.
        global_batch_size: Global batch size.
        is_prefill: True if prefill phase, False if decode phase.
        strategy: Sharding strategy.

    Returns:
        Total bytes accessed in one layer across the whole system.
    '''
    B = global_batch_size
    S = seq_len
    H = config.hidden_size
    N_kv = config.num_kv_heads
    D = config.attn_head_dim
    E = config.num_experts
    I = config.intermediate_size
    K = config.num_activated_experts
    bytes_per_param = config.bytes_per_param
    
    kv_hidden = N_kv * D
    q_hidden = config.num_q_heads * config.attn_head_dim
    
    # Weights size in bytes
    attn_weights = (2 * H * q_hidden + 2 * H * kv_hidden) * bytes_per_param
    moe_weights = (H * E + 3 * E * H * I) * bytes_per_param
    
    if strategy:
        attn_weights *= strategy.attn_dp_degree
            
    total_weights = attn_weights + moe_weights
    
    if config.kv_cache_precision == 'fp8':
        bytes_per_kv_param = 1
    elif config.kv_cache_precision == 'bf16':
        bytes_per_kv_param = 2
    else:
        raise ValueError(f"Unsupported kv_cache_precision: {config.kv_cache_precision}")
        
    # Activation access: Read input, Write after Attn, Read before MoE, Write after MoE
    # Approx 4 accesses to the activation tensor of size B * (S if is_prefill else 1) * H
    S_eff = S if is_prefill else 1
    activation_access = 4 * B * S_eff * H * bytes_per_param
    
    # MoE intermediate activation access (not fused): Read and Write between gate_up and down
    # Size: B * S_eff * K * I
    moe_inter_access = 2 * B * S_eff * K * I * bytes_per_param
    
    if is_prefill:
        # Read weights (once per layer)
        # Write KV cache for S tokens
        kv_write = 2 * B * S * kv_hidden * bytes_per_kv_param
        
        return float(total_weights + kv_write + activation_access + moe_inter_access)
    else:
        # Decode
        # Read weights (for every token step)
        # Read KV cache for S tokens (past + current)
        kv_read = 2 * B * S * kv_hidden * bytes_per_kv_param
        # Write new KV cache for 1 token
        kv_write = 2 * B * 1 * kv_hidden * bytes_per_kv_param
        
        return float(total_weights + kv_read + kv_write + activation_access + moe_inter_access)

def calculate_communication_latency(config: ModelConfig, strategy: ShardingStrategy, hardware: HardwareSpec, seq_len: int, global_batch_size: int, is_prefill: bool) -> float:
    '''Calculates the communication latency in seconds for a single layer.

    Args:
        config: Model configuration.
        strategy: Sharding strategy.
        hardware: Hardware specifications.
        seq_len: Current sequence length.
        global_batch_size: Global batch size.
        is_prefill: True if prefill phase, False if decode phase.

    Returns:
        Communication latency in seconds.
    '''
    if strategy is None:
        return 0.0
        
    B = global_batch_size
    S = seq_len if is_prefill else 1
    H = config.hidden_size
    bytes_per_param = config.bytes_per_param
    
    comm_latency = 0.0
    
    bw_link_ar = hardware.ici_ar_ag_bandwidth
    bw_link_a2a = hardware.ici_a2a_bandwidth
    
    # Attention Communication
    P = strategy.attn_tp_degree
    if P > 1 and bw_link_ar > 0:
        # All-Reduce after output projection
        # Data size = (B / attn_dp_degree) * S * H * bytes_per_param
        data_size = (B / strategy.attn_dp_degree) * S * H * bytes_per_param
        comm_latency += 2 * ((P - 1) / P) * data_size / 1e9 / bw_link_ar
        
    # MoE Communication
    # Total tokens in system = B * S
    total_tokens = B * S
    
    # MoE All-Reduce
    P = strategy.moe_tp_degree
    if P > 1 and bw_link_ar > 0:
        # All-Reduce after down projection
        # Data size handled by this group = (Total Tokens / moe_ep_degree) * H * bytes_per_param
        data_size = (total_tokens / strategy.moe_ep_degree) * H * bytes_per_param
        comm_latency += 2 * ((P - 1) / P) * data_size / 1e9 / bw_link_ar
        
    # MoE All-to-All
    P = strategy.moe_ep_degree
    if P > 1 and bw_link_a2a > 0:
        # All-to-All communication to route tokens
        # Optimized: Tokens are sent at most once per destination group.
        # Expected number of remote groups a token visits: (EP - 1) * (1 - (1 - 1/EP)^K)
        K = config.num_activated_experts
        ep = strategy.moe_ep_degree
        prob_visit_remote = 1 - (1 - 1/ep)**K
        data_size = (total_tokens / ep) * (ep - 1) * prob_visit_remote * H * bytes_per_param
        
        # All-to-All latency approx data_size / bw_link
        comm_latency += data_size / 1e9 / bw_link_a2a
        
        # All-to-All communication to unroute tokens (send back)
        # Assuming same data size for return trip
        comm_latency += data_size / 1e9 / bw_link_a2a
        
    return comm_latency

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


def calculate_weights_storage(config: ModelConfig, strategy: ShardingStrategy) -> float:
    '''Calculates the model weights storage in bytes per chip.

    Args:
        config: Model configuration.
        strategy: Sharding strategy.

    Returns:
        Weights storage in bytes per chip.
    '''
    H = config.hidden_size
    N_kv = config.num_kv_heads
    D = config.attn_head_dim
    E = config.num_experts
    I = config.intermediate_size
    bytes_per_param = config.bytes_per_param
    
    kv_hidden = N_kv * D
    q_hidden = config.num_q_heads * config.attn_head_dim
    
    # Attention weights (one replica)
    attn_weights = (2 * H * q_hidden + 2 * H * kv_hidden) * bytes_per_param
    
    # Sharded by TP for attention
    attn_weights_per_chip = attn_weights / strategy.attn_tp_degree
    
    # MoE weights
    # Gating is replicated
    moe_gating = (H * E) * bytes_per_param
    # Experts are distributed across all chips (EP * TP = num_chips)
    moe_experts = (3 * E * H * I) * bytes_per_param
    
    moe_weights_per_chip = moe_gating + (moe_experts / strategy.num_chips)
    
    total_weights_per_chip = (attn_weights_per_chip + moe_weights_per_chip) * config.num_layers
    
    return float(total_weights_per_chip)


def calculate_roofline(config: ModelConfig, hardware: HardwareSpec, seq_len: int, batch_size: int, is_prefill: bool, strategy: ShardingStrategy = None) -> Dict[str, Any]:
    '''Calculates the roofline performance and latency.

    Args:
        config: Model configuration.
        hardware: Hardware specifications.
        seq_len: Current sequence length.
        batch_size: Global batch size.
        is_prefill: True if prefill phase, False if decode phase.
        strategy: Sharding strategy.

    Returns:
        Dictionary containing calculated metrics.
    '''
    # Validation
    if strategy:
        if strategy.attn_tp_degree * strategy.attn_dp_degree != strategy.num_chips:
            raise ValueError(f"Attn degrees product ({strategy.attn_tp_degree} * {strategy.attn_dp_degree}) != num_chips ({strategy.num_chips})")
        if strategy.moe_tp_degree * strategy.moe_ep_degree != strategy.num_chips:
            raise ValueError(f"MoE degrees product ({strategy.moe_tp_degree} * {strategy.moe_ep_degree}) != num_chips ({strategy.num_chips})")
        num_chips = strategy.num_chips
    else:
        num_chips = 1
        
    # B is now global batch size
    B = batch_size
    
    attn_flops = calculate_attention_flops(config, seq_len, B, is_prefill)
    moe_flops = calculate_moe_flops(config, seq_len, B, is_prefill)
    
    # Calculate total FLOPs first
    total_fp8_flops = (attn_flops['fp8_flops'] + moe_flops['fp8_flops']) * config.num_layers
    total_bf16_flops = (attn_flops['bf16_flops'] + moe_flops['bf16_flops']) * config.num_layers
    
    # Per-chip FLOPs
    fp8_flops = total_fp8_flops / num_chips
    bf16_flops = total_bf16_flops / num_chips
    
    total_flops = fp8_flops + bf16_flops
    
    # Total memory access
    total_mem_access = calculate_memory_access(config, seq_len, B, is_prefill, strategy) * config.num_layers
    # Per-chip memory access
    mem_access = total_mem_access / num_chips
    
    # Convert to GB
    gb_access = mem_access / 1e9
    
    compute_latency = (fp8_flops / 1e12 / hardware.peak_fp8_flops) + (bf16_flops / 1e12 / hardware.peak_bf16_flops)
    memory_latency = gb_access / hardware.hbm_bandwidth  # seconds
    
    comm_latency = 0.0
    if strategy:
        comm_latency = calculate_communication_latency(config, strategy, hardware, seq_len, B, is_prefill) * config.num_layers
        
    roofline_latency = max(compute_latency, memory_latency)
    total_latency = roofline_latency + comm_latency
    
    # Total KV cache in system
    kv_cache_size = calculate_kv_cache_size(config, seq_len, B)
    
    # Calculate TTFT and TPOT
    ttft = total_latency if is_prefill else 0.0
    tpot = total_latency if not is_prefill else 0.0
    
    # Calculate throughput per chip
    if is_prefill:
        throughput_per_chip = (B * seq_len) / (total_latency * num_chips) if total_latency > 0 else 0.0
    else:
        throughput_per_chip = B / (total_latency * num_chips) if total_latency > 0 else 0.0
        
    # HBM Usage Calculation
    effective_strategy = strategy if strategy else ShardingStrategy(num_chips=1)
    weights_storage = calculate_weights_storage(config, effective_strategy)
    kv_cache_per_chip = kv_cache_size / num_chips
    hbm_usage_bytes = weights_storage + kv_cache_per_chip
    hbm_usage_gb = hbm_usage_bytes / 1e9
        
    return {
        'flops_per_chip': total_flops,
        'weights_per_chip_gb': weights_storage / 1e9,
        'kv_cache_per_chip_gb': kv_cache_per_chip / 1e9,
        'hbm_usage_gb': hbm_usage_gb,
        'hbm_capacity_gb': hardware.hbm_capacity,
        'mem_access_bytes_per_chip': mem_access,
        'compute_latency_ms': compute_latency * 1000,
        'memory_latency_ms': memory_latency * 1000,
        'comm_latency_ms': comm_latency * 1000,
        'roofline_latency_ms': roofline_latency * 1000,
        'total_latency_ms': total_latency * 1000,
        'bound_by': 'compute' if compute_latency > memory_latency else 'memory',
        'kv_cache_size_gb': kv_cache_size / 1e9,
        'ttft_ms': ttft * 1000 if is_prefill else 0.0,
        'tpot_ms': tpot * 1000 if not is_prefill else 0.0,
        'throughput_per_chip': throughput_per_chip
    }

def print_markdown_table(headers: List[str], rows: List[List[Any]]) -> None:
    '''Prints a list of rows as a markdown table.
    
    Args:
        headers: List of column headers.
        rows: List of rows, where each row is a list of values.
    '''
    if not headers or not rows:
        return
        
    # Calculate max width for each column
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    # Print header
    header_str = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    print(f"| {header_str} |")
    
    # Print separator
    sep_str = " | ".join("-" * widths[i] for i in range(len(headers)))
    print(f"| {sep_str} |")
    
    # Print rows
    for row in rows:
        row_str = " | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(row))
        print(f"| {row_str} |")

def generate_strategies(num_chips: int) -> List[ShardingStrategy]:
    '''Generates all valid ShardingStrategy combinations for a given num_chips.
    
    A strategy is valid if:
    - attn_tp_degree * attn_dp_degree == num_chips
    - moe_tp_degree * moe_ep_degree == num_chips
    
    Args:
        num_chips: Total number of chips.
        
    Returns:
        A list of valid ShardingStrategy instances.
    '''
    if num_chips <= 0:
        raise ValueError("num_chips must be a positive integer.")
        
    factors = [i for i in range(1, num_chips + 1) if num_chips % i == 0]
    
    attn_pairs = [(tp, num_chips // tp) for tp in factors]
    moe_pairs = [(tp, num_chips // tp) for tp in factors]
    
    strategies = []
    for attn_tp, attn_dp in attn_pairs:
        for moe_tp, moe_ep in moe_pairs:
            strategies.append(
                ShardingStrategy(
                    num_chips=num_chips,
                    attn_tp_degree=attn_tp,
                    attn_dp_degree=attn_dp,
                    moe_tp_degree=moe_tp,
                    moe_ep_degree=moe_ep
                )
            )
    return strategies

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate roofline for LLM deployment.')
    parser.add_argument('--model_config', type=str, help='Path to model config JSON file.')
    parser.add_argument('--hardware_spec', type=str, nargs='+', help='Path to hardware spec JSON file(s).')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length.')
    prefill_group = parser.add_mutually_exclusive_group()
    prefill_group.add_argument('--prefill_batch_size', type=int, default=1, help='Global batch size for prefill phase. Conflicts with --prefill_local_batch_size.')
    prefill_group.add_argument('--prefill_local_batch_size', type=int, help='Local batch size per DP group for prefill phase. Conflicts with --prefill_batch_size.')
    
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument('--decode_batch_size', type=int, default=1, help='Global batch size for decode phase. Conflicts with --decode_local_batch_size.')
    decode_group.add_argument('--decode_local_batch_size', type=int, help='Local batch size per DP group for decode phase. Conflicts with --decode_batch_size.')
    parser.add_argument('--table', action='store_true', help='Output results in markdown table format.')
    parser.add_argument('--num_chips', type=int, default=4, help='Number of chips for grid search (fallback if not in spec).')
    
    args = parser.parse_args()
    
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
        
    hw_specs = []
    spec_paths = []
    if args.hardware_spec:
        for spec_path in args.hardware_spec:
            if os.path.isdir(spec_path):
                json_files = glob.glob(os.path.join(spec_path, "*.json"))
                json_files.sort()
                for json_file in json_files:
                    hw_specs.append(load_hardware_spec(json_file))
                    spec_paths.append(json_file)
            else:
                hw_specs.append(load_hardware_spec(spec_path))
                spec_paths.append(spec_path)
    else:
        print("Warning: No hardware spec file provided. Using dummy values.")
        hw_specs.append(HardwareSpec(
            peak_bf16_flops=2307.0,
            peak_fp8_flops=4614.0,
            hbm_bandwidth=7380.0,
            ici_ar_ag_bandwidth=600.0,
            ici_a2a_bandwidth=200.0,
            num_chips=args.num_chips
        ))
        spec_paths.append("Default")
        
    prefill_results = []
    decode_results = []
    
    for hw_spec, spec_path in zip(hw_specs, spec_paths):
        strategies = generate_strategies(hw_spec.num_chips)
        spec_name = os.path.basename(spec_path) if spec_path != "Default" else "Default"
        
        for strategy in strategies:
            strategy_str = f"Spec={spec_name}, Chips={strategy.num_chips}, Attn(TP={strategy.attn_tp_degree},DP={strategy.attn_dp_degree}), MoE(TP={strategy.moe_tp_degree},EP={strategy.moe_ep_degree})"
            
            if args.prefill_local_batch_size is not None:
                prefill_batch = args.prefill_local_batch_size * strategy.attn_dp_degree
            else:
                prefill_batch = args.prefill_batch_size
                
            if args.decode_local_batch_size is not None:
                decode_batch = args.decode_local_batch_size * strategy.attn_dp_degree
            else:
                decode_batch = args.decode_batch_size
                
            try:
                prefill_res = calculate_roofline(model_cfg, hw_spec, seq_len=args.seq_len, batch_size=prefill_batch, is_prefill=True, strategy=strategy)
                prefill_results.append((strategy_str, prefill_batch, prefill_res))
            except ValueError as e:
                print(f"Error calculating prefill for {strategy_str}: {e}")
                
            try:
                decode_res = calculate_roofline(model_cfg, hw_spec, seq_len=args.seq_len, batch_size=decode_batch, is_prefill=False, strategy=strategy)
                decode_results.append((strategy_str, decode_batch, decode_res))
            except ValueError as e:
                print(f"Error calculating decode for {strategy_str}: {e}")

    if args.table:
        print("\n=== Prefill Phase Table ===")
        headers = ["Strategy", "Batch Size", "Throughput/Chip", "TTFT (ms)", "Bound By", "Total Latency (ms)", "KV Cache (GB)"]
        rows = []
        for strategy_str, batch, res in prefill_results:
            rows.append([
                strategy_str,
                batch,
                f"{res['throughput_per_chip']:.2f}",
                f"{res['ttft_ms']:.2f}",
                res['bound_by'],
                f"{res['total_latency_ms']:.2f}",
                f"{res['kv_cache_size_gb']:.2f}"
            ])
        print_markdown_table(headers, rows)
        
        print("\n=== Decode Phase Table ===")
        headers = ["Strategy", "Batch Size", "Throughput/Chip", "TPOT (ms)", "Bound By", "Total Latency (ms)", "KV Cache (GB)"]
        rows = []
        for strategy_str, batch, res in decode_results:
            rows.append([
                strategy_str,
                batch,
                f"{res['throughput_per_chip']:.2f}",
                f"{res['tpot_ms']:.2f}",
                res['bound_by'],
                f"{res['total_latency_ms']:.2f}",
                f"{res['kv_cache_size_gb']:.2f}"
            ])
        print_markdown_table(headers, rows)
        
        print("\n=== Latency Comparison Table ===")
        headers = ["Strategy", "Phase", "Batch Size", "Compute Latency (ms)", "Memory Latency (ms)", "ICI Latency (ms)", "Gap (ms)", "Bound By"]
        rows = []
        for strategy_str, batch, res in prefill_results:
            gap = abs(res['compute_latency_ms'] - res['memory_latency_ms'])
            rows.append([
                strategy_str,
                "Prefill",
                batch,
                f"{res['compute_latency_ms']:.2f}",
                f"{res['memory_latency_ms']:.2f}",
                f"{res['comm_latency_ms']:.2f}",
                f"{gap:.2f}",
                res['bound_by']
            ])
        for strategy_str, batch, res in decode_results:
            gap = abs(res['compute_latency_ms'] - res['memory_latency_ms'])
            rows.append([
                strategy_str,
                "Decode",
                batch,
                f"{res['compute_latency_ms']:.2f}",
                f"{res['memory_latency_ms']:.2f}",
                f"{res['comm_latency_ms']:.2f}",
                f"{gap:.2f}",
                res['bound_by']
            ])
        print_markdown_table(headers, rows)
        
        print("\n=== HBM Usage Table ===")
        headers = ["Strategy", "Phase", "Batch Size", "Weights/Chip (GB)", "KV Cache/Chip (GB)", "Total HBM/Chip (GB)", "Capacity (GB)", "Util (%)"]
        rows = []
        for strategy_str, batch, res in prefill_results:
            util = (res['hbm_usage_gb'] / res['hbm_capacity_gb']) * 100 if res['hbm_capacity_gb'] > 0 else 0.0
            rows.append([
                strategy_str,
                "Prefill",
                batch,
                f"{res['weights_per_chip_gb']:.2f}",
                f"{res['kv_cache_per_chip_gb']:.2f}",
                f"{res['hbm_usage_gb']:.2f}",
                f"{res['hbm_capacity_gb']:.2f}",
                f"{util:.2f}"
            ])
        for strategy_str, batch, res in decode_results:
            util = (res['hbm_usage_gb'] / res['hbm_capacity_gb']) * 100 if res['hbm_capacity_gb'] > 0 else 0.0
            rows.append([
                strategy_str,
                "Decode",
                batch,
                f"{res['weights_per_chip_gb']:.2f}",
                f"{res['kv_cache_per_chip_gb']:.2f}",
                f"{res['hbm_usage_gb']:.2f}",
                f"{res['hbm_capacity_gb']:.2f}",
                f"{util:.2f}"
            ])
        print_markdown_table(headers, rows)
        
        print("\n=== Prefill Throughput/Chip Sorted Table ===")
        headers = ["Strategy", "Batch Size", "Throughput/Chip", "Total Latency (ms)", "Bound By", "HBM Util (%)"]
        
        prefill_sorted = sorted(prefill_results, key=lambda x: x[2]['throughput_per_chip'], reverse=True)
        rows = []
        for strategy_str, batch, res in prefill_sorted:
            util = (res['hbm_usage_gb'] / res['hbm_capacity_gb']) * 100 if res['hbm_capacity_gb'] > 0 else 0.0
            rows.append([
                strategy_str,
                batch,
                f"{res['throughput_per_chip']:.2f}",
                f"{res['total_latency_ms']:.2f}",
                res['bound_by'],
                f"{util:.2f}"
            ])
        print_markdown_table(headers, rows)
        
        print("\n=== Decode Throughput/Chip Sorted Table ===")
        headers = ["Strategy", "Batch Size", "Throughput/Chip", "Total Latency (ms)", "Bound By", "HBM Util (%)"]
        
        decode_sorted = sorted(decode_results, key=lambda x: x[2]['throughput_per_chip'], reverse=True)
        rows = []
        for strategy_str, batch, res in decode_sorted:
            util = (res['hbm_usage_gb'] / res['hbm_capacity_gb']) * 100 if res['hbm_capacity_gb'] > 0 else 0.0
            rows.append([
                strategy_str,
                batch,
                f"{res['throughput_per_chip']:.2f}",
                f"{res['total_latency_ms']:.2f}",
                res['bound_by'],
                f"{util:.2f}"
            ])
        print_markdown_table(headers, rows)
    else:
        for strategy_str, res in prefill_results:
            print(f"\n=== Strategy: {strategy_str} ===")
            print(f"--- Prefill Phase (Seq Len {args.seq_len}, Batch {prefill_batch}) ---")
            for k, v in res.items():
                print(f'{k}: {v}')
                
        for strategy_str, res in decode_results:
            print(f"\n=== Strategy: {strategy_str} ===")
            print(f"--- Decode Phase (Seq Len {args.seq_len}, Batch {decode_batch}, 1 step) ---")
            for k, v in res.items():
                print(f'{k}: {v}')
