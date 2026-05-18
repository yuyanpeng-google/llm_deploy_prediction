'''Calculations for roofline model.

This module contains pure functions for calculating FLOPs, memory access,
communication latency, and overall roofline performance for LLM deployment.
'''

from dataclasses import dataclass
from typing import Any, Dict, Optional
from llm_deploy_prediction.roofline.config import HardwareSpec, ModelConfig, ShardingStrategy


def calculate_attention_flops(
    config: ModelConfig, seq_len: int, global_batch_size: int, is_prefill: bool
) -> Dict[str, float]:
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
        score_flops = (
            2 * B * config.num_q_heads * S * S * config.attn_head_dim
        )
        # Value Score * V
        value_flops = (
            2 * B * config.num_q_heads * S * S * config.attn_head_dim
        )
        # Output projection
        out_flops = 2 * B * S * q_hidden * H
    else:
        # Decode (S is current length, but we only process 1 new token)
        # Projections for 1 new token
        q_flops = 2 * B * 1 * H * q_hidden
        k_flops = 2 * B * 1 * H * kv_hidden
        v_flops = 2 * B * 1 * H * kv_hidden
        # Score QK^T (Q is 1 token, K is S tokens)
        score_flops = (
            2 * B * config.num_q_heads * 1 * S * config.attn_head_dim
        )
        # Value Score * V
        value_flops = (
            2 * B * config.num_q_heads * 1 * S * config.attn_head_dim
        )
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
            raise ValueError(
                f"Unsupported bytes_per_param: {config.bytes_per_param}"
            )
    elif config.attn_op_precision == 'fp8':
        if config.bytes_per_param == 1:
            fp8_flops = float(proj_flops + attn_op_flops)
            bf16_flops = 0.0
        elif config.bytes_per_param == 2:
            fp8_flops = float(attn_op_flops)
            bf16_flops = float(proj_flops)
        else:
            raise ValueError(
                f"Unsupported bytes_per_param: {config.bytes_per_param}"
            )
    else:
        raise ValueError(
            f"Unsupported attn_op_precision: {config.attn_op_precision}"
        )

    return {'fp8_flops': fp8_flops, 'bf16_flops': bf16_flops}


def calculate_moe_flops(
    config: ModelConfig, seq_len: int, global_batch_size: int, is_prefill: bool
) -> Dict[str, float]:
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
        raise ValueError(
            f"Unsupported bytes_per_param: {config.bytes_per_param}"
        )

    return {'fp8_flops': fp8_flops, 'bf16_flops': bf16_flops}


def calculate_memory_access(
    config: ModelConfig,
    seq_len: int,
    global_batch_size: int,
    is_prefill: bool,
    strategy: Optional[ShardingStrategy] = None,
) -> float:
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
        raise ValueError(
            f"Unsupported kv_cache_precision: {config.kv_cache_precision}"
        )

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

        return float(
            total_weights + kv_write + activation_access + moe_inter_access
        )
    else:
        # Decode
        # Read weights (for every token step)
        # Read KV cache for S tokens (past + current)
        kv_read = 2 * B * S * kv_hidden * bytes_per_kv_param
        # Write new KV cache for 1 token
        kv_write = 2 * B * 1 * kv_hidden * bytes_per_kv_param

        return float(
            total_weights
            + kv_read
            + kv_write
            + activation_access
            + moe_inter_access
        )


def calculate_communication_latency(
    config: ModelConfig,
    strategy: ShardingStrategy,
    hardware: HardwareSpec,
    seq_len: int,
    global_batch_size: int,
    is_prefill: bool,
) -> float:
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

    # MoE Communication - Phase 1: Before MoE (Routing/Gathering)
    moe_comm_type = strategy.moe_comm_type
    
    if strategy.attn_dp_degree > 1:
        if moe_comm_type == 'all_gather' and bw_link_ar > 0:
            # All-Gather inputs
            G = strategy.attn_dp_degree
            data_size = B * S * H * bytes_per_param
            comm_latency += ((G - 1) / G) * data_size / 1e9 / bw_link_ar
        elif moe_comm_type == 'a2a' and bw_link_a2a > 0:
            # Forward A2A (routing)
            ep_eff = strategy.moe_ep_degree if strategy.moe_ep_degree > 1 else strategy.attn_dp_degree
            K = config.num_activated_experts
            prob_visit_remote = 1 - (1 - 1 / ep_eff) ** K
            data_size = (
                (total_tokens / ep_eff)
                * (ep_eff - 1)
                * prob_visit_remote
                * H
                * bytes_per_param
            )
            comm_latency += data_size / 1e9 / bw_link_a2a

    # MoE Communication - Phase 2: After MoE (Unrouting/Reduction)
    if moe_comm_type == 'all_gather' and bw_link_ar > 0:
        # Option A: Full size All-Reduce for both TP and EP
        if strategy.attn_dp_degree > 1 or strategy.moe_ep_degree > 1:
            G = strategy.moe_ep_degree if strategy.moe_ep_degree > 1 else strategy.attn_dp_degree
            data_size = B * S * H * bytes_per_param
            comm_latency += 2 * ((G - 1) / G) * data_size / 1e9 / bw_link_ar
            
    elif moe_comm_type == 'a2a':
        # Option B: AR for TP and then A2A for EP
        
        # AR for TP
        P = strategy.moe_tp_degree
        if P > 1 and bw_link_ar > 0:
            # All-Reduce after down projection
            data_size = (total_tokens / strategy.moe_ep_degree) * H * bytes_per_param
            comm_latency += 2 * ((P - 1) / P) * data_size / 1e9 / bw_link_ar
            
        # A2A for EP (Unrouting)
        if strategy.attn_dp_degree > 1 or strategy.moe_ep_degree > 1:
            if bw_link_a2a > 0:
                ep_eff = strategy.moe_ep_degree if strategy.moe_ep_degree > 1 else strategy.attn_dp_degree
                K = config.num_activated_experts
                prob_visit_remote = 1 - (1 - 1 / ep_eff) ** K
                data_size = (
                    (total_tokens / ep_eff)
                    * (ep_eff - 1)
                    * prob_visit_remote
                    * H
                    * bytes_per_param
                )
                comm_latency += data_size / 1e9 / bw_link_a2a

    return comm_latency


def calculate_kv_cache_size(
    config: ModelConfig, seq_len: int, batch_size: int
) -> float:
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
        raise ValueError(
            f"Unsupported kv_cache_precision: {config.kv_cache_precision}"
        )

    kv_hidden = config.num_kv_heads * config.attn_head_dim
    # Factor of 2 for K and V
    return float(
        2
        * batch_size
        * seq_len
        * kv_hidden
        * bytes_per_kv_param
        * config.num_layers
    )


def calculate_weights_storage(
    config: ModelConfig, strategy: ShardingStrategy
) -> float:
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

    total_weights_per_chip = (
        attn_weights_per_chip + moe_weights_per_chip
    ) * config.num_layers

    return float(total_weights_per_chip)


@dataclass
class RooflineResult:
    '''Result of roofline analysis.

    Attributes:
        flops_per_chip: Total FLOPs per chip.
        weights_per_chip_gb: Weights storage per chip in GB.
        kv_cache_per_chip_gb: KV cache storage per chip in GB.
        hbm_usage_gb: Total HBM usage per chip in GB.
        hbm_capacity_gb: HBM capacity per chip in GB.
        mem_access_bytes_per_chip: Total memory access in bytes per chip.
        compute_latency_ms: Compute latency in milliseconds.
        memory_latency_ms: Memory latency in milliseconds.
        comm_latency_ms: Communication latency in milliseconds.
        roofline_latency_ms: Roofline latency (max of compute and memory) in milliseconds.
        total_latency_ms: Total latency (roofline + communication) in milliseconds.
        bound_by: Which resource bounds the performance ('compute' or 'memory').
        kv_cache_size_gb: Total KV cache size in GB across all chips.
        ttft_ms: Time to first token in milliseconds (if prefill).
        tpot_ms: Time per output token in milliseconds (if decode).
        throughput_per_chip: Throughput per chip (tokens/s).
    '''

    flops_per_chip: float
    weights_per_chip_gb: float
    kv_cache_per_chip_gb: float
    hbm_usage_gb: float
    hbm_capacity_gb: float
    mem_access_bytes_per_chip: float
    compute_latency_ms: float
    memory_latency_ms: float
    comm_latency_ms: float
    roofline_latency_ms: float
    total_latency_ms: float
    bound_by: str
    kv_cache_size_gb: float
    ttft_ms: float
    tpot_ms: float
    throughput_per_chip: float


def calculate_roofline(
    config: ModelConfig,
    hardware: HardwareSpec,
    seq_len: int,
    batch_size: int,
    is_prefill: bool,
    strategy: Optional[ShardingStrategy] = None,
) -> RooflineResult:
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
        if (
            strategy.attn_tp_degree * strategy.attn_dp_degree
            != strategy.num_chips
        ):
            raise ValueError(
                f"Attn degrees product ({strategy.attn_tp_degree} * {strategy.attn_dp_degree}) != num_chips ({strategy.num_chips})"
            )
        if (
            strategy.moe_tp_degree * strategy.moe_ep_degree
            != strategy.num_chips
        ):
            raise ValueError(
                f"MoE degrees product ({strategy.moe_tp_degree} * {strategy.moe_ep_degree}) != num_chips ({strategy.num_chips})"
            )
        num_chips = strategy.num_chips
    else:
        num_chips = 1

    # B is now global batch size
    B = batch_size

    attn_flops = calculate_attention_flops(config, seq_len, B, is_prefill)
    moe_flops = calculate_moe_flops(config, seq_len, B, is_prefill)

    # Calculate total FLOPs first
    total_fp8_flops = (
        attn_flops['fp8_flops'] + moe_flops['fp8_flops']
    ) * config.num_layers
    total_bf16_flops = (
        attn_flops['bf16_flops'] + moe_flops['bf16_flops']
    ) * config.num_layers

    # Per-chip FLOPs
    fp8_flops = total_fp8_flops / num_chips
    bf16_flops = total_bf16_flops / num_chips

    total_flops = fp8_flops + bf16_flops

    # Total memory access
    total_mem_access = (
        calculate_memory_access(config, seq_len, B, is_prefill, strategy)
        * config.num_layers
    )
    # Per-chip memory access
    mem_access = total_mem_access / num_chips

    # Convert to GB
    gb_access = mem_access / 1e9

    compute_latency = (fp8_flops / 1e12 / hardware.peak_fp8_flops) + (
        bf16_flops / 1e12 / hardware.peak_bf16_flops
    )
    memory_latency = gb_access / hardware.hbm_bandwidth  # seconds

    comm_latency = 0.0
    if strategy:
        comm_latency = (
            calculate_communication_latency(
                config, strategy, hardware, seq_len, B, is_prefill
            )
            * config.num_layers
        )

    roofline_latency = max(compute_latency, memory_latency)
    total_latency = roofline_latency + comm_latency

    # Total KV cache in system
    kv_cache_size = calculate_kv_cache_size(config, seq_len, B)

    # Calculate TTFT and TPOT
    ttft = total_latency if is_prefill else 0.0
    tpot = total_latency if not is_prefill else 0.0

    # Calculate throughput per chip
    if is_prefill:
        throughput_per_chip = (
            (B * seq_len) / (total_latency * num_chips)
            if total_latency > 0
            else 0.0
        )
    else:
        throughput_per_chip = (
            B / (total_latency * num_chips) if total_latency > 0 else 0.0
        )

    # HBM Usage Calculation
    effective_strategy = (
        strategy if strategy else ShardingStrategy(num_chips=1)
    )
    weights_storage = calculate_weights_storage(config, effective_strategy)
    kv_cache_per_chip = kv_cache_size / num_chips
    hbm_usage_bytes = weights_storage + kv_cache_per_chip
    hbm_usage_gb = hbm_usage_bytes / 1e9

    return RooflineResult(
        flops_per_chip=total_flops,
        weights_per_chip_gb=weights_storage / 1e9,
        kv_cache_per_chip_gb=kv_cache_per_chip / 1e9,
        hbm_usage_gb=hbm_usage_gb,
        hbm_capacity_gb=hardware.hbm_capacity,
        mem_access_bytes_per_chip=mem_access,
        compute_latency_ms=compute_latency * 1000,
        memory_latency_ms=memory_latency * 1000,
        comm_latency_ms=comm_latency * 1000,
        roofline_latency_ms=roofline_latency * 1000,
        total_latency_ms=total_latency * 1000,
        bound_by='compute' if compute_latency > memory_latency else 'memory',
        kv_cache_size_gb=kv_cache_size / 1e9,
        ttft_ms=ttft * 1000 if is_prefill else 0.0,
        tpot_ms=tpot * 1000 if not is_prefill else 0.0,
        throughput_per_chip=throughput_per_chip,
    )
