'''Unit tests for roofline.py.

This module contains tests to verify the calculations in roofline.py.
'''

import unittest
from typing import Dict, Any

from llm_deploy_prediction.roofline.config import ModelConfig, HardwareSpec, ShardingStrategy
from llm_deploy_prediction.roofline.calculations import (
    calculate_attention_flops,
    calculate_moe_flops,
    calculate_memory_access,
    calculate_communication_latency,
    calculate_roofline,
    RooflineResult,
)
from llm_deploy_prediction.roofline.cli import parse_args, main

def get_default_config(**kwargs) -> ModelConfig:
    '''Returns a default ModelConfig for testing.'''
    defaults = dict(
        hidden_size=1024,
        num_q_heads=8,
        num_kv_heads=8,
        attn_head_dim=128,
        num_layers=1,
        vocab_size=1000,
        num_experts=4,
        num_activated_experts=2,
        intermediate_size=2048,
        bytes_per_param=2,
        attn_op_precision='bf16',
        kv_cache_precision='bf16'
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)

def get_default_hardware(**kwargs) -> HardwareSpec:
    '''Returns a default HardwareSpec for testing.'''
    defaults = dict(
        peak_bf16_flops=100.0,
        peak_fp8_flops=200.0,
        hbm_bandwidth=50.0,
        ici_ar_ag_bandwidth=10.0,
        ici_a2a_bandwidth=5.0,
        num_chips=4
    )
    defaults.update(kwargs)
    return HardwareSpec(**defaults)






class TestRooflineCalculations(unittest.TestCase):
    '''Test case for roofline calculations.'''


    def test_calculate_attention_flops_mha_prefill(self) -> None:
        '''Test attention FLOPs calculation for prefill phase.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=8,
            attn_head_dim=128,
            bytes_per_param=2,
            attn_op_precision='bf16'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=True)
        

        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * S * H * q_hidden
        k_flops = 2 * B * S * H * kv_hidden
        v_flops = 2 * B * S * H * kv_hidden
        score_flops = 2 * B * N_q * S * S * D
        value_flops = 2 * B * N_q * S * S * D
        out_flops = 2 * B * S * q_hidden * H
        
        expected_bf16 = float(q_flops + k_flops + v_flops + score_flops + value_flops + out_flops)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_attention_flops_fp8_all(self) -> None:
        '''Test attention FLOPs calculation for all FP8 precision.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=8,
            attn_head_dim=128,
            bytes_per_param=1,
            attn_op_precision='fp8'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=True)
        
        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * S * H * q_hidden
        k_flops = 2 * B * S * H * kv_hidden
        v_flops = 2 * B * S * H * kv_hidden
        score_flops = 2 * B * N_q * S * S * D
        value_flops = 2 * B * N_q * S * S * D
        out_flops = 2 * B * S * q_hidden * H
        
        expected_fp8 = float(q_flops + k_flops + v_flops + score_flops + value_flops + out_flops)
        self.assertAlmostEqual(flops['fp8_flops'], expected_fp8)
        self.assertEqual(flops['bf16_flops'], 0.0)

    def test_calculate_attention_flops_fp8_mixed(self) -> None:
        '''Test attention FLOPs calculation for mixed FP8/BF16 precision.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=8,
            attn_head_dim=128,
            bytes_per_param=2,
            attn_op_precision='fp8'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=True)
        
        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * S * H * q_hidden
        k_flops = 2 * B * S * H * kv_hidden
        v_flops = 2 * B * S * H * kv_hidden
        score_flops = 2 * B * N_q * S * S * D
        value_flops = 2 * B * N_q * S * S * D
        out_flops = 2 * B * S * q_hidden * H
        
        # With bytes_per_param=2 and attn_op_precision='fp8':
        # proj_flops are BF16, attn_op_flops (score + value) are FP8
        expected_bf16 = float(q_flops + k_flops + v_flops + out_flops)
        expected_fp8 = float(score_flops + value_flops)
        
        self.assertAlmostEqual(flops['fp8_flops'], expected_fp8)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)

    def test_calculate_attention_flops_w8a16(self) -> None:
        '''Test attention FLOPs calculation for W8A16 simulated by bytes_per_param=1 and attn_op_precision='bf16'.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=8,
            attn_head_dim=128,
            bytes_per_param=1,
            attn_op_precision='bf16'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=True)
        
        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * S * H * q_hidden
        k_flops = 2 * B * S * H * kv_hidden
        v_flops = 2 * B * S * H * kv_hidden
        score_flops = 2 * B * N_q * S * S * D
        value_flops = 2 * B * N_q * S * S * D
        out_flops = 2 * B * S * q_hidden * H
        
        # With bytes_per_param=1 and attn_op_precision='bf16':
        # proj_flops are FP8, attn_op_flops are BF16
        expected_fp8 = float(q_flops + k_flops + v_flops + out_flops)
        expected_bf16 = float(score_flops + value_flops)
        
        self.assertAlmostEqual(flops['fp8_flops'], expected_fp8)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)

    def test_calculate_attention_flops_gqa_prefill(self) -> None:
        '''Test attention FLOPs calculation for GQA in prefill phase.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=2,
            attn_head_dim=128,
            bytes_per_param=2,
            attn_op_precision='bf16'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=True)
        

        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * S * H * q_hidden
        k_flops = 2 * B * S * H * kv_hidden
        v_flops = 2 * B * S * H * kv_hidden
        score_flops = 2 * B * N_q * S * S * D
        value_flops = 2 * B * N_q * S * S * D
        out_flops = 2 * B * S * q_hidden * H
        
        expected_bf16 = float(q_flops + k_flops + v_flops + score_flops + value_flops + out_flops)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_attention_flops_mqa_prefill(self) -> None:
        '''Test attention FLOPs calculation for MQA in prefill phase.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=1,
            attn_head_dim=128,
            bytes_per_param=2,
            attn_op_precision='bf16'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=True)
        

        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * S * H * q_hidden
        k_flops = 2 * B * S * H * kv_hidden
        v_flops = 2 * B * S * H * kv_hidden
        score_flops = 2 * B * N_q * S * S * D
        value_flops = 2 * B * N_q * S * S * D
        out_flops = 2 * B * S * q_hidden * H
        
        expected_bf16 = float(q_flops + k_flops + v_flops + score_flops + value_flops + out_flops)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_attention_flops_mha_decode(self) -> None:
        '''Test attention FLOPs calculation for MHA in decode phase.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=8,
            attn_head_dim=128,
            bytes_per_param=2,
            attn_op_precision='bf16'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=False)
        

        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * 1 * H * q_hidden
        k_flops = 2 * B * 1 * H * kv_hidden
        v_flops = 2 * B * 1 * H * kv_hidden
        score_flops = 2 * B * N_q * 1 * S * D
        value_flops = 2 * B * N_q * 1 * S * D
        out_flops = 2 * B * 1 * q_hidden * H
        
        expected_bf16 = float(q_flops + k_flops + v_flops + score_flops + value_flops + out_flops)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_attention_flops_gqa_decode(self) -> None:
        '''Test attention FLOPs calculation for GQA in decode phase.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=2,
            attn_head_dim=128,
            bytes_per_param=2,
            attn_op_precision='bf16'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=False)
        

        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * 1 * H * q_hidden
        k_flops = 2 * B * 1 * H * kv_hidden
        v_flops = 2 * B * 1 * H * kv_hidden
        score_flops = 2 * B * N_q * 1 * S * D
        value_flops = 2 * B * N_q * 1 * S * D
        out_flops = 2 * B * 1 * q_hidden * H
        
        expected_bf16 = float(q_flops + k_flops + v_flops + score_flops + value_flops + out_flops)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_attention_flops_mqa_decode(self) -> None:
        '''Test attention FLOPs calculation for MQA in decode phase.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=1,
            attn_head_dim=128,
            bytes_per_param=2,
            attn_op_precision='bf16'
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(config, seq_len, batch_size, is_prefill=False)
        

        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        q_flops = 2 * B * 1 * H * q_hidden
        k_flops = 2 * B * 1 * H * kv_hidden
        v_flops = 2 * B * 1 * H * kv_hidden
        score_flops = 2 * B * N_q * 1 * S * D
        value_flops = 2 * B * N_q * 1 * S * D
        out_flops = 2 * B * 1 * q_hidden * H
        
        expected_bf16 = float(q_flops + k_flops + v_flops + score_flops + value_flops + out_flops)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_moe_flops_prefill(self) -> None:
        '''Test MoE FLOPs calculation for prefill phase.'''
        config = get_default_config(
            hidden_size=1024,
            num_experts=4,
            num_activated_experts=2,
            intermediate_size=2048,
            bytes_per_param=2
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_moe_flops(config, seq_len, batch_size, is_prefill=True)
        

        B = batch_size
        S = seq_len
        H = config.hidden_size
        E = config.num_experts
        K = config.num_activated_experts
        I = config.intermediate_size
        
        gate_flops = 2 * B * S * H * E
        expert_flops = B * S * K * 6 * H * I
        
        expected_bf16 = float(gate_flops + expert_flops)
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_moe_flops_fp8(self) -> None:
        '''Test MoE FLOPs calculation for FP8 precision.'''
        config = get_default_config(
            hidden_size=1024,
            num_experts=4,
            num_activated_experts=2,
            intermediate_size=2048,
            bytes_per_param=1
        )
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_moe_flops(config, seq_len, batch_size, is_prefill=True)
        
        B = batch_size
        S = seq_len
        H = config.hidden_size
        E = config.num_experts
        K = config.num_activated_experts
        I = config.intermediate_size
        
        gate_flops = 2 * B * S * H * E
        expert_flops = B * S * K * 6 * H * I
        
        expected_fp8 = float(gate_flops + expert_flops)
        self.assertAlmostEqual(flops['fp8_flops'], expected_fp8)
        self.assertEqual(flops['bf16_flops'], 0.0)

    def test_calculate_communication_latency_tp_attn(self) -> None:
        '''Test communication latency for TP in attention.'''
        config = get_default_config(
            hidden_size=1024,
            bytes_per_param=2
        )
        hardware = HardwareSpec(
            peak_bf16_flops=100.0,
            peak_fp8_flops=200.0,
            hbm_bandwidth=50.0,
            ici_ar_ag_bandwidth=10.0,
            ici_a2a_bandwidth=5.0,
            num_chips=4
        )
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=4, attn_dp_degree=1, moe_tp_degree=1, moe_ep_degree=1)
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(config, strategy, hardware, seq_len, batch_size, is_prefill=True)
        

        P = strategy.attn_tp_degree
        bw = hardware.ici_ar_ag_bandwidth
        data_size = (batch_size / strategy.attn_dp_degree) * seq_len * config.hidden_size * config.bytes_per_param
        expected_latency = 2 * ((P - 1) / P) * data_size / 1e9 / bw
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_communication_latency_tp_moe(self) -> None:
        '''Test communication latency for TP in MoE.'''
        config = get_default_config(
            hidden_size=1024,
            bytes_per_param=2
        )
        hardware = HardwareSpec(
            peak_bf16_flops=100.0,
            peak_fp8_flops=200.0,
            hbm_bandwidth=50.0,
            ici_ar_ag_bandwidth=10.0,
            ici_a2a_bandwidth=5.0,
            num_chips=4
        )
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=1, attn_dp_degree=1, moe_tp_degree=4, moe_ep_degree=1)
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(config, strategy, hardware, seq_len, batch_size, is_prefill=True)
        

        P = strategy.moe_tp_degree
        bw = hardware.ici_ar_ag_bandwidth
        total_tokens = batch_size * seq_len
        data_size = (total_tokens / strategy.moe_ep_degree) * config.hidden_size * config.bytes_per_param
        expected_latency = 2 * ((P - 1) / P) * data_size / 1e9 / bw
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_communication_latency_ep_moe(self) -> None:
        '''Test communication latency for EP in MoE.'''
        config = get_default_config(
            hidden_size=1024,
            num_activated_experts=2,
            bytes_per_param=2
        )
        hardware = HardwareSpec(
            peak_bf16_flops=100.0,
            peak_fp8_flops=200.0,
            hbm_bandwidth=50.0,
            ici_ar_ag_bandwidth=10.0,
            ici_a2a_bandwidth=5.0,
            num_chips=4
        )
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=1, attn_dp_degree=4, moe_tp_degree=1, moe_ep_degree=4)
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(config, strategy, hardware, seq_len, batch_size, is_prefill=True)
        

        K = config.num_activated_experts
        ep = strategy.moe_ep_degree
        bw = hardware.ici_a2a_bandwidth
        total_tokens = batch_size * seq_len
        prob_visit_remote = 1 - (1 - 1/ep)**K
        data_size = (total_tokens / ep) * (ep - 1) * prob_visit_remote * config.hidden_size * config.bytes_per_param
        expected_latency = 2 * (data_size / 1e9 / bw)
        
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_communication_latency_moe_tp_with_attn_dp_a2a(self) -> None:
        '''Test communication latency for moe_tp with attn_dp using a2a.'''
        config = get_default_config(
            hidden_size=1024,
            num_activated_experts=2,
            bytes_per_param=2
        )
        hardware = HardwareSpec(
            peak_bf16_flops=100.0,
            peak_fp8_flops=200.0,
            hbm_bandwidth=50.0,
            ici_ar_ag_bandwidth=10.0,
            ici_a2a_bandwidth=5.0,
            num_chips=4
        )
        # attn_dp=4, moe_tp=4 (so moe_ep=1)
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=1, attn_dp_degree=4, moe_tp_degree=4, moe_ep_degree=1, moe_comm_type='a2a')
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(config, strategy, hardware, seq_len, batch_size, is_prefill=True)
        
        # 1. MoE All-Reduce (from moe_tp=4)
        P_tp = strategy.moe_tp_degree
        bw_ar = hardware.ici_ar_ag_bandwidth
        total_tokens = batch_size * seq_len
        data_size_ar = (total_tokens / strategy.moe_ep_degree) * config.hidden_size * config.bytes_per_param
        latency_ar = 2 * ((P_tp - 1) / P_tp) * data_size_ar / 1e9 / bw_ar
        
        # 2. MoE A2A Routing (due to attn_dp=4 and moe_tp=4)
        ep_eff = strategy.attn_dp_degree # fallback to attn_dp
        K = config.num_activated_experts
        bw_a2a = hardware.ici_a2a_bandwidth
        prob_visit_remote = 1 - (1 - 1 / ep_eff) ** K
        data_size_a2a = (total_tokens / ep_eff) * (ep_eff - 1) * prob_visit_remote * config.hidden_size * config.bytes_per_param
        latency_a2a = 2 * data_size_a2a / 1e9 / bw_a2a
        
        expected_latency = latency_ar + latency_a2a
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_communication_latency_moe_tp_with_attn_dp_all_gather(self) -> None:
        '''Test communication latency for moe_tp with attn_dp using all_gather.'''
        config = get_default_config(
            hidden_size=1024,
            bytes_per_param=2
        )
        hardware = HardwareSpec(
            peak_bf16_flops=100.0,
            peak_fp8_flops=200.0,
            hbm_bandwidth=50.0,
            ici_ar_ag_bandwidth=10.0,
            ici_a2a_bandwidth=5.0,
            num_chips=4
        )
        # attn_dp=4, moe_tp=4
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=1, attn_dp_degree=4, moe_tp_degree=4, moe_ep_degree=1, moe_comm_type='all_gather')
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(config, strategy, hardware, seq_len, batch_size, is_prefill=True)
        
        # 1. MoE All-Reduce (from moe_tp=4)
        P_tp = strategy.moe_tp_degree
        bw_ar = hardware.ici_ar_ag_bandwidth
        total_tokens = batch_size * seq_len
        data_size_ar = (total_tokens / strategy.moe_ep_degree) * config.hidden_size * config.bytes_per_param
        latency_ar = 2 * ((P_tp - 1) / P_tp) * data_size_ar / 1e9 / bw_ar
        
        # 2. MoE All-Gather inputs (needed because attn_dp=4 > 1)
        G = strategy.attn_dp_degree
        data_size_ag = batch_size * seq_len * config.hidden_size * config.bytes_per_param
        latency_ag = ((G - 1) / G) * data_size_ag / 1e9 / bw_ar
        
        expected_latency = latency_ar + latency_ag
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_communication_latency_moe_ep_with_attn_tp_full_sharding(self) -> None:
        '''Test communication latency for moe_ep with attn_tp (full sharding, attn_dp=1).'''
        config = get_default_config(
            hidden_size=1024,
            num_activated_experts=2,
            bytes_per_param=2
        )
        hardware = HardwareSpec(
            peak_bf16_flops=100.0,
            peak_fp8_flops=200.0,
            hbm_bandwidth=50.0,
            ici_ar_ag_bandwidth=10.0,
            ici_a2a_bandwidth=5.0,
            num_chips=4
        )
        # attn_tp=4, attn_dp=1, moe_tp=1, moe_ep=4
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=4, attn_dp_degree=1, moe_tp_degree=1, moe_ep_degree=4, moe_comm_type='a2a')
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(config, strategy, hardware, seq_len, batch_size, is_prefill=True)
        
        # 1. Attention All-Reduce (from attn_tp=4)
        P_attn = strategy.attn_tp_degree
        bw_ar = hardware.ici_ar_ag_bandwidth
        data_size_attn = (batch_size / strategy.attn_dp_degree) * seq_len * config.hidden_size * config.bytes_per_param
        latency_attn = 2 * ((P_attn - 1) / P_attn) * data_size_attn / 1e9 / bw_ar
        
        # 2. MoE A2A Routing (due to moe_ep=4 > 1)
        ep_eff = strategy.moe_ep_degree
        K = config.num_activated_experts
        bw_a2a = hardware.ici_a2a_bandwidth
        total_tokens = batch_size * seq_len
        prob_visit_remote = 1 - (1 - 1 / ep_eff) ** K
        data_size_a2a = (total_tokens / ep_eff) * (ep_eff - 1) * prob_visit_remote * config.hidden_size * config.bytes_per_param
        latency_a2a = data_size_a2a / 1e9 / bw_a2a
        
        expected_latency = latency_attn + latency_a2a
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_memory_access_prefill(self) -> None:
        '''Test memory access calculation for prefill phase.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=8,
            attn_head_dim=128,
            num_experts=4,
            num_activated_experts=2,
            intermediate_size=2048,
            bytes_per_param=2,
            kv_cache_precision='bf16'
        )
        seq_len = 128
        batch_size = 1
        mem_access: float = calculate_memory_access(config, seq_len, batch_size, is_prefill=True)
        

        B = batch_size
        S = seq_len
        H = config.hidden_size
        N_q = config.num_q_heads
        N_kv = config.num_kv_heads
        D = config.attn_head_dim
        E = config.num_experts
        K = config.num_activated_experts
        I = config.intermediate_size
        bytes_per_param = config.bytes_per_param
        
        q_hidden = N_q * D
        kv_hidden = N_kv * D
        
        attn_weights = (2 * H * q_hidden + 2 * H * kv_hidden) * bytes_per_param
        moe_weights = (H * E + 3 * E * H * I) * bytes_per_param
        total_weights = attn_weights + moe_weights
        
        bytes_per_kv_param = 2  # kv_cache_precision='bf16'
        kv_write = 2 * B * S * kv_hidden * bytes_per_kv_param
        activation_access = 4 * B * S * H * bytes_per_param
        moe_inter_access = 2 * B * S * K * I * bytes_per_param
        
        expected_mem_access = float(total_weights + kv_write + activation_access + moe_inter_access)
        self.assertAlmostEqual(mem_access, expected_mem_access)

    def test_calculate_roofline_simple(self) -> None:
        '''Test full roofline calculation for a simple case.'''
        config = get_default_config(
            hidden_size=1024,
            num_q_heads=8,
            num_kv_heads=8,
            attn_head_dim=128,
            num_layers=1,
            num_experts=4,
            num_activated_experts=2,
            intermediate_size=2048,
            bytes_per_param=2,
            attn_op_precision='bf16',
            kv_cache_precision='bf16'
        )
        hardware = HardwareSpec(
            peak_bf16_flops=100.0,
            peak_fp8_flops=200.0,
            hbm_bandwidth=50.0,
            ici_ar_ag_bandwidth=10.0,
            ici_a2a_bandwidth=5.0,
            num_chips=4
        )
        seq_len = 128
        batch_size = 1
        results: RooflineResult = calculate_roofline(config, hardware, seq_len, batch_size, is_prefill=True)
        

        # We reuse calculations from previous tests or reproduce them here.
        # From test_calculate_attention_flops_mha_prefill: 1140850688
        # From test_calculate_moe_flops_prefill: 3222274048
        total_flops = 1140850688 + 3222274048
        # From test_calculate_memory_access_prefill: 62398464
        mem_access = 62398464
        
        compute_latency = total_flops / 1e12 / hardware.peak_bf16_flops
        memory_latency = (mem_access / 1e9) / hardware.hbm_bandwidth
        roofline_latency = max(compute_latency, memory_latency)
        total_latency = roofline_latency + 0.0  # No communication in this simple test
        
        self.assertAlmostEqual(results.flops_per_chip, 4363124736.0)
        self.assertAlmostEqual(results.mem_access_bytes_per_chip, 62398464.0)
        self.assertAlmostEqual(results.compute_latency_ms, 4.363124736e-5 * 1000, places=4)
        self.assertAlmostEqual(results.memory_latency_ms, 0.00124796928 * 1000, places=4)
        self.assertAlmostEqual(results.total_latency_ms, 0.00124796928 * 1000, places=4)
        self.assertEqual(results.bound_by, 'memory')

    def test_parse_args_defaults(self) -> None:
        '''Test parse_args with default values.'''
        args = parse_args([])
        self.assertEqual(args.seq_len, 1024)
        self.assertEqual(args.prefill_batch_size, 1)
        self.assertEqual(args.decode_batch_size, 1)
        self.assertFalse(args.table)
        self.assertEqual(args.num_chips, 4)

    def test_parse_args_custom(self) -> None:
        '''Test parse_args with custom values.'''
        args = parse_args(['--seq_len', '2048', '--prefill_batch_size', '2', '--table'])
        self.assertEqual(args.seq_len, 2048)
        self.assertEqual(args.prefill_batch_size, 2)
        self.assertTrue(args.table)

    def test_main_smoke(self) -> None:
        '''Smoke test for main function to ensure it runs without error.'''
        try:
            main([])
        except Exception as e:
            self.fail(f"main([]) raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()
