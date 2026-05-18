'''Unit tests for roofline.py.

This module contains tests to verify the calculations in roofline.py.
'''

import unittest
from typing import Dict, Any

from src.llm_deploy_prediction.roofline.roofline import (
    ModelConfig,
    HardwareSpec,
    ShardingStrategy,
    calculate_attention_flops,
    calculate_moe_flops,
    calculate_memory_access,
    calculate_communication_latency,
    calculate_roofline,
    parse_args,
    main,
)

class TestRooflineCalculations(unittest.TestCase):
    '''Test case for roofline calculations.'''

    def setUp(self) -> None:
        '''Set up dummy configurations for testing.'''
        self.config = ModelConfig(
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
        self.hardware = HardwareSpec(
            peak_bf16_flops=100.0,
            peak_fp8_flops=200.0,
            hbm_bandwidth=50.0,
            ici_ar_ag_bandwidth=10.0,
            ici_a2a_bandwidth=5.0,
            num_chips=4
        )

    def test_calculate_attention_flops_prefill(self) -> None:
        '''Test attention FLOPs calculation for prefill phase.'''
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_attention_flops(self.config, seq_len, batch_size, is_prefill=True)
        
        # Manual calculation:
        # B = 1, S = 128, H = 1024, N_kv = 8, D = 128
        # q_hidden = 8 * 128 = 1024
        # kv_hidden = 8 * 128 = 1024
        # q_flops = 2 * 1 * 128 * 1024 * 1024 = 268435456
        # k_flops = 2 * 1 * 128 * 1024 * 1024 = 268435456
        # v_flops = 2 * 1 * 128 * 1024 * 1024 = 268435456
        # score_flops = 2 * 1 * 8 * 128 * 128 * 128 = 33554432
        # value_flops = 2 * 1 * 8 * 128 * 128 * 128 = 33554432
        # out_flops = 2 * 1 * 128 * 1024 * 1024 = 268435456
        # Total proj = q + k + v + out = 1073741824
        # Total attn = score + value = 67108864
        # Total = 1140850688
        
        expected_bf16 = 1140850688.0
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_moe_flops_prefill(self) -> None:
        '''Test MoE FLOPs calculation for prefill phase.'''
        seq_len = 128
        batch_size = 1
        flops: Dict[str, float] = calculate_moe_flops(self.config, seq_len, batch_size, is_prefill=True)
        
        # Manual calculation:
        # B = 1, S = 128, H = 1024, E = 4, K = 2, I = 2048
        # gate_flops = 2 * 1 * 128 * 1024 * 4 = 1048576
        # expert_flops = 1 * 128 * 2 * 6 * 1024 * 2048 = 3221225472
        # Total = 3222274048
        
        expected_bf16 = 3222274048.0
        self.assertAlmostEqual(flops['bf16_flops'], expected_bf16)
        self.assertEqual(flops['fp8_flops'], 0.0)

    def test_calculate_communication_latency_tp_attn(self) -> None:
        '''Test communication latency for TP in attention.'''
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=4, attn_dp_degree=1, moe_tp_degree=1, moe_ep_degree=1)
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(self.config, strategy, self.hardware, seq_len, batch_size, is_prefill=True)
        
        # Manual calculation:
        # P = 4, bw = 10
        # data_size = (1 / 1) * 128 * 1024 * 2 = 262144 bytes
        # latency = 2 * ((4-1)/4) * 262144 / 1e9 / 10 = 1.5 * 262144 / 1e10 = 393216 / 1e10 = 0.0000393216 seconds
        
        expected_latency = 2 * (3/4) * (1 * 128 * 1024 * 2) / 1e9 / 10
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_communication_latency_tp_moe(self) -> None:
        '''Test communication latency for TP in MoE.'''
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=1, attn_dp_degree=1, moe_tp_degree=4, moe_ep_degree=1)
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(self.config, strategy, self.hardware, seq_len, batch_size, is_prefill=True)
        
        # Manual calculation:
        # P = 4, bw = 10
        # total_tokens = 1 * 128 = 128
        # data_size = (128 / 1) * 1024 * 2 = 262144 bytes
        # latency = 2 * ((4-1)/4) * 262144 / 1e9 / 10 = 0.0000393216 seconds
        
        expected_latency = 2 * (3/4) * (128 * 1024 * 2) / 1e9 / 10
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_communication_latency_ep_moe(self) -> None:
        '''Test communication latency for EP in MoE.'''
        strategy = ShardingStrategy(num_chips=4, attn_tp_degree=1, attn_dp_degree=1, moe_tp_degree=1, moe_ep_degree=4)
        seq_len = 128
        batch_size = 1
        latency: float = calculate_communication_latency(self.config, strategy, self.hardware, seq_len, batch_size, is_prefill=True)
        
        # Manual calculation:
        # K = 2, ep = 4, bw = 5
        # total_tokens = 128
        # prob_visit_remote = 1 - (1 - 1/4)^2 = 1 - (3/4)^2 = 1 - 9/16 = 7/16 = 0.4375
        # data_size = (128 / 4) * (4 - 1) * 0.4375 * 1024 * 2 = 32 * 3 * 0.4375 * 2048 = 86016 bytes
        # latency = 2 * (data_size / 1e9 / 5) = 2 * 86016 / 5e9 = 172032 / 5e9 = 0.0000344064 seconds
        
        prob_visit_remote = 1 - (1 - 1/4)**2
        data_size = (128 / 4) * (4 - 1) * prob_visit_remote * 1024 * 2
        expected_latency = 2 * (data_size / 1e9 / 5)
        
        self.assertAlmostEqual(latency, expected_latency)

    def test_calculate_memory_access_prefill(self) -> None:
        '''Test memory access calculation for prefill phase.'''
        seq_len = 128
        batch_size = 1
        mem_access: float = calculate_memory_access(self.config, seq_len, batch_size, is_prefill=True)
        
        # Manual calculation:
        # total_weights = 58728448
        # kv_write = 524288
        # activation_access = 1048576
        # moe_inter_access = 2097152
        # Total = 62398464
        
        expected_mem_access = 62398464.0
        self.assertAlmostEqual(mem_access, expected_mem_access)

    def test_calculate_roofline_simple(self) -> None:
        '''Test full roofline calculation for a simple case.'''
        seq_len = 128
        batch_size = 1
        results: Dict[str, Any] = calculate_roofline(self.config, self.hardware, seq_len, batch_size, is_prefill=True)
        
        # Manual calculation:
        # total_flops = 4363124736
        # mem_access = 62398464
        # compute_latency = 4.363124736e-5
        # memory_latency = gb_access / hbm_bandwidth = (62398464 / 1e9) / 50 = 0.062398464 / 50 = 0.00124796928 seconds
        # roofline_latency = max(compute, memory) = 0.00124796928
        # total_latency = 0.00124796928
        
        self.assertAlmostEqual(results['flops_per_chip'], 4363124736.0)
        self.assertAlmostEqual(results['mem_access_bytes_per_chip'], 62398464.0)
        self.assertAlmostEqual(results['compute_latency_ms'], 4.363124736e-5 * 1000, places=4)
        self.assertAlmostEqual(results['memory_latency_ms'], 0.00124796928 * 1000, places=4)
        self.assertAlmostEqual(results['total_latency_ms'], 0.00124796928 * 1000, places=4)
        self.assertEqual(results['bound_by'], 'memory')

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
