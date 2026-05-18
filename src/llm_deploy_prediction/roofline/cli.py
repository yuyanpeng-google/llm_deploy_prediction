'''Command line interface for roofline analysis.

This module handles argument parsing and the main execution flow for the
roofline analysis tool.
'''

import argparse
from dataclasses import asdict
import glob
import os
import sys
from typing import Any, List, Optional, Tuple

from llm_deploy_prediction.roofline.calculations import calculate_roofline, RooflineResult
from llm_deploy_prediction.roofline.config import (
    HardwareSpec,
    ModelConfig,
    ShardingStrategy,
    load_hardware_spec,
    load_model_config,
)
from llm_deploy_prediction.roofline.output import print_markdown_table


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
            for comm_type in ['a2a', 'all_gather']:
                strategies.append(
                    ShardingStrategy(
                        num_chips=num_chips,
                        attn_tp_degree=attn_tp,
                        attn_dp_degree=attn_dp,
                        moe_tp_degree=moe_tp,
                        moe_ep_degree=moe_ep,
                        moe_comm_type=comm_type,
                    )
                )
    return strategies


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    '''Parses command line arguments.

    Args:
        argv: List of arguments to parse. If None, uses sys.argv.

    Returns:
        Parsed arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Calculate roofline for LLM deployment.'
    )
    parser.add_argument(
        '--model_config', type=str, help='Path to model config JSON file.'
    )
    parser.add_argument(
        '--hardware_spec',
        type=str,
        nargs='+',
        help='Path to hardware spec JSON file(s).',
    )
    parser.add_argument(
        '--seq_len', type=int, default=1024, help='Sequence length.'
    )

    prefill_group = parser.add_mutually_exclusive_group()
    prefill_group.add_argument(
        '--prefill_batch_size',
        type=int,
        default=1,
        help='Global batch size for prefill phase. Conflicts with --prefill_local_batch_size.',
    )
    prefill_group.add_argument(
        '--prefill_local_batch_size',
        type=int,
        help='Local batch size per DP group for prefill phase. Conflicts with --prefill_batch_size.',
    )

    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument(
        '--decode_batch_size',
        type=int,
        default=1,
        help='Global batch size for decode phase. Conflicts with --decode_local_batch_size.',
    )
    decode_group.add_argument(
        '--decode_local_batch_size',
        type=int,
        help='Local batch size per DP group for decode phase. Conflicts with --decode_batch_size.',
    )

    parser.add_argument(
        '--table',
        action='store_true',
        help='Output results in markdown table format.',
    )
    parser.add_argument(
        '--num_chips',
        type=int,
        default=4,
        help='Number of chips for grid search (fallback if not in spec).',
    )

    return parser.parse_args(argv)


def _load_model_cfg(args: argparse.Namespace) -> ModelConfig:
    '''Loads model configuration from file or returns default values.

    Args:
        args: Parsed command line arguments.

    Returns:
        ModelConfig instance.
    '''
    if args.model_config:
        return load_model_config(args.model_config)
    
    print("Warning: No model config file provided. Using dummy values.")
    return ModelConfig(
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
        kv_cache_precision='fp8',
    )


def _load_hw_specs(args: argparse.Namespace) -> Tuple[List[HardwareSpec], List[str]]:
    '''Loads hardware specifications from files or returns default values.

    Args:
        args: Parsed command line arguments.

    Returns:
        Tuple containing a list of HardwareSpec instances and a list of their paths/names.
    '''
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
        print(
            "Warning: No hardware spec file provided. Using dummy values."
        )
        hw_specs.append(
            HardwareSpec(
                peak_bf16_flops=2307.0,
                peak_fp8_flops=4614.0,
                hbm_bandwidth=7380.0,
                ici_ar_ag_bandwidth=600.0,
                ici_a2a_bandwidth=200.0,
                num_chips=args.num_chips,
            )
        )
        spec_paths.append("Default")
    return hw_specs, spec_paths


def _calculate_results(
    model_cfg: ModelConfig,
    hw_specs: List[HardwareSpec],
    spec_paths: List[str],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, int, RooflineResult]], List[Tuple[str, int, RooflineResult]]]:
    '''Runs roofline calculations for all hardware specs and strategies.

    Args:
        model_cfg: Model configuration.
        hw_specs: List of hardware specifications.
        spec_paths: List of paths/names for hardware specs.
        args: Parsed command line arguments.

    Returns:
        Tuple containing prefill results and decode results.
    '''
    prefill_results: List[Tuple[str, int, RooflineResult]] = []
    decode_results: List[Tuple[str, int, RooflineResult]] = []

    for hw_spec, spec_path in zip(hw_specs, spec_paths):
        strategies = generate_strategies(hw_spec.num_chips)
        spec_name = (
            os.path.basename(spec_path) if spec_path != "Default" else "Default"
        )

        for strategy in strategies:
            strategy_str = f"Spec={spec_name}, Chips={strategy.num_chips}, Attn(TP={strategy.attn_tp_degree},DP={strategy.attn_dp_degree}), MoE(TP={strategy.moe_tp_degree},EP={strategy.moe_ep_degree},Comm={strategy.moe_comm_type})"

            if args.prefill_local_batch_size is not None:
                prefill_batch = (
                    args.prefill_local_batch_size * strategy.attn_dp_degree
                )
            else:
                prefill_batch = args.prefill_batch_size

            if args.decode_local_batch_size is not None:
                decode_batch = (
                    args.decode_local_batch_size * strategy.attn_dp_degree
                )
            else:
                decode_batch = args.decode_batch_size

            try:
                prefill_res = calculate_roofline(
                    model_cfg,
                    hw_spec,
                    seq_len=args.seq_len,
                    batch_size=prefill_batch,
                    is_prefill=True,
                    strategy=strategy,
                )
                prefill_results.append(
                    (strategy_str, prefill_batch, prefill_res)
                )
            except ValueError as e:
                print(
                    f"Error calculating prefill for {strategy_str}: {e}"
                )

            try:
                decode_res = calculate_roofline(
                    model_cfg,
                    hw_spec,
                    seq_len=args.seq_len,
                    batch_size=decode_batch,
                    is_prefill=False,
                    strategy=strategy,
                )
                decode_results.append(
                    (strategy_str, decode_batch, decode_res)
                )
            except ValueError as e:
                print(f"Error calculating decode for {strategy_str}: {e}")

    return prefill_results, decode_results


def _print_prefill_table(results: List[Tuple[str, int, RooflineResult]]) -> None:
    '''Prints prefill phase table.'''
    print("\n=== Prefill Phase Table ===")
    headers = [
        "Strategy",
        "Batch Size",
        "Throughput/Chip",
        "TTFT (ms)",
        "Bound By",
        "Total Latency (ms)",
        "KV Cache (GB)",
    ]
    rows = []
    for strategy_str, batch, res in results:
        rows.append(
            [
                strategy_str,
                batch,
                f"{res.throughput_per_chip:.2f}",
                f"{res.ttft_ms:.2f}",
                res.bound_by,
                f"{res.total_latency_ms:.2f}",
                f"{res.kv_cache_size_gb:.2f}",
            ]
        )
    print_markdown_table(headers, rows)


def _print_decode_table(results: List[Tuple[str, int, RooflineResult]]) -> None:
    '''Prints decode phase table.'''
    print("\n=== Decode Phase Table ===")
    headers = [
        "Strategy",
        "Batch Size",
        "Throughput/Chip",
        "TPOT (ms)",
        "Bound By",
        "Total Latency (ms)",
        "KV Cache (GB)",
    ]
    rows = []
    for strategy_str, batch, res in results:
        rows.append(
            [
                strategy_str,
                batch,
                f"{res.throughput_per_chip:.2f}",
                f"{res.tpot_ms:.2f}",
                res.bound_by,
                f"{res.total_latency_ms:.2f}",
                f"{res.kv_cache_size_gb:.2f}",
            ]
        )
    print_markdown_table(headers, rows)


def _print_latency_table(
    prefill_results: List[Tuple[str, int, RooflineResult]],
    decode_results: List[Tuple[str, int, RooflineResult]],
) -> None:
    '''Prints latency comparison table.'''
    print("\n=== Latency Comparison Table ===")
    headers = [
        "Strategy",
        "Phase",
        "Batch Size",
        "Compute Latency (ms)",
        "Memory Latency (ms)",
        "ICI Latency (ms)",
        "Gap (ms)",
        "Bound By",
    ]
    rows = []
    for strategy_str, batch, res in prefill_results:
        gap = abs(res.compute_latency_ms - res.memory_latency_ms)
        rows.append(
            [
                strategy_str,
                "Prefill",
                batch,
                f"{res.compute_latency_ms:.2f}",
                f"{res.memory_latency_ms:.2f}",
                f"{res.comm_latency_ms:.2f}",
                f"{gap:.2f}",
                res.bound_by,
            ]
        )
    for strategy_str, batch, res in decode_results:
        gap = abs(res.compute_latency_ms - res.memory_latency_ms)
        rows.append(
            [
                strategy_str,
                "Decode",
                batch,
                f"{res.compute_latency_ms:.2f}",
                f"{res.memory_latency_ms:.2f}",
                f"{res.comm_latency_ms:.2f}",
                f"{gap:.2f}",
                res.bound_by,
            ]
        )
    print_markdown_table(headers, rows)


def _print_hbm_table(
    prefill_results: List[Tuple[str, int, RooflineResult]],
    decode_results: List[Tuple[str, int, RooflineResult]],
) -> None:
    '''Prints HBM usage table.'''
    print("\n=== HBM Usage Table ===")
    headers = [
        "Strategy",
        "Phase",
        "Batch Size",
        "Weights/Chip (GB)",
        "KV Cache/Chip (GB)",
        "Total HBM/Chip (GB)",
        "Capacity (GB)",
        "Util (%)",
    ]
    rows = []
    for strategy_str, batch, res in prefill_results:
        util = (
            (res.hbm_usage_gb / res.hbm_capacity_gb) * 100
            if res.hbm_capacity_gb > 0
            else 0.0
        )
        rows.append(
            [
                strategy_str,
                "Prefill",
                batch,
                f"{res.weights_per_chip_gb:.2f}",
                f"{res.kv_cache_per_chip_gb:.2f}",
                f"{res.hbm_usage_gb:.2f}",
                f"{res.hbm_capacity_gb:.2f}",
                f"{util:.2f}",
            ]
        )
    for strategy_str, batch, res in decode_results:
        util = (
            (res.hbm_usage_gb / res.hbm_capacity_gb) * 100
            if res.hbm_capacity_gb > 0
            else 0.0
        )
        rows.append(
            [
                strategy_str,
                "Decode",
                batch,
                f"{res.weights_per_chip_gb:.2f}",
                f"{res.kv_cache_per_chip_gb:.2f}",
                f"{res.hbm_usage_gb:.2f}",
                f"{res.hbm_capacity_gb:.2f}",
                f"{util:.2f}",
            ]
        )
    print_markdown_table(headers, rows)


def _print_sorted_prefill_table(results: List[Tuple[str, int, RooflineResult]]) -> None:
    '''Prints prefill throughput sorted table.'''
    print("\n=== Prefill Throughput/Chip Sorted Table ===")
    headers = [
        "Strategy",
        "Batch Size",
        "Throughput/Chip",
        "Total Latency (ms)",
        "Bound By",
        "HBM Util (%)",
    ]

    sorted_results = sorted(
        results,
        key=lambda x: x[2].throughput_per_chip,
        reverse=True,
    )
    rows = []
    for strategy_str, batch, res in sorted_results:
        util = (
            (res.hbm_usage_gb / res.hbm_capacity_gb) * 100
            if res.hbm_capacity_gb > 0
            else 0.0
        )
        rows.append(
            [
                strategy_str,
                batch,
                f"{res.throughput_per_chip:.2f}",
                f"{res.total_latency_ms:.2f}",
                res.bound_by,
                f"{util:.2f}",
            ]
        )
    print_markdown_table(headers, rows)


def _print_sorted_decode_table(results: List[Tuple[str, int, RooflineResult]]) -> None:
    '''Prints decode throughput sorted table.'''
    print("\n=== Decode Throughput/Chip Sorted Table ===")
    headers = [
        "Strategy",
        "Batch Size",
        "Throughput/Chip",
        "Total Latency (ms)",
        "Bound By",
        "HBM Util (%)",
    ]

    sorted_results = sorted(
        results,
        key=lambda x: x[2].throughput_per_chip,
        reverse=True,
    )
    rows = []
    for strategy_str, batch, res in sorted_results:
        util = (
            (res.hbm_usage_gb / res.hbm_capacity_gb) * 100
            if res.hbm_capacity_gb > 0
            else 0.0
        )
        rows.append(
            [
                strategy_str,
                batch,
                f"{res.throughput_per_chip:.2f}",
                f"{res.total_latency_ms:.2f}",
                res.bound_by,
                f"{util:.2f}",
            ]
        )
    print_markdown_table(headers, rows)


def _print_detailed_results(
    prefill_results: List[Tuple[str, int, RooflineResult]],
    decode_results: List[Tuple[str, int, RooflineResult]],
    seq_len: int,
) -> None:
    '''Prints detailed results in non-table format.'''
    for strategy_str, prefill_batch, res in prefill_results:
        print(f"\n=== Strategy: {strategy_str} ===")
        print(f"--- Prefill Phase (Seq Len {seq_len}, Batch {prefill_batch}) ---")
        for k, v in asdict(res).items():
            print(f"{k}: {v}")

    for strategy_str, decode_batch, res in decode_results:
        print(f"\n=== Strategy: {strategy_str} ===")
        print(f"--- Decode Phase (Seq Len {seq_len}, Batch {decode_batch}, 1 step) ---")
        for k, v in asdict(res).items():
            print(f"{k}: {v}")


def _print_results(
    prefill_results: List[Tuple[str, int, RooflineResult]],
    decode_results: List[Tuple[str, int, RooflineResult]],
    args: argparse.Namespace,
) -> None:
    '''Prints the results in markdown tables or detailed format.'''
    if args.table:
        _print_prefill_table(prefill_results)
        _print_decode_table(decode_results)
        _print_latency_table(prefill_results, decode_results)
        _print_hbm_table(prefill_results, decode_results)
        _print_sorted_prefill_table(prefill_results)
        _print_sorted_decode_table(decode_results)
    else:
        _print_detailed_results(prefill_results, decode_results, args.seq_len)


def main(argv: Optional[List[str]] = None) -> None:
    '''Main entry point for calculating roofline.

    Args:
        argv: List of arguments to parse. If None, uses sys.argv.
    '''
    args = parse_args(argv)

    model_cfg = _load_model_cfg(args)

    hw_specs, spec_paths = _load_hw_specs(args)

    prefill_results, decode_results = _calculate_results(
        model_cfg, hw_specs, spec_paths, args
    )

    _print_results(prefill_results, decode_results, args)


if __name__ == '__main__':
    main(sys.argv[1:])
