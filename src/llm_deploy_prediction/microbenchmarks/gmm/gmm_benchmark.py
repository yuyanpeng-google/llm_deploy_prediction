import os
import glob
import gzip
import json
import time
import functools
import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

# Hardware specs (Ironwood per core/device)
PEAK_FP8_TFLOPS = 2307.0
PEAK_BW_GB_S = 3690.0

# Model parameters (Qwen3-Coder-480B)
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2560
NUM_EXPERTS = 160
TOP_K = 8
FUSE_ACT = "silu"
BLOCK_SIZE = 256 # Use 256 to enable FP8 matmul on Ironwood (MXU column size = 256)

NUM_RUNS = 10

@dataclass
class BenchmarkConfig:
    seq_len: int
    ep: int
    kernel_type: str
    size_m: int
    size_m_raw: int
    size_group: int
    size_k: int
    size_n: int
    fuse_act: Optional[str]
    gs: np.ndarray
    num_active: int
    # JAX inputs
    lhs: jax.Array
    rhs: jax.Array
    rhs_scale: jax.Array
    group_sizes: jax.Array
    group_offset: jax.Array

# Trace parsing functions
def find_latest_json_gz(root_dir, min_time=None):
    files = glob.glob(os.path.join(root_dir, "**", "*.json.gz"), recursive=True)
    if not files:
        return None
    if min_time:
        files = [f for f in files if os.path.getmtime(f) > min_time]
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def extract_all_results(json_file_path):
    results = {} # (g, m, k, act, n) -> {latencies: [], tiling: (tm, tk, tn)}
    try:
        with gzip.open(json_file_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            trace_events = data.get("traceEvents", [])
            for item in trace_events:
                name = item.get("name")
                if name and "gmm_v2" in name:
                    duration = item.get("args", {}).get("device_duration_ps")
                    if duration:
                        # Regex to match: gmm_v2-g_20-m_16-k_6144-act_silu-n_5120-tm_128-tk_64-tn_128
                        pattern = r"gmm_v2-g_(\d+)-m_(\d+)-k_(\d+)-act_(\w+|None)-n_(\d+)-tm_(\d+)-tk_(\d+)-tn_(\d+)"
                        m = re.search(pattern, name)
                        if m:
                            g, m_val, k, act, n, tm, tk, tn = m.groups()
                            g, m_val, k, n = map(int, [g, m_val, k, n])
                            tm, tk, tn = map(int, [tm, tk, tn])
                            act = None if act == "None" else act
                            
                            key = (g, m_val, k, act, n)
                            if key not in results:
                                results[key] = {"latencies": [], "tiling": (tm, tk, tn)}
                            results[key]["latencies"].append(float(duration) / 1_000_000)
    except Exception as e:
        print(f"Error parsing {json_file_path}: {e}")
    return results

def calculate_theoretical_refined(size_m, size_k, size_n, size_group, gs, fuse_act, tm, tk, tn):
    if tm is None:
        return 0, 0, 0, 0, 0
    
    # size_n is the width of the RHS (fused gate + up if fuse_act is silu)
    output_n = size_n // 2 if fuse_act else size_n
    
    # FLOPs: 2 * M * K * N. SiLU fusion doesn't change total matmul FLOPs.
    flops = 2 * size_m * size_k * size_n
    theo_flops_us = (flops / (PEAK_FP8_TFLOPS * 1e12)) * 1e6
    
    # LHS: Read from HBM. Assuming read once per N-tile if not fitting in VMEM.
    lhs_bytes = size_m * size_k * 2 # bf16
    
    # Weights: Read once per active expert (per N-tile).
    num_active_experts = np.count_nonzero(gs)
    rhs_bytes = num_active_experts * size_k * size_n * 1 # FP8
    
    # rhs_scale: float32 (4 bytes).
    rhs_scale_bytes = num_active_experts * (size_k // BLOCK_SIZE) * size_n * 4
    
    # Output: write bf16 (2 bytes).
    out_bytes = size_m * output_n * 2
    
    total_bytes = lhs_bytes + rhs_bytes + rhs_scale_bytes + out_bytes
    theo_bw_us = (total_bytes / (PEAK_BW_GB_S * 1e9)) * 1e6
    
    return max(theo_flops_us, theo_bw_us), theo_flops_us, theo_bw_us, flops, total_bytes

def prepare_benchmark_config(seq_len, ep, kernel_type="gate_up") -> BenchmarkConfig:
    num_experts_per_shard = -(-NUM_EXPERTS // ep) # Ceiling division
    total_tokens_all_shards = seq_len * TOP_K
    
    # Workload per shard (tokens)
    size_m_raw = total_tokens_all_shards // ep
    if size_m_raw == 0 and total_tokens_all_shards > 0:
        size_m_raw = 1
    
    # Minimum M alignment for the kernel is usually 16.
    size_m = max(16, int(np.ceil(size_m_raw / 16) * 16))
    size_group = num_experts_per_shard
    
    if kernel_type == "gate_up":
        size_k = HIDDEN_SIZE
        size_n = INTERMEDIATE_SIZE * 2
        fuse_act = "silu"
    else: # down
        size_k = INTERMEDIATE_SIZE
        size_n = HIDDEN_SIZE
        fuse_act = None
    
    # Realistic expert distribution
    num_active_experts = min(size_m_raw, size_group)
    if num_active_experts == 0 and size_m_raw > 0:
        num_active_experts = 1
    
    gs = np.zeros((size_group,), dtype=np.int32)
    if num_active_experts > 0:
        tokens_per_active_expert = size_m // num_active_experts
        remainder = size_m % num_active_experts
        gs[:num_active_experts] = tokens_per_active_expert
        gs[:remainder] += 1
    
    group_sizes = jnp.array(gs)
    group_offset = jnp.array([0], dtype=jnp.int32)
    
    key = jax.random.PRNGKey(0)
    lhs = jax.random.normal(key, (size_m, size_k), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (size_group, size_k, size_n), dtype=jnp.bfloat16).astype(jnp.float8_e4m3fn)
    rhs_scale = jnp.ones((size_group, size_k // BLOCK_SIZE, 1, size_n), dtype=jnp.float32)

    return BenchmarkConfig(
        seq_len=seq_len, ep=ep, kernel_type=kernel_type,
        size_m=size_m, size_m_raw=size_m_raw, size_group=size_group,
        size_k=size_k, size_n=size_n, fuse_act=fuse_act,
        gs=gs, num_active=num_active_experts,
        lhs=lhs, rhs=rhs, rhs_scale=rhs_scale,
        group_sizes=group_sizes, group_offset=group_offset
    )

@jax.jit(static_argnames=["fuse_act"])
def run_gmm(l, r, gs_in, rs, go, fuse_act):
    return gmm_v2(l, r, gs_in, rhs_scale=rs, group_offset=go, fuse_act=fuse_act)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, nargs='+', help="List of sequence lengths", default=None)
    parser.add_argument("--ep", type=int, nargs='+', help="List of expert parallel sizes", default=None)
    parser.add_argument("--kernel", type=str, choices=["gate_up", "down", "all"], default="all")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs per configuration")
    parser.add_argument("--prof-dir", type=str, default="/dev/shm/vllm-prof", help="Directory to save profiles")
    args = parser.parse_args()

    if args.seq_len is not None and args.ep is not None:
        if len(args.seq_len) != len(args.ep):
            raise ValueError("--seq-len and --ep must have the same number of arguments")
        comparison_configs = list(zip(args.seq_len, args.ep))
    else:
        # Comparison cases: Fixed M per shard
        comparison_configs = [
            # (1, 8), (4, 32),
            # (256, 8), (1024, 32),
            # (512, 8), (2048, 32),
            # (4096, 8), (16384, 32)
            # (1024, 8), (2048, 8), (4096, 8), (8192, 8), (16384, 8), (32768, 8), (65536, 8)
            (256, 8), (512, 16), (1024, 32), (2048, 64), (4096, 128),
        ]
    
    if args.kernel == "all":
        kernels = ["gate_up", "down"]
    else:
        kernels = [args.kernel]
    
    configs = []
    for kernel_type in kernels:
        for seq_len, ep in comparison_configs:
            configs.append(prepare_benchmark_config(seq_len, ep, kernel_type))

    print(f"Prepared {len(configs)} configurations.")
    
    print("Warming up all configurations...")
    for cfg in configs:
        out = run_gmm(cfg.lhs, cfg.rhs, cfg.group_sizes, cfg.rhs_scale, cfg.group_offset, cfg.fuse_act)
        out.block_until_ready()
    
    search_dir = args.prof_dir
    if not os.path.exists(search_dir):
        os.makedirs(search_dir)
    
    start_time = time.time()
    print(f"Profiling {args.runs} runs for each configuration in one trace...")
    with jax.profiler.trace(search_dir):
        for cfg in configs:
            for _ in range(args.runs):
                out = run_gmm(cfg.lhs, cfg.rhs, cfg.group_sizes, cfg.rhs_scale, cfg.group_offset, cfg.fuse_act)
            out.block_until_ready()
    
    # Wait for trace to be written
    time.sleep(2)
    latest_trace = find_latest_json_gz(search_dir, min_time=start_time)
    if not latest_trace:
        print("Failed to find trace file.")
        exit(1)
    
    print(f"Parsing trace: {latest_trace}")
    trace_results = extract_all_results(latest_trace)
    
    all_results = []
    for cfg in configs:
        key = (cfg.size_group, cfg.size_m, cfg.size_k, cfg.fuse_act, cfg.size_n)
        if key in trace_results:
            latencies = trace_results[key]["latencies"]
            if not latencies:
                continue
            
            # Remove outliers or just average?
            # Usually the first run after JIT might still be slower sometimes, but we did warmup.
            # Let's take the median or average of the middle 80%.
            latencies.sort()
            if len(latencies) >= 5:
                # Remove best and worst
                trimmed = latencies[1:-1]
                latency_us = sum(trimmed) / len(trimmed)
            else:
                latency_us = sum(latencies) / len(latencies)
            
            tm, tk, tn = trace_results[key]["tiling"]
            theo_us, theo_comp_us, theo_mem_us, flops, total_bytes = calculate_theoretical_refined(
                cfg.size_m, cfg.size_k, cfg.size_n, cfg.size_group, cfg.gs, cfg.fuse_act, tm, tk, tn)
            
            tflops_rate = flops / (latency_us * 1e-6) / 1e12
            theo_tflops_rate = flops / (theo_us * 1e-6) / 1e12
            total_tflops = flops / 1e12
            total_gb = total_bytes / 1e9
            mfu = (tflops_rate / PEAK_FP8_TFLOPS) * 100
            
            all_results.append({
                "kernel_type": cfg.kernel_type,
                "seq_len": cfg.seq_len,
                "ep": cfg.ep,
                "size_m": cfg.size_m,
                "num_active": cfg.num_active,
                "size_group": cfg.size_group,
                "tm": tm,
                "tk": tk,
                "tn": tn,
                "latency_us": latency_us,
                "theo_us": theo_us,
                "theo_comp_us": theo_comp_us,
                "theo_mem_us": theo_mem_us,
                "total_tflops": total_tflops,
                "tflops_rate": tflops_rate,
                "total_gb": total_gb,
                "mfu": mfu,
            })
        else:
            print(f"Warning: No trace data found for {cfg.kernel_type} SeqLen={cfg.seq_len}, EP={cfg.ep}")

    print("\n" + "="*210)
    print("GMM Benchmark Summary (FP8 w8a8) - Ironwood Core")
    print("="*210)
    print(f"{'Kernel':>8} | {'SeqLen':>8} | {'EP':>4} | {'M':>6} | {'G_act/G':>10} | {'TM':>4} | {'TK':>4} | {'TN':>4} | {'Latency(us)':>12} | {'Theo(us)':>10} | {'Comp(us)':>10} | {'Mem(us)':>10} | {'TFLOPs':>10} | {'TFLOPS/s':>10} | {'HBM(GB)':>10} | {'MFU(%)':>8}")
    print("-" * 210)
    for r in all_results:
        g_info = f"{r['num_active']}/{r['size_group']}"
        print(f"{r['kernel_type']:>8} | {r['seq_len']:8d} | {r['ep']:4d} | {r['size_m']:6d} | {g_info:>10} | {r['tm']:4d} | {r['tk']:4d} | {r['tn']:4d} | {r['latency_us']:12.2f} | {r['theo_us']:10.2f} | {r['theo_comp_us']:10.2f} | {r['theo_mem_us']:10.2f} | {r['total_tflops']:10.4f} | {r['tflops_rate']:10.2f} | {r['total_gb']:10.4f} | {r['mfu']:7.2f}%")

    print("\n" + "="*50)
    print("Debug: Tiling Parameters")
    print("="*50)
    for r in all_results:
        print(f"Kernel: {r['kernel_type']:8} | SeqLen: {r['seq_len']:6d} | M: {r['size_m']:6d} | TM: {r['tm']:4d} | TK: {r['tk']:4d} | TN: {r['tn']:4d}")
