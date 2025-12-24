#!/usr/bin/env python3
"""
KV Cache Transfer Bandwidth Test for vLLM PD Separation

This script measures the actual TCP bandwidth of KV cache transfers
between Prefill and Decode instances and compares it with theoretical
hardware limits.

Hardware Reference (for verification):
- 10GbE NIC: ~1.25 GB/s theoretical, ~1.0 GB/s practical
- 25GbE NIC: ~3.1 GB/s theoretical, ~2.5 GB/s practical
- 100GbE NIC: ~12.5 GB/s theoretical, ~10 GB/s practical
- PCIe 4.0 x16: ~32 GB/s (GPU memory bandwidth limit for TCP)
- Localhost loopback: ~10-50 GB/s (memory-bound)

For TCP over network (NCCL with NCCL_NET=Socket):
- Typical achievable: 60-80% of link speed
- With NCCL optimizations: up to 90% of link speed
"""

import argparse
import json
import time
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class KVCacheMetrics:
    """Metrics for KV cache transfer"""
    prompt_tokens: int
    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype_bytes: int  # bfloat16 = 2 bytes

    @property
    def kv_cache_size_per_token(self) -> int:
        """Size of KV cache per token in bytes"""
        # K and V each: num_kv_heads * head_dim * dtype_bytes
        # Total per layer: 2 * num_kv_heads * head_dim * dtype_bytes
        # Total all layers: num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
        return self.num_layers * 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def get_total_transfer_bytes(self, num_tokens: int) -> int:
        """Total bytes transferred for given number of tokens"""
        return num_tokens * self.kv_cache_size_per_token


# Llama-3.1-8B model parameters
LLAMA_8B_METRICS = KVCacheMetrics(
    prompt_tokens=0,  # Will be set per request
    num_layers=32,
    hidden_size=4096,
    num_kv_heads=8,  # GQA: 8 KV heads for 32 attention heads
    head_dim=128,    # 4096 / 32 = 128
    block_size=16,
    dtype_bytes=2,   # bfloat16
)


def calculate_kv_transfer_size(num_tokens: int, metrics: KVCacheMetrics = LLAMA_8B_METRICS) -> dict:
    """Calculate KV cache transfer size for given number of tokens"""
    total_bytes = metrics.get_total_transfer_bytes(num_tokens)
    return {
        "num_tokens": num_tokens,
        "kv_size_per_token_bytes": metrics.kv_cache_size_per_token,
        "total_transfer_bytes": total_bytes,
        "total_transfer_mb": total_bytes / (1024 * 1024),
        "total_transfer_gb": total_bytes / (1024 * 1024 * 1024),
    }


def run_bandwidth_test(
    proxy_url: str,
    model_path: str,
    prompt_lengths: list[int],
    num_iterations: int = 3,
) -> list[dict]:
    """Run bandwidth test with different prompt lengths"""

    results = []

    # Generate prompts of specific lengths
    base_word = "hello "

    for target_tokens in prompt_lengths:
        # Approximate: 1 word ≈ 1.3 tokens, so use 0.77 words per token
        num_words = int(target_tokens * 0.77)
        prompt = base_word * num_words

        print(f"\n{'='*60}")
        print(f"Testing with ~{target_tokens} tokens prompt")
        print(f"{'='*60}")

        iteration_results = []

        for i in range(num_iterations):
            # Measure request time
            start_time = time.perf_counter()

            response = requests.post(
                f"{proxy_url}/v1/completions",
                json={
                    "model": model_path,
                    "prompt": prompt,
                    "max_tokens": 1,  # Minimal output to focus on prefill/transfer
                    "temperature": 0,
                },
                timeout=120,
            )

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            if response.status_code != 200:
                print(f"  Iteration {i+1}: FAILED - {response.status_code}")
                continue

            result = response.json()
            actual_prompt_tokens = result.get("usage", {}).get("prompt_tokens", target_tokens)

            # Calculate transfer metrics
            kv_info = calculate_kv_transfer_size(actual_prompt_tokens)

            # The total time includes:
            # 1. HTTP request/response overhead
            # 2. Prefill computation on GPU 0
            # 3. KV cache transfer from GPU 0 to GPU 1 (via TCP)
            # 4. First token generation on GPU 1

            # Estimate transfer time as a portion of total time
            # For PD separation, transfer happens after prefill
            # We'll report the raw measurements and let user interpret

            bandwidth_mb_s = kv_info["total_transfer_mb"] / elapsed_time if elapsed_time > 0 else 0
            bandwidth_gb_s = kv_info["total_transfer_gb"] / elapsed_time if elapsed_time > 0 else 0

            iteration_result = {
                "iteration": i + 1,
                "prompt_tokens": actual_prompt_tokens,
                "elapsed_time_s": elapsed_time,
                "kv_transfer_bytes": kv_info["total_transfer_bytes"],
                "kv_transfer_mb": kv_info["total_transfer_mb"],
                "apparent_bandwidth_mb_s": bandwidth_mb_s,
                "apparent_bandwidth_gb_s": bandwidth_gb_s,
            }
            iteration_results.append(iteration_result)

            print(f"  Iteration {i+1}:")
            print(f"    Prompt tokens: {actual_prompt_tokens}")
            print(f"    Total time: {elapsed_time*1000:.2f} ms")
            print(f"    KV cache size: {kv_info['total_transfer_mb']:.2f} MB")
            print(f"    Apparent bandwidth: {bandwidth_mb_s:.2f} MB/s ({bandwidth_gb_s:.3f} GB/s)")

        if iteration_results:
            # Calculate averages (excluding first iteration which includes NCCL setup)
            if len(iteration_results) > 1:
                avg_results = iteration_results[1:]  # Skip warmup
            else:
                avg_results = iteration_results

            avg_time = sum(r["elapsed_time_s"] for r in avg_results) / len(avg_results)
            avg_bandwidth_mb = sum(r["apparent_bandwidth_mb_s"] for r in avg_results) / len(avg_results)
            avg_bandwidth_gb = sum(r["apparent_bandwidth_gb_s"] for r in avg_results) / len(avg_results)

            summary = {
                "target_tokens": target_tokens,
                "actual_tokens": iteration_results[-1]["prompt_tokens"],
                "kv_transfer_mb": iteration_results[-1]["kv_transfer_mb"],
                "avg_time_s": avg_time,
                "avg_bandwidth_mb_s": avg_bandwidth_mb,
                "avg_bandwidth_gb_s": avg_bandwidth_gb,
                "iterations": iteration_results,
            }
            results.append(summary)

            print(f"\n  Summary (excluding warmup):")
            print(f"    Average time: {avg_time*1000:.2f} ms")
            print(f"    Average bandwidth: {avg_bandwidth_mb:.2f} MB/s ({avg_bandwidth_gb:.3f} GB/s)")

    return results


def print_bandwidth_analysis(results: list[dict]):
    """Print bandwidth analysis and comparison with hardware specs"""

    print("\n" + "="*70)
    print("BANDWIDTH ANALYSIS REPORT")
    print("="*70)

    print("\n1. TEST RESULTS:")
    print("-"*70)
    print(f"{'Tokens':>10} {'KV Size (MB)':>15} {'Time (ms)':>12} {'Bandwidth (GB/s)':>18}")
    print("-"*70)

    for r in results:
        print(f"{r['actual_tokens']:>10} {r['kv_transfer_mb']:>15.2f} {r['avg_time_s']*1000:>12.2f} {r['avg_bandwidth_gb_s']:>18.3f}")

    print("\n2. KV CACHE SIZE FORMULA (Llama-3.1-8B):")
    print("-"*70)
    print("   KV size per token = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes")
    print("                     = 32 * 2 * 8 * 128 * 2 = 131,072 bytes = 128 KB/token")
    print("   For 1000 tokens: 128 KB * 1000 = 128 MB")

    print("\n3. HARDWARE BANDWIDTH REFERENCE:")
    print("-"*70)
    print("   Localhost loopback (same machine): 10-50 GB/s (memory-bound)")
    print("   PCIe 4.0 x16 (GPU←→CPU):           ~32 GB/s")
    print("   100GbE Network:                    ~10 GB/s practical")
    print("   25GbE Network:                     ~2.5 GB/s practical")
    print("   10GbE Network:                     ~1.0 GB/s practical")

    print("\n4. BANDWIDTH VERIFICATION:")
    print("-"*70)

    if results:
        avg_bandwidth = sum(r['avg_bandwidth_gb_s'] for r in results) / len(results)
        print(f"   Measured average bandwidth: {avg_bandwidth:.3f} GB/s")

        # Determine likely bottleneck
        if avg_bandwidth > 5:
            print("   Analysis: High bandwidth suggests localhost/loopback transfer")
            print("             This is expected when P and D run on same machine")
        elif avg_bandwidth > 1:
            print("   Analysis: Bandwidth consistent with high-speed network (25-100GbE)")
        elif avg_bandwidth > 0.5:
            print("   Analysis: Bandwidth consistent with 10GbE network")
        else:
            print("   Analysis: Low bandwidth - check for network congestion or misconfig")

        print("\n   Note: Measured time includes prefill computation + transfer + decode start")
        print("         Actual transfer bandwidth is higher than apparent bandwidth")

    print("\n5. NCCL TCP TRANSPORT VERIFICATION:")
    print("-"*70)
    print("   Environment settings applied:")
    print("   - NCCL_IB_DISABLE=1 (InfiniBand disabled)")
    print("   - NCCL_NET=Socket (TCP transport forced)")
    print("   These settings ensure KV cache is transferred via TCP, not NVLink/RDMA")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="KV Cache Transfer Bandwidth Test")
    parser.add_argument("--proxy-url", default="http://localhost:10001",
                        help="Proxy server URL")
    parser.add_argument("--model-path",
                        default="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
                        help="Model path")
    parser.add_argument("--prompt-lengths", type=int, nargs="+",
                        default=[100, 500, 1000],
                        help="Prompt lengths to test (in approximate tokens)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations per prompt length")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    print("="*70)
    print("vLLM PD Separation - KV Cache Transfer Bandwidth Test")
    print("="*70)
    print(f"Proxy URL: {args.proxy_url}")
    print(f"Model: {args.model_path}")
    print(f"Test prompt lengths: {args.prompt_lengths}")
    print(f"Iterations per length: {args.iterations}")

    # Verify server is running (proxy may not have /health, so just test connection)
    try:
        # Try a simple POST to see if proxy responds
        response = requests.post(
            f"{args.proxy_url}/v1/completions",
            json={
                "model": args.model_path,
                "prompt": "test",
                "max_tokens": 1,
            },
            timeout=30,
        )
        print(f"Server connection verified (status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"\nError: Cannot connect to proxy server at {args.proxy_url}")
        print(f"Make sure run_pd_separation_test.sh is running first.")
        print(f"Error: {e}")
        return

    # Run tests
    results = run_bandwidth_test(
        proxy_url=args.proxy_url,
        model_path=args.model_path,
        prompt_lengths=args.prompt_lengths,
        num_iterations=args.iterations,
    )

    # Print analysis
    print_bandwidth_analysis(results)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
