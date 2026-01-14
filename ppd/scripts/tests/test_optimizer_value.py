#!/usr/bin/env python3
"""
Optimizer Value Verification Experiment

Core Question: Can dynamic mode switching benefit overcome capacity reduction?

Setup Comparison:
1. Pure Replica (4 workers): Full capacity, no mode switching
2. Pure PPD (2P+2D): Full PD capacity, no mode switching
3. Simulated Optimizer (2P+2pD): Half replica capacity, but can switch

Test Scenarios:
- Scenario A: All requests favor Replica (tiny requests, high QPS)
- Scenario B: All requests favor PPD (multi-turn, large context)
- Scenario C: Mixed workload (50% each)

Key Metrics:
- Throughput (requests/sec)
- Average TTFT
- P99 TTFT
"""

import argparse
import asyncio
import json
import time
import random
from dataclasses import dataclass, field
from typing import Optional
import aiohttp


@dataclass
class RequestResult:
    request_id: int
    workload_type: str  # "tiny" or "large_context"
    ttft_ms: float
    e2e_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    name: str
    mode: str
    total_requests: int
    successful_requests: int
    total_time_s: float
    throughput_rps: float
    avg_ttft_ms: float
    p99_ttft_ms: float
    results: list[RequestResult] = field(default_factory=list)


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    workload_type: str,
) -> RequestResult:
    """Send a single request and measure latency."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    start_time = time.perf_counter()
    ttft = None
    error = None

    try:
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            if resp.status == 200:
                # For non-streaming, TTFT ≈ E2E
                await resp.json()
                ttft = (time.perf_counter() - start_time) * 1000
            else:
                error = f"HTTP {resp.status}"
    except Exception as e:
        error = str(e)

    e2e = (time.perf_counter() - start_time) * 1000
    ttft = ttft or e2e

    return RequestResult(
        request_id=request_id,
        workload_type=workload_type,
        ttft_ms=ttft,
        e2e_ms=e2e,
        success=error is None,
        error=error,
    )


async def run_workload(
    url: str,
    model: str,
    workload_mix: list[tuple[str, str, int]],  # [(type, prompt, max_tokens), ...]
    target_qps: float,
    duration_s: float,
    scenario_name: str,
    mode_name: str,
) -> ScenarioResult:
    """Run a workload at target QPS for specified duration."""

    print(f"\n  Running: {scenario_name} @ {target_qps} QPS for {duration_s}s...")

    results = []
    request_id = 0
    start_time = time.perf_counter()
    interval = 1.0 / target_qps

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        tasks = []
        next_send_time = start_time

        while time.perf_counter() - start_time < duration_s:
            # Select workload based on mix
            workload_type, prompt, max_tokens = random.choice(workload_mix)

            # Create task
            task = asyncio.create_task(
                send_request(session, url, model, prompt, max_tokens, request_id, workload_type)
            )
            tasks.append(task)
            request_id += 1

            # Wait for next send time (Poisson-like arrival)
            next_send_time += interval
            wait_time = next_send_time - time.perf_counter()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Wait for all tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.perf_counter() - start_time

    # Filter successful results
    valid_results = [r for r in results if isinstance(r, RequestResult) and r.success]
    ttfts = [r.ttft_ms for r in valid_results]

    return ScenarioResult(
        name=scenario_name,
        mode=mode_name,
        total_requests=len(results),
        successful_requests=len(valid_results),
        total_time_s=total_time,
        throughput_rps=len(valid_results) / total_time if total_time > 0 else 0,
        avg_ttft_ms=sum(ttfts) / len(ttfts) if ttfts else 0,
        p99_ttft_ms=sorted(ttfts)[int(len(ttfts) * 0.99)] if len(ttfts) > 10 else (max(ttfts) if ttfts else 0),
        results=valid_results,
    )


async def main():
    parser = argparse.ArgumentParser(description="Optimizer Value Verification")
    parser.add_argument("--mode", choices=["replica", "ppd", "optimizer"], required=True,
                        help="Test mode")
    parser.add_argument("--url", required=True, help="Server URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--qps", type=float, default=2.0, help="Target QPS")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration (seconds)")
    args = parser.parse_args()

    # Define workloads
    TINY_PROMPT = "Hi"  # ~2 tokens
    TINY_MAX_TOKENS = 8

    LARGE_CONTEXT_PROMPT = "A " * 256 + "Summarize:"  # ~256 tokens
    LARGE_CONTEXT_MAX_TOKENS = 32

    # Workload mixes for different scenarios
    SCENARIO_A_MIX = [("tiny", TINY_PROMPT, TINY_MAX_TOKENS)] * 10  # 100% tiny
    SCENARIO_B_MIX = [("large_context", LARGE_CONTEXT_PROMPT, LARGE_CONTEXT_MAX_TOKENS)] * 10  # 100% large
    SCENARIO_C_MIX = [
        ("tiny", TINY_PROMPT, TINY_MAX_TOKENS),
        ("large_context", LARGE_CONTEXT_PROMPT, LARGE_CONTEXT_MAX_TOKENS),
    ] * 5  # 50-50 mix

    print("=" * 70)
    print(f"OPTIMIZER VALUE VERIFICATION - Mode: {args.mode.upper()}")
    print("=" * 70)
    print(f"URL: {args.url}")
    print(f"QPS: {args.qps}")
    print(f"Duration: {args.duration}s per scenario")

    all_results = []

    # Run Scenario A: All tiny requests
    result_a = await run_workload(
        args.url, args.model,
        SCENARIO_A_MIX,
        args.qps, args.duration,
        "Scenario_A_AllTiny", args.mode
    )
    all_results.append(result_a)
    print(f"    Throughput: {result_a.throughput_rps:.2f} rps, Avg TTFT: {result_a.avg_ttft_ms:.1f}ms")

    await asyncio.sleep(2)

    # Run Scenario B: All large context requests
    result_b = await run_workload(
        args.url, args.model,
        SCENARIO_B_MIX,
        args.qps, args.duration,
        "Scenario_B_AllLargeCtx", args.mode
    )
    all_results.append(result_b)
    print(f"    Throughput: {result_b.throughput_rps:.2f} rps, Avg TTFT: {result_b.avg_ttft_ms:.1f}ms")

    await asyncio.sleep(2)

    # Run Scenario C: Mixed workload
    result_c = await run_workload(
        args.url, args.model,
        SCENARIO_C_MIX,
        args.qps, args.duration,
        "Scenario_C_Mixed", args.mode
    )
    all_results.append(result_c)
    print(f"    Throughput: {result_c.throughput_rps:.2f} rps, Avg TTFT: {result_c.avg_ttft_ms:.1f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Throughput':<12} {'Avg TTFT':<12} {'P99 TTFT':<12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r.name:<25} {r.throughput_rps:<12.2f} {r.avg_ttft_ms:<12.1f} {r.p99_ttft_ms:<12.1f}")

    # Save results
    output_file = f"results/optimizer_value_{args.mode}.json"
    with open(output_file, "w") as f:
        json.dump({
            "mode": args.mode,
            "qps": args.qps,
            "duration": args.duration,
            "scenarios": [
                {
                    "name": r.name,
                    "throughput_rps": r.throughput_rps,
                    "avg_ttft_ms": r.avg_ttft_ms,
                    "p99_ttft_ms": r.p99_ttft_ms,
                    "total_requests": r.total_requests,
                    "successful_requests": r.successful_requests,
                }
                for r in all_results
            ]
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
