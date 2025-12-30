#!/usr/bin/env python3
"""
QPS vs Latency Benchmark for Replication Mode (Data Parallelism Baseline)

================================================================================
REPLICATION MODE BENCHMARK
================================================================================

This benchmark tests the Replication mode (Data Parallelism) baseline.
It uses the SAME methodology and data format as qps_benchmark.py for PD/PPD,
allowing results to be merged for comparison.

Key Differences from PD/PPD Benchmark:
- Uses separate replication proxy (port 10002)
- No Turn 1 setup needed (no conversation state for D-direct)
- Each request is independent

================================================================================
WORKLOAD DEFINITIONS (Same as qps_benchmark.py)
================================================================================

W1: Chat (Balanced) - Input: 512, Output: 256
W2: RAG (Read-Heavy) - Input: 4096, Output: 128
W3: Agent (Write-Heavy) - Input: 256, Output: 1024
W4: Limit (Context Wall) - Input: 8192, Output: 64

================================================================================
"""

import argparse
import asyncio
import json
import time
import numpy as np
import aiohttp
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

PROXY_URL = "http://localhost:10002"  # Replication proxy port
MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    req_id: str
    mode: str
    workload: str
    qps_target: float
    input_len: int
    output_len: int
    start_time: float
    ttft_ms: float
    e2e_latency_ms: float
    output_tokens: int
    throughput_tps: float
    success: bool
    error: Optional[str] = None


@dataclass
class QPSStepResult:
    """Aggregated result for a single QPS step."""
    workload: str
    input_tokens: int
    output_tokens: int
    target_qps: float
    mode: str
    duration_s: int

    # Sample statistics
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    real_qps: float = 0.0

    # TTFT statistics (ms)
    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p90_ttft: float = 0.0
    p99_ttft: float = 0.0
    min_ttft: float = 0.0
    max_ttft: float = 0.0

    # E2E latency statistics (ms)
    avg_e2e: float = 0.0
    p50_e2e: float = 0.0
    p90_e2e: float = 0.0
    p99_e2e: float = 0.0

    # Throughput
    avg_throughput_tps: float = 0.0
    total_tokens: int = 0

    # Raw metrics for debugging
    raw_metrics: list = field(default_factory=list)


@dataclass
class WorkloadConfig:
    """Configuration for a workload type."""
    name: str
    description: str
    input_tokens: int
    output_tokens: int
    expected_winner: str


# Standard workload definitions (SAME as qps_benchmark.py)
WORKLOADS = [
    WorkloadConfig(
        name="W1_Chat_Balanced",
        description="Short I/O, latency sensitive. Tests TCP overhead vs isolation benefit.",
        input_tokens=512,
        output_tokens=256,
        expected_winner="Replication at low load"
    ),
    WorkloadConfig(
        name="W2_RAG_ReadHeavy",
        description="Long input, severe HOL blocking candidate. PD's main battlefield.",
        input_tokens=4096,
        output_tokens=128,
        expected_winner="PD at high load"
    ),
    WorkloadConfig(
        name="W3_Agent_WriteHeavy",
        description="Long output, decode bound. Tests ITL vulnerability to prefill.",
        input_tokens=256,
        output_tokens=1024,
        expected_winner="Tie"
    ),
    WorkloadConfig(
        name="W4_Limit_Context",
        description="Extreme context, bandwidth/memory stress test. Tests system boundaries.",
        input_tokens=8192,
        output_tokens=64,
        expected_winner="System dependent"
    ),
]

# QPS sweep configuration (SAME as qps_benchmark.py)
QPS_SWEEP = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


def generate_prompt(num_tokens: int, unique_id: str = "") -> str:
    """Generate a prompt with approximately the specified number of tokens."""
    base_text = (
        "This is a comprehensive benchmark test for the vLLM disaggregated serving system. "
        "The system separates prefill and decode phases across different GPU instances. "
        "We are measuring latency under various load conditions to understand performance characteristics. "
    )
    target_chars = num_tokens * 4
    repeated = (base_text * ((target_chars // len(base_text)) + 1))[:target_chars]
    if unique_id:
        return f"[REQ:{unique_id}] Analyze and respond: {repeated}"
    return repeated


async def clear_state(session: aiohttp.ClientSession):
    """Clear proxy state."""
    try:
        await session.post(f"{PROXY_URL}/metrics/clear")
    except Exception:
        pass


async def send_request(
    session: aiohttp.ClientSession,
    req_id: str,
    workload: str,
    qps: float,
    input_len: int,
    output_len: int,
) -> RequestMetrics:
    """Send a single request and measure metrics."""
    prompt = generate_prompt(input_len, req_id)
    full_prompt = f"User: {prompt}\nAssistant:"

    start_t = time.perf_counter()
    ttft = 0.0
    first_token = False
    tokens_received = 0

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": full_prompt,
                "max_tokens": output_len,
                "temperature": 0.8,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                return RequestMetrics(
                    req_id=req_id, mode="replication", workload=workload, qps_target=qps,
                    input_len=input_len, output_len=output_len, start_time=start_t,
                    ttft_ms=0, e2e_latency_ms=0, output_tokens=0, throughput_tps=0,
                    success=False, error=f"HTTP {resp.status}"
                )

            async for line in resp.content:
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    try:
                        chunk = json.loads(decoded[6:])
                        if not first_token and chunk.get("choices"):
                            if chunk["choices"][0].get("text"):
                                ttft = (time.perf_counter() - start_t) * 1000
                                first_token = True
                        if chunk.get("choices") and chunk["choices"][0].get("text"):
                            tokens_received += 1
                        if chunk.get("usage"):
                            tokens_received = chunk["usage"].get("completion_tokens", tokens_received)
                    except json.JSONDecodeError:
                        continue

    except asyncio.TimeoutError:
        return RequestMetrics(
            req_id=req_id, mode="replication", workload=workload, qps_target=qps,
            input_len=input_len, output_len=output_len, start_time=start_t,
            ttft_ms=0, e2e_latency_ms=0, output_tokens=0, throughput_tps=0,
            success=False, error="Timeout"
        )
    except Exception as e:
        return RequestMetrics(
            req_id=req_id, mode="replication", workload=workload, qps_target=qps,
            input_len=input_len, output_len=output_len, start_time=start_t,
            ttft_ms=0, e2e_latency_ms=0, output_tokens=0, throughput_tps=0,
            success=False, error=str(e)
        )

    end_t = time.perf_counter()
    latency = (end_t - start_t) * 1000
    tps = (tokens_received / (latency / 1000)) if latency > 0 else 0

    return RequestMetrics(
        req_id=req_id, mode="replication", workload=workload, qps_target=qps,
        input_len=input_len, output_len=output_len, start_time=start_t,
        ttft_ms=ttft, e2e_latency_ms=latency, output_tokens=tokens_received,
        throughput_tps=tps, success=True
    )


async def run_qps_step(
    workload: WorkloadConfig,
    qps: float,
    duration_s: int,
) -> QPSStepResult:
    """Run a specific QPS load for a duration using Poisson arrival process."""
    print(f"    [REPLICATION] Target QPS: {qps}, Duration: {duration_s}s...")

    result = QPSStepResult(
        workload=workload.name,
        input_tokens=workload.input_tokens,
        output_tokens=workload.output_tokens,
        target_qps=qps,
        mode="replication",
        duration_s=duration_s,
    )

    async with aiohttp.ClientSession() as session:
        await clear_state(session)
        await asyncio.sleep(1)

        # Generate Poisson arrival times
        expected_requests = int(qps * duration_s)
        if expected_requests == 0:
            expected_requests = 1

        # Inter-arrival time follows Exponential distribution
        arrival_intervals = np.random.exponential(1.0 / qps, expected_requests)
        arrival_times = np.cumsum(arrival_intervals)

        # Only use arrivals within duration
        arrival_times = arrival_times[arrival_times < duration_s]
        actual_requests = len(arrival_times)

        print(f"      Sending {actual_requests} requests with Poisson arrival...")

        metrics_list = []
        tasks = []

        start_benchmark = time.perf_counter()

        for i, delay in enumerate(arrival_times):
            now = time.perf_counter() - start_benchmark
            wait = delay - now
            if wait > 0:
                await asyncio.sleep(wait)

            req_id = f"{workload.name}_repl_{qps}_{i}"
            task = asyncio.create_task(
                send_request(
                    session, req_id, workload.name, qps,
                    workload.input_tokens, workload.output_tokens
                )
            )
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, RequestMetrics):
                metrics_list.append(r)

        # Calculate statistics
        successful = [m for m in metrics_list if m.success]
        failed = [m for m in metrics_list if not m.success]

        result.sample_count = len(metrics_list)
        result.success_count = len(successful)
        result.failure_count = len(failed)
        result.real_qps = len(successful) / duration_s if duration_s > 0 else 0

        if successful:
            ttfts = [m.ttft_ms for m in successful if m.ttft_ms > 0]
            e2es = [m.e2e_latency_ms for m in successful]
            throughputs = [m.throughput_tps for m in successful]

            if ttfts:
                result.avg_ttft = np.mean(ttfts)
                result.p50_ttft = np.percentile(ttfts, 50)
                result.p90_ttft = np.percentile(ttfts, 90)
                result.p99_ttft = np.percentile(ttfts, 99)
                result.min_ttft = np.min(ttfts)
                result.max_ttft = np.max(ttfts)

            if e2es:
                result.avg_e2e = np.mean(e2es)
                result.p50_e2e = np.percentile(e2es, 50)
                result.p90_e2e = np.percentile(e2es, 90)
                result.p99_e2e = np.percentile(e2es, 99)

            if throughputs:
                result.avg_throughput_tps = np.mean(throughputs)

            result.total_tokens = sum(m.output_tokens for m in successful)
            result.raw_metrics = [asdict(m) for m in successful[:10]]

        print(f"      Complete. Success: {result.success_count}/{result.sample_count}, "
              f"P99 TTFT: {result.p99_ttft:.2f}ms, Avg E2E: {result.avg_e2e:.2f}ms")

    return result


async def run_workload_sweep(
    workload: WorkloadConfig,
    qps_list: list[float],
    duration_s: int = 45,
) -> list[QPSStepResult]:
    """Run a complete QPS sweep for a workload."""
    print(f"\n{'#'*70}")
    print(f"WORKLOAD: {workload.name}")
    print(f"  {workload.description}")
    print(f"  Input: {workload.input_tokens}, Output: {workload.output_tokens}")
    print(f"{'#'*70}")

    results = []

    for qps in qps_list:
        print(f"\n  --- QPS: {qps} ---")

        step_result = await run_qps_step(workload, qps, duration_s)
        results.append(step_result)

        # Cooldown between runs
        await asyncio.sleep(5)

    return results


def print_summary(all_results: list[QPSStepResult]):
    """Print summary of all results."""
    print("\n" + "=" * 80)
    print("REPLICATION BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by workload
    workloads = {}
    for r in all_results:
        if r.workload not in workloads:
            workloads[r.workload] = []
        workloads[r.workload].append(r)

    for wk_name, results in workloads.items():
        print(f"\n{wk_name}")
        print("-" * 80)
        print(f"{'QPS':<6} {'Success':<10} {'P50 TTFT':<12} {'P99 TTFT':<12} {'Avg E2E':<12} {'Real QPS':<10}")
        print("-" * 80)

        for r in sorted(results, key=lambda x: x.target_qps):
            print(f"{r.target_qps:<6.1f} "
                  f"{r.success_count}/{r.sample_count:<6} "
                  f"{r.p50_ttft:<12.2f} {r.p99_ttft:<12.2f} "
                  f"{r.avg_e2e:<12.2f} {r.real_qps:<10.2f}")


async def main():
    parser = argparse.ArgumentParser(description="Replication Mode QPS Benchmark")
    parser.add_argument("--duration", type=int, default=45,
                        help="Duration per QPS point in seconds (default: 45)")
    parser.add_argument("--workload", type=str, choices=["W1", "W2", "W3", "W4", "all"],
                        default="all", help="Workload to run (default: all)")
    parser.add_argument("--qps", type=str, default="0.5,1.0,2.0,3.0,4.0,6.0,8.0",
                        help="Comma-separated QPS values to sweep")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--list", action="store_true", help="List available workloads")

    args = parser.parse_args()

    if args.list:
        print("Available workloads:")
        for wk in WORKLOADS:
            print(f"\n  {wk.name}")
            print(f"    {wk.description}")
            print(f"    Input: {wk.input_tokens}, Output: {wk.output_tokens}")
        return

    # Parse QPS list
    qps_list = [float(q.strip()) for q in args.qps.split(",")]

    # Select workloads
    if args.workload == "all":
        workloads = WORKLOADS
    else:
        workloads = [wk for wk in WORKLOADS if wk.name.startswith(args.workload)]

    print("=" * 70)
    print("Replication Mode QPS Benchmark")
    print("=" * 70)
    print(f"Mode: replication (Data Parallelism)")
    print(f"Workloads: {len(workloads)}")
    print(f"QPS sweep: {qps_list}")
    print(f"Duration per point: {args.duration}s")
    print(f"Total experiments: {len(workloads) * len(qps_list)}")
    print(f"Estimated time: ~{len(workloads) * len(qps_list) * (args.duration + 10) / 60:.0f} minutes")
    print("=" * 70)

    # Check connection
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status == 200:
                    status = await resp.json()
                    print(f"Connected: {status}")
    except Exception as e:
        print(f"Cannot connect to replication proxy at {PROXY_URL}: {e}")
        print("Start servers with: ./scripts/start_replication_servers.sh")
        return

    # Run benchmark
    all_results = []
    for workload in workloads:
        results = await run_workload_sweep(workload, qps_list, args.duration)
        all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or str(RESULTS_DIR / f"replication_benchmark_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "workloads": [asdict(wk) for wk in workloads],
                "qps_sweep": qps_list,
                "duration_s": args.duration,
            },
            "results": [asdict(r) for r in all_results],
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
