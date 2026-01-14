#!/usr/bin/env python3
"""
PD Mode Benchmark Script (V5)

Usage:
    python benchmark_pd.py --run-id 1 [--workloads S_a,M_b] [--duration 10]

This script benchmarks PD (Prefill-Decode disaggregation) mode without cache affinity.
"""

import argparse
import asyncio
import json
import time
import numpy as np
import aiohttp
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmark_common import (
    MODEL_PATH, RESULTS_DIR, DECODE_URLS,
    WorkloadConfig, BenchmarkResult, RequestMetrics, CacheSnapshot,
    CONTEXT_CONFIGS, T2_CONFIGS, QPS_BY_CONTEXT_PD, CORE_WORKLOADS,
    MIN_SUCCESS_RATE,
    create_workload, send_turn1, send_turn2, compute_turn_metrics,
    get_all_cache_snapshots, compute_cache_delta,
    check_all_servers_health, wait_for_servers_recovery,
)

PROXY_URL = "http://localhost:10001"
MODE = "pd"

# Server restart configuration
# PD mode accumulates KV cache across workloads, causing OOM on larger contexts
# Restart servers between context size groups to clear KV cache
RESTART_BETWEEN_CONTEXT_GROUPS = True
CONTEXT_ORDER = ["XS", "S", "M", "L", "XL"]  # Order to run context sizes


async def restart_servers():
    """Restart PD servers to clear KV cache."""
    import subprocess
    print("\n  [Server Restart] Clearing KV cache by restarting servers...")

    # Stop servers
    subprocess.run(
        ["bash", "scripts/server/stop_servers_4gpu.sh"],
        capture_output=True,
        cwd="/net/projects2/ds3lab/zongzel/vllm/ppd"
    )
    await asyncio.sleep(5)

    # Start servers in PD mode
    subprocess.run(
        ["bash", "scripts/server/start_servers_4gpu.sh", "ppd"],
        capture_output=True,
        cwd="/net/projects2/ds3lab/zongzel/vllm/ppd"
    )

    # Wait for servers to be ready
    print("  [Server Restart] Waiting for servers to be ready...")
    for attempt in range(30):
        await asyncio.sleep(5)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{PROXY_URL}/status", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        # Also verify we can set PD mode
                        await session.post(f"{PROXY_URL}/mode/pd")
                        print(f"  [Server Restart] Servers ready after {(attempt+1)*5}s")
                        return True
        except:
            pass

    print("  [Server Restart] WARNING: Servers may not be fully ready")
    return False


def save_results(results: list[BenchmarkResult], output_dir: Path, run_id: int):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{MODE}_run{run_id}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump({
            "benchmark_type": f"{MODE}_benchmark_v5",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(r) for r in results]
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


async def warmup(session: aiohttp.ClientSession):
    """Warmup the system."""
    print("Warming up...")

    for _ in range(3):
        conv_id = f"warmup_{int(time.time())}_{np.random.randint(10000)}"
        workload = create_workload("S", "a", 2)
        success, history, _ = await send_turn1(session, PROXY_URL, conv_id, workload, MODE)
        if success:
            await send_turn2(session, PROXY_URL, conv_id, history, workload, MODE)
        await asyncio.sleep(0.5)

    print("Warmup complete.")


async def run_qps_test(
    session: aiohttp.ClientSession,
    workload: WorkloadConfig,
    qps: float,
    duration_s: int,
    run_id: int,
) -> BenchmarkResult:
    """Run a single QPS test with per-turn cache metrics collection."""
    print(f"  QPS={qps:.1f}, duration={duration_s}s...")

    # Generate Poisson arrival times
    expected_requests = max(int(qps * duration_s), 1)
    arrival_intervals = np.random.exponential(1.0 / qps, expected_requests * 2)
    arrival_times = np.cumsum(arrival_intervals)
    arrival_times = arrival_times[arrival_times < duration_s]

    all_metrics: list[RequestMetrics] = []
    raw_metrics: list[dict] = [{"run": run_id}]
    histories: dict[str, str] = {}

    # ========== Phase 1: Turn 1 ==========
    # Get cache snapshot before Turn 1
    cache_before_t1 = await get_all_cache_snapshots()

    t1_tasks = []
    t1_conv_ids = []
    start_time = time.perf_counter()

    for i, arrival in enumerate(arrival_times):
        now = time.perf_counter() - start_time
        if arrival > now:
            await asyncio.sleep(arrival - now)

        conv_id = f"conv_{i}_{int(time.time())}_{np.random.randint(10000)}"
        t1_conv_ids.append(conv_id)
        task = asyncio.create_task(
            send_turn1(session, PROXY_URL, conv_id, workload, MODE, qps)
        )
        t1_tasks.append(task)

    t1_results = await asyncio.gather(*t1_tasks, return_exceptions=True)

    for conv_id, result in zip(t1_conv_ids, t1_results):
        if isinstance(result, tuple) and len(result) == 3:
            success, history, metrics = result
            all_metrics.append(metrics)
            raw_metrics.append(asdict(metrics))
            if success:
                histories[conv_id] = history

    # Get cache snapshot after Turn 1
    cache_after_t1 = await get_all_cache_snapshots()
    t1_cache_delta = compute_cache_delta(cache_before_t1, cache_after_t1)

    # ========== Phase 2: Turn 2 ==========
    # Get cache snapshot before Turn 2
    cache_before_t2 = await get_all_cache_snapshots()

    t2_tasks = []
    t2_conv_ids = []

    for conv_id, history in histories.items():
        task = asyncio.create_task(
            send_turn2(session, PROXY_URL, conv_id, history, workload, MODE, qps, 2)
        )
        t2_tasks.append(task)
        t2_conv_ids.append(conv_id)

    if t2_tasks:
        t2_results = await asyncio.gather(*t2_tasks, return_exceptions=True)

        for conv_id, result in zip(t2_conv_ids, t2_results):
            if isinstance(result, tuple) and len(result) == 3:
                success, history, metrics = result
                all_metrics.append(metrics)
                raw_metrics.append(asdict(metrics))
                if success:
                    histories[conv_id] = history

    # Get cache snapshot after Turn 2
    cache_after_t2 = await get_all_cache_snapshots()
    t2_cache_delta = compute_cache_delta(cache_before_t2, cache_after_t2)

    # Phase 3-5: Additional turns if num_turns > 2
    for turn_num in range(3, workload.num_turns + 1):
        turn_tasks = []
        turn_conv_ids = []

        for conv_id, history in histories.items():
            task = asyncio.create_task(
                send_turn2(session, PROXY_URL, conv_id, history, workload, MODE, qps, turn_num)
            )
            turn_tasks.append(task)
            turn_conv_ids.append(conv_id)

        if turn_tasks:
            turn_results = await asyncio.gather(*turn_tasks, return_exceptions=True)

            for conv_id, result in zip(turn_conv_ids, turn_results):
                if isinstance(result, tuple) and len(result) == 3:
                    success, history, metrics = result
                    all_metrics.append(metrics)
                    raw_metrics.append(asdict(metrics))
                    if success:
                        histories[conv_id] = history

    elapsed = time.perf_counter() - start_time

    # Total cache stats (for backward compatibility)
    total_cache_stats = compute_cache_delta(cache_before_t1, cache_after_t2)

    # Compute turn metrics
    t1_metrics = [m for m in all_metrics if m.turn == 1]
    t2_metrics = [m for m in all_metrics if m.turn == 2]

    t1_success = [m for m in t1_metrics if m.success]
    t2_success = [m for m in t2_metrics if m.success]

    # Turn 1 metrics (with per-turn cache info)
    turn1_metrics = {
        "turn": 1,
        "sample_count": len(t1_metrics),
        "success_count": len(t1_success),
        "cache": t1_cache_delta,
    }
    if t1_success:
        ttfts = [m.ttft_ms for m in t1_success if m.ttft_ms > 0]
        e2es = [m.e2e_latency_ms for m in t1_success]
        tpots = [m.tpot_ms for m in t1_success if m.tpot_ms > 0]
        if ttfts:
            turn1_metrics["avg_ttft"] = np.mean(ttfts)
            turn1_metrics["p50_ttft"] = np.percentile(ttfts, 50)
            turn1_metrics["p90_ttft"] = np.percentile(ttfts, 90)
            turn1_metrics["p99_ttft"] = np.percentile(ttfts, 99)
        if tpots:
            turn1_metrics["avg_tpot"] = np.mean(tpots)
        if e2es:
            turn1_metrics["avg_e2e"] = np.mean(e2es)
            turn1_metrics["p50_e2e"] = np.percentile(e2es, 50)
            turn1_metrics["p99_e2e"] = np.percentile(e2es, 99)
        turn1_metrics["total_tokens"] = sum(m.output_tokens for m in t1_success)

    # Turn 2 metrics (with per-turn cache info)
    # Calculate KV reuse from T1 (for model fitting)
    # In PD mode, T2 also goes through P->D, so prefix cache may show hits
    t1_context_tokens = workload.turn1.input_tokens + workload.turn1.output_tokens
    t2_total_context = t1_context_tokens + workload.turn2.input_tokens
    kv_reuse_pct = (t1_context_tokens / t2_total_context * 100) if t2_total_context > 0 else 0

    turn2_metrics = {
        "turn": 2,
        "sample_count": len(t2_metrics),
        "success_count": len(t2_success),
        "cache": t2_cache_delta,
        "kv_reuse_tokens": t1_context_tokens,  # Tokens that could be reused from T1
        "kv_reuse_pct": kv_reuse_pct,  # Percentage of T2 context from T1
    }
    if t2_success:
        ttfts = [m.ttft_ms for m in t2_success if m.ttft_ms > 0]
        tpots = [m.tpot_ms for m in t2_success if m.tpot_ms > 0]
        e2es = [m.e2e_latency_ms for m in t2_success]
        throughputs = [m.throughput_tps for m in t2_success]

        if ttfts:
            turn2_metrics["avg_ttft"] = np.mean(ttfts)
            turn2_metrics["p50_ttft"] = np.percentile(ttfts, 50)
            turn2_metrics["p90_ttft"] = np.percentile(ttfts, 90)
            turn2_metrics["p99_ttft"] = np.percentile(ttfts, 99)
        if tpots:
            turn2_metrics["avg_tpot"] = np.mean(tpots)
            turn2_metrics["p50_tpot"] = np.percentile(tpots, 50)
            turn2_metrics["p99_tpot"] = np.percentile(tpots, 99)
        if e2es:
            turn2_metrics["avg_e2e"] = np.mean(e2es)
            turn2_metrics["p50_e2e"] = np.percentile(e2es, 50)
            turn2_metrics["p99_e2e"] = np.percentile(e2es, 99)
        if throughputs:
            turn2_metrics["avg_throughput_tps"] = np.mean(throughputs)
        turn2_metrics["total_tokens"] = sum(m.output_tokens for m in t2_success)

    # Aggregate metrics (T2 focused)
    total_requests = len(all_metrics)
    successful = sum(1 for m in all_metrics if m.success)
    actual_qps = total_requests / elapsed if elapsed > 0 else 0

    # T2 aggregates for main metrics
    t2_ttfts = [m.ttft_ms for m in t2_success if m.ttft_ms > 0]
    t2_e2es = [m.e2e_latency_ms for m in t2_success]
    total_e2es = [m.e2e_latency_ms for m in t1_success] + t2_e2es

    result = BenchmarkResult(
        mode=MODE,
        workload=workload.name,
        target_qps=qps,
        duration_s=duration_s,
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        t1_input=workload.turn1.input_tokens,
        t1_output=workload.turn1.output_tokens,
        t2_input=workload.turn2.input_tokens,
        t2_output=workload.turn2.output_tokens,
        num_turns=workload.num_turns,
        turn1_metrics=turn1_metrics,
        turn2_metrics=turn2_metrics,
        sample_count=len(t2_metrics),
        success_count=len(t2_success),
        failure_count=len(t2_metrics) - len(t2_success),
        real_qps=actual_qps,
        cache_stats=total_cache_stats,
        raw_metrics=raw_metrics,
    )

    # Fill in aggregate T2 metrics
    if t2_ttfts:
        result.avg_ttft = np.mean(t2_ttfts)
        result.p50_ttft = np.percentile(t2_ttfts, 50)
        result.p90_ttft = np.percentile(t2_ttfts, 90)
        result.p99_ttft = np.percentile(t2_ttfts, 99)
        result.min_ttft = np.min(t2_ttfts)
        result.max_ttft = np.max(t2_ttfts)

    if t2_e2es:
        result.avg_e2e = np.mean(t2_e2es)
        result.p50_e2e = np.percentile(t2_e2es, 50)
        result.p90_e2e = np.percentile(t2_e2es, 90)
        result.p99_e2e = np.percentile(t2_e2es, 99)

    if total_e2es:
        result.avg_total_e2e = np.mean(total_e2es)
        result.p50_total_e2e = np.percentile(total_e2es, 50)
        result.p99_total_e2e = np.percentile(total_e2es, 99)

    if t2_success:
        result.avg_throughput_tps = np.mean([m.throughput_tps for m in t2_success])
        result.total_tokens = sum(m.output_tokens for m in t2_success)

    print(f"    Completed: {successful}/{total_requests} OK, QPS={actual_qps:.2f}")
    print(f"    T1 Cache: local_hits={t1_cache_delta['local_cache_hits']}, local_rate={t1_cache_delta['local_hit_rate_pct']:.1f}%")
    print(f"    T2 Cache: local_hits={t2_cache_delta['local_cache_hits']}, local_rate={t2_cache_delta['local_hit_rate_pct']:.1f}%, ext_hits={t2_cache_delta['external_cache_hits']}")

    return result


async def run_workload(
    session: aiohttp.ClientSession,
    workload: WorkloadConfig,
    run_id: int,
    duration: int,
) -> list[BenchmarkResult]:
    """Run all QPS levels for a workload with adaptive stopping."""
    results = []

    # Get context size from workload name
    context_size = workload.name.split("_")[0]
    qps_levels = QPS_BY_CONTEXT_PD.get(context_size, [0.5, 1, 2, 4])

    print(f"\nWorkload: {workload.name} ({workload.description})")
    print(f"  T1: {workload.turn1.input_tokens}→{workload.turn1.output_tokens}")
    print(f"  T2: {workload.turn2.input_tokens}→{workload.turn2.output_tokens}")
    print(f"  Turns: {workload.num_turns}")
    print(f"  QPS levels: {qps_levels}")

    consecutive_failures = 0
    for qps in qps_levels:
        # Check server health before each QPS test
        healthy, msg = await check_all_servers_health(PROXY_URL)
        if not healthy:
            print(f"    Server unhealthy: {msg}")
            print(f"    Waiting for recovery...")
            if not await wait_for_servers_recovery(PROXY_URL, max_wait_s=60):
                print(f"    Server did not recover, skipping remaining QPS levels")
                break

        result = await run_qps_test(session, workload, qps, duration, run_id)
        results.append(result)

        # Check success rate for adaptive stopping
        success_rate = result.success_count / result.sample_count if result.sample_count > 0 else 0
        if success_rate < MIN_SUCCESS_RATE:
            consecutive_failures += 1
            print(f"    WARNING: Low success rate ({success_rate*100:.1f}%), consecutive failures: {consecutive_failures}")
            if consecutive_failures >= 2:
                print(f"    Stopping workload early due to repeated low success rate")
                break
        else:
            consecutive_failures = 0  # Reset on success

        await asyncio.sleep(3)  # Longer pause between QPS levels for PD mode

    return results


async def main():
    parser = argparse.ArgumentParser(description="PD Mode Benchmark")
    parser.add_argument("--run-id", type=int, default=1, help="Run ID (1, 2, or 3)")
    parser.add_argument("--workloads", type=str, default="",
                        help="Comma-separated workload names (e.g., S_a,M_b). Empty=all core workloads")
    parser.add_argument("--duration", type=int, default=10, help="Duration per QPS level in seconds")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print(f"PD Mode Benchmark (Run {args.run_id})")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check proxy
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status != 200:
                    print("ERROR: Proxy not available")
                    return
                status = await resp.json()
                if status.get("mode") != "pd":
                    # Switch to PD mode
                    await session.post(f"{PROXY_URL}/mode/pd")
                    await asyncio.sleep(1)
                    print("Switched to PD mode")
    except Exception as e:
        print(f"ERROR: Cannot connect to proxy: {e}")
        return

    # Determine workloads
    if args.workloads:
        workload_names = [w.strip() for w in args.workloads.split(",")]
        workloads = []
        for name in workload_names:
            parts = name.split("_")
            if len(parts) >= 2:
                ctx, t2type = parts[0], parts[1]
                num_turns = int(parts[2]) if len(parts) > 2 else 2
                workloads.append(create_workload(ctx, t2type, num_turns))
    else:
        # Use core workloads
        workloads = [create_workload(ctx, t2, turns) for ctx, t2, turns in CORE_WORKLOADS]

    print(f"Workloads to test: {[w.name for w in workloads]}")

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "pd"

    # Group workloads by context size for server restart between groups
    if RESTART_BETWEEN_CONTEXT_GROUPS:
        workloads_by_context = {}
        for w in workloads:
            ctx = w.name.split("_")[0]
            if ctx not in workloads_by_context:
                workloads_by_context[ctx] = []
            workloads_by_context[ctx].append(w)

        # Order by CONTEXT_ORDER
        ordered_contexts = [c for c in CONTEXT_ORDER if c in workloads_by_context]
        print(f"Context groups (with restart between): {ordered_contexts}")
    else:
        workloads_by_context = {"all": workloads}
        ordered_contexts = ["all"]

    all_results = []

    for ctx_idx, ctx in enumerate(ordered_contexts):
        ctx_workloads = workloads_by_context[ctx]
        print(f"\n{'='*60}")
        print(f"Context Group: {ctx} ({len(ctx_workloads)} workloads)")
        print(f"{'='*60}")

        # Restart servers between context groups (except first)
        if RESTART_BETWEEN_CONTEXT_GROUPS and ctx_idx > 0:
            await restart_servers()

        async with aiohttp.ClientSession() as session:
            if ctx_idx == 0:
                await warmup(session)

            for i, workload in enumerate(ctx_workloads):
                # Health check between workloads within same context group
                if i > 0:
                    print(f"\n  Checking server health before next workload...")
                    healthy, msg = await check_all_servers_health(PROXY_URL)
                    if not healthy:
                        print(f"    Server unhealthy: {msg}")
                        if not await wait_for_servers_recovery(PROXY_URL, max_wait_s=120):
                            print(f"    Server did not recover, skipping remaining workloads in {ctx}")
                            break
                        print(f"    Server recovered, continuing...")

                results = await run_workload(session, workload, args.run_id, args.duration)
                all_results.extend(results)

    save_results(all_results, output_dir, args.run_id)

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
