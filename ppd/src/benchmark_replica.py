#!/usr/bin/env python3
"""
Replica Mode Benchmark Script (V5)

Usage:
    python benchmark_replica.py --run-id 1 [--workloads S_a,M_b] [--duration 10]

This script benchmarks Replica mode (4 independent vLLM instances with load balancing).
Requires starting replica servers with start_replication_servers_4gpu.sh first.
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
from typing import Dict

from benchmark_common import (
    MODEL_PATH, RESULTS_DIR,
    WorkloadConfig, BenchmarkResult, RequestMetrics, CacheSnapshot,
    CONTEXT_CONFIGS, T2_CONFIGS, QPS_BY_CONTEXT, CORE_WORKLOADS,
    MIN_SUCCESS_RATE,
    create_workload, compute_turn_metrics,
    generate_prompt, get_cache_snapshot,
    check_server_health, check_proxy_health,
)

# Replica uses different proxy port
PROXY_URL = "http://localhost:10002"
MODE = "replica"

# Replica server URLs (4 independent vLLM instances)
# Must match ports in start_replication_servers_4gpu.sh: 8300, 8400, 8500, 8600
REPLICA_URLS = [
    "http://localhost:8300",
    "http://localhost:8400",
    "http://localhost:8500",
    "http://localhost:8600",
]


async def check_replica_servers_health() -> tuple[bool, str]:
    """Check health of all replica servers."""
    # Check proxy
    if not await check_proxy_health(PROXY_URL):
        return False, "Replica proxy not responding"

    # Check replica servers
    for i, url in enumerate(REPLICA_URLS):
        if not await check_server_health(url):
            return False, f"Replica server {i} ({url}) not responding"

    return True, "All replica servers healthy"


async def wait_for_replica_recovery(max_wait_s: int = 60, check_interval_s: int = 5) -> bool:
    """Wait for replica servers to recover."""
    import time
    start = time.time()
    while time.time() - start < max_wait_s:
        healthy, msg = await check_replica_servers_health()
        if healthy:
            return True
        print(f"    Waiting for server recovery... ({msg})")
        await asyncio.sleep(check_interval_s)
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


async def get_replica_cache_snapshots() -> Dict[str, CacheSnapshot]:
    """Get cache snapshots from all replica servers."""
    snapshots = {}
    for i, url in enumerate(REPLICA_URLS):
        snapshots[f"R{i}"] = await get_cache_snapshot(url)
    return snapshots


def compute_replica_cache_delta(before: Dict[str, CacheSnapshot], after: Dict[str, CacheSnapshot]) -> Dict:
    """Compute cache metrics delta for replica servers."""
    total_local_queries = 0
    total_local_hits = 0
    max_kv_usage = 0.0

    for key in after:
        if key in before:
            total_local_queries += after[key].local_queries - before[key].local_queries
            total_local_hits += after[key].local_hits - before[key].local_hits
        max_kv_usage = max(max_kv_usage, after[key].kv_usage_pct)

    local_hit_rate = (total_local_hits / total_local_queries * 100) if total_local_queries > 0 else 0

    return {
        "local_cache_queries": total_local_queries,
        "local_cache_hits": total_local_hits,
        "local_hit_rate_pct": local_hit_rate,
        "external_cache_queries": 0,
        "external_cache_hits": 0,
        "external_hit_rate_pct": 0,
        "max_kv_usage_pct": max_kv_usage,
    }


async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    output_tokens: int,
    conv_id: str,
    workload_name: str,
    input_tokens: int,
    qps: float,
    turn: int,
) -> RequestMetrics:
    """Send a request to replica proxy."""
    start_t = time.perf_counter()
    ttft = 0.0
    tokens_received = 0

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": prompt,
                "max_tokens": output_tokens,
                "temperature": 0.8,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                return RequestMetrics(
                    req_id=conv_id, mode=MODE, workload=workload_name, qps_target=qps,
                    input_len=input_tokens, output_len=output_tokens,
                    start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
                    throughput_tps=0, success=False, error=f"HTTP {resp.status}", turn=turn
                )

            first_token = False
            buffer = ""
            response_text = ""

            async for chunk_bytes in resp.content.iter_any():
                if not chunk_bytes:
                    continue
                buffer += chunk_bytes.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            if not first_token and chunk.get("choices"):
                                text = chunk["choices"][0].get("text", "")
                                if text:
                                    ttft = (time.perf_counter() - start_t) * 1000
                                    first_token = True
                                    response_text += text
                            elif chunk.get("choices"):
                                response_text += chunk["choices"][0].get("text", "")
                            if chunk.get("usage"):
                                tokens_received = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

            end_t = time.perf_counter()
            latency = (end_t - start_t) * 1000
            tps = (tokens_received / (latency / 1000)) if latency > 0 and tokens_received > 0 else 0
            tpot = (latency - ttft) / max(tokens_received - 1, 1) if tokens_received > 1 else 0

            return RequestMetrics(
                req_id=conv_id, mode=MODE, workload=workload_name, qps_target=qps,
                input_len=input_tokens, output_len=output_tokens,
                start_time=start_t, ttft_ms=ttft, e2e_latency_ms=latency, output_tokens=tokens_received,
                throughput_tps=tps, tpot_ms=tpot, success=True, turn=turn
            ), response_text

    except asyncio.TimeoutError:
        return RequestMetrics(
            req_id=conv_id, mode=MODE, workload=workload_name, qps_target=qps,
            input_len=input_tokens, output_len=output_tokens,
            start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
            throughput_tps=0, success=False, error="Timeout", turn=turn
        ), ""
    except Exception as e:
        return RequestMetrics(
            req_id=conv_id, mode=MODE, workload=workload_name, qps_target=qps,
            input_len=input_tokens, output_len=output_tokens,
            start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
            throughput_tps=0, success=False, error=str(e)[:100], turn=turn
        ), ""


async def warmup(session: aiohttp.ClientSession):
    """Warmup the system."""
    print("Warming up...")

    for _ in range(3):
        prompt = generate_prompt(256, "warmup_")
        result = await send_request(
            session, prompt, 64, "warmup", "warmup", 256, 0, 1
        )
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
    cache_before_t1 = await get_replica_cache_snapshots()

    t1_tasks = []
    t1_conv_ids = []
    start_time = time.perf_counter()

    for i, arrival in enumerate(arrival_times):
        now = time.perf_counter() - start_time
        if arrival > now:
            await asyncio.sleep(arrival - now)

        conv_id = f"conv_{i}_{int(time.time())}_{np.random.randint(10000)}"
        t1_conv_ids.append(conv_id)

        prefix = f"CONV_{conv_id[:8]}_T1_"
        prompt = generate_prompt(workload.turn1.input_tokens, prefix)

        task = asyncio.create_task(
            send_request(
                session, prompt, workload.turn1.output_tokens,
                conv_id, workload.name, workload.turn1.input_tokens, qps, 1
            )
        )
        t1_tasks.append((task, prompt))

    t1_results = await asyncio.gather(*[t[0] for t in t1_tasks], return_exceptions=True)

    for i, (conv_id, result) in enumerate(zip(t1_conv_ids, t1_results)):
        if isinstance(result, tuple) and len(result) == 2:
            metrics, response = result
            all_metrics.append(metrics)
            raw_metrics.append(asdict(metrics))
            if metrics.success:
                histories[conv_id] = t1_tasks[i][1] + response

    # Get cache snapshot after Turn 1
    cache_after_t1 = await get_replica_cache_snapshots()
    t1_cache_delta = compute_replica_cache_delta(cache_before_t1, cache_after_t1)

    # ========== Phase 2: Turn 2 ==========
    # Get cache snapshot before Turn 2
    cache_before_t2 = await get_replica_cache_snapshots()
    t2_tasks = []
    t2_conv_ids = []

    for conv_id, history in histories.items():
        prefix = f"T2_followup_"
        new_input = generate_prompt(workload.turn2.input_tokens, prefix)
        prompt = history + "\n" + new_input

        task = asyncio.create_task(
            send_request(
                session, prompt, workload.turn2.output_tokens,
                conv_id, workload.name, workload.turn2.input_tokens, qps, 2
            )
        )
        t2_tasks.append((task, prompt))
        t2_conv_ids.append(conv_id)

    if t2_tasks:
        t2_results = await asyncio.gather(*[t[0] for t in t2_tasks], return_exceptions=True)

        for i, (conv_id, result) in enumerate(zip(t2_conv_ids, t2_results)):
            if isinstance(result, tuple) and len(result) == 2:
                metrics, response = result
                all_metrics.append(metrics)
                raw_metrics.append(asdict(metrics))
                if metrics.success:
                    histories[conv_id] = t2_tasks[i][1] + response

    # Get cache snapshot after Turn 2
    cache_after_t2 = await get_replica_cache_snapshots()
    t2_cache_delta = compute_replica_cache_delta(cache_before_t2, cache_after_t2)

    # Phase 3-5: Additional turns
    for turn_num in range(3, workload.num_turns + 1):
        turn_tasks = []
        turn_conv_ids = []

        for conv_id, history in histories.items():
            prefix = f"T{turn_num}_followup_"
            new_input = generate_prompt(workload.turn2.input_tokens, prefix)
            prompt = history + "\n" + new_input

            task = asyncio.create_task(
                send_request(
                    session, prompt, workload.turn2.output_tokens,
                    conv_id, workload.name, workload.turn2.input_tokens, qps, turn_num
                )
            )
            turn_tasks.append((task, prompt))
            turn_conv_ids.append(conv_id)

        if turn_tasks:
            turn_results = await asyncio.gather(*[t[0] for t in turn_tasks], return_exceptions=True)

            for i, (conv_id, result) in enumerate(zip(turn_conv_ids, turn_results)):
                if isinstance(result, tuple) and len(result) == 2:
                    metrics, response = result
                    all_metrics.append(metrics)
                    raw_metrics.append(asdict(metrics))
                    if metrics.success:
                        histories[conv_id] = turn_tasks[i][1] + response

    elapsed = time.perf_counter() - start_time

    # Total cache stats (for backward compatibility)
    total_cache_stats = compute_replica_cache_delta(cache_before_t1, cache_after_t2)

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
    # In Replica mode, T2 goes to same replica as T1, so prefix cache should show hits
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

    print(f"    Completed: {successful}/{total_requests} OK, QPS={actual_qps:.2f}")
    print(f"    T1 Cache: local_hits={t1_cache_delta['local_cache_hits']}, local_rate={t1_cache_delta['local_hit_rate_pct']:.1f}%")
    print(f"    T2 Cache: local_hits={t2_cache_delta['local_cache_hits']}, local_rate={t2_cache_delta['local_hit_rate_pct']:.1f}%")

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

    return result


async def run_workload(
    session: aiohttp.ClientSession,
    workload: WorkloadConfig,
    run_id: int,
    duration: int,
) -> list[BenchmarkResult]:
    """Run all QPS levels for a workload with adaptive stopping."""
    results = []

    context_size = workload.name.split("_")[0]
    qps_levels = QPS_BY_CONTEXT.get(context_size, [0.5, 1, 2, 4])

    print(f"\nWorkload: {workload.name} ({workload.description})")
    print(f"  T1: {workload.turn1.input_tokens}→{workload.turn1.output_tokens}")
    print(f"  T2: {workload.turn2.input_tokens}→{workload.turn2.output_tokens}")
    print(f"  Turns: {workload.num_turns}")
    print(f"  QPS levels: {qps_levels}")

    consecutive_failures = 0
    for qps in qps_levels:
        # Check server health before each QPS test
        healthy, msg = await check_replica_servers_health()
        if not healthy:
            print(f"    Server unhealthy: {msg}")
            print(f"    Waiting for recovery...")
            if not await wait_for_replica_recovery(max_wait_s=60):
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

        await asyncio.sleep(2)

    return results


async def main():
    parser = argparse.ArgumentParser(description="Replica Mode Benchmark")
    parser.add_argument("--run-id", type=int, default=1, help="Run ID (1, 2, or 3)")
    parser.add_argument("--workloads", type=str, default="",
                        help="Comma-separated workload names. Empty=all core workloads")
    parser.add_argument("--duration", type=int, default=10, help="Duration per QPS level")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Replica Mode Benchmark (Run {args.run_id})")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check proxy
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status != 200:
                    print("ERROR: Replica proxy not available at port 10002")
                    print("Please start replica servers with: scripts/server/start_replication_servers_4gpu.sh")
                    return
    except Exception as e:
        print(f"ERROR: Cannot connect to replica proxy: {e}")
        print("Please start replica servers with: scripts/server/start_replication_servers_4gpu.sh")
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
        workloads = [create_workload(ctx, t2, turns) for ctx, t2, turns in CORE_WORKLOADS]

    print(f"Workloads to test: {[w.name for w in workloads]}")

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "replica"

    async with aiohttp.ClientSession() as session:
        await warmup(session)

        all_results = []
        for i, workload in enumerate(workloads):
            # Health check between workloads
            if i > 0:
                print(f"\n  Checking server health before next workload...")
                healthy, msg = await check_replica_servers_health()
                if not healthy:
                    print(f"    Server unhealthy: {msg}")
                    if not await wait_for_replica_recovery(max_wait_s=120):
                        print(f"    Server did not recover, stopping benchmark")
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
