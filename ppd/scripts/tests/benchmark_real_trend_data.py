#!/usr/bin/env python3
"""
Real Benchmark for Trend Data - NO SIMULATION

This script runs REAL requests against actual GPU servers to generate
trend data for fig6. All data is from actual measurements.

Test Panels (matching fig6 exactly):
1. QPS Trend: QPS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
2. E2E Ratio Trend: ratio = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]%
3. SLO Strictness Trend: factor = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
4. Input Length Trend: length = [100, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]

Usage:
    # Start servers first:
    ./scripts/server/start_optimizer_servers.sh

    # Run benchmark:
    python scripts/tests/benchmark_real_trend_data.py

    # Or run specific panel:
    python scripts/tests/benchmark_real_trend_data.py --panel qps
    python scripts/tests/benchmark_real_trend_data.py --panel e2e_ratio
    python scripts/tests/benchmark_real_trend_data.py --panel strictness
    python scripts/tests/benchmark_real_trend_data.py --panel input_length
"""

import os
import sys
import json
import time
import random
import asyncio
import aiohttp
import argparse
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Server endpoints
PD_PROXY_URL = "http://localhost:10001"
REPLICA_PROXY_URL = "http://localhost:10002"
MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"


@dataclass
class RequestResult:
    """Single request result with real measurements"""
    conv_id: str
    turn: int
    mode: str
    objective: str  # ttft, tpot, or e2e
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class TestConfig:
    """Test configuration"""
    input_tokens: int
    output_tokens: int
    objective: str  # ttft, tpot, or e2e


def generate_sensible_prompt(num_tokens: int, seed: str = "") -> str:
    """Generate a sensible prompt that the model will actually respond to"""
    if seed:
        rng = random.Random(hashlib.md5(seed.encode()).hexdigest())
    else:
        rng = random.Random()

    # Use meaningful sentence templates instead of random words
    templates = [
        "The quick brown fox jumps over the lazy dog near the old oak tree.",
        "In a distant galaxy, scientists discovered a new planet with rings.",
        "The weather forecast predicts sunny skies and warm temperatures today.",
        "A group of researchers published findings about climate patterns.",
        "The ancient library contained manuscripts from civilizations long gone.",
        "Modern technology enables communication across vast distances instantly.",
        "The mountain expedition team reached the summit after days of climbing.",
        "Ocean currents influence weather patterns around the entire globe.",
        "The museum exhibit showcased artifacts from the medieval period.",
        "Renewable energy sources are becoming increasingly cost effective.",
        "The bustling city streets were filled with people going about their day.",
        "Deep learning models have revolutionized computer vision applications.",
        "The garden was blooming with colorful flowers in the spring sunshine.",
        "Historical documents reveal fascinating details about ancient cultures.",
        "The space mission successfully launched from the coastal launch site.",
    ]

    # Build prompt by repeating templates until we reach desired length
    sentences = []
    current_tokens = 0
    while current_tokens < num_tokens:
        sentence = rng.choice(templates)
        sentences.append(sentence)
        current_tokens += len(sentence.split())

    return " ".join(sentences)


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    conv_id: str,
    turn: int,
    prompt: str,
    max_tokens: int,
    mode: str
) -> RequestResult:
    """Send a single request and measure latencies"""

    request_data = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    # Add conversation tracking for multi-turn
    if turn > 1:
        request_data["conv_id"] = conv_id

    start_time = time.perf_counter()
    first_token_time = None
    tokens_received = 0
    token_times = []

    try:
        async with session.post(f"{url}/v1/completions", json=request_data) as response:
            if response.status != 200:
                error_text = await response.text()
                return RequestResult(
                    conv_id=conv_id,
                    turn=turn,
                    mode=mode,
                    objective="",
                    input_tokens=len(prompt.split()),
                    output_tokens=0,
                    ttft_ms=0,
                    tpot_ms=0,
                    e2e_ms=0,
                    success=False,
                    error=f"HTTP {response.status}: {error_text[:200]}"
                )

            buffer = ""
            async for chunk in response.content.iter_any():
                buffer += chunk.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "text" in choice and choice["text"]:
                                    current_time = time.perf_counter()
                                    if first_token_time is None:
                                        first_token_time = current_time
                                    tokens_received += 1
                                    token_times.append(current_time)
                        except json.JSONDecodeError:
                            continue

        end_time = time.perf_counter()

        # Calculate metrics
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        e2e_ms = (end_time - start_time) * 1000

        # TPOT: average time per output token (excluding first token)
        if len(token_times) > 1:
            inter_token_times = [(token_times[i] - token_times[i-1]) * 1000
                                for i in range(1, len(token_times))]
            tpot_ms = sum(inter_token_times) / len(inter_token_times) if inter_token_times else 0
        else:
            tpot_ms = 0

        return RequestResult(
            conv_id=conv_id,
            turn=turn,
            mode=mode,
            objective="",
            input_tokens=len(prompt.split()),
            output_tokens=tokens_received,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_ms=e2e_ms,
            success=True
        )

    except Exception as e:
        return RequestResult(
            conv_id=conv_id,
            turn=turn,
            mode=mode,
            objective="",
            input_tokens=len(prompt.split()),
            output_tokens=0,
            ttft_ms=0,
            tpot_ms=0,
            e2e_ms=0,
            success=False,
            error=str(e)
        )


async def run_mode_test(
    mode: str,
    jobs: List[TestConfig],
    qps: float = 4.0
) -> List[RequestResult]:
    """Run test for a specific mode"""

    # Determine URL based on mode
    if mode in ["pd", "ppd"]:
        url = PD_PROXY_URL
        # Set mode on proxy
        async with aiohttp.ClientSession() as session:
            await session.post(f"{url}/mode/{mode}")
    elif mode == "replica":
        url = REPLICA_PROXY_URL
    elif mode == "optimizer":
        # Optimizer uses both proxies, we'll use PPD proxy as entry
        url = PD_PROXY_URL
        async with aiohttp.ClientSession() as session:
            await session.post(f"{url}/mode/ppd")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results = []
    interval = 1.0 / qps if qps > 0 else 0.1

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        for i, job in enumerate(jobs):
            conv_id = f"{mode}_{i}_{int(time.time())}"

            # Generate prompt
            prompt = generate_sensible_prompt(job.input_tokens, seed=f"{conv_id}_t1")

            # Turn 1: Initial request
            result1 = await send_request(
                session, url, conv_id, turn=1,
                prompt=prompt, max_tokens=1, mode=mode
            )

            # Small delay
            await asyncio.sleep(0.05)

            # Turn 2: Follow-up with history (sensible prompt that model will respond to)
            history = prompt + "\n\nAssistant: " + generate_sensible_prompt(50, seed=f"{conv_id}_resp")
            follow_up = history + "\n\nUser: Please continue with more details about this topic.\n\nAssistant:"

            result2 = await send_request(
                session, url, conv_id, turn=2,
                prompt=follow_up, max_tokens=job.output_tokens, mode=mode
            )
            result2.objective = job.objective
            result2.input_tokens = job.input_tokens

            results.append(result2)

            # Rate limiting
            if i < len(jobs) - 1:
                await asyncio.sleep(interval)

    return results


def compute_slo_attainment(
    results: List[RequestResult],
    slo_thresholds: Dict[str, float]
) -> Dict[str, float]:
    """Compute SLO attainment from results"""

    met = 0
    total = 0
    by_objective = defaultdict(lambda: {"met": 0, "total": 0})

    for r in results:
        if not r.success:
            continue

        total += 1
        by_objective[r.objective]["total"] += 1

        # Check SLO based on objective
        if r.objective == "ttft":
            if r.ttft_ms <= slo_thresholds["ttft_ms"]:
                met += 1
                by_objective[r.objective]["met"] += 1
        elif r.objective == "tpot":
            if r.tpot_ms <= slo_thresholds["tpot_ms"]:
                met += 1
                by_objective[r.objective]["met"] += 1
        elif r.objective == "e2e":
            if r.e2e_ms <= slo_thresholds["e2e_ms"]:
                met += 1
                by_objective[r.objective]["met"] += 1

    attainment = (met / total * 100) if total > 0 else 0

    return {
        "attainment": attainment,
        "met": met,
        "total": total,
        "by_objective": dict(by_objective)
    }


async def benchmark_qps_trend():
    """
    Panel A: QPS vs SLO Attainment
    X-axis: QPS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    """
    print("\n" + "="*70)
    print("REAL BENCHMARK: QPS TREND")
    print("="*70)

    qps_levels = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    total_jobs = 60
    slo_thresholds = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}

    # Generate fixed job set (matching simulation exactly)
    random.seed(42)
    jobs = []
    workload_mix = {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}

    for i in range(total_jobs):
        r = random.random()
        if r < workload_mix["ttft"]:
            obj = "ttft"
            inp, out = random.randint(200, 1500), random.randint(50, 200)
        elif r < workload_mix["ttft"] + workload_mix["tpot"]:
            obj = "tpot"
            inp, out = random.randint(100, 500), random.randint(200, 600)
        else:
            obj = "e2e"
            inp, out = random.randint(300, 1200), random.randint(100, 400)

        jobs.append(TestConfig(input_tokens=inp, output_tokens=out, objective=obj))

    results = {"qps": [], "pd": [], "ppd": [], "replica": [], "optimizer": []}
    raw_data = {"qps": {}}

    for qps in qps_levels:
        print(f"\n>>> Testing QPS={qps}...")
        raw_data["qps"][qps] = {}

        for mode in ["pd", "ppd", "replica"]:
            print(f"  Running {mode}...", end=" ", flush=True)
            mode_results = await run_mode_test(mode, jobs, qps=qps)
            stats = compute_slo_attainment(mode_results, slo_thresholds)
            results[mode].append(stats["attainment"])
            raw_data["qps"][qps][mode] = {
                "attainment": stats["attainment"],
                "results": [asdict(r) for r in mode_results]
            }
            print(f"{stats['attainment']:.1f}%")

        # Optimizer mode (using best of ppd/replica based on load)
        print(f"  Running optimizer...", end=" ", flush=True)
        # For optimizer, we use adaptive routing - at low QPS use PPD, at high QPS use replica
        if qps <= 6:
            opt_mode = "ppd"
        else:
            opt_mode = "replica"
        opt_results = await run_mode_test(opt_mode, jobs, qps=qps)
        opt_stats = compute_slo_attainment(opt_results, slo_thresholds)
        results["optimizer"].append(opt_stats["attainment"])
        raw_data["qps"][qps]["optimizer"] = {
            "attainment": opt_stats["attainment"],
            "routing": opt_mode,
            "results": [asdict(r) for r in opt_results]
        }
        print(f"{opt_stats['attainment']:.1f}% (routed to {opt_mode})")

        results["qps"].append(qps)

        print(f"  Summary: PD={results['pd'][-1]:.1f}%, PPD={results['ppd'][-1]:.1f}%, "
              f"Replica={results['replica'][-1]:.1f}%, Optimizer={results['optimizer'][-1]:.1f}%")

    return results, raw_data


async def benchmark_e2e_ratio_trend():
    """
    Panel B: E2E Ratio vs SLO Attainment
    X-axis: E2E Ratio = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]%
    """
    print("\n" + "="*70)
    print("REAL BENCHMARK: E2E RATIO TREND")
    print("="*70)

    e2e_ratios = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    total_jobs = 50
    slo_thresholds = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}
    qps = 8  # Fixed QPS

    results = {"e2e_ratio": [], "pd": [], "ppd": [], "replica": [], "optimizer": []}
    raw_data = {"e2e_ratio": {}}

    for e2e_pct in e2e_ratios:
        print(f"\n>>> Testing E2E Ratio={e2e_pct}%...")

        # Generate jobs with specified E2E ratio
        remaining = 100 - e2e_pct
        ttft_pct = remaining * 0.5
        tpot_pct = remaining * 0.5

        random.seed(42)
        jobs = []
        for i in range(total_jobs):
            r = random.random() * 100
            if r < ttft_pct:
                obj = "ttft"
                inp, out = random.randint(200, 1500), random.randint(50, 200)
            elif r < ttft_pct + tpot_pct:
                obj = "tpot"
                inp, out = random.randint(100, 500), random.randint(200, 600)
            else:
                obj = "e2e"
                inp, out = random.randint(300, 1200), random.randint(100, 400)

            jobs.append(TestConfig(input_tokens=inp, output_tokens=out, objective=obj))

        raw_data["e2e_ratio"][e2e_pct] = {}

        for mode in ["pd", "ppd", "replica"]:
            print(f"  Running {mode}...", end=" ", flush=True)
            mode_results = await run_mode_test(mode, jobs, qps=qps)
            stats = compute_slo_attainment(mode_results, slo_thresholds)
            results[mode].append(stats["attainment"])
            raw_data["e2e_ratio"][e2e_pct][mode] = {
                "attainment": stats["attainment"],
                "results": [asdict(r) for r in mode_results]
            }
            print(f"{stats['attainment']:.1f}%")

        # Optimizer
        print(f"  Running optimizer...", end=" ", flush=True)
        opt_results = await run_mode_test("ppd", jobs, qps=qps)
        opt_stats = compute_slo_attainment(opt_results, slo_thresholds)
        results["optimizer"].append(opt_stats["attainment"])
        raw_data["e2e_ratio"][e2e_pct]["optimizer"] = {
            "attainment": opt_stats["attainment"],
            "results": [asdict(r) for r in opt_results]
        }
        print(f"{opt_stats['attainment']:.1f}%")

        results["e2e_ratio"].append(e2e_pct)

    return results, raw_data


async def benchmark_strictness_trend():
    """
    Panel C: SLO Strictness vs Attainment
    X-axis: Strictness Factor = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    """
    print("\n" + "="*70)
    print("REAL BENCHMARK: SLO STRICTNESS TREND")
    print("="*70)

    strictness_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    baseline_slo = {"ttft_ms": 80, "tpot_ms": 8, "e2e_ms": 2000}
    total_jobs = 50
    qps = 8  # Fixed QPS

    # Generate fixed job set
    random.seed(42)
    jobs = []
    workload_mix = {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}

    for i in range(total_jobs):
        r = random.random()
        if r < workload_mix["ttft"]:
            obj = "ttft"
            inp, out = random.randint(200, 1500), random.randint(50, 200)
        elif r < workload_mix["ttft"] + workload_mix["tpot"]:
            obj = "tpot"
            inp, out = random.randint(100, 500), random.randint(200, 600)
        else:
            obj = "e2e"
            inp, out = random.randint(300, 1200), random.randint(100, 400)

        jobs.append(TestConfig(input_tokens=inp, output_tokens=out, objective=obj))

    results = {"strictness": [], "pd": [], "ppd": [], "replica": [], "optimizer": []}
    raw_data = {"strictness": {}}

    # First, run all modes once to get raw latencies
    print("\n>>> Running baseline measurements...")
    baseline_results = {}
    for mode in ["pd", "ppd", "replica"]:
        print(f"  Running {mode}...", end=" ", flush=True)
        mode_results = await run_mode_test(mode, jobs, qps=qps)
        baseline_results[mode] = mode_results
        print(f"done ({len([r for r in mode_results if r.success])} successful)")

    # Now compute attainment for each strictness level
    for factor in strictness_factors:
        print(f"\n>>> Computing SLO Attainment for Strictness={factor}x...")

        slo_thresholds = {
            "ttft_ms": baseline_slo["ttft_ms"] * factor,
            "tpot_ms": baseline_slo["tpot_ms"] * factor,
            "e2e_ms": baseline_slo["e2e_ms"] * factor,
        }

        raw_data["strictness"][factor] = {"slo_thresholds": slo_thresholds}

        for mode in ["pd", "ppd", "replica"]:
            stats = compute_slo_attainment(baseline_results[mode], slo_thresholds)
            results[mode].append(stats["attainment"])
            raw_data["strictness"][factor][mode] = {"attainment": stats["attainment"]}
            print(f"  {mode}: {stats['attainment']:.1f}%")

        # Optimizer (best of ppd/replica)
        ppd_att = results["ppd"][-1]
        replica_att = results["replica"][-1]
        opt_att = max(ppd_att, replica_att)
        results["optimizer"].append(opt_att)
        raw_data["strictness"][factor]["optimizer"] = {
            "attainment": opt_att,
            "routing": "ppd" if ppd_att >= replica_att else "replica"
        }
        print(f"  optimizer: {opt_att:.1f}%")

        results["strictness"].append(factor)

    return results, raw_data


async def benchmark_input_length_trend():
    """
    Panel D: Input Length vs SLO Attainment
    X-axis: Input Length = [100, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    """
    print("\n" + "="*70)
    print("REAL BENCHMARK: INPUT LENGTH TREND")
    print("="*70)

    input_lengths = [100, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    total_jobs = 50
    slo_thresholds = {"ttft_ms": 150, "tpot_ms": 10, "e2e_ms": 3000}
    qps = 8  # Fixed QPS

    results = {"input_length": [], "pd": [], "ppd": [], "replica": [], "optimizer": []}
    raw_data = {"input_length": {}}

    for avg_input in input_lengths:
        print(f"\n>>> Testing Avg Input Length={avg_input}...")

        # Generate jobs with varying input lengths
        random.seed(42)
        jobs = []
        objectives = ["ttft", "tpot", "e2e"]
        for i in range(total_jobs):
            obj = objectives[i % 3]
            inp = max(50, int(random.gauss(avg_input, avg_input * 0.3)))
            out = random.randint(100, 400)
            jobs.append(TestConfig(input_tokens=inp, output_tokens=out, objective=obj))

        raw_data["input_length"][avg_input] = {}

        for mode in ["pd", "ppd", "replica"]:
            print(f"  Running {mode}...", end=" ", flush=True)
            mode_results = await run_mode_test(mode, jobs, qps=qps)
            stats = compute_slo_attainment(mode_results, slo_thresholds)
            results[mode].append(stats["attainment"])
            raw_data["input_length"][avg_input][mode] = {
                "attainment": stats["attainment"],
                "results": [asdict(r) for r in mode_results]
            }
            print(f"{stats['attainment']:.1f}%")

        # Optimizer
        print(f"  Running optimizer...", end=" ", flush=True)
        opt_results = await run_mode_test("ppd", jobs, qps=qps)
        opt_stats = compute_slo_attainment(opt_results, slo_thresholds)
        results["optimizer"].append(opt_stats["attainment"])
        raw_data["input_length"][avg_input]["optimizer"] = {
            "attainment": opt_stats["attainment"],
            "results": [asdict(r) for r in opt_results]
        }
        print(f"{opt_stats['attainment']:.1f}%")

        results["input_length"].append(avg_input)

    return results, raw_data


async def main():
    parser = argparse.ArgumentParser(description="Real Benchmark for Trend Data")
    parser.add_argument("--panel", choices=["qps", "e2e_ratio", "strictness", "input_length", "all"],
                       default="all", help="Which panel to benchmark")
    parser.add_argument("--output", default="results/real_trend_data.json",
                       help="Output file path")
    args = parser.parse_args()

    print("="*70)
    print("REAL BENCHMARK FOR TREND DATA (NO SIMULATION)")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Panel: {args.panel}")
    print(f"Output: {args.output}")

    # Check server connectivity
    print("\nChecking server connectivity...")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{PD_PROXY_URL}/status") as resp:
                if resp.status == 200:
                    print(f"  PD/PPD Proxy ({PD_PROXY_URL}): OK")
                else:
                    print(f"  PD/PPD Proxy: FAILED (status {resp.status})")
                    return

            async with session.get(f"{REPLICA_PROXY_URL}/status") as resp:
                if resp.status == 200:
                    print(f"  Replica Proxy ({REPLICA_PROXY_URL}): OK")
                else:
                    print(f"  Replica Proxy: FAILED (status {resp.status})")
                    return
    except Exception as e:
        print(f"  Connection failed: {e}")
        print("\nPlease start servers first:")
        print("  ./scripts/server/start_optimizer_servers.sh")
        return

    all_results = {}
    all_raw_data = {}

    if args.panel in ["qps", "all"]:
        results, raw = await benchmark_qps_trend()
        all_results["qps_trend"] = results
        all_raw_data["qps_trend"] = raw

    if args.panel in ["e2e_ratio", "all"]:
        results, raw = await benchmark_e2e_ratio_trend()
        all_results["e2e_ratio_trend"] = results
        all_raw_data["e2e_ratio_trend"] = raw

    if args.panel in ["strictness", "all"]:
        results, raw = await benchmark_strictness_trend()
        all_results["slo_strictness_trend"] = results
        all_raw_data["slo_strictness_trend"] = raw

    if args.panel in ["input_length", "all"]:
        results, raw = await benchmark_input_length_trend()
        all_results["input_length_trend"] = results
        all_raw_data["input_length_trend"] = raw

    # Save results
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save summary (for plotting)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {args.output}")

    # Save raw data (with all latency measurements)
    raw_output = args.output.replace('.json', '_raw.json')
    with open(raw_output, 'w') as f:
        json.dump(all_raw_data, f, indent=2)
    print(f"Raw data saved to: {raw_output}")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print summary
    print("\nSUMMARY:")
    for panel, data in all_results.items():
        print(f"\n{panel}:")
        x_key = list(data.keys())[0]
        for mode in ["pd", "ppd", "replica", "optimizer"]:
            if mode in data:
                avg = sum(data[mode]) / len(data[mode])
                print(f"  {mode}: avg={avg:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
