#!/usr/bin/env python3
"""
Mode Switching Cost Verification

Tests:
1. PD <-> PPD switching (within same D, should preserve cache)
2. PD/PPD <-> Replica switching (different GPUs, cache miss expected)

Measures:
- TTFT for each turn
- Cache hit indicators
- Total switching cost

Server Configuration:
- PD/PPD: GPU 0 (P), GPU 1 (D) - Proxy port 10001
- Replica: GPU 2 (R0), GPU 3 (R1) - Proxy port 10002
"""

import asyncio
import aiohttp
import time
import json
from dataclasses import dataclass, field
from typing import Optional

MODEL = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"

# Server URLs
PD_PROXY = "http://localhost:10001"
REPLICA_PROXY = "http://localhost:10002"
DECODE_DIRECT = "http://localhost:8200"
REPLICA0_DIRECT = "http://localhost:8300"
REPLICA1_DIRECT = "http://localhost:8400"


@dataclass
class TurnResult:
    turn: int
    mode: str
    url: str
    ttft_ms: float
    e2e_ms: float
    tokens: int


@dataclass
class ExperimentResult:
    name: str
    description: str
    turns: list = field(default_factory=list)
    total_e2e_ms: float = 0.0


async def set_pd_mode(mode: str):
    """Set PD proxy mode (pd or ppd)"""
    async with aiohttp.ClientSession() as session:
        await session.post(f"{PD_PROXY}/mode/{mode}")
        await asyncio.sleep(0.2)


async def clear_pd_conversations():
    """Clear PD proxy conversation state"""
    async with aiohttp.ClientSession() as session:
        await session.post(f"{PD_PROXY}/conversations/clear")
        await session.post(f"{PD_PROXY}/cache_stats/reset")


async def send_request(url: str, prompt: str, max_tokens: int = 32) -> tuple:
    """Send request and measure latency"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            result = await resp.json()

    e2e = (time.perf_counter() - start) * 1000
    ttft = e2e  # For non-streaming, TTFT ≈ E2E

    text = result.get("choices", [{}])[0].get("text", "")
    tokens = len(text.split())

    return ttft, e2e, tokens, text


async def run_experiment(name: str, description: str, turn_configs: list) -> ExperimentResult:
    """
    Run a multi-turn experiment with specified mode/URL for each turn.

    turn_configs: list of (mode, url, prompt_suffix) tuples
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    result = ExperimentResult(name=name, description=description)

    # Clear state
    await clear_pd_conversations()

    # Base context (same for all turns to enable cache hit)
    BASE_CONTEXT = "A " * 256 + "Remember this context."
    conversation_history = BASE_CONTEXT

    total_start = time.perf_counter()

    for turn_idx, (mode, url, prompt_suffix) in enumerate(turn_configs):
        turn_num = turn_idx + 1

        # Set mode if using PD proxy
        if "10001" in url:
            await set_pd_mode(mode)

        # Build prompt
        if turn_num == 1:
            prompt = conversation_history + " Say OK."
        else:
            prompt = conversation_history + f" {prompt_suffix}"

        # Send request
        ttft, e2e, tokens, text = await send_request(url, prompt, max_tokens=32)

        # Update history
        conversation_history = prompt + " " + text

        turn_result = TurnResult(
            turn=turn_num,
            mode=mode,
            url=url.split(":")[-1],  # Just port
            ttft_ms=ttft,
            e2e_ms=e2e,
            tokens=tokens
        )
        result.turns.append(turn_result)

        print(f"  Turn {turn_num} [{mode}@{turn_result.url}]: TTFT={ttft:.0f}ms")

    result.total_e2e_ms = (time.perf_counter() - total_start) * 1000

    return result


async def main():
    print("=" * 70)
    print("MODE SWITCHING COST VERIFICATION")
    print("=" * 70)
    print(f"\nServer Configuration:")
    print(f"  PD/PPD: Prefill(8100) + Decode(8200), Proxy(10001)")
    print(f"  Replica: R0(8300) + R1(8400), Proxy(10002)")

    experiments = []

    # ================================================================
    # Test 1: Pure PD (baseline)
    # ================================================================
    exp = await run_experiment(
        "Pure_PD",
        "All turns use PD mode (P→D)",
        [
            ("pd", PD_PROXY, ""),
            ("pd", PD_PROXY, "Continue."),
            ("pd", PD_PROXY, "Summarize."),
        ]
    )
    experiments.append(exp)
    await asyncio.sleep(1)

    # ================================================================
    # Test 2: Pure PPD (baseline)
    # ================================================================
    exp = await run_experiment(
        "Pure_PPD",
        "All turns use PPD mode (T1:P→D, T2+:D-direct)",
        [
            ("ppd", PD_PROXY, ""),
            ("ppd", PD_PROXY, "Continue."),
            ("ppd", PD_PROXY, "Summarize."),
        ]
    )
    experiments.append(exp)
    await asyncio.sleep(1)

    # ================================================================
    # Test 3: Pure Replica (baseline)
    # ================================================================
    exp = await run_experiment(
        "Pure_Replica",
        "All turns to same Replica worker",
        [
            ("replica", REPLICA0_DIRECT, ""),
            ("replica", REPLICA0_DIRECT, "Continue."),
            ("replica", REPLICA0_DIRECT, "Summarize."),
        ]
    )
    experiments.append(exp)
    await asyncio.sleep(1)

    # ================================================================
    # Test 4: PD → PPD → PD (switching within PD group)
    # ================================================================
    exp = await run_experiment(
        "PD_PPD_PD",
        "Switch: PD → PPD → PD",
        [
            ("pd", PD_PROXY, ""),
            ("ppd", PD_PROXY, "Continue."),
            ("pd", PD_PROXY, "Summarize."),
        ]
    )
    experiments.append(exp)
    await asyncio.sleep(1)

    # ================================================================
    # Test 5: PPD → PD → PPD (switching within PD group)
    # ================================================================
    exp = await run_experiment(
        "PPD_PD_PPD",
        "Switch: PPD → PD → PPD",
        [
            ("ppd", PD_PROXY, ""),
            ("pd", PD_PROXY, "Continue."),
            ("ppd", PD_PROXY, "Summarize."),
        ]
    )
    experiments.append(exp)
    await asyncio.sleep(1)

    # ================================================================
    # Test 6: PD → Replica (cross-group switch, cache miss expected)
    # ================================================================
    exp = await run_experiment(
        "PD_to_Replica",
        "Switch: PD → Replica (cross-GPU, cache miss)",
        [
            ("pd", PD_PROXY, ""),
            ("replica", REPLICA0_DIRECT, "Continue."),
            ("replica", REPLICA0_DIRECT, "Summarize."),
        ]
    )
    experiments.append(exp)
    await asyncio.sleep(1)

    # ================================================================
    # Test 7: Replica → PD (cross-group switch, cache miss expected)
    # ================================================================
    exp = await run_experiment(
        "Replica_to_PD",
        "Switch: Replica → PD (cross-GPU, cache miss)",
        [
            ("replica", REPLICA0_DIRECT, ""),
            ("pd", PD_PROXY, "Continue."),
            ("pd", PD_PROXY, "Summarize."),
        ]
    )
    experiments.append(exp)
    await asyncio.sleep(1)

    # ================================================================
    # Test 8: PPD → Replica → PPD (round trip cross-group)
    # ================================================================
    exp = await run_experiment(
        "PPD_Replica_PPD",
        "Switch: PPD → Replica → PPD (round trip)",
        [
            ("ppd", PD_PROXY, ""),
            ("replica", REPLICA0_DIRECT, "Continue."),
            ("ppd", PD_PROXY, "Summarize."),
        ]
    )
    experiments.append(exp)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Experiment':<20} {'T1(ms)':<10} {'T2(ms)':<10} {'T3(ms)':<10} {'Avg(ms)':<10}")
    print("-" * 70)

    for exp in experiments:
        t1 = exp.turns[0].ttft_ms
        t2 = exp.turns[1].ttft_ms
        t3 = exp.turns[2].ttft_ms
        avg = (t1 + t2 + t3) / 3
        print(f"{exp.name:<20} {t1:<10.0f} {t2:<10.0f} {t3:<10.0f} {avg:<10.0f}")

    # ================================================================
    # Switching Cost Analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("SWITCHING COST ANALYSIS")
    print("=" * 70)

    # Get baseline values
    pure_pd = next(e for e in experiments if e.name == "Pure_PD")
    pure_ppd = next(e for e in experiments if e.name == "Pure_PPD")
    pure_replica = next(e for e in experiments if e.name == "Pure_Replica")
    pd_ppd_pd = next(e for e in experiments if e.name == "PD_PPD_PD")
    ppd_pd_ppd = next(e for e in experiments if e.name == "PPD_PD_PPD")
    pd_to_replica = next(e for e in experiments if e.name == "PD_to_Replica")
    replica_to_pd = next(e for e in experiments if e.name == "Replica_to_PD")
    ppd_replica_ppd = next(e for e in experiments if e.name == "PPD_Replica_PPD")

    print("\n1. PD <-> PPD Switching (same D GPU, cache should persist):")
    # Compare T2 of PD→PPD switch vs pure PPD T2
    pd_to_ppd_cost = pd_ppd_pd.turns[1].ttft_ms - pure_ppd.turns[1].ttft_ms
    ppd_to_pd_cost = ppd_pd_ppd.turns[1].ttft_ms - pure_pd.turns[1].ttft_ms
    print(f"   PD→PPD switch cost (T2): {pd_to_ppd_cost:+.0f}ms")
    print(f"   PPD→PD switch cost (T2): {ppd_to_pd_cost:+.0f}ms")

    print("\n2. PD/PPD <-> Replica Switching (cross-GPU, cache miss):")
    # Compare T2 of PD→Replica vs pure Replica T2
    pd_to_replica_cost = pd_to_replica.turns[1].ttft_ms - pure_replica.turns[1].ttft_ms
    replica_to_pd_cost = replica_to_pd.turns[1].ttft_ms - pure_pd.turns[1].ttft_ms
    print(f"   PD→Replica switch cost (T2): {pd_to_replica_cost:+.0f}ms")
    print(f"   Replica→PD switch cost (T2): {replica_to_pd_cost:+.0f}ms")

    print("\n3. Round-trip switching:")
    # PPD→Replica→PPD
    ppd_replica_ppd_t3_cost = ppd_replica_ppd.turns[2].ttft_ms - pure_ppd.turns[2].ttft_ms
    print(f"   PPD→Replica→PPD T3 cost: {ppd_replica_ppd_t3_cost:+.0f}ms")

    # ================================================================
    # Conclusion
    # ================================================================
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    intra_group_cost = max(abs(pd_to_ppd_cost), abs(ppd_to_pd_cost))
    cross_group_cost = max(abs(pd_to_replica_cost), abs(replica_to_pd_cost))

    print(f"\n  PD<->PPD switching cost:      ~{intra_group_cost:.0f}ms (within same D)")
    print(f"  PD/PPD<->Replica switching:   ~{cross_group_cost:.0f}ms (cross GPU)")

    if cross_group_cost > 100:
        print("\n  ⚠️  Cross-group switching has significant cost!")
        print("  RECOMMENDATION: Lock mode group at Turn 1")
        print("    - If T1 uses PD/PPD → stay in PD/PPD group")
        print("    - If T1 uses Replica → stay in Replica")
    else:
        print("\n  ✅ Switching costs are acceptable")
        print("  Free to switch between any modes")

    # Save results
    results_data = {
        "experiments": [
            {
                "name": e.name,
                "description": e.description,
                "turns": [
                    {"turn": t.turn, "mode": t.mode, "url": t.url, "ttft_ms": t.ttft_ms}
                    for t in e.turns
                ],
                "total_e2e_ms": e.total_e2e_ms
            }
            for e in experiments
        ],
        "analysis": {
            "pd_to_ppd_cost_ms": pd_to_ppd_cost,
            "ppd_to_pd_cost_ms": ppd_to_pd_cost,
            "pd_to_replica_cost_ms": pd_to_replica_cost,
            "replica_to_pd_cost_ms": replica_to_pd_cost,
            "intra_group_max_cost_ms": intra_group_cost,
            "cross_group_max_cost_ms": cross_group_cost,
        }
    }

    with open("results/mode_switching_cost.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: results/mode_switching_cost.json")


if __name__ == "__main__":
    asyncio.run(main())
