#!/usr/bin/env python3
"""
Mode Switching Verification Experiments

This script verifies:
1. PD <-> PPD switching: Does cache remain valid?
2. Cache cost of switching: What's the latency penalty?

Experimental Design:
- Experiment 1: Pure PD (3 turns) - baseline
- Experiment 2: Pure PPD (3 turns) - baseline
- Experiment 3: PD -> PPD -> PD (switching)
- Experiment 4: PPD -> PD -> PPD (switching)

For each experiment, we measure:
- TTFT for each turn
- Cache hit rate
- Total E2E latency

Usage:
    python scripts/tests/verify_mode_switching.py --proxy-url http://localhost:10001
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import aiohttp


@dataclass
class TurnResult:
    turn: int
    mode: str
    ttft_ms: float
    e2e_ms: float
    output_tokens: int
    cache_info: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    name: str
    description: str
    turns: list[TurnResult] = field(default_factory=list)
    total_e2e_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    cache_hit_rate: float = 0.0


async def set_proxy_mode(proxy_url: str, mode: str) -> bool:
    """Set proxy routing mode (pd or ppd)."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{proxy_url}/mode/{mode}") as resp:
                result = await resp.json()
                print(f"  Mode set to: {result.get('mode')}")
                return result.get('mode') == mode
        except Exception as e:
            print(f"  Error setting mode: {e}")
            return False


async def get_cache_stats(proxy_url: str) -> dict:
    """Get cache affinity statistics from proxy."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{proxy_url}/cache_stats") as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}


async def reset_cache_stats(proxy_url: str):
    """Reset cache statistics."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{proxy_url}/cache_stats/reset") as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}


async def clear_conversations(proxy_url: str):
    """Clear conversation state in proxy."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{proxy_url}/conversations/clear") as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}


async def send_turn(
    proxy_url: str,
    conversation_history: list[dict],
    new_message: str,
    max_tokens: int = 64
) -> tuple[float, float, int, str]:
    """
    Send a turn in a multi-turn conversation.

    Returns: (ttft_ms, e2e_ms, output_tokens, response_text)
    """
    # Build messages
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": new_message})

    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    ttft = None
    start_time = time.perf_counter()
    output_tokens = 0
    response_text = ""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{proxy_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if not line or not line.startswith('data:'):
                    continue

                if line == 'data: [DONE]':
                    break

                try:
                    data = json.loads(line[5:].strip())
                    if 'choices' in data and data['choices']:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            if ttft is None:
                                ttft = (time.perf_counter() - start_time) * 1000
                            response_text += delta['content']
                            output_tokens += 1
                except json.JSONDecodeError:
                    continue

    e2e = (time.perf_counter() - start_time) * 1000
    ttft = ttft or e2e

    return ttft, e2e, output_tokens, response_text


async def run_multi_turn_experiment(
    proxy_url: str,
    name: str,
    description: str,
    mode_sequence: list[str],  # e.g., ["pd", "ppd", "pd"]
    context_tokens: int = 512,
    t2_input_tokens: int = 128,
    t2_output_tokens: int = 64,
) -> ExperimentResult:
    """
    Run a multi-turn experiment with specified mode switching.
    """
    result = ExperimentResult(name=name, description=description)

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Description: {description}")
    print(f"Mode sequence: {' -> '.join(mode_sequence)}")
    print(f"{'='*60}")

    # Clear state
    await clear_conversations(proxy_url)
    await reset_cache_stats(proxy_url)

    # Generate conversation content
    # Turn 1: Establish context
    turn1_content = "A" * context_tokens  # Simulate context
    turn1_prompt = f"Remember this context for our conversation: {turn1_content}. Just say 'OK, I remember.'"

    conversation_history = []
    total_start = time.perf_counter()

    for turn_idx, mode in enumerate(mode_sequence):
        turn_num = turn_idx + 1

        # Set mode for this turn
        print(f"\n[Turn {turn_num}] Setting mode: {mode}")
        await set_proxy_mode(proxy_url, mode)
        await asyncio.sleep(0.5)  # Let proxy stabilize

        # Prepare turn content
        if turn_num == 1:
            prompt = turn1_prompt
            max_tokens = 32
        else:
            # Subsequent turns reference the context
            turn_content = "B" * t2_input_tokens
            prompt = f"Based on the context you remembered, process this: {turn_content}"
            max_tokens = t2_output_tokens

        # Send turn
        print(f"[Turn {turn_num}] Sending request...")
        ttft, e2e, tokens, response = await send_turn(
            proxy_url, conversation_history, prompt, max_tokens
        )

        # Update conversation history
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": response})

        # Get cache stats
        cache_stats = await get_cache_stats(proxy_url)

        turn_result = TurnResult(
            turn=turn_num,
            mode=mode,
            ttft_ms=ttft,
            e2e_ms=e2e,
            output_tokens=tokens,
            cache_info=cache_stats
        )
        result.turns.append(turn_result)

        print(f"[Turn {turn_num}] TTFT: {ttft:.1f}ms, E2E: {e2e:.1f}ms, Tokens: {tokens}")
        print(f"[Turn {turn_num}] Cache stats: {cache_stats}")

    result.total_e2e_ms = (time.perf_counter() - total_start) * 1000
    result.avg_ttft_ms = sum(t.ttft_ms for t in result.turns) / len(result.turns)

    # Calculate cache hit rate from final stats
    final_stats = await get_cache_stats(proxy_url)
    turn2plus = final_stats.get("turn2plus_requests", 0)
    hits = final_stats.get("cache_affinity_hits", 0)
    result.cache_hit_rate = (hits / turn2plus * 100) if turn2plus > 0 else 0.0

    return result


async def main():
    parser = argparse.ArgumentParser(description="Mode Switching Verification")
    parser.add_argument("--proxy-url", default="http://localhost:10001",
                        help="Proxy server URL")
    parser.add_argument("--context-tokens", type=int, default=512,
                        help="Context size for Turn 1")
    parser.add_argument("--t2-input", type=int, default=128,
                        help="Turn 2+ input tokens")
    parser.add_argument("--t2-output", type=int, default=64,
                        help="Turn 2+ output tokens")
    args = parser.parse_args()

    print("=" * 70)
    print("MODE SWITCHING VERIFICATION EXPERIMENTS")
    print("=" * 70)
    print(f"Proxy URL: {args.proxy_url}")
    print(f"Context tokens: {args.context_tokens}")
    print(f"T2 input/output: {args.t2_input}/{args.t2_output}")

    experiments = []

    # Experiment 1: Pure PD (baseline)
    exp1 = await run_multi_turn_experiment(
        args.proxy_url,
        name="Pure_PD",
        description="All turns use PD mode (P->D with KV transfer)",
        mode_sequence=["pd", "pd", "pd"],
        context_tokens=args.context_tokens,
        t2_input_tokens=args.t2_input,
        t2_output_tokens=args.t2_output,
    )
    experiments.append(exp1)

    await asyncio.sleep(2)  # Cooldown

    # Experiment 2: Pure PPD (baseline)
    exp2 = await run_multi_turn_experiment(
        args.proxy_url,
        name="Pure_PPD",
        description="All turns use PPD mode (Turn1: P->D, Turn2+: D-direct)",
        mode_sequence=["ppd", "ppd", "ppd"],
        context_tokens=args.context_tokens,
        t2_input_tokens=args.t2_input,
        t2_output_tokens=args.t2_output,
    )
    experiments.append(exp2)

    await asyncio.sleep(2)

    # Experiment 3: PD -> PPD -> PD (switching)
    exp3 = await run_multi_turn_experiment(
        args.proxy_url,
        name="PD_PPD_PD",
        description="Switch: PD -> PPD -> PD",
        mode_sequence=["pd", "ppd", "pd"],
        context_tokens=args.context_tokens,
        t2_input_tokens=args.t2_input,
        t2_output_tokens=args.t2_output,
    )
    experiments.append(exp3)

    await asyncio.sleep(2)

    # Experiment 4: PPD -> PD -> PPD (switching)
    exp4 = await run_multi_turn_experiment(
        args.proxy_url,
        name="PPD_PD_PPD",
        description="Switch: PPD -> PD -> PPD",
        mode_sequence=["ppd", "pd", "ppd"],
        context_tokens=args.context_tokens,
        t2_input_tokens=args.t2_input,
        t2_output_tokens=args.t2_output,
    )
    experiments.append(exp4)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<20} {'Avg TTFT':<12} {'Total E2E':<12} {'Cache Hit%':<12}")
    print("-" * 70)

    for exp in experiments:
        print(f"{exp.name:<20} {exp.avg_ttft_ms:<12.1f} {exp.total_e2e_ms:<12.1f} {exp.cache_hit_rate:<12.1f}")

    print("\n" + "=" * 70)
    print("DETAILED TURN-BY-TURN ANALYSIS")
    print("=" * 70)

    for exp in experiments:
        print(f"\n{exp.name}: {exp.description}")
        print(f"{'Turn':<6} {'Mode':<8} {'TTFT(ms)':<12} {'E2E(ms)':<12} {'Tokens':<8}")
        print("-" * 50)
        for turn in exp.turns:
            print(f"{turn.turn:<6} {turn.mode:<8} {turn.ttft_ms:<12.1f} {turn.e2e_ms:<12.1f} {turn.output_tokens:<8}")

    # Analysis
    print("\n" + "=" * 70)
    print("SWITCHING COST ANALYSIS")
    print("=" * 70)

    # Compare Turn 2 TTFT between pure and switching experiments
    pure_pd_t2 = exp1.turns[1].ttft_ms
    pure_ppd_t2 = exp2.turns[1].ttft_ms
    switch_pd_ppd_t2 = exp3.turns[1].ttft_ms  # PPD turn after PD
    switch_ppd_pd_t2 = exp4.turns[1].ttft_ms  # PD turn after PPD

    print(f"\nTurn 2 TTFT Comparison:")
    print(f"  Pure PD Turn 2:        {pure_pd_t2:.1f}ms")
    print(f"  Pure PPD Turn 2:       {pure_ppd_t2:.1f}ms")
    print(f"  PD->PPD Turn 2:        {switch_pd_ppd_t2:.1f}ms (PPD after PD)")
    print(f"  PPD->PD Turn 2:        {switch_ppd_pd_t2:.1f}ms (PD after PPD)")

    # Calculate switching cost
    ppd_switch_cost = switch_pd_ppd_t2 - pure_ppd_t2
    pd_switch_cost = switch_ppd_pd_t2 - pure_pd_t2

    print(f"\nSwitching Cost Estimation:")
    print(f"  PD->PPD switch cost:   {ppd_switch_cost:+.1f}ms")
    print(f"  PPD->PD switch cost:   {pd_switch_cost:+.1f}ms")

    if abs(ppd_switch_cost) < 50 and abs(pd_switch_cost) < 50:
        print("\n✅ CONCLUSION: PD <-> PPD switching has minimal cost!")
        print("   Cache affinity is preserved across mode switches.")
    else:
        print("\n⚠️ CONCLUSION: PD <-> PPD switching has notable cost.")
        print("   Consider restricting mode switches or accepting the overhead.")

    # Save results
    results_data = {
        "experiments": [
            {
                "name": exp.name,
                "description": exp.description,
                "avg_ttft_ms": exp.avg_ttft_ms,
                "total_e2e_ms": exp.total_e2e_ms,
                "cache_hit_rate": exp.cache_hit_rate,
                "turns": [
                    {
                        "turn": t.turn,
                        "mode": t.mode,
                        "ttft_ms": t.ttft_ms,
                        "e2e_ms": t.e2e_ms,
                        "output_tokens": t.output_tokens,
                    }
                    for t in exp.turns
                ]
            }
            for exp in experiments
        ],
        "analysis": {
            "ppd_switch_cost_ms": ppd_switch_cost,
            "pd_switch_cost_ms": pd_switch_cost,
        }
    }

    with open("results/mode_switching_verification.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: results/mode_switching_verification.json")


if __name__ == "__main__":
    asyncio.run(main())
