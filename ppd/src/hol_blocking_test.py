#!/usr/bin/env python3
"""
Head-of-Line (HOL) Blocking Test for PD vs PPD

================================================================================
TERMINOLOGY
================================================================================

Victim:
  - Requests already in DECODE phase, generating output tokens
  - Represents: Users waiting for their responses to complete
  - Example: User asked "write a story", system is outputting token by token

Intruder:
  - NEW requests arriving with LARGE prompts needing PREFILL
  - Represents: New users submitting requests while others are being served
  - Example: User suddenly sends a 4096-token RAG document for summarization

HOL Blocking:
  - In PPD mode: D is decoding for victims → intruder arrives → D must STOP
    victim decode to prefill intruder → victims are BLOCKED
  - In PD mode: D is decoding for victims → intruder arrives → P handles
    intruder prefill, D continues → victims UNAFFECTED

================================================================================
TEST SCENARIOS
================================================================================

Each scenario represents a different load pattern:

1. LIGHT LOAD: Few intruders, small prompts
   - PPD should handle well (minimal HOL blocking)
   - Tests: PPD's efficiency under normal conditions

2. MEDIUM LOAD: Moderate intruders, medium prompts
   - Transition point between PPD and PD
   - Tests: Where the trade-off begins

3. HEAVY LOAD: Many intruders, large prompts
   - PD should clearly win (significant HOL blocking in PPD)
   - Tests: PD's isolation advantage

4. EXTREME LOAD: Maximum stress
   - PD's advantage should be most pronounced
   - Tests: System behavior under peak load

================================================================================
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

PROXY_URL = "http://localhost:10001"
MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Test scenarios: (name, description, victims, intruders, victim_tokens, intruder_tokens, expected_winner)
TEST_SCENARIOS = [
    {
        "name": "01_Light_Load",
        "description": "Few small intruders - PPD should handle well",
        "victims": 4,
        "intruders": 1,
        "victim_tokens": 300,
        "intruder_tokens": 1024,
        "intruder_delay_ms": 400,
        "expected": "PPD",  # Low interference, PPD's speed advantage wins
    },
    {
        "name": "02_Light_Medium",
        "description": "Few medium intruders - PPD likely still OK",
        "victims": 4,
        "intruders": 2,
        "victim_tokens": 300,
        "intruder_tokens": 2048,
        "intruder_delay_ms": 400,
        "expected": "PPD",
    },
    {
        "name": "03_Medium_Load",
        "description": "Moderate intruders - transition point",
        "victims": 6,
        "intruders": 3,
        "victim_tokens": 400,
        "intruder_tokens": 4096,
        "intruder_delay_ms": 500,
        "expected": "Close",
    },
    {
        "name": "04_Heavy_Load",
        "description": "Many large intruders - PD isolation advantage",
        "victims": 6,
        "intruders": 5,
        "victim_tokens": 400,
        "intruder_tokens": 8192,
        "intruder_delay_ms": 300,
        "expected": "PD",
    },
    {
        "name": "05_Extreme_Load",
        "description": "Maximum stress - PD should clearly win",
        "victims": 8,
        "intruders": 6,
        "victim_tokens": 500,
        "intruder_tokens": 10000,
        "intruder_delay_ms": 200,
        "expected": "PD",
    },
]


def generate_prompt(num_tokens: int, unique_id: str = "") -> str:
    """Generate a prompt with approximately the specified number of tokens."""
    base = "The quick brown fox jumps over the lazy dog repeatedly. "
    target_chars = int(num_tokens * 4)
    repeated = (base * ((target_chars // len(base)) + 1))[:target_chars]
    if unique_id:
        return f"[REQ:{unique_id}] Analyze: {repeated}"
    return repeated


@dataclass
class TurnResult:
    """Result of a single turn."""
    conversation_id: str
    turn: int
    role: str
    total_time_ms: float
    ttft_ms: float
    tokens_generated: int
    prompt_tokens: int
    throughput: float
    success: bool
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Result of a single scenario."""
    scenario_name: str
    description: str
    expected_winner: str

    # Config
    num_victims: int
    num_intruders: int
    victim_output_tokens: int
    intruder_prompt_tokens: int

    # PD results
    pd_baseline_ms: float = 0.0
    pd_interference_ms: float = 0.0
    pd_slowdown_pct: float = 0.0
    pd_intruder_ttft_ms: float = 0.0

    # PPD results
    ppd_baseline_ms: float = 0.0
    ppd_interference_ms: float = 0.0
    ppd_slowdown_pct: float = 0.0
    ppd_intruder_ttft_ms: float = 0.0

    # Winner
    actual_winner: str = ""


async def set_mode(mode: str) -> bool:
    """Set the proxy routing mode."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{PROXY_URL}/mode/{mode}") as resp:
                return resp.status == 200
        except Exception:
            return False


async def clear_state():
    """Clear proxy state."""
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(f"{PROXY_URL}/conversations/clear")
            await session.post(f"{PROXY_URL}/metrics/clear")
        except Exception:
            pass


async def do_turn(
    session: aiohttp.ClientSession,
    conversation_id: str,
    conversation_history: str,
    new_prompt: str,
    max_tokens: int,
    turn: int,
    role: str,
) -> tuple[TurnResult, str]:
    """Execute a single turn and return result + updated history."""
    if conversation_history:
        full_prompt = f"{conversation_history}\nUser: {new_prompt}\nAssistant:"
    else:
        full_prompt = f"User: {new_prompt}\nAssistant:"

    start_time = time.perf_counter()
    ttft_ms = 0.0
    first_token_received = False
    response_text = ""
    tokens = 0
    prompt_tokens = 0

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                end_time = time.perf_counter()
                return TurnResult(
                    conversation_id=conversation_id, turn=turn, role=role,
                    total_time_ms=(end_time - start_time) * 1000,
                    ttft_ms=0, tokens_generated=0, prompt_tokens=0,
                    throughput=0, success=False, error=f"HTTP {resp.status}",
                ), conversation_history

            async for line in resp.content:
                if not line:
                    continue
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    if not first_token_received and chunk.get("choices"):
                        if chunk["choices"][0].get("text"):
                            ttft_ms = (time.perf_counter() - start_time) * 1000
                            first_token_received = True
                    if chunk.get("choices") and chunk["choices"][0].get("text"):
                        response_text += chunk["choices"][0]["text"]
                        tokens += 1
                    if chunk.get("usage"):
                        tokens = chunk["usage"].get("completion_tokens", tokens)
                        prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        throughput = (tokens / total_time_ms * 1000) if total_time_ms > 0 else 0
        new_history = f"{full_prompt}{response_text}"

        return TurnResult(
            conversation_id=conversation_id, turn=turn, role=role,
            total_time_ms=total_time_ms, ttft_ms=ttft_ms,
            tokens_generated=tokens, prompt_tokens=prompt_tokens,
            throughput=throughput, success=True,
        ), new_history

    except Exception as e:
        end_time = time.perf_counter()
        return TurnResult(
            conversation_id=conversation_id, turn=turn, role=role,
            total_time_ms=(end_time - start_time) * 1000,
            ttft_ms=0, tokens_generated=0, prompt_tokens=0,
            throughput=0, success=False, error=str(e),
        ), conversation_history


async def run_baseline(
    mode: str, num_victims: int, victim_tokens: int, run_id: str
) -> float:
    """Run baseline: victims only, no intruders. Returns mean victim time."""
    await set_mode(mode)
    await clear_state()
    await asyncio.sleep(0.3)

    async with aiohttp.ClientSession() as session:
        # Turn 1: establish conversations
        histories = {}
        turn1_tasks = []
        for i in range(num_victims):
            conv_id = f"v{i}_{run_id}"
            prompt = f"Hello, topic {i}. Brief reply."
            turn1_tasks.append(do_turn(session, conv_id, "", prompt, 20, 1, "victim"))

        results = await asyncio.gather(*turn1_tasks)
        for r, h in results:
            histories[r.conversation_id] = h

        await asyncio.sleep(0.2)

        # Turn 2: measure baseline
        turn2_tasks = []
        for i in range(num_victims):
            conv_id = f"v{i}_{run_id}"
            prompt = f"Write a detailed story about adventure {i}. Many characters and plot twists."
            turn2_tasks.append(do_turn(session, conv_id, histories[conv_id], prompt, victim_tokens, 2, "victim"))

        results = await asyncio.gather(*turn2_tasks)
        times = [r.total_time_ms for r, _ in results if r.success]

        return statistics.mean(times) if times else 0


async def run_interference(
    mode: str, num_victims: int, num_intruders: int,
    victim_tokens: int, intruder_tokens: int, intruder_delay_ms: float, run_id: str
) -> tuple[float, float]:
    """Run interference test. Returns (victim_mean_ms, intruder_mean_ms)."""
    await set_mode(mode)
    await clear_state()
    await asyncio.sleep(0.3)

    async with aiohttp.ClientSession() as session:
        # Turn 1: establish ALL conversations
        histories = {}
        turn1_tasks = []

        for i in range(num_victims):
            conv_id = f"v{i}_{run_id}"
            turn1_tasks.append(do_turn(session, conv_id, "", f"Hello victim {i}", 20, 1, "victim"))

        for i in range(num_intruders):
            conv_id = f"i{i}_{run_id}"
            turn1_tasks.append(do_turn(session, conv_id, "", f"Hello intruder {i}", 20, 1, "intruder"))

        results = await asyncio.gather(*turn1_tasks)
        for r, h in results:
            histories[r.conversation_id] = h

        await asyncio.sleep(0.2)

        # Turn 2: victims first, then intruders after delay
        victim_tasks = []
        for i in range(num_victims):
            conv_id = f"v{i}_{run_id}"
            prompt = f"Write a detailed story about adventure {i}. Many characters and plot twists."
            victim_tasks.append(do_turn(session, conv_id, histories[conv_id], prompt, victim_tokens, 2, "victim"))

        intruder_tasks = []
        for i in range(num_intruders):
            conv_id = f"i{i}_{run_id}"
            large_prompt = generate_prompt(intruder_tokens, conv_id)
            intruder_tasks.append(do_turn(session, conv_id, histories[conv_id], large_prompt, 50, 2, "intruder"))

        # Launch victims
        victim_futures = [asyncio.create_task(t) for t in victim_tasks]

        # Wait, then launch intruders
        await asyncio.sleep(intruder_delay_ms / 1000)
        intruder_futures = [asyncio.create_task(t) for t in intruder_tasks]

        # Wait for all
        all_results = await asyncio.gather(*victim_futures, *intruder_futures)

        victim_times = []
        intruder_times = []
        for r, _ in all_results:
            if r.success:
                if r.role == "victim":
                    victim_times.append(r.total_time_ms)
                else:
                    intruder_times.append(r.total_time_ms)

        victim_mean = statistics.mean(victim_times) if victim_times else 0
        intruder_mean = statistics.mean(intruder_times) if intruder_times else 0

        return victim_mean, intruder_mean


async def run_scenario(scenario: dict, num_rounds: int = 2) -> ScenarioResult:
    """Run a single scenario with multiple rounds for reliability."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Victims: {scenario['victims']} x {scenario['victim_tokens']} tokens")
    print(f"  Intruders: {scenario['intruders']} x {scenario['intruder_tokens']} tokens")
    print(f"  Expected winner: {scenario['expected']}")
    print(f"{'='*70}")

    result = ScenarioResult(
        scenario_name=scenario["name"],
        description=scenario["description"],
        expected_winner=scenario["expected"],
        num_victims=scenario["victims"],
        num_intruders=scenario["intruders"],
        victim_output_tokens=scenario["victim_tokens"],
        intruder_prompt_tokens=scenario["intruder_tokens"],
    )

    pd_baselines = []
    pd_interferences = []
    pd_intruder_ttfts = []
    ppd_baselines = []
    ppd_interferences = []
    ppd_intruder_ttfts = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n  Round {round_num}/{num_rounds}:")

        for mode in ["pd", "ppd"]:
            run_id = f"{scenario['name']}_{mode}_r{round_num}_{int(time.time())}"

            # Baseline
            baseline = await run_baseline(
                mode, scenario["victims"], scenario["victim_tokens"], run_id + "_base"
            )

            await asyncio.sleep(0.5)

            # Interference
            victim_ms, intruder_ms = await run_interference(
                mode, scenario["victims"], scenario["intruders"],
                scenario["victim_tokens"], scenario["intruder_tokens"],
                scenario["intruder_delay_ms"], run_id + "_int"
            )

            slowdown = ((victim_ms - baseline) / baseline * 100) if baseline > 0 else 0

            if mode == "pd":
                pd_baselines.append(baseline)
                pd_interferences.append(victim_ms)
                pd_intruder_ttfts.append(intruder_ms)
                print(f"    PD:  baseline={baseline:.0f}ms, interference={victim_ms:.0f}ms, slowdown={slowdown:.1f}%")
            else:
                ppd_baselines.append(baseline)
                ppd_interferences.append(victim_ms)
                ppd_intruder_ttfts.append(intruder_ms)
                print(f"    PPD: baseline={baseline:.0f}ms, interference={victim_ms:.0f}ms, slowdown={slowdown:.1f}%")

            await asyncio.sleep(0.5)

    # Aggregate results (use mean across rounds)
    result.pd_baseline_ms = statistics.mean(pd_baselines)
    result.pd_interference_ms = statistics.mean(pd_interferences)
    result.pd_slowdown_pct = ((result.pd_interference_ms - result.pd_baseline_ms) / result.pd_baseline_ms * 100) if result.pd_baseline_ms > 0 else 0
    result.pd_intruder_ttft_ms = statistics.mean(pd_intruder_ttfts)

    result.ppd_baseline_ms = statistics.mean(ppd_baselines)
    result.ppd_interference_ms = statistics.mean(ppd_interferences)
    result.ppd_slowdown_pct = ((result.ppd_interference_ms - result.ppd_baseline_ms) / result.ppd_baseline_ms * 100) if result.ppd_baseline_ms > 0 else 0
    result.ppd_intruder_ttft_ms = statistics.mean(ppd_intruder_ttfts)

    # Determine winner based on slowdown
    if abs(result.pd_slowdown_pct - result.ppd_slowdown_pct) < 5:
        result.actual_winner = "Close"
    elif result.pd_slowdown_pct < result.ppd_slowdown_pct:
        result.actual_winner = "PD"
    else:
        result.actual_winner = "PPD"

    return result


def print_summary(results: list[ScenarioResult]):
    """Print summary table of all scenarios."""
    print(f"\n{'='*100}")
    print("HOL BLOCKING TEST SUMMARY")
    print(f"{'='*100}")
    print("""
TERMINOLOGY:
  Victim   = Ongoing requests in decode phase (users waiting for responses)
  Intruder = New requests with large prompts (new users with big inputs)
  Slowdown = How much victims are delayed due to intruders (lower is better)
""")
    print(f"{'='*100}\n")

    # Header
    print(f"{'Scenario':<25} {'Victims':>8} {'Intruders':>10} {'PD Slow%':>10} {'PPD Slow%':>10} {'Winner':>10} {'Expected':>10}")
    print("-" * 100)

    for r in results:
        config = f"{r.num_victims}x{r.victim_output_tokens}"
        intruder_config = f"{r.num_intruders}x{r.intruder_prompt_tokens}"
        match = "✓" if r.actual_winner == r.expected_winner or r.expected_winner == "Close" else "✗"

        print(f"{r.scenario_name:<25} {config:>8} {intruder_config:>10} {r.pd_slowdown_pct:>9.1f}% {r.ppd_slowdown_pct:>9.1f}% {r.actual_winner:>10} {r.expected_winner:>8} {match}")

    print("-" * 100)

    # Analysis
    print(f"\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}")

    ppd_wins = [r for r in results if r.actual_winner == "PPD"]
    pd_wins = [r for r in results if r.actual_winner == "PD"]
    close = [r for r in results if r.actual_winner == "Close"]

    print(f"""
RESULTS:
  PPD wins: {len(ppd_wins)} scenarios (light load, PPD's speed advantage outweighs HOL blocking)
  PD wins:  {len(pd_wins)} scenarios (heavy load, PD's isolation prevents HOL blocking)
  Close:    {len(close)} scenarios (trade-off point)

KEY INSIGHT:
  - Under LIGHT LOAD: PPD is better (minimal HOL blocking, faster D-direct)
  - Under HEAVY LOAD: PD is better (P/D isolation prevents victim slowdown)

RECOMMENDATION:
  Dynamic mode switching based on system load:
  - Low load → Use PPD for speed
  - High load → Switch to PD for isolation
""")


async def main():
    parser = argparse.ArgumentParser(description="HOL Blocking Test Suite")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds per scenario")
    parser.add_argument("--scenario", type=str, help="Run specific scenario by name")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for s in TEST_SCENARIOS:
            print(f"  {s['name']}: {s['description']}")
        return

    # Check connection
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status == 200:
                    status = await resp.json()
                    print(f"Connected: {status}")
    except Exception as e:
        print(f"Cannot connect to proxy at {PROXY_URL}: {e}")
        return

    # Select scenarios
    if args.scenario:
        scenarios = [s for s in TEST_SCENARIOS if s["name"] == args.scenario]
        if not scenarios:
            print(f"Scenario '{args.scenario}' not found")
            return
    else:
        scenarios = TEST_SCENARIOS

    # Run all scenarios
    results = []
    for scenario in scenarios:
        result = await run_scenario(scenario, args.rounds)
        results.append(result)

    # Print summary
    print_summary(results)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"hol_blocking_suite_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "num_rounds": args.rounds,
            "scenarios": [asdict(r) for r in results],
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
