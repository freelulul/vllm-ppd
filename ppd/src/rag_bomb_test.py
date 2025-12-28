#!/usr/bin/env python3
"""
RAG Bomb & Batch Congestion Test (Turn 2 Version)

================================================================================
TERMINOLOGY
================================================================================

Victim:
  - Requests in Turn 2+ doing D-direct decode (in PPD mode)
  - Represents: Users in ongoing conversations waiting for responses

Intruder:
  - New Turn 2 requests with LARGE prompts arriving during victim decode
  - Represents: Users sending large RAG/context documents mid-conversation

HOL Blocking in RAG Bomb:
  - Victim is happily decoding on D (Turn 2, D-direct)
  - Intruder arrives with 4K token RAG document (Turn 2, needs D-direct prefill)
  - In PPD: D must STOP victim decode to prefill intruder → victim stutters
  - In PD: P handles intruder prefill, D continues → victim unaffected

================================================================================
SCENARIOS
================================================================================

Scenario 1: "The RAG Bomb" (Single Victim Interference)
  - 1 Victim in Turn 2, generating 500 tokens
  - 1 Intruder in Turn 2 with 4096 token prompt arrives mid-generation
  - Measure: Victim ITL spike, Intruder TTFT

Scenario 2: "The Batch Congestion" (Multi-Victim Interference)
  - 4 Victims in Turn 2, generating concurrently
  - 1 Intruder in Turn 2 with 2048 token prompt arrives
  - Measure: Average ITL spike across all victims

Usage:
    python src/rag_bomb_test.py --scenario rag_bomb
    python src/rag_bomb_test.py --scenario batch_congestion
    python src/rag_bomb_test.py  # Run both
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

PROXY_URL = "http://localhost:10001"
MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def generate_large_prompt(num_tokens: int, unique_id: str = "") -> str:
    """Generate a prompt with approximately the specified number of tokens."""
    base = "The quick brown fox jumps over the lazy dog repeatedly for testing purposes. "
    target_chars = int(num_tokens * 4)
    repeated = (base * ((target_chars // len(base)) + 1))[:target_chars]
    if unique_id:
        return f"[REQ:{unique_id}] Summarize this document: {repeated}"
    return repeated


@dataclass
class StreamingMetrics:
    """Metrics from a streaming request."""
    request_id: str
    role: str
    total_time_ms: float
    ttft_ms: float
    tokens_generated: int
    itl_values: list[float] = field(default_factory=list)
    itl_timestamps: list[float] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

    @property
    def mean_itl(self) -> float:
        return statistics.mean(self.itl_values) if self.itl_values else 0

    @property
    def max_itl(self) -> float:
        return max(self.itl_values) if self.itl_values else 0


async def set_mode(mode: str) -> bool:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{PROXY_URL}/mode/{mode}") as resp:
                return resp.status == 200
        except:
            return False


async def clear_state():
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(f"{PROXY_URL}/conversations/clear")
            await session.post(f"{PROXY_URL}/metrics/clear")
        except:
            pass


async def do_turn1(session: aiohttp.ClientSession, conv_id: str, prompt: str) -> str:
    """Do Turn 1 to establish conversation. Returns conversation history."""
    full_prompt = f"User: {prompt}\nAssistant:"

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": full_prompt,
                "max_tokens": 20,
                "temperature": 0.8,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                response_text = data["choices"][0]["text"]
                return f"{full_prompt}{response_text}"
    except:
        pass
    return full_prompt


async def streaming_turn2_with_itl(
    session: aiohttp.ClientSession,
    history: str,
    new_prompt: str,
    max_tokens: int,
    request_id: str,
    role: str,
) -> StreamingMetrics:
    """Send Turn 2 streaming request and record ITL with timestamps."""
    full_prompt = f"{history}\nUser: {new_prompt}\nAssistant:"

    start_time = time.perf_counter()
    itl_values = []
    itl_timestamps = []
    last_token_time = None
    tokens_received = 0
    ttft_ms = 0

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                return StreamingMetrics(
                    request_id=request_id, role=role,
                    total_time_ms=(time.perf_counter() - start_time) * 1000,
                    ttft_ms=0, tokens_generated=0, success=False,
                    error=f"HTTP {resp.status}",
                )

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices and choices[0].get("text"):
                        current_time = time.perf_counter()

                        if last_token_time is None:
                            ttft_ms = (current_time - start_time) * 1000
                        else:
                            itl = (current_time - last_token_time) * 1000
                            itl_values.append(itl)
                            itl_timestamps.append((current_time - start_time) * 1000)

                        last_token_time = current_time
                        tokens_received += 1
                except json.JSONDecodeError:
                    continue

            return StreamingMetrics(
                request_id=request_id, role=role,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                ttft_ms=ttft_ms, tokens_generated=tokens_received,
                itl_values=itl_values, itl_timestamps=itl_timestamps,
                success=True,
            )

    except Exception as e:
        return StreamingMetrics(
            request_id=request_id, role=role,
            total_time_ms=(time.perf_counter() - start_time) * 1000,
            ttft_ms=0, tokens_generated=0, success=False, error=str(e),
        )


async def run_rag_bomb_scenario(mode: str, runs: int = 3) -> dict:
    """
    Scenario 1: "The RAG Bomb" (Turn 2 version)

    Both victim and intruder use Turn 2 to trigger D-direct in PPD mode.
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO 1: RAG BOMB (Turn 2) | Mode: {mode.upper()}")
    print(f"{'='*70}")
    print(f"  Victim: Turn 2 decode, generating 500 tokens")
    print(f"  Intruder: Turn 2 with 4096 token prompt, arrives mid-generation")
    print(f"  In PPD: Both use D-direct → HOL blocking expected")

    await set_mode(mode)

    all_victim_max_itl = []
    all_intruder_ttft = []

    for run in range(runs):
        await clear_state()
        await asyncio.sleep(0.5)

        run_id = f"{mode}_rag_{run}_{int(time.time())}"

        async with aiohttp.ClientSession() as session:
            # Phase 1: Establish conversations (Turn 1)
            victim_history = await do_turn1(session, f"victim_{run_id}", "Hello, I want a story.")
            intruder_history = await do_turn1(session, f"intruder_{run_id}", "Hello, I have a document.")

            await asyncio.sleep(0.3)

            # Phase 2: Start victim Turn 2
            victim_prompt = "Now write a very long and detailed story about a brave knight. Include many characters, plot twists, and vivid descriptions."
            victim_task = asyncio.create_task(
                streaming_turn2_with_itl(
                    session, victim_history, victim_prompt, 500,
                    f"victim_{run_id}", "victim"
                )
            )

            # Wait for victim to be in decode phase
            await asyncio.sleep(1.0)

            # Phase 3: Launch intruder Turn 2 with large prompt
            intruder_prompt = generate_large_prompt(4096, f"rag_{run}")
            intruder_task = asyncio.create_task(
                streaming_turn2_with_itl(
                    session, intruder_history, intruder_prompt, 30,
                    f"intruder_{run_id}", "intruder"
                )
            )

            victim_result, intruder_result = await asyncio.gather(victim_task, intruder_task)

        if victim_result.success and intruder_result.success:
            all_victim_max_itl.append(victim_result.max_itl)
            all_intruder_ttft.append(intruder_result.ttft_ms)

            print(f"\n  [Run {run+1}]")
            print(f"    Victim Max ITL: {victim_result.max_itl:.1f} ms")
            print(f"    Victim tokens: {victim_result.tokens_generated}")
            print(f"    Intruder TTFT: {intruder_result.ttft_ms:.1f} ms")

    return {
        "mode": mode,
        "scenario": "rag_bomb",
        "victim_max_itl_mean": statistics.mean(all_victim_max_itl) if all_victim_max_itl else 0,
        "intruder_ttft_mean": statistics.mean(all_intruder_ttft) if all_intruder_ttft else 0,
    }


async def run_batch_congestion_scenario(mode: str, runs: int = 3) -> dict:
    """
    Scenario 2: "The Batch Congestion" (Turn 2 version)

    Multiple victims and intruder all in Turn 2.
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO 2: BATCH CONGESTION (Turn 2) | Mode: {mode.upper()}")
    print(f"{'='*70}")
    print(f"  Victims: 4 concurrent Turn 2 decodes, 300 tokens each")
    print(f"  Intruder: Turn 2 with 2048 token prompt, arrives mid-decode")

    await set_mode(mode)

    all_victim_max_itl = []
    all_intruder_ttft = []

    for run in range(runs):
        await clear_state()
        await asyncio.sleep(0.5)

        run_id = f"{mode}_batch_{run}_{int(time.time())}"

        async with aiohttp.ClientSession() as session:
            # Phase 1: Establish all conversations (Turn 1)
            victim_histories = []
            for i in range(4):
                h = await do_turn1(session, f"victim_{i}_{run_id}", f"Hello, topic {i}.")
                victim_histories.append(h)

            intruder_history = await do_turn1(session, f"intruder_{run_id}", "Hello, I have data.")

            await asyncio.sleep(0.3)

            # Phase 2: Start all victim Turn 2s
            victim_tasks = []
            for i in range(4):
                prompt = f"Write a detailed story {i} about adventures. Include characters and plot."
                task = asyncio.create_task(
                    streaming_turn2_with_itl(
                        session, victim_histories[i], prompt, 300,
                        f"victim_{i}_{run_id}", "victim"
                    )
                )
                victim_tasks.append(task)

            # Wait for victims to be in decode
            await asyncio.sleep(0.5)

            # Phase 3: Launch intruder Turn 2
            intruder_prompt = generate_large_prompt(2048, f"batch_{run}")
            intruder_task = asyncio.create_task(
                streaming_turn2_with_itl(
                    session, intruder_history, intruder_prompt, 30,
                    f"intruder_{run_id}", "intruder"
                )
            )

            results = await asyncio.gather(*victim_tasks, intruder_task)

        victim_results = [r for r in results[:-1] if r.success]
        intruder_result = results[-1]

        if victim_results and intruder_result.success:
            run_max_itl = [v.max_itl for v in victim_results]
            all_victim_max_itl.extend(run_max_itl)
            all_intruder_ttft.append(intruder_result.ttft_ms)

            print(f"\n  [Run {run+1}]")
            print(f"    Victim Max ITLs: {[f'{x:.1f}' for x in run_max_itl]} ms")
            print(f"    Intruder TTFT: {intruder_result.ttft_ms:.1f} ms")

    return {
        "mode": mode,
        "scenario": "batch_congestion",
        "victim_max_itl_mean": statistics.mean(all_victim_max_itl) if all_victim_max_itl else 0,
        "intruder_ttft_mean": statistics.mean(all_intruder_ttft) if all_intruder_ttft else 0,
    }


def print_comparison(scenario: str, pd_result: dict, ppd_result: dict):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {scenario.upper()}")
    print(f"{'='*80}")

    print(f"\n{'Metric':<30} {'PD':>15} {'PPD':>15} {'Winner':>10}")
    print("-" * 70)

    # Victim Max ITL (lower is better for isolation)
    pd_itl = pd_result['victim_max_itl_mean']
    ppd_itl = ppd_result['victim_max_itl_mean']
    winner = "PD" if pd_itl < ppd_itl else "PPD"
    print(f"{'Victim Max ITL (ms)':<30} {pd_itl:>15.1f} {ppd_itl:>15.1f} {winner:>10}")

    # Intruder TTFT (lower is better for new requests)
    pd_ttft = pd_result['intruder_ttft_mean']
    ppd_ttft = ppd_result['intruder_ttft_mean']
    winner = "PPD" if ppd_ttft < pd_ttft else "PD"
    print(f"{'Intruder TTFT (ms)':<30} {pd_ttft:>15.1f} {ppd_ttft:>15.1f} {winner:>10}")

    print("-" * 70)

    # Analysis
    if ppd_itl > pd_itl * 1.3:
        print(f"\n  HOL Blocking detected in PPD mode!")
        print(f"  PPD victim ITL: {ppd_itl:.1f}ms vs PD: {pd_itl:.1f}ms")
        print(f"  PD's P/D isolation protects ongoing decodes.")
    else:
        print(f"\n  No significant HOL blocking difference.")


async def main():
    parser = argparse.ArgumentParser(description="RAG Bomb & Batch Congestion Tests (Turn 2)")
    parser.add_argument("--scenario", choices=["rag_bomb", "batch_congestion", "both"],
                        default="both", help="Which scenario to run")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    args = parser.parse_args()

    # Check connection
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status == 200:
                    print(f"Connected: {await resp.json()}")
    except Exception as e:
        print(f"Cannot connect: {e}")
        return

    all_results = {}

    if args.scenario in ["rag_bomb", "both"]:
        print("\n" + "="*80)
        print("SCENARIO 1: RAG BOMB (Turn 2)")
        print("="*80)

        pd_rag = await run_rag_bomb_scenario("pd", args.runs)
        await asyncio.sleep(2)
        ppd_rag = await run_rag_bomb_scenario("ppd", args.runs)

        print_comparison("rag_bomb", pd_rag, ppd_rag)
        all_results["rag_bomb"] = {"pd": pd_rag, "ppd": ppd_rag}

        await asyncio.sleep(3)

    if args.scenario in ["batch_congestion", "both"]:
        print("\n" + "="*80)
        print("SCENARIO 2: BATCH CONGESTION (Turn 2)")
        print("="*80)

        pd_batch = await run_batch_congestion_scenario("pd", args.runs)
        await asyncio.sleep(2)
        ppd_batch = await run_batch_congestion_scenario("ppd", args.runs)

        print_comparison("batch_congestion", pd_batch, ppd_batch)
        all_results["batch_congestion"] = {"pd": pd_batch, "ppd": ppd_batch}

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Turn 2 HOL Blocking Analysis")
    print("="*80)
    print("""
  Key Insight:
    - Turn 1: Both PD and PPD use P for prefill (no difference)
    - Turn 2+: PPD uses D-direct (prefill ON D) → potential HOL blocking

  Expected Results:
    - PPD should show higher victim ITL (D blocked by intruder prefill)
    - PD should show lower victim ITL (P handles intruder, D continues)
    - PPD may show faster intruder TTFT (no KV transfer)
""")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"rag_bomb_turn2_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
