#!/usr/bin/env python3
"""
Compare PD vs PPD Mode for Multi-Turn Dialogue

This script compares the performance of:
- PD Mode: Always route P → D (every turn)
- PPD Mode: First turn P → D, subsequent turns direct to D

It runs the same multi-turn conversation in both modes and compares:
- Per-turn latency
- Total latency
- Response correctness
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dataclasses import dataclass, field

# Get project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


@dataclass
class TurnResult:
    turn: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    response_text: str


@dataclass
class ConversationResult:
    mode: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / len(self.turns) if self.turns else 0


def set_proxy_mode(proxy_url: str, mode: str) -> bool:
    """Set the proxy routing mode."""
    try:
        response = requests.post(f"{proxy_url}/mode/{mode}", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to set mode: {e}")
        return False


def clear_conversations(proxy_url: str) -> bool:
    """Clear conversation state in proxy."""
    try:
        response = requests.post(f"{proxy_url}/conversations/clear", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to clear conversations: {e}")
        return False


def run_multi_turn_conversation(
    proxy_url: str,
    model_path: str,
    num_turns: int,
    output_tokens: int = 30,
) -> list[TurnResult]:
    """Run a multi-turn conversation and collect results."""

    results = []
    conversation_text = ""

    prompts = [
        "Tell me about artificial intelligence.",
        "What are the main types of machine learning?",
        "Explain how neural networks work.",
        "What is deep learning and how is it different?",
        "What are transformers in NLP?",
        "How do large language models work?",
        "What are the limitations of current AI?",
        "What is the future of AI research?",
    ]

    for turn in range(1, num_turns + 1):
        # Build prompt
        user_prompt = prompts[(turn - 1) % len(prompts)]

        if turn == 1:
            full_prompt = f"User: {user_prompt}\nAssistant:"
        else:
            full_prompt = conversation_text + f"\nUser: {user_prompt}\nAssistant:"

        # Make request
        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{proxy_url}/v1/completions",
                json={
                    "model": model_path,
                    "prompt": full_prompt,
                    "max_tokens": output_tokens,
                    "temperature": 0.7,
                    "stop": ["\nUser:", "\n\nUser:"],
                },
                timeout=120,
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            if response.status_code != 200:
                print(f"  Turn {turn}: Error {response.status_code}")
                continue

            result = response.json()
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            response_text = result["choices"][0]["text"]

            # Update conversation
            conversation_text = full_prompt + response_text

            results.append(TurnResult(
                turn=turn,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                response_text=response_text[:100],  # Truncate for display
            ))

        except Exception as e:
            print(f"  Turn {turn}: Exception {e}")
            continue

    return results


def run_comparison(
    proxy_url: str,
    model_path: str,
    num_turns: int,
    output_tokens: int,
) -> tuple[ConversationResult, ConversationResult]:
    """Run the same conversation in both PD and PPD modes."""

    print("\n" + "=" * 70)
    print("PD vs PPD Mode Comparison Test")
    print("=" * 70)
    print(f"Turns: {num_turns}")
    print(f"Output tokens per turn: {output_tokens}")
    print("=" * 70)

    # Test PD Mode
    print("\n>>> Testing PD Mode (always P → D)...")
    if not set_proxy_mode(proxy_url, "pd"):
        print("Failed to set PD mode")
        return None, None

    clear_conversations(proxy_url)
    time.sleep(1)

    pd_results = run_multi_turn_conversation(
        proxy_url, model_path, num_turns, output_tokens
    )
    pd_conv = ConversationResult(mode="pd", turns=pd_results)

    print(f"PD Mode: {len(pd_results)} turns completed")
    for t in pd_results:
        print(f"  Turn {t.turn}: {t.latency_ms:.1f}ms, {t.prompt_tokens} prompt tokens")

    # Wait a bit between tests
    time.sleep(2)

    # Test PPD Mode
    print("\n>>> Testing PPD Mode (first turn P → D, rest direct to D)...")
    if not set_proxy_mode(proxy_url, "ppd"):
        print("Failed to set PPD mode")
        return pd_conv, None

    clear_conversations(proxy_url)
    time.sleep(1)

    ppd_results = run_multi_turn_conversation(
        proxy_url, model_path, num_turns, output_tokens
    )
    ppd_conv = ConversationResult(mode="ppd", turns=ppd_results)

    print(f"PPD Mode: {len(ppd_results)} turns completed")
    for t in ppd_results:
        print(f"  Turn {t.turn}: {t.latency_ms:.1f}ms, {t.prompt_tokens} prompt tokens")

    return pd_conv, ppd_conv


def print_comparison_report(pd: ConversationResult, ppd: ConversationResult):
    """Print detailed comparison report."""

    print("\n" + "=" * 70)
    print("COMPARISON REPORT: PD vs PPD Mode")
    print("=" * 70)

    print("\n1. PER-TURN LATENCY COMPARISON:")
    print("-" * 70)
    print(f"{'Turn':>5} {'PD (ms)':>12} {'PPD (ms)':>12} {'Speedup':>12} {'Note':>20}")
    print("-" * 70)

    total_pd = 0
    total_ppd = 0

    for i, (pd_turn, ppd_turn) in enumerate(zip(pd.turns, ppd.turns)):
        pd_lat = pd_turn.latency_ms
        ppd_lat = ppd_turn.latency_ms
        total_pd += pd_lat
        total_ppd += ppd_lat

        speedup = pd_lat / ppd_lat if ppd_lat > 0 else 0
        note = "P→D" if i == 0 else ("D-direct" if pd.mode == "ppd" else "P→D")

        # For PPD mode, turn 1 is P→D, rest are D-direct
        ppd_note = "P→D" if i == 0 else "D-direct"

        print(f"{i+1:>5} {pd_lat:>12.1f} {ppd_lat:>12.1f} {speedup:>11.2f}x {ppd_note:>20}")

    print("-" * 70)

    total_speedup = total_pd / total_ppd if total_ppd > 0 else 0
    print(f"{'Total':>5} {total_pd:>12.1f} {total_ppd:>12.1f} {total_speedup:>11.2f}x")

    print("\n2. SUMMARY:")
    print("-" * 70)
    print(f"  PD Mode Total Latency:  {pd.total_latency_ms:.1f} ms")
    print(f"  PPD Mode Total Latency: {ppd.total_latency_ms:.1f} ms")
    print(f"  Latency Reduction:      {pd.total_latency_ms - ppd.total_latency_ms:.1f} ms")
    print(f"  Speedup Factor:         {total_speedup:.2f}x")

    print("\n3. ANALYSIS:")
    print("-" * 70)
    print("""
    PD Mode (Always P → D):
    - Every turn: Proxy → P (prefill all) → [KV Transfer] → D (decode)
    - P must recompute KV for entire conversation each turn
    - Full KV cache transferred every turn

    PPD Mode (D-Direct for subsequent turns):
    - Turn 1: Normal P → D flow
    - Turn 2+: Direct to D, D uses local prefix cache
    - No redundant KV transfer for cached tokens
    - D only computes prefill for new tokens

    Expected behavior:
    - Turn 1: Similar latency (both do P → D)
    - Turn 2+: PPD should be faster (no P involvement, no transfer)
    """)

    print("\n4. RESPONSE VERIFICATION:")
    print("-" * 70)
    # Check if responses are similar (not exact due to sampling)
    for i, (pd_turn, ppd_turn) in enumerate(zip(pd.turns, ppd.turns)):
        pd_text = pd_turn.response_text[:50]
        ppd_text = ppd_turn.response_text[:50]
        match = "OK" if pd_turn.prompt_tokens == ppd_turn.prompt_tokens else "DIFF"
        print(f"  Turn {i+1}: Prompt tokens match: {match}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare PD vs PPD Mode")
    parser.add_argument("--proxy-url", default="http://localhost:10001",
                        help="Proxy server URL")
    parser.add_argument("--model-path",
                        default="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
                        help="Model path")
    parser.add_argument("--turns", type=int, default=5,
                        help="Number of conversation turns")
    parser.add_argument("--output-tokens", type=int, default=30,
                        help="Output tokens per turn")
    parser.add_argument("--output", type=str,
                        default=str(RESULTS_DIR / "comparison_results.json"),
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Verify server
    try:
        response = requests.get(f"{args.proxy_url}/mode", timeout=5)
        if response.status_code != 200:
            print(f"Cannot connect to proxy at {args.proxy_url}")
            return
        print(f"Connected to proxy: {response.json()}")
    except Exception as e:
        print(f"Cannot connect to proxy: {e}")
        print("Make sure to run: ./run_ppd_test.sh ppd")
        return

    # Run comparison
    pd_conv, ppd_conv = run_comparison(
        args.proxy_url,
        args.model_path,
        args.turns,
        args.output_tokens,
    )

    if pd_conv and ppd_conv:
        print_comparison_report(pd_conv, ppd_conv)

        # Save results
        if args.output:
            # Ensure results directory exists
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            results = {
                "config": {
                    "turns": args.turns,
                    "output_tokens": args.output_tokens,
                },
                "pd_mode": {
                    "total_latency_ms": pd_conv.total_latency_ms,
                    "turns": [
                        {
                            "turn": t.turn,
                            "prompt_tokens": t.prompt_tokens,
                            "completion_tokens": t.completion_tokens,
                            "latency_ms": t.latency_ms,
                        }
                        for t in pd_conv.turns
                    ],
                },
                "ppd_mode": {
                    "total_latency_ms": ppd_conv.total_latency_ms,
                    "turns": [
                        {
                            "turn": t.turn,
                            "prompt_tokens": t.prompt_tokens,
                            "completion_tokens": t.completion_tokens,
                            "latency_ms": t.latency_ms,
                        }
                        for t in ppd_conv.turns
                    ],
                },
                "speedup": pd_conv.total_latency_ms / ppd_conv.total_latency_ms if ppd_conv.total_latency_ms > 0 else 0,
            }
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
