#!/usr/bin/env python3
"""
Multi-Turn Dialogue Test for vLLM PD Separation

This script tests multi-turn dialogue scenarios in PD separation to understand
how vLLM handles KV cache across conversation turns.

Key Concepts:
=============

1. Standard PD Separation Flow (Single Turn):
   Client → Proxy → Prefill (GPU 0) → [KV Transfer via TCP] → Decode (GPU 1) → Response

2. Multi-Turn Dialogue Challenge:
   - Turn 1: Normal PD flow, KV cache transferred from P to D
   - Turn 2+: Where should the NEW prompt go?
     - Option A: Back to P (current proxy behavior) - P computes new KV, transfers to D
     - Option B: Directly to D - D can use prefix cache for history, compute new tokens locally

3. vLLM's Mechanism:
   - D machine retains KV cache via prefix cache (lazy eviction with block_hash)
   - get_num_new_matched_tokens() checks how many tokens are already cached
   - External prefix cache hit rate shows cache utilization from P→D transfers

4. Current Proxy Behavior (xpyd):
   - ALWAYS routes to P first (max_tokens=1 for prefill)
   - Then forwards to D with original max_tokens
   - This means: every turn does a full P→D transfer

5. What This Test Measures:
   - Per-turn latency breakdown
   - KV cache sizes per turn
   - External prefix cache hit rate on D (from logs)
   - Whether D is reusing KV cache from previous turns

Usage:
======
    # Start PD servers first
    ./run_pd_separation_test.sh &

    # Wait for servers, then run multi-turn test
    python multi_turn_test.py --turns 5 --input-tokens 100 --output-tokens 50
"""

import argparse
import json
import time
import requests
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn"""
    turn_number: int
    input_tokens: int          # New tokens in this turn's prompt
    output_tokens: int         # Generated tokens
    cumulative_tokens: int     # Total tokens in conversation so far
    kv_cache_size_mb: float    # KV cache size for new tokens only
    cumulative_kv_mb: float    # Total KV cache size for full conversation
    latency_ms: float          # Response latency
    apparent_bandwidth_gb_s: float  # Apparent bandwidth for this turn


@dataclass
class ConversationMetrics:
    """Metrics for entire conversation"""
    num_turns: int
    turns: list[TurnMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_kv_transfer_mb: float = 0.0
    total_latency_ms: float = 0.0

    def add_turn(self, turn: TurnMetrics):
        self.turns.append(turn)
        self.total_input_tokens += turn.input_tokens
        self.total_output_tokens += turn.output_tokens
        self.total_kv_transfer_mb += turn.kv_cache_size_mb
        self.total_latency_ms += turn.latency_ms


# Llama-3.1-8B KV cache size: 128 KB per token
KV_BYTES_PER_TOKEN = 32 * 2 * 8 * 128 * 2  # 131,072 bytes = 128 KB


def calculate_kv_size_mb(num_tokens: int) -> float:
    """Calculate KV cache size in MB for given number of tokens"""
    return (num_tokens * KV_BYTES_PER_TOKEN) / (1024 * 1024)


def generate_prompt(word_count: int, turn_number: int) -> str:
    """Generate a prompt with approximately the specified word count"""
    # Approximate: 1 word ≈ 1.3 tokens
    base_prompts = [
        "Tell me about artificial intelligence and machine learning. ",
        "Explain the concept of neural networks in deep learning. ",
        "Describe how transformers work in natural language processing. ",
        "What are the key innovations in large language models? ",
        "How does attention mechanism improve model performance? ",
    ]
    base = base_prompts[turn_number % len(base_prompts)]
    padding = "Please elaborate with examples. " * (word_count // 5)
    return base + padding


class MultiTurnDialogueTest:
    """Multi-turn dialogue test client"""

    def __init__(
        self,
        proxy_url: str,
        model_path: str,
        num_turns: int,
        input_tokens_per_turn: int,
        output_tokens_per_turn: int,
    ):
        self.proxy_url = proxy_url
        self.model_path = model_path
        self.num_turns = num_turns
        self.input_tokens_per_turn = input_tokens_per_turn
        self.output_tokens_per_turn = output_tokens_per_turn

        # Conversation history for chat format
        self.messages: list[dict] = []
        # Accumulated prompt for completion format
        self.conversation_text = ""

    def run_chat_format(self) -> ConversationMetrics:
        """Run multi-turn test using chat/completions format"""
        metrics = ConversationMetrics(num_turns=self.num_turns)
        cumulative_tokens = 0

        print("\n" + "="*70)
        print("Multi-Turn Dialogue Test (Chat Format)")
        print("="*70)
        print(f"Turns: {self.num_turns}")
        print(f"Input tokens per turn: ~{self.input_tokens_per_turn}")
        print(f"Output tokens per turn: {self.output_tokens_per_turn}")
        print("="*70)

        for turn in range(1, self.num_turns + 1):
            print(f"\n--- Turn {turn} ---")

            # Generate user message
            word_count = int(self.input_tokens_per_turn * 0.77)
            user_message = generate_prompt(word_count, turn)

            # Add to conversation history
            self.messages.append({"role": "user", "content": user_message})

            # Make request
            start_time = time.perf_counter()

            try:
                response = requests.post(
                    f"{self.proxy_url}/v1/chat/completions",
                    json={
                        "model": self.model_path,
                        "messages": self.messages,
                        "max_tokens": self.output_tokens_per_turn,
                        "temperature": 0.7,
                    },
                    timeout=120,
                )

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                if response.status_code != 200:
                    print(f"  Error: {response.status_code}")
                    continue

                result = response.json()

                # Extract metrics
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Add assistant response to history
                assistant_message = result["choices"][0]["message"]["content"]
                self.messages.append({"role": "assistant", "content": assistant_message})

                # Calculate KV cache metrics
                # For current turn, new KV is for the new user message only
                # (previous turns' KV should be cached on D via prefix cache)
                new_input_tokens = self.input_tokens_per_turn  # Approximate
                cumulative_tokens = prompt_tokens

                new_kv_mb = calculate_kv_size_mb(new_input_tokens)
                cumulative_kv_mb = calculate_kv_size_mb(cumulative_tokens)

                # Calculate bandwidth (apparent, includes all overhead)
                apparent_bandwidth = new_kv_mb / (latency_ms / 1000) / 1024 if latency_ms > 0 else 0

                turn_metrics = TurnMetrics(
                    turn_number=turn,
                    input_tokens=new_input_tokens,
                    output_tokens=completion_tokens,
                    cumulative_tokens=cumulative_tokens,
                    kv_cache_size_mb=new_kv_mb,
                    cumulative_kv_mb=cumulative_kv_mb,
                    latency_ms=latency_ms,
                    apparent_bandwidth_gb_s=apparent_bandwidth,
                )
                metrics.add_turn(turn_metrics)

                print(f"  Prompt tokens: {prompt_tokens} (cumulative)")
                print(f"  New input tokens: ~{new_input_tokens}")
                print(f"  Output tokens: {completion_tokens}")
                print(f"  Latency: {latency_ms:.2f} ms")
                print(f"  New KV size: {new_kv_mb:.2f} MB")
                print(f"  Cumulative KV size: {cumulative_kv_mb:.2f} MB")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        return metrics

    def run_completion_format(self) -> ConversationMetrics:
        """Run multi-turn test using completions format (simpler, no chat template)"""
        metrics = ConversationMetrics(num_turns=self.num_turns)
        cumulative_tokens = 0

        print("\n" + "="*70)
        print("Multi-Turn Dialogue Test (Completion Format)")
        print("="*70)
        print(f"Turns: {self.num_turns}")
        print(f"Input tokens per turn: ~{self.input_tokens_per_turn}")
        print(f"Output tokens per turn: {self.output_tokens_per_turn}")
        print("="*70)

        for turn in range(1, self.num_turns + 1):
            print(f"\n--- Turn {turn} ---")

            # Generate user prompt
            word_count = int(self.input_tokens_per_turn * 0.77)
            user_prompt = generate_prompt(word_count, turn)

            # Build conversation context
            if turn == 1:
                full_prompt = f"User: {user_prompt}\nAssistant:"
            else:
                full_prompt = self.conversation_text + f"\nUser: {user_prompt}\nAssistant:"

            # Make request
            start_time = time.perf_counter()

            try:
                response = requests.post(
                    f"{self.proxy_url}/v1/completions",
                    json={
                        "model": self.model_path,
                        "prompt": full_prompt,
                        "max_tokens": self.output_tokens_per_turn,
                        "temperature": 0.7,
                        "stop": ["\nUser:", "\n\nUser:"],
                    },
                    timeout=120,
                )

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                if response.status_code != 200:
                    print(f"  Error: {response.status_code} - {response.text[:200]}")
                    continue

                result = response.json()

                # Extract metrics
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Add response to conversation
                assistant_response = result["choices"][0]["text"]
                self.conversation_text = full_prompt + assistant_response

                # Calculate KV cache metrics
                new_input_tokens = prompt_tokens - cumulative_tokens if turn > 1 else prompt_tokens
                cumulative_tokens = prompt_tokens + completion_tokens

                new_kv_mb = calculate_kv_size_mb(new_input_tokens)
                cumulative_kv_mb = calculate_kv_size_mb(prompt_tokens)

                # Calculate bandwidth
                apparent_bandwidth = new_kv_mb / (latency_ms / 1000) / 1024 if latency_ms > 0 else 0

                turn_metrics = TurnMetrics(
                    turn_number=turn,
                    input_tokens=new_input_tokens,
                    output_tokens=completion_tokens,
                    cumulative_tokens=prompt_tokens,
                    kv_cache_size_mb=new_kv_mb,
                    cumulative_kv_mb=cumulative_kv_mb,
                    latency_ms=latency_ms,
                    apparent_bandwidth_gb_s=apparent_bandwidth,
                )
                metrics.add_turn(turn_metrics)

                print(f"  Prompt tokens: {prompt_tokens} (cumulative context)")
                print(f"  New tokens this turn: {new_input_tokens}")
                print(f"  Output tokens: {completion_tokens}")
                print(f"  Latency: {latency_ms:.2f} ms")
                print(f"  New KV size: {new_kv_mb:.2f} MB")
                print(f"  Cumulative KV size: {cumulative_kv_mb:.2f} MB")

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        return metrics


def print_analysis(metrics: ConversationMetrics):
    """Print analysis of multi-turn dialogue metrics"""

    print("\n" + "="*70)
    print("MULTI-TURN DIALOGUE ANALYSIS")
    print("="*70)

    print("\n1. PER-TURN METRICS:")
    print("-"*70)
    print(f"{'Turn':>5} {'New Tokens':>12} {'Cumulative':>12} {'New KV (MB)':>12} {'Latency (ms)':>14}")
    print("-"*70)

    for t in metrics.turns:
        print(f"{t.turn_number:>5} {t.input_tokens:>12} {t.cumulative_tokens:>12} {t.kv_cache_size_mb:>12.2f} {t.latency_ms:>14.2f}")

    print("\n2. TOTALS:")
    print("-"*70)
    print(f"  Total turns: {metrics.num_turns}")
    print(f"  Total input tokens: {metrics.total_input_tokens}")
    print(f"  Total output tokens: {metrics.total_output_tokens}")
    print(f"  Total KV transferred: {metrics.total_kv_transfer_mb:.2f} MB")
    print(f"  Total latency: {metrics.total_latency_ms:.2f} ms")

    if metrics.turns:
        avg_latency = metrics.total_latency_ms / len(metrics.turns)
        print(f"  Average latency per turn: {avg_latency:.2f} ms")

    print("\n3. KEY OBSERVATIONS FOR PD SEPARATION:")
    print("-"*70)
    print("""
    Current Proxy Behavior (xpyd):
    - Every turn routes to Prefill first (max_tokens=1)
    - Then forwards to Decode with original request
    - This means: FULL KV cache is transferred every turn

    What This Implies:
    - Turn N transfers ALL cumulative context (not just new tokens)
    - Prefix cache on Decode helps with LOCAL cache reuse
    - But cross-request KV transfer still happens for full context

    Efficiency Consideration:
    - Ideal: Only transfer KV for NEW tokens each turn
    - Current: Transfer KV for ALL tokens each turn
    - This is where PPD optimization can help

    Check Decode Logs For:
    - 'External prefix cache hit rate' - shows cache utilization
    - Higher rate = better KV cache reuse from P→D transfers
    """)

    print("\n4. LATENCY BREAKDOWN EXPECTATION:")
    print("-"*70)
    print("""
    Total Request Time = Prefill Time + KV Transfer Time + Decode Time + Overhead

    For Turn N with C cumulative tokens:
    - Prefill Time: O(C) - must process all context
    - KV Transfer: O(C) - must transfer all KV cache
    - Decode Time: O(output_tokens)

    As conversation grows, both Prefill and Transfer scale linearly!
    This is the key motivation for PPD optimization.
    """)


def main():
    parser = argparse.ArgumentParser(description="Multi-Turn Dialogue Test for PD Separation")
    parser.add_argument("--proxy-url", default="http://localhost:10001",
                        help="Proxy server URL")
    parser.add_argument("--model-path",
                        default="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
                        help="Model path")
    parser.add_argument("--turns", type=int, default=5,
                        help="Number of conversation turns")
    parser.add_argument("--input-tokens", type=int, default=100,
                        help="Approximate input tokens per turn")
    parser.add_argument("--output-tokens", type=int, default=50,
                        help="Output tokens per turn")
    parser.add_argument("--format", choices=["chat", "completion"], default="completion",
                        help="API format to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Verify server is running
    try:
        response = requests.post(
            f"{args.proxy_url}/v1/completions",
            json={"model": args.model_path, "prompt": "test", "max_tokens": 1},
            timeout=30,
        )
        print(f"Server connection verified (status: {response.status_code})")
    except Exception as e:
        print(f"Error: Cannot connect to proxy server at {args.proxy_url}")
        print(f"Make sure run_pd_separation_test.sh is running first.")
        print(f"Error: {e}")
        return

    # Run test
    test = MultiTurnDialogueTest(
        proxy_url=args.proxy_url,
        model_path=args.model_path,
        num_turns=args.turns,
        input_tokens_per_turn=args.input_tokens,
        output_tokens_per_turn=args.output_tokens,
    )

    if args.format == "chat":
        metrics = test.run_chat_format()
    else:
        metrics = test.run_completion_format()

    # Print analysis
    print_analysis(metrics)

    # Save results
    if args.output:
        results = {
            "config": {
                "num_turns": args.turns,
                "input_tokens_per_turn": args.input_tokens,
                "output_tokens_per_turn": args.output_tokens,
                "format": args.format,
            },
            "totals": {
                "total_input_tokens": metrics.total_input_tokens,
                "total_output_tokens": metrics.total_output_tokens,
                "total_kv_transfer_mb": metrics.total_kv_transfer_mb,
                "total_latency_ms": metrics.total_latency_ms,
            },
            "turns": [
                {
                    "turn": t.turn_number,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "cumulative_tokens": t.cumulative_tokens,
                    "kv_cache_size_mb": t.kv_cache_size_mb,
                    "cumulative_kv_mb": t.cumulative_kv_mb,
                    "latency_ms": t.latency_ms,
                }
                for t in metrics.turns
            ],
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
