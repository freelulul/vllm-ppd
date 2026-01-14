#!/usr/bin/env python3
"""
Optimizer Comparison Test

Runs real tests comparing PD, PPD, Replica, and Optimizer modes.
Evaluates optimizer effectiveness against oracle (best actual mode).
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import sys

# Add optimizer to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'optimizer'))

from models.rule_based_selector import (
    RuleBasedSelector, RequestFeatures, OptimizationObjective, Mode
)


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"

# Server endpoints
SERVERS = {
    "pd": "http://localhost:10001",      # PD via proxy
    "ppd": "http://localhost:10001",     # PPD via proxy (same proxy, mode switch)
    "replica": "http://localhost:10002", # Replica via proxy
    "decode_direct": "http://localhost:8200",  # Direct to decode for PPD T2+
    "replica0_direct": "http://localhost:8300",
    "replica1_direct": "http://localhost:8400",
}

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=300)


# =============================================================================
# Test Workload Definitions
# =============================================================================

@dataclass
class TestWorkload:
    """Definition of a test workload"""
    name: str
    description: str
    t1_input: int      # Turn 1 input tokens
    t1_output: int     # Turn 1 output tokens
    t2_input: int      # Turn 2 additional input tokens
    t2_output: int     # Turn 2 output tokens
    num_turns: int = 2
    num_conversations: int = 3  # Number of conversations to test

    @property
    def is_big_paste(self) -> bool:
        return self.t1_input > 1000 and self.t1_output < 256

    @property
    def is_long_generation(self) -> bool:
        return self.t1_output > 512 or self.t2_output > 512


# Define diverse test workloads
TEST_WORKLOADS = [
    # Small workloads
    TestWorkload("XS_balanced", "Very small, balanced I/O",
                 t1_input=128, t1_output=128, t2_input=32, t2_output=64),

    # Medium workloads
    TestWorkload("M_balanced", "Medium balanced",
                 t1_input=512, t1_output=256, t2_input=64, t2_output=128),
    TestWorkload("M_long_output", "Medium input, long output",
                 t1_input=256, t1_output=512, t2_input=32, t2_output=256),

    # Big-Paste scenarios (long input, short output)
    TestWorkload("L_big_paste", "Large input, short output (Big-Paste)",
                 t1_input=2000, t1_output=128, t2_input=32, t2_output=64),
    TestWorkload("XL_big_paste", "Very large input, short output",
                 t1_input=4000, t1_output=64, t2_input=16, t2_output=32),

    # Long generation scenarios
    TestWorkload("M_generation", "Medium input, very long output",
                 t1_input=256, t1_output=1024, t2_input=32, t2_output=512),

    # Multi-turn heavy (context accumulation)
    TestWorkload("Multi_turn", "Multi-turn conversation",
                 t1_input=512, t1_output=256, t2_input=128, t2_output=256, num_turns=4),
]


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class TurnMetrics:
    """Metrics for a single turn"""
    turn: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    output_tokens: int
    success: bool
    error: Optional[str] = None


@dataclass
class ConversationMetrics:
    """Metrics for a full conversation"""
    conversation_id: str
    mode: str
    workload: str
    turns: List[TurnMetrics] = field(default_factory=list)
    total_e2e_ms: float = 0.0

    @property
    def avg_ttft(self) -> float:
        valid = [t.ttft_ms for t in self.turns if t.success]
        return statistics.mean(valid) if valid else float('inf')

    @property
    def avg_tpot(self) -> float:
        valid = [t.tpot_ms for t in self.turns if t.success]
        return statistics.mean(valid) if valid else float('inf')

    @property
    def t2_ttft(self) -> float:
        """Turn 2 TTFT (key metric for PPD)"""
        t2 = [t for t in self.turns if t.turn == 2 and t.success]
        return t2[0].ttft_ms if t2 else float('inf')


@dataclass
class ModeResult:
    """Aggregated results for a mode on a workload"""
    mode: str
    workload: str
    conversations: List[ConversationMetrics] = field(default_factory=list)

    @property
    def avg_t2_ttft(self) -> float:
        valid = [c.t2_ttft for c in self.conversations if c.t2_ttft < float('inf')]
        return statistics.mean(valid) if valid else float('inf')

    @property
    def avg_tpot(self) -> float:
        valid = [c.avg_tpot for c in self.conversations if c.avg_tpot < float('inf')]
        return statistics.mean(valid) if valid else float('inf')

    @property
    def avg_e2e(self) -> float:
        valid = [c.total_e2e_ms for c in self.conversations if c.total_e2e_ms > 0]
        return statistics.mean(valid) if valid else float('inf')

    @property
    def throughput(self) -> float:
        """Total tokens / total time"""
        total_tokens = sum(
            sum(t.output_tokens for t in c.turns if t.success)
            for c in self.conversations
        )
        total_time = sum(c.total_e2e_ms for c in self.conversations) / 1000
        return total_tokens / total_time if total_time > 0 else 0


# =============================================================================
# Test Runner
# =============================================================================

class TestRunner:
    """Runs comparison tests across all modes"""

    def __init__(self):
        self.selector = RuleBasedSelector(verbose=False)
        self.results: Dict[str, Dict[str, ModeResult]] = {}  # workload -> mode -> result

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        prompt: str,
        max_tokens: int,
        mode: str = None
    ) -> Tuple[float, float, float, int, bool, str]:
        """Send a single request and measure metrics"""
        payload = {
            "model": MODEL_PATH,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        # Set mode for PD/PPD proxy
        if mode in ["pd", "ppd"] and "10001" in url:
            payload["mode"] = mode

        start_time = time.perf_counter()
        ttft = None
        tokens_received = 0
        full_response = ""

        try:
            async with session.post(f"{url}/v1/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return 0, 0, 0, 0, False, f"HTTP {response.status}: {error_text[:100]}"

                async for line in response.content:
                    if ttft is None:
                        ttft = (time.perf_counter() - start_time) * 1000

                    line = line.decode('utf-8').strip()
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        try:
                            data = json.loads(line[6:])
                            if 'choices' in data and data['choices']:
                                text = data['choices'][0].get('text', '')
                                full_response += text
                                tokens_received += 1
                        except json.JSONDecodeError:
                            pass

            e2e = (time.perf_counter() - start_time) * 1000

            # Calculate TPOT (excluding first token)
            if tokens_received > 1 and ttft:
                tpot = (e2e - ttft) / (tokens_received - 1)
            else:
                tpot = 0

            return ttft or 0, tpot, e2e, tokens_received, True, None

        except Exception as e:
            return 0, 0, 0, 0, False, str(e)

    def _generate_context(self, num_tokens: int) -> str:
        """Generate context of approximately num_tokens"""
        # Each "A " is roughly 1 token
        return "A " * num_tokens

    async def _run_conversation(
        self,
        session: aiohttp.ClientSession,
        workload: TestWorkload,
        mode: str,
        conv_idx: int
    ) -> ConversationMetrics:
        """Run a single multi-turn conversation"""
        conv_id = f"{workload.name}_{mode}_{conv_idx}_{int(time.time())}"
        metrics = ConversationMetrics(
            conversation_id=conv_id,
            mode=mode,
            workload=workload.name
        )

        # Determine URL based on mode
        if mode == "pd":
            url = SERVERS["pd"]
        elif mode == "ppd":
            url = SERVERS["ppd"]
        elif mode == "replica":
            # Use direct access for proper cache affinity
            url = SERVERS[f"replica{conv_idx % 2}_direct"]
        elif mode == "optimizer":
            # Optimizer mode - will be handled separately
            url = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build conversation history
        base_context = self._generate_context(workload.t1_input)
        history = base_context + " Remember this context."

        total_start = time.perf_counter()

        # Track where Turn 1 went for cache affinity
        optimizer_cached_gpu = None
        optimizer_cached_url = None

        for turn in range(1, workload.num_turns + 1):
            # Add turn-specific prompt
            if turn == 1:
                prompt = history + " Briefly summarize and say OK."
                output_tokens = workload.t1_output
            else:
                # Add previous response to history (simulated)
                history += " OK. "
                history += self._generate_context(workload.t2_input)
                prompt = history + f" Continue turn {turn}."
                output_tokens = workload.t2_output

            # For optimizer mode, select best mode dynamically
            if mode == "optimizer":
                features = RequestFeatures(
                    input_length=len(prompt) // 4,  # Rough token estimate
                    output_length=output_tokens,
                    turn_number=turn,
                    has_cache=(turn > 1 and optimizer_cached_gpu is not None),
                    cached_gpu=optimizer_cached_gpu,  # Use ACTUAL Turn 1 location
                    queue_depths={"gpu1": 0, "gpu2": 0, "gpu3": 0},
                    objective=OptimizationObjective.TTFT  # Default: TTFT optimization
                )
                decision = self.selector.select_mode(features)
                selected_mode = decision.mode.value

                # Turn 1: Select best mode and remember cache location
                if turn == 1:
                    if selected_mode == "pd":
                        url = SERVERS["pd"]
                        optimizer_cached_gpu = "gpu1"
                    elif selected_mode == "ppd":
                        url = SERVERS["ppd"]
                        optimizer_cached_gpu = "gpu1"
                    else:  # replica
                        replica_idx = conv_idx % 2
                        url = SERVERS[f"replica{replica_idx}_direct"]
                        optimizer_cached_gpu = f"gpu{2 + replica_idx}"
                    optimizer_cached_url = url
                else:
                    # Turn 2+: Follow cache affinity - go to same GPU!
                    # This is the KEY: always use the cached location for Turn 2+
                    url = optimizer_cached_url

            ttft, tpot, e2e, tokens, success, error = await self._send_request(
                session, url, prompt, output_tokens,
                mode if mode in ["pd", "ppd"] else None
            )

            metrics.turns.append(TurnMetrics(
                turn=turn,
                ttft_ms=ttft,
                tpot_ms=tpot,
                e2e_ms=e2e,
                output_tokens=tokens,
                success=success,
                error=error
            ))

            # Update history with response length
            history += " " + "Response " * (tokens // 2)

        metrics.total_e2e_ms = (time.perf_counter() - total_start) * 1000
        return metrics

    async def run_workload_test(
        self,
        workload: TestWorkload,
        modes: List[str] = ["pd", "ppd", "replica", "optimizer"]
    ) -> Dict[str, ModeResult]:
        """Run tests for all modes on a single workload"""
        print(f"\n{'='*60}")
        print(f"Testing: {workload.name}")
        print(f"  {workload.description}")
        print(f"  T1: {workload.t1_input} in → {workload.t1_output} out")
        print(f"  T2: {workload.t2_input} in → {workload.t2_output} out")
        print(f"  Turns: {workload.num_turns}, Conversations: {workload.num_conversations}")
        print('='*60)

        results = {}

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            for mode in modes:
                print(f"\n  [{mode.upper()}]", end=" ", flush=True)

                mode_result = ModeResult(mode=mode, workload=workload.name)

                for conv_idx in range(workload.num_conversations):
                    try:
                        metrics = await self._run_conversation(
                            session, workload, mode, conv_idx
                        )
                        mode_result.conversations.append(metrics)

                        # Print progress
                        success = all(t.success for t in metrics.turns)
                        print("✓" if success else "✗", end="", flush=True)

                    except Exception as e:
                        print(f"E({e})", end="", flush=True)

                results[mode] = mode_result

                # Print mode summary
                print(f" | T2_TTFT: {mode_result.avg_t2_ttft:.0f}ms, "
                      f"TPOT: {mode_result.avg_tpot:.1f}ms, "
                      f"E2E: {mode_result.avg_e2e:.0f}ms")

        self.results[workload.name] = results
        return results

    async def run_all_tests(self) -> Dict:
        """Run tests for all workloads"""
        all_results = {}

        for workload in TEST_WORKLOADS:
            try:
                results = await self.run_workload_test(workload)
                all_results[workload.name] = results
            except Exception as e:
                print(f"Error testing {workload.name}: {e}")

        return all_results

    def analyze_results(self) -> Dict:
        """Analyze results and determine optimizer effectiveness"""
        analysis = {
            "workloads": {},
            "optimizer_accuracy": {},
            "summary": {}
        }

        correct_ttft = 0
        correct_tpot = 0
        correct_throughput = 0
        total = 0

        print("\n" + "="*80)
        print("RESULTS ANALYSIS")
        print("="*80)

        print(f"\n{'Workload':<20} {'PD':<15} {'PPD':<15} {'Replica':<15} {'Optimizer':<15} {'Best':<10}")
        print("-"*80)

        for workload_name, mode_results in self.results.items():
            if not mode_results:
                continue

            # Get metrics for each mode
            metrics = {}
            for mode, result in mode_results.items():
                metrics[mode] = {
                    "t2_ttft": result.avg_t2_ttft,
                    "tpot": result.avg_tpot,
                    "e2e": result.avg_e2e,
                    "throughput": result.throughput
                }

            # Find oracle (best mode for each objective)
            oracle_ttft = min(["pd", "ppd", "replica"],
                             key=lambda m: metrics.get(m, {}).get("t2_ttft", float('inf')))
            oracle_tpot = min(["pd", "ppd", "replica"],
                             key=lambda m: metrics.get(m, {}).get("tpot", float('inf')))
            oracle_throughput = max(["pd", "ppd", "replica"],
                                   key=lambda m: metrics.get(m, {}).get("throughput", 0))

            # Check optimizer choice
            opt_ttft = metrics.get("optimizer", {}).get("t2_ttft", float('inf'))
            opt_tpot = metrics.get("optimizer", {}).get("tpot", float('inf'))

            # Compare optimizer to oracle
            total += 1
            if opt_ttft <= metrics.get(oracle_ttft, {}).get("t2_ttft", float('inf')) * 1.1:
                correct_ttft += 1
            if opt_tpot <= metrics.get(oracle_tpot, {}).get("tpot", float('inf')) * 1.1:
                correct_tpot += 1

            # Print row
            pd_ttft = metrics.get("pd", {}).get("t2_ttft", float('inf'))
            ppd_ttft = metrics.get("ppd", {}).get("t2_ttft", float('inf'))
            rep_ttft = metrics.get("replica", {}).get("t2_ttft", float('inf'))
            opt_ttft_val = metrics.get("optimizer", {}).get("t2_ttft", float('inf'))

            best = oracle_ttft
            print(f"{workload_name:<20} "
                  f"{pd_ttft:>6.0f}ms      "
                  f"{ppd_ttft:>6.0f}ms      "
                  f"{rep_ttft:>6.0f}ms      "
                  f"{opt_ttft_val:>6.0f}ms      "
                  f"{best:<10}")

            analysis["workloads"][workload_name] = {
                "metrics": metrics,
                "oracle_ttft": oracle_ttft,
                "oracle_tpot": oracle_tpot,
                "oracle_throughput": oracle_throughput,
            }

        # Summary
        print("\n" + "-"*80)
        print("OPTIMIZER EFFECTIVENESS (within 10% of oracle)")
        print("-"*80)

        if total > 0:
            ttft_acc = correct_ttft / total
            tpot_acc = correct_tpot / total
            print(f"  TTFT optimization: {correct_ttft}/{total} ({ttft_acc:.1%})")
            print(f"  TPOT optimization: {correct_tpot}/{total} ({tpot_acc:.1%})")

            analysis["summary"] = {
                "total_workloads": total,
                "ttft_accuracy": ttft_acc,
                "tpot_accuracy": tpot_acc,
            }

        return analysis

    def save_results(self, output_path: str):
        """Save results to JSON file"""
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workloads": {},
        }

        for workload_name, mode_results in self.results.items():
            output["workloads"][workload_name] = {}
            for mode, result in mode_results.items():
                output["workloads"][workload_name][mode] = {
                    "avg_t2_ttft": result.avg_t2_ttft,
                    "avg_tpot": result.avg_tpot,
                    "avg_e2e": result.avg_e2e,
                    "throughput": result.throughput,
                    "conversations": [
                        {
                            "id": c.conversation_id,
                            "turns": [
                                {
                                    "turn": t.turn,
                                    "ttft_ms": t.ttft_ms,
                                    "tpot_ms": t.tpot_ms,
                                    "e2e_ms": t.e2e_ms,
                                    "output_tokens": t.output_tokens,
                                    "success": t.success,
                                }
                                for t in c.turns
                            ]
                        }
                        for c in result.conversations
                    ]
                }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_path}")


async def main():
    print("="*80)
    print("OPTIMIZER COMPARISON TEST")
    print("="*80)
    print(f"Testing {len(TEST_WORKLOADS)} workloads across 4 modes")
    print("Modes: PD, PPD, Replica, Optimizer (rule-based)")

    runner = TestRunner()

    # Run all tests
    await runner.run_all_tests()

    # Analyze results
    analysis = runner.analyze_results()

    # Save results
    output_path = "results/optimizer_comparison.json"
    runner.save_results(output_path)

    return analysis


if __name__ == "__main__":
    asyncio.run(main())
