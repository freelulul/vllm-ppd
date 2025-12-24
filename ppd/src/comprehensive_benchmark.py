#!/usr/bin/env python3
"""
Comprehensive PD vs PPD Benchmark Suite

This script performs detailed comparison between PD and PPD modes with:
- Various conversation turn counts (1-20)
- Various input/output token length combinations
- Fine-grained timing breakdown (P prefill, D decode, KV transfer estimation)
- Cache hit rate analysis from vLLM logs
- Multi-run support with warmup for statistical significance

Usage:
    python comprehensive_benchmark.py --config benchmark_config.json
    python comprehensive_benchmark.py --quick  # Quick test with minimal configs
    python comprehensive_benchmark.py --extended  # Extended test with large tokens
    python comprehensive_benchmark.py --runs 3 --warmup 1  # 3 runs with 1 warmup
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"


@dataclass
class TurnMetrics:
    """Metrics for a single turn."""
    turn: int
    mode: str  # pd, ppd_turn1, ppd_d_direct

    # Timing (ms)
    total_time_ms: float = 0.0
    p_prefill_time_ms: float = 0.0  # Time on P (PD mode)
    d_time_ms: float = 0.0  # Time on D

    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_tokens_this_turn: int = 0  # New tokens added this turn

    # KV Cache
    estimated_kv_size_mb: float = 0.0
    cumulative_kv_size_mb: float = 0.0

    # Response
    response_text: str = ""


@dataclass
class ConversationMetrics:
    """Metrics for a complete conversation."""
    config_name: str
    mode: str
    num_turns: int
    input_tokens_per_turn: int
    output_tokens_per_turn: int

    turns: list[TurnMetrics] = field(default_factory=list)

    # Aggregates
    total_time_ms: float = 0.0
    total_p_time_ms: float = 0.0
    total_d_time_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_kv_transferred_mb: float = 0.0

    # Cache metrics (from logs)
    local_cache_hit_rate: float = 0.0
    external_cache_hit_rate: float = 0.0

    timestamp: str = ""


@dataclass
class MultiRunResult:
    """Aggregated results from multiple runs of a single config."""
    config_name: str
    num_turns: int
    input_tokens: int
    output_tokens: int
    num_runs: int

    # PD timing stats
    pd_times_ms: list[float] = field(default_factory=list)
    pd_mean_ms: float = 0.0
    pd_std_ms: float = 0.0

    # PPD timing stats
    ppd_times_ms: list[float] = field(default_factory=list)
    ppd_mean_ms: float = 0.0
    ppd_std_ms: float = 0.0

    # Speedup stats
    speedup_mean: float = 0.0
    speedup_std: float = 0.0

    def calculate_stats(self):
        """Calculate mean and std from collected times."""
        if self.pd_times_ms:
            self.pd_mean_ms = statistics.mean(self.pd_times_ms)
            self.pd_std_ms = statistics.stdev(self.pd_times_ms) if len(self.pd_times_ms) > 1 else 0.0
        if self.ppd_times_ms:
            self.ppd_mean_ms = statistics.mean(self.ppd_times_ms)
            self.ppd_std_ms = statistics.stdev(self.ppd_times_ms) if len(self.ppd_times_ms) > 1 else 0.0
        if self.pd_mean_ms > 0 and self.ppd_mean_ms > 0:
            speedups = [pd / ppd for pd, ppd in zip(self.pd_times_ms, self.ppd_times_ms) if ppd > 0]
            if speedups:
                self.speedup_mean = statistics.mean(speedups)
                self.speedup_std = statistics.stdev(speedups) if len(speedups) > 1 else 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    name: str
    timestamp: str
    config: dict

    pd_results: list[ConversationMetrics] = field(default_factory=list)
    ppd_results: list[ConversationMetrics] = field(default_factory=list)

    # Multi-run aggregated results
    multi_run_results: list[MultiRunResult] = field(default_factory=list)

    comparison: dict = field(default_factory=dict)


class ComprehensiveBenchmark:
    """Comprehensive benchmark suite for PD vs PPD comparison."""

    def __init__(
        self,
        proxy_url: str = "http://localhost:10001",
        model_path: str = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
    ):
        self.proxy_url = proxy_url
        self.model_path = model_path
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)

    def set_mode(self, mode: str) -> bool:
        """Set proxy routing mode."""
        try:
            resp = requests.post(f"{self.proxy_url}/mode/{mode}", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"Error setting mode: {e}")
            return False

    def clear_state(self):
        """Clear conversation and metrics state."""
        try:
            requests.post(f"{self.proxy_url}/conversations/clear", timeout=5)
            requests.post(f"{self.proxy_url}/metrics/clear", timeout=5)
        except:
            pass

    def get_metrics(self) -> list[dict]:
        """Get metrics from proxy."""
        try:
            resp = requests.get(f"{self.proxy_url}/metrics", timeout=10)
            return resp.json()
        except:
            return []

    def generate_prompt(self, base_tokens: int, turn: int) -> str:
        """Generate a prompt with approximately the specified number of tokens."""
        # Approximate: 1 token ≈ 4 characters
        base_text = (
            "This is a test prompt for benchmarking the vLLM disaggregated serving system. "
            "The system separates prefill and decode phases across different GPU instances. "
        )
        # Repeat to get desired length
        target_chars = base_tokens * 4
        repeated = (base_text * ((target_chars // len(base_text)) + 1))[:target_chars]
        return f"Turn {turn}: {repeated}"

    def run_single_turn(
        self,
        conversation_history: str,
        new_prompt: str,
        output_tokens: int,
    ) -> tuple[str, dict, float]:
        """
        Run a single turn and return (response_text, usage_dict, latency_ms).
        """
        if conversation_history:
            full_prompt = f"{conversation_history}\nUser: {new_prompt}\nAssistant:"
        else:
            full_prompt = f"User: {new_prompt}\nAssistant:"

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.proxy_url}/v1/completions",
                json={
                    "model": self.model_path,
                    "prompt": full_prompt,
                    "max_tokens": output_tokens,
                    "temperature": 0.7,
                    "stop": ["\nUser:", "\n\nUser:"],
                },
                timeout=600,  # Increased timeout for large outputs
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code != 200:
                return "", {}, latency_ms

            result = response.json()
            usage = result.get("usage", {})
            text = result["choices"][0]["text"]

            return text, usage, latency_ms

        except Exception as e:
            print(f"Error in turn: {e}")
            return "", {}, 0.0

    def run_conversation(
        self,
        mode: str,
        num_turns: int,
        input_tokens_per_turn: int,
        output_tokens_per_turn: int,
        config_name: str = "",
    ) -> ConversationMetrics:
        """Run a complete multi-turn conversation."""

        metrics = ConversationMetrics(
            config_name=config_name,
            mode=mode,
            num_turns=num_turns,
            input_tokens_per_turn=input_tokens_per_turn,
            output_tokens_per_turn=output_tokens_per_turn,
            timestamp=datetime.now().isoformat(),
        )

        conversation_history = ""
        cumulative_tokens = 0

        for turn in range(1, num_turns + 1):
            # Generate prompt for this turn
            new_prompt = self.generate_prompt(input_tokens_per_turn, turn)

            # Run turn
            response_text, usage, latency_ms = self.run_single_turn(
                conversation_history, new_prompt, output_tokens_per_turn
            )

            if not response_text:
                print(f"  Turn {turn}: Failed")
                continue

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Calculate KV size
            kv_size_mb = self._calculate_kv_size(prompt_tokens)
            cumulative_tokens += prompt_tokens

            turn_metrics = TurnMetrics(
                turn=turn,
                mode=f"{mode}_turn{turn}" if mode == "ppd" and turn == 1 else (
                    "ppd_d_direct" if mode == "ppd" else "pd"
                ),
                total_time_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_tokens_this_turn=input_tokens_per_turn,
                estimated_kv_size_mb=kv_size_mb,
                cumulative_kv_size_mb=self._calculate_kv_size(cumulative_tokens),
                response_text=response_text[:100],
            )

            metrics.turns.append(turn_metrics)
            metrics.total_time_ms += latency_ms
            metrics.total_prompt_tokens += prompt_tokens
            metrics.total_completion_tokens += completion_tokens

            # Update conversation history
            conversation_history = f"User: {new_prompt}\nAssistant:{response_text}"
            if turn > 1:
                conversation_history = f"{conversation_history}\n{conversation_history}"

        return metrics

    def _calculate_kv_size(self, num_tokens: int) -> float:
        """Calculate KV cache size in MB for Llama-3.1-8B."""
        # 32 layers * 2 (K+V) * 8 heads * 128 dim * 2 bytes = 128 KB/token
        kv_bytes = num_tokens * 32 * 2 * 8 * 128 * 2
        return kv_bytes / (1024 * 1024)

    def parse_cache_rates_from_logs(self) -> tuple[float, float]:
        """Parse cache hit rates from decode server logs."""
        decode_log = LOGS_DIR / "decode.log"
        if not decode_log.exists():
            return 0.0, 0.0

        try:
            with open(decode_log, "r") as f:
                content = f.read()

            # Find the last cache hit rate entries
            local_pattern = r"Prefix cache hit rate: ([\d.]+)%"
            external_pattern = r"External prefix cache hit rate: ([\d.]+)%"

            local_matches = re.findall(local_pattern, content)
            external_matches = re.findall(external_pattern, content)

            local_rate = float(local_matches[-1]) if local_matches else 0.0
            external_rate = float(external_matches[-1]) if external_matches else 0.0

            return local_rate, external_rate
        except:
            return 0.0, 0.0

    def run_benchmark_config(
        self,
        config: dict,
    ) -> tuple[ConversationMetrics, ConversationMetrics]:
        """Run a single benchmark configuration in both modes."""

        num_turns = config["turns"]
        input_tokens = config["input_tokens"]
        output_tokens = config["output_tokens"]
        config_name = config.get("name", f"t{num_turns}_i{input_tokens}_o{output_tokens}")

        print(f"\n--- Config: {config_name} ---")
        print(f"    Turns: {num_turns}, Input: {input_tokens}, Output: {output_tokens}")

        # Run PD mode
        print("  Running PD mode...")
        self.set_mode("pd")
        self.clear_state()
        time.sleep(1)

        pd_metrics = self.run_conversation(
            mode="pd",
            num_turns=num_turns,
            input_tokens_per_turn=input_tokens,
            output_tokens_per_turn=output_tokens,
            config_name=config_name,
        )
        pd_local, pd_external = self.parse_cache_rates_from_logs()
        pd_metrics.local_cache_hit_rate = pd_local
        pd_metrics.external_cache_hit_rate = pd_external

        print(f"    PD: {pd_metrics.total_time_ms:.1f}ms total")

        # Wait between modes
        time.sleep(2)

        # Run PPD mode
        print("  Running PPD mode...")
        self.set_mode("ppd")
        self.clear_state()
        time.sleep(1)

        ppd_metrics = self.run_conversation(
            mode="ppd",
            num_turns=num_turns,
            input_tokens_per_turn=input_tokens,
            output_tokens_per_turn=output_tokens,
            config_name=config_name,
        )
        ppd_local, ppd_external = self.parse_cache_rates_from_logs()
        ppd_metrics.local_cache_hit_rate = ppd_local
        ppd_metrics.external_cache_hit_rate = ppd_external

        print(f"    PPD: {ppd_metrics.total_time_ms:.1f}ms total")

        # Get detailed metrics from proxy
        proxy_metrics = self.get_metrics()
        self._enrich_metrics_from_proxy(pd_metrics, ppd_metrics, proxy_metrics)

        return pd_metrics, ppd_metrics

    def run_benchmark_config_multi(
        self,
        config: dict,
        num_runs: int = 3,
        warmup_runs: int = 1,
    ) -> MultiRunResult:
        """Run a benchmark configuration multiple times and aggregate results."""

        num_turns = config["turns"]
        input_tokens = config["input_tokens"]
        output_tokens = config["output_tokens"]
        config_name = config.get("name", f"t{num_turns}_i{input_tokens}_o{output_tokens}")

        print(f"\n--- Config: {config_name} ---")
        print(f"    Turns: {num_turns}, Input: {input_tokens}, Output: {output_tokens}")
        print(f"    Runs: {num_runs} (warmup: {warmup_runs})")

        multi_result = MultiRunResult(
            config_name=config_name,
            num_turns=num_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_runs=num_runs,
        )

        total_iterations = warmup_runs + num_runs

        for i in range(total_iterations):
            is_warmup = i < warmup_runs
            run_label = f"warmup {i+1}" if is_warmup else f"run {i+1-warmup_runs}"

            # Run PD mode
            self.set_mode("pd")
            self.clear_state()
            time.sleep(0.5)

            pd_metrics = self.run_conversation(
                mode="pd",
                num_turns=num_turns,
                input_tokens_per_turn=input_tokens,
                output_tokens_per_turn=output_tokens,
                config_name=config_name,
            )

            time.sleep(1)

            # Run PPD mode
            self.set_mode("ppd")
            self.clear_state()
            time.sleep(0.5)

            ppd_metrics = self.run_conversation(
                mode="ppd",
                num_turns=num_turns,
                input_tokens_per_turn=input_tokens,
                output_tokens_per_turn=output_tokens,
                config_name=config_name,
            )

            if is_warmup:
                print(f"    [{run_label}] PD: {pd_metrics.total_time_ms:.1f}ms, PPD: {ppd_metrics.total_time_ms:.1f}ms (discarded)")
            else:
                multi_result.pd_times_ms.append(pd_metrics.total_time_ms)
                multi_result.ppd_times_ms.append(ppd_metrics.total_time_ms)
                speedup = pd_metrics.total_time_ms / ppd_metrics.total_time_ms if ppd_metrics.total_time_ms > 0 else 0
                print(f"    [{run_label}] PD: {pd_metrics.total_time_ms:.1f}ms, PPD: {ppd_metrics.total_time_ms:.1f}ms, speedup: {speedup:.2f}x")

            time.sleep(1)

        # Calculate statistics
        multi_result.calculate_stats()

        print(f"    => Mean: PD {multi_result.pd_mean_ms:.1f}±{multi_result.pd_std_ms:.1f}ms, "
              f"PPD {multi_result.ppd_mean_ms:.1f}±{multi_result.ppd_std_ms:.1f}ms, "
              f"speedup {multi_result.speedup_mean:.2f}±{multi_result.speedup_std:.2f}x")

        return multi_result

    def _enrich_metrics_from_proxy(
        self,
        pd_metrics: ConversationMetrics,
        ppd_metrics: ConversationMetrics,
        proxy_metrics: list[dict],
    ):
        """Enrich conversation metrics with detailed proxy metrics."""
        # Split proxy metrics by mode
        pd_proxy = [m for m in proxy_metrics if m.get("mode") == "pd"]
        ppd_proxy = [m for m in proxy_metrics if m.get("mode") in ("ppd_turn1", "ppd_d_direct")]

        # Update PD metrics
        for i, pm in enumerate(pd_proxy):
            if i < len(pd_metrics.turns):
                pd_metrics.turns[i].p_prefill_time_ms = pm.get("p_prefill_time_ms", 0)
                pd_metrics.turns[i].d_time_ms = pm.get("d_time_ms", 0)
                pd_metrics.total_p_time_ms += pm.get("p_prefill_time_ms", 0)
                pd_metrics.total_d_time_ms += pm.get("d_time_ms", 0)
                pd_metrics.total_kv_transferred_mb += pm.get("estimated_kv_size_mb", 0)

        # Update PPD metrics
        for i, pm in enumerate(ppd_proxy):
            if i < len(ppd_metrics.turns):
                ppd_metrics.turns[i].p_prefill_time_ms = pm.get("p_prefill_time_ms", 0)
                ppd_metrics.turns[i].d_time_ms = pm.get("d_time_ms", 0)
                ppd_metrics.total_p_time_ms += pm.get("p_prefill_time_ms", 0)
                ppd_metrics.total_d_time_ms += pm.get("d_time_ms", 0)
                if pm.get("mode") == "ppd_turn1":
                    ppd_metrics.total_kv_transferred_mb += pm.get("estimated_kv_size_mb", 0)

    def generate_comparison(
        self,
        pd_results: list[ConversationMetrics],
        ppd_results: list[ConversationMetrics],
    ) -> dict:
        """Generate comparison statistics."""
        comparison = {
            "by_config": [],
            "overall": {},
            "analysis": {},
        }

        total_pd_time = 0
        total_ppd_time = 0
        total_pd_p_time = 0
        total_ppd_p_time = 0
        total_pd_kv = 0
        total_ppd_kv = 0

        pd_wins = 0
        ppd_wins = 0

        for pd, ppd in zip(pd_results, ppd_results):
            speedup = pd.total_time_ms / ppd.total_time_ms if ppd.total_time_ms > 0 else 0
            kv_reduction = 1 - (ppd.total_kv_transferred_mb / pd.total_kv_transferred_mb) if pd.total_kv_transferred_mb > 0 else 0

            if speedup > 1.0:
                ppd_wins += 1
            else:
                pd_wins += 1

            config_comparison = {
                "config": pd.config_name,
                "turns": pd.num_turns,
                "input_tokens": pd.input_tokens_per_turn,
                "output_tokens": pd.output_tokens_per_turn,
                "pd_total_ms": pd.total_time_ms,
                "ppd_total_ms": ppd.total_time_ms,
                "speedup": speedup,
                "winner": "PPD" if speedup > 1.0 else "PD",
                "pd_p_time_ms": pd.total_p_time_ms,
                "ppd_p_time_ms": ppd.total_p_time_ms,
                "p_time_reduction": 1 - (ppd.total_p_time_ms / pd.total_p_time_ms) if pd.total_p_time_ms > 0 else 0,
                "pd_kv_mb": pd.total_kv_transferred_mb,
                "ppd_kv_mb": ppd.total_kv_transferred_mb,
                "kv_reduction": kv_reduction,
                "pd_cache_local": pd.local_cache_hit_rate,
                "pd_cache_external": pd.external_cache_hit_rate,
                "ppd_cache_local": ppd.local_cache_hit_rate,
                "ppd_cache_external": ppd.external_cache_hit_rate,
            }
            comparison["by_config"].append(config_comparison)

            total_pd_time += pd.total_time_ms
            total_ppd_time += ppd.total_time_ms
            total_pd_p_time += pd.total_p_time_ms
            total_ppd_p_time += ppd.total_p_time_ms
            total_pd_kv += pd.total_kv_transferred_mb
            total_ppd_kv += ppd.total_kv_transferred_mb

        comparison["overall"] = {
            "total_pd_time_ms": total_pd_time,
            "total_ppd_time_ms": total_ppd_time,
            "overall_speedup": total_pd_time / total_ppd_time if total_ppd_time > 0 else 0,
            "total_pd_kv_mb": total_pd_kv,
            "total_ppd_kv_mb": total_ppd_kv,
            "kv_transfer_reduction": 1 - (total_ppd_kv / total_pd_kv) if total_pd_kv > 0 else 0,
            "pd_wins": pd_wins,
            "ppd_wins": ppd_wins,
        }

        # Analysis by category
        comparison["analysis"] = self._analyze_results(comparison["by_config"])

        return comparison

    def _analyze_results(self, by_config: list[dict]) -> dict:
        """Analyze patterns in results."""
        analysis = {
            "by_turns": {},
            "by_input_tokens": {},
            "by_output_tokens": {},
            "pd_advantage_cases": [],
            "ppd_advantage_cases": [],
        }

        for cfg in by_config:
            # Track by turns
            turns = cfg["turns"]
            if turns not in analysis["by_turns"]:
                analysis["by_turns"][turns] = {"count": 0, "avg_speedup": 0, "speedups": []}
            analysis["by_turns"][turns]["count"] += 1
            analysis["by_turns"][turns]["speedups"].append(cfg["speedup"])

            # Track by input tokens
            inp = cfg["input_tokens"]
            if inp not in analysis["by_input_tokens"]:
                analysis["by_input_tokens"][inp] = {"count": 0, "avg_speedup": 0, "speedups": []}
            analysis["by_input_tokens"][inp]["count"] += 1
            analysis["by_input_tokens"][inp]["speedups"].append(cfg["speedup"])

            # Track by output tokens
            out = cfg["output_tokens"]
            if out not in analysis["by_output_tokens"]:
                analysis["by_output_tokens"][out] = {"count": 0, "avg_speedup": 0, "speedups": []}
            analysis["by_output_tokens"][out]["count"] += 1
            analysis["by_output_tokens"][out]["speedups"].append(cfg["speedup"])

            # Categorize wins
            if cfg["speedup"] < 1.0:
                analysis["pd_advantage_cases"].append({
                    "config": cfg["config"],
                    "speedup": cfg["speedup"],
                    "turns": cfg["turns"],
                    "input": cfg["input_tokens"],
                    "output": cfg["output_tokens"],
                })
            elif cfg["speedup"] > 1.1:
                analysis["ppd_advantage_cases"].append({
                    "config": cfg["config"],
                    "speedup": cfg["speedup"],
                    "turns": cfg["turns"],
                    "input": cfg["input_tokens"],
                    "output": cfg["output_tokens"],
                })

        # Calculate averages
        for cat in ["by_turns", "by_input_tokens", "by_output_tokens"]:
            for key in analysis[cat]:
                speedups = analysis[cat][key]["speedups"]
                analysis[cat][key]["avg_speedup"] = sum(speedups) / len(speedups) if speedups else 0
                del analysis[cat][key]["speedups"]  # Remove raw data

        return analysis

    def run_full_benchmark(self, configs: list[dict]) -> BenchmarkResult:
        """Run complete benchmark suite."""
        print("=" * 70)
        print("Comprehensive PD vs PPD Benchmark")
        print("=" * 70)
        print(f"Configurations: {len(configs)}")
        print("=" * 70)

        result = BenchmarkResult(
            name="pd_vs_ppd_benchmark",
            timestamp=datetime.now().isoformat(),
            config={"configs": configs},
        )

        for config in configs:
            pd_metrics, ppd_metrics = self.run_benchmark_config(config)
            result.pd_results.append(pd_metrics)
            result.ppd_results.append(ppd_metrics)

        result.comparison = self.generate_comparison(
            result.pd_results, result.ppd_results
        )

        return result

    def run_full_benchmark_multi(
        self,
        configs: list[dict],
        num_runs: int = 3,
        warmup_runs: int = 1,
    ) -> BenchmarkResult:
        """Run complete benchmark suite with multiple runs per config."""
        print("=" * 70)
        print("Comprehensive PD vs PPD Benchmark (Multi-Run)")
        print("=" * 70)
        print(f"Configurations: {len(configs)}")
        print(f"Runs per config: {num_runs} (warmup: {warmup_runs})")
        print("=" * 70)

        result = BenchmarkResult(
            name="pd_vs_ppd_benchmark_multi",
            timestamp=datetime.now().isoformat(),
            config={"configs": configs, "num_runs": num_runs, "warmup_runs": warmup_runs},
        )

        for config in configs:
            multi_result = self.run_benchmark_config_multi(config, num_runs, warmup_runs)
            result.multi_run_results.append(multi_result)

        return result

    def print_report_multi(self, result: BenchmarkResult):
        """Print detailed benchmark report for multi-run results."""
        print("\n" + "=" * 70)
        print("BENCHMARK REPORT: PD vs PPD Mode (Multi-Run)")
        print("=" * 70)

        print("\n1. PER-CONFIGURATION RESULTS (mean ± std):")
        print("-" * 110)
        print(f"{'Config':<25} {'Turns':<6} {'In':<5} {'Out':<5} {'PD (ms)':<18} {'PPD (ms)':<18} {'Speedup':<14} {'Winner':<6}")
        print("-" * 110)

        pd_advantage = []
        ppd_advantage = []

        for mr in result.multi_run_results:
            winner = "PPD" if mr.speedup_mean > 1.0 else "PD"
            pd_str = f"{mr.pd_mean_ms:.1f}±{mr.pd_std_ms:.1f}"
            ppd_str = f"{mr.ppd_mean_ms:.1f}±{mr.ppd_std_ms:.1f}"
            speedup_str = f"{mr.speedup_mean:.2f}±{mr.speedup_std:.2f}x"

            print(f"{mr.config_name:<25} {mr.num_turns:<6} {mr.input_tokens:<5} {mr.output_tokens:<5} "
                  f"{pd_str:<18} {ppd_str:<18} {speedup_str:<14} {winner:<6}")

            if mr.speedup_mean < 1.0:
                pd_advantage.append(mr)
            elif mr.speedup_mean > 1.1:
                ppd_advantage.append(mr)

        print("-" * 110)

        # Overall summary
        total_pd = sum(mr.pd_mean_ms for mr in result.multi_run_results)
        total_ppd = sum(mr.ppd_mean_ms for mr in result.multi_run_results)
        overall_speedup = total_pd / total_ppd if total_ppd > 0 else 0

        print(f"\n2. OVERALL SUMMARY:")
        print("-" * 70)
        print(f"  Total PD Time (mean):   {total_pd:.1f} ms")
        print(f"  Total PPD Time (mean):  {total_ppd:.1f} ms")
        print(f"  Overall Speedup:        {overall_speedup:.2f}x")
        print(f"  PD Wins:                {sum(1 for mr in result.multi_run_results if mr.speedup_mean < 1.0)} configs")
        print(f"  PPD Wins:               {sum(1 for mr in result.multi_run_results if mr.speedup_mean >= 1.0)} configs")

        # PD advantage cases
        print(f"\n3. PD ADVANTAGE CASES (speedup < 1.0):")
        print("-" * 70)
        for mr in sorted(pd_advantage, key=lambda x: x.speedup_mean):
            print(f"  {mr.config_name}: {mr.speedup_mean:.2f}±{mr.speedup_std:.2f}x "
                  f"(turns={mr.num_turns}, in={mr.input_tokens}, out={mr.output_tokens})")

        # PPD strong advantage cases
        print(f"\n4. PPD STRONG ADVANTAGE CASES (speedup > 1.1x):")
        print("-" * 70)
        for mr in sorted(ppd_advantage, key=lambda x: -x.speedup_mean):
            print(f"  {mr.config_name}: {mr.speedup_mean:.2f}±{mr.speedup_std:.2f}x "
                  f"(turns={mr.num_turns}, in={mr.input_tokens}, out={mr.output_tokens})")

        print("\n" + "=" * 70)

    def print_report(self, result: BenchmarkResult):
        """Print detailed benchmark report."""
        print("\n" + "=" * 70)
        print("BENCHMARK REPORT: PD vs PPD Mode")
        print("=" * 70)

        print("\n1. PER-CONFIGURATION RESULTS:")
        print("-" * 90)
        print(f"{'Config':<25} {'Turns':<6} {'Input':<7} {'Output':<7} {'PD(ms)':<10} {'PPD(ms)':<10} {'Speedup':<8} {'Winner':<6}")
        print("-" * 90)

        for comp in result.comparison["by_config"]:
            print(f"{comp['config']:<25} {comp['turns']:<6} {comp['input_tokens']:<7} {comp['output_tokens']:<7} "
                  f"{comp['pd_total_ms']:<10.1f} {comp['ppd_total_ms']:<10.1f} "
                  f"{comp['speedup']:<8.2f}x {comp['winner']:<6}")

        print("-" * 90)

        # Summary stats
        overall = result.comparison["overall"]
        print(f"\n2. OVERALL SUMMARY:")
        print("-" * 70)
        print(f"  Total PD Time:      {overall['total_pd_time_ms']:.1f} ms")
        print(f"  Total PPD Time:     {overall['total_ppd_time_ms']:.1f} ms")
        print(f"  Overall Speedup:    {overall['overall_speedup']:.2f}x")
        print(f"  PD Wins:            {overall['pd_wins']} configs")
        print(f"  PPD Wins:           {overall['ppd_wins']} configs")

        # Analysis
        analysis = result.comparison["analysis"]
        print(f"\n3. ANALYSIS BY CATEGORY:")
        print("-" * 70)

        print("\n  By Turn Count:")
        for turns, data in sorted(analysis["by_turns"].items()):
            print(f"    {turns} turns: avg speedup = {data['avg_speedup']:.2f}x (n={data['count']})")

        print("\n  By Input Tokens:")
        for inp, data in sorted(analysis["by_input_tokens"].items()):
            print(f"    {inp} tokens: avg speedup = {data['avg_speedup']:.2f}x (n={data['count']})")

        print("\n  By Output Tokens:")
        for out, data in sorted(analysis["by_output_tokens"].items()):
            print(f"    {out} tokens: avg speedup = {data['avg_speedup']:.2f}x (n={data['count']})")

        if analysis["pd_advantage_cases"]:
            print(f"\n4. PD ADVANTAGE CASES (speedup < 1.0):")
            print("-" * 70)
            for case in analysis["pd_advantage_cases"]:
                print(f"  {case['config']}: speedup={case['speedup']:.2f}x "
                      f"(turns={case['turns']}, in={case['input']}, out={case['output']})")

        if analysis["ppd_advantage_cases"]:
            print(f"\n5. PPD STRONG ADVANTAGE CASES (speedup > 1.1x):")
            print("-" * 70)
            for case in analysis["ppd_advantage_cases"]:
                print(f"  {case['config']}: speedup={case['speedup']:.2f}x "
                      f"(turns={case['turns']}, in={case['input']}, out={case['output']})")

        print("\n" + "=" * 70)

    def save_results(self, result: BenchmarkResult, output_file: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.results_dir / f"benchmark_{timestamp}.json")

        # Convert to dict for JSON serialization
        result_dict = {
            "name": result.name,
            "timestamp": result.timestamp,
            "config": result.config,
            "pd_results": [asdict(r) for r in result.pd_results],
            "ppd_results": [asdict(r) for r in result.ppd_results],
            "comparison": result.comparison,
        }

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return output_file


def get_default_configs() -> list[dict]:
    """Get default benchmark configurations."""
    configs = []

    # Vary turns (1-20) with fixed tokens
    for turns in [1, 2, 3, 5, 10, 15, 20]:
        configs.append({
            "name": f"turns_{turns}",
            "turns": turns,
            "input_tokens": 50,
            "output_tokens": 30,
        })

    # Vary input tokens with fixed turns
    for input_tokens in [20, 50, 100, 200, 500]:
        configs.append({
            "name": f"input_{input_tokens}",
            "turns": 5,
            "input_tokens": input_tokens,
            "output_tokens": 30,
        })

    # Vary output tokens with fixed turns
    for output_tokens in [10, 30, 50, 100, 200]:
        configs.append({
            "name": f"output_{output_tokens}",
            "turns": 5,
            "input_tokens": 50,
            "output_tokens": output_tokens,
        })

    # Mixed configurations
    mixed_configs = [
        {"turns": 3, "input_tokens": 100, "output_tokens": 50},
        {"turns": 5, "input_tokens": 200, "output_tokens": 100},
        {"turns": 10, "input_tokens": 100, "output_tokens": 30},
        {"turns": 10, "input_tokens": 50, "output_tokens": 100},
        {"turns": 20, "input_tokens": 50, "output_tokens": 20},
    ]
    for i, cfg in enumerate(mixed_configs):
        cfg["name"] = f"mixed_{i+1}"
        configs.append(cfg)

    return configs


def get_extended_configs() -> list[dict]:
    """Get extended benchmark configurations with larger tokens."""
    configs = []

    # Turn variations with larger tokens
    for turns in [1, 2, 3, 5, 10, 15, 20]:
        configs.append({
            "name": f"turns_{turns}_large",
            "turns": turns,
            "input_tokens": 200,
            "output_tokens": 100,
        })

    # Large input tokens (simulate long context)
    for input_tokens in [100, 200, 500, 800, 1000]:
        configs.append({
            "name": f"input_{input_tokens}_5t",
            "turns": 5,
            "input_tokens": input_tokens,
            "output_tokens": 50,
        })

    # Large output tokens (simulate verbose responses)
    for output_tokens in [50, 100, 200, 300, 500]:
        configs.append({
            "name": f"output_{output_tokens}_5t",
            "turns": 5,
            "input_tokens": 100,
            "output_tokens": output_tokens,
        })

    # Single turn with very large tokens (PD should win here)
    for input_tokens in [200, 500, 1000]:
        configs.append({
            "name": f"single_large_in{input_tokens}",
            "turns": 1,
            "input_tokens": input_tokens,
            "output_tokens": 100,
        })

    # Few turns with large context (PD might win)
    for input_tokens in [300, 500, 800]:
        configs.append({
            "name": f"few_turns_large_{input_tokens}",
            "turns": 2,
            "input_tokens": input_tokens,
            "output_tokens": 100,
        })

    # Many turns with small tokens (PPD should win)
    for turns in [5, 10, 15, 20]:
        configs.append({
            "name": f"many_turns_{turns}_small",
            "turns": turns,
            "input_tokens": 30,
            "output_tokens": 20,
        })

    # Extreme cases
    extreme_configs = [
        # Very long single turn
        {"name": "single_very_long", "turns": 1, "input_tokens": 1000, "output_tokens": 200},
        # Many short turns
        {"name": "many_very_short", "turns": 20, "input_tokens": 20, "output_tokens": 10},
        # Balanced medium
        {"name": "balanced_medium", "turns": 10, "input_tokens": 100, "output_tokens": 50},
        # Long output short input
        {"name": "long_output", "turns": 5, "input_tokens": 50, "output_tokens": 300},
        # Short output long input
        {"name": "long_input", "turns": 5, "input_tokens": 500, "output_tokens": 30},
    ]
    configs.extend(extreme_configs)

    return configs


def get_quick_configs() -> list[dict]:
    """Get quick test configurations."""
    return [
        {"name": "quick_1", "turns": 2, "input_tokens": 30, "output_tokens": 20},
        {"name": "quick_2", "turns": 5, "input_tokens": 50, "output_tokens": 30},
        {"name": "quick_3", "turns": 3, "input_tokens": 100, "output_tokens": 50},
    ]


def main():
    parser = argparse.ArgumentParser(description="Comprehensive PD vs PPD Benchmark")
    parser.add_argument("--proxy-url", default="http://localhost:10001")
    parser.add_argument("--model-path",
                        default="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--extended", action="store_true", help="Run extended test with larger tokens")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per config (default: 1)")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup runs (default: 0)")

    args = parser.parse_args()

    # Verify proxy connection
    try:
        resp = requests.get(f"{args.proxy_url}/mode", timeout=5)
        if resp.status_code != 200:
            print(f"Cannot connect to proxy at {args.proxy_url}")
            return
        print(f"Connected to proxy: {resp.json()}")
    except Exception as e:
        print(f"Cannot connect to proxy: {e}")
        print("Start servers with: ./scripts/start_servers.sh ppd --benchmark")
        return

    # Load configurations
    if args.config:
        with open(args.config, "r") as f:
            configs = json.load(f)
    elif args.quick:
        configs = get_quick_configs()
    elif args.extended:
        configs = get_extended_configs()
    else:
        configs = get_default_configs()

    print(f"\nLoaded {len(configs)} configurations")

    # Run benchmark
    benchmark = ComprehensiveBenchmark(
        proxy_url=args.proxy_url,
        model_path=args.model_path,
    )

    # Use multi-run if runs > 1
    if args.runs > 1:
        print(f"Running with {args.runs} runs per config, {args.warmup} warmup runs")
        result = benchmark.run_full_benchmark_multi(configs, args.runs, args.warmup)
        benchmark.print_report_multi(result)
    else:
        result = benchmark.run_full_benchmark(configs)
        benchmark.print_report(result)

    benchmark.save_results(result, args.output)


if __name__ == "__main__":
    main()
