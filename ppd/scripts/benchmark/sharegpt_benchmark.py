#!/usr/bin/env python3
"""
ShareGPT Multi-turn Benchmark for PPD Evaluation.

Tests GPU configurations with real multi-turn conversations from ShareGPT dataset.
Key features:
- Uses actual conversation content and output lengths from ShareGPT
- Records metrics for ALL turns, but aggregates focus on Turn 2+ (T2+)
- Supports dynamic PPD mode testing
- Compatible with comprehensive_benchmark.py metrics format

Usage:
    # Baseline test (no PPD)
    python sharegpt_benchmark.py --config 2P_2D --num-conversations 1000 --qps 4

    # With dynamic PPD mode enabled
    python sharegpt_benchmark.py --config 2P_2D --num-conversations 1000 --qps 4 --enable-ppd

    # Test multiple QPS points
    python sharegpt_benchmark.py --config 2P_2pD --qps 1 2 4 8 --num-conversations 500
"""

import os
import sys
import gc

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

import json
import time
import random
import asyncio
import aiohttp
import argparse
import subprocess
import traceback
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from transformers import AutoTokenizer

from src.config import MODEL_PATH


# Configuration
PROXY_URL = "http://localhost:10001"
SHAREGPT_PATH = Path(PROJECT_DIR) / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json"

# Benchmark parameters
REQUEST_TIMEOUT_SEC = 180
WARMUP_REQUESTS = 5
TEST_POINT_TIMEOUT_SEC = 600
HEALTH_CHECK_TIMEOUT_SEC = 10
SERVER_RESTART_WAIT_SEC = 120

# Recovery parameters (same as comprehensive_benchmark)
MAX_SERVER_RESTARTS = 2
MAX_CONSECUTIVE_FAILURES = 3

# QPS test points (same as comprehensive_benchmark)
QPS_POINTS = [0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20]

# Token limits
MAX_INPUT_TOKENS = 2048
MAX_OUTPUT_TOKENS = 2048
MAX_CONTEXT_TOKENS = 8192


@dataclass
class TurnResult:
    """Result of a single turn."""
    turn: int
    input_tokens: int
    output_tokens: int
    context_tokens: int  # Accumulated context from previous turns
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    success: bool
    completion_tokens: int = 0
    ppd_decision: Optional[bool] = None  # PPD decision (if enabled)
    error: Optional[str] = None


@dataclass
class ConversationResult:
    """Result of a complete conversation."""
    conv_id: str
    num_turns: int
    turns: List[TurnResult] = field(default_factory=list)
    success: bool = True


@dataclass
class LatencyStats:
    """Latency statistics."""
    count: int = 0
    avg_ttft_ms: float = 0
    avg_tpot_ms: float = 0
    avg_e2e_ms: float = 0
    p50_ttft_ms: float = 0
    p50_tpot_ms: float = 0
    p50_e2e_ms: float = 0
    p90_ttft_ms: float = 0
    p90_tpot_ms: float = 0
    p90_e2e_ms: float = 0
    p99_ttft_ms: float = 0
    p99_tpot_ms: float = 0
    p99_e2e_ms: float = 0

    @classmethod
    def from_turns(cls, turns: List[TurnResult]) -> "LatencyStats":
        """Compute stats from turn results."""
        successful = [t for t in turns if t.success]
        if not successful:
            return cls()

        ttfts = [t.ttft_ms for t in successful]
        tpots = [t.tpot_ms for t in successful if t.tpot_ms > 0]
        e2es = [t.e2e_ms for t in successful]

        return cls(
            count=len(successful),
            avg_ttft_ms=float(np.mean(ttfts)),
            avg_tpot_ms=float(np.mean(tpots)) if tpots else 0,
            avg_e2e_ms=float(np.mean(e2es)),
            p50_ttft_ms=float(np.percentile(ttfts, 50)),
            p50_tpot_ms=float(np.percentile(tpots, 50)) if tpots else 0,
            p50_e2e_ms=float(np.percentile(e2es, 50)),
            p90_ttft_ms=float(np.percentile(ttfts, 90)),
            p90_tpot_ms=float(np.percentile(tpots, 90)) if tpots else 0,
            p90_e2e_ms=float(np.percentile(e2es, 90)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            p99_tpot_ms=float(np.percentile(tpots, 99)) if tpots else 0,
            p99_e2e_ms=float(np.percentile(e2es, 99)),
        )


class ShareGPTDataLoader:
    """Loads and processes ShareGPT dataset."""

    def __init__(self, data_path: str, tokenizer_path: str = None):
        self.data_path = Path(data_path)
        self.tokenizer = None
        self.conversations = []

        # Load tokenizer
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            # Try to load from model path
            try:
                model_dir = Path(MODEL_PATH).parent if "/" in MODEL_PATH else Path(MODEL_PATH)
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            except Exception:
                print("Warning: Could not load tokenizer, using character-based estimation")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Fallback: rough estimation (~4 chars per token)
        return len(text) // 4

    def load_data(self, min_turns: int = 2, max_turns: int = 20) -> List[dict]:
        """Load and filter ShareGPT or WildChat conversations.

        Supports two formats:
        - ShareGPT: {from: "human"/"gpt", value: "..."}
        - WildChat: {role: "user"/"assistant", content: "..."}

        Args:
            min_turns: Minimum number of turns (default 2 for multi-turn)
            max_turns: Maximum number of turns to include

        Returns:
            List of processed conversations
        """
        print(f"Loading data from {self.data_path}...")

        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)

        print(f"Total raw conversations: {len(raw_data)}")

        # Detect format from first conversation
        is_wildchat = False
        if raw_data and raw_data[0].get("conversations"):
            first_msg = raw_data[0]["conversations"][0]
            if "role" in first_msg and "content" in first_msg:
                is_wildchat = True
                print("Detected WildChat format")
            else:
                print("Detected ShareGPT format")

        processed = []
        skipped_reasons = {"too_few_turns": 0, "invalid_format": 0, "too_long": 0}

        for conv in raw_data:
            conv_id = conv.get("id", "")
            messages = conv.get("conversations", [])

            # Validate message format
            if len(messages) < min_turns * 2:
                skipped_reasons["too_few_turns"] += 1
                continue

            # Process turns
            turns = []
            context_tokens = 0
            valid = True

            for i in range(0, len(messages) - 1, 2):
                # Each turn: human message + gpt response
                if i + 1 >= len(messages):
                    break

                human_msg = messages[i]
                gpt_msg = messages[i + 1]

                # Validate roles based on format
                if is_wildchat:
                    if human_msg.get("role") != "user" or gpt_msg.get("role") != "assistant":
                        valid = False
                        break
                    human_text = human_msg.get("content", "")
                    gpt_text = gpt_msg.get("content", "")
                else:
                    if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
                        valid = False
                        break
                    human_text = human_msg.get("value", "")
                    gpt_text = gpt_msg.get("value", "")

                if not human_text or not gpt_text:
                    valid = False
                    break

                input_tokens = self.count_tokens(human_text)
                output_tokens = self.count_tokens(gpt_text)

                # Skip if tokens exceed limits
                if input_tokens > MAX_INPUT_TOKENS or output_tokens > MAX_OUTPUT_TOKENS:
                    skipped_reasons["too_long"] += 1
                    valid = False
                    break

                if context_tokens + input_tokens > MAX_CONTEXT_TOKENS:
                    # Stop adding more turns but keep existing ones
                    break

                turns.append({
                    "turn": len(turns) + 1,
                    "human_text": human_text,
                    "gpt_text": gpt_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "context_tokens": context_tokens,
                })

                # Update context for next turn
                context_tokens += input_tokens + output_tokens

                if len(turns) >= max_turns:
                    break

            if not valid:
                skipped_reasons["invalid_format"] += 1
                continue

            if len(turns) >= min_turns:
                processed.append({
                    "id": conv_id,
                    "turns": turns,
                })

        print(f"Processed conversations: {len(processed)}")
        print(f"Skipped: {skipped_reasons}")

        self.conversations = processed
        return processed

    def sample_conversations(self, n: int, seed: int = 42) -> List[dict]:
        """Sample n conversations randomly."""
        random.seed(seed)
        if n >= len(self.conversations):
            return self.conversations.copy()
        return random.sample(self.conversations, n)


async def check_server_health() -> Tuple[bool, Optional[str]]:
    """Check if proxy and backend servers are healthy."""
    try:
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_SEC)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status != 200:
                    return False, f"Proxy returned status {resp.status}"
                data = await resp.json()
                instances = data.get("instances", {})
                total_instances = sum(len(v) for v in instances.values())
                if total_instances == 0:
                    return False, "No backend instances registered"
                return True, None
    except asyncio.TimeoutError:
        return False, "Health check timed out"
    except aiohttp.ClientError as e:
        return False, f"Connection error: {str(e)[:50]}"
    except Exception as e:
        return False, f"Health check failed: {str(e)[:50]}"


async def warmup_servers() -> bool:
    """Send warmup requests."""
    print("  Warming up servers...")
    success_count = 0

    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i in range(WARMUP_REQUESTS):
                try:
                    async with session.post(
                        f"{PROXY_URL}/v1/completions",
                        json={
                            "model": MODEL_PATH,
                            "prompt": f"User: Warmup {i}\nAssistant:",
                            "max_tokens": 10,
                            "temperature": 0.7,
                        }
                    ) as resp:
                        await resp.read()
                        if resp.status == 200:
                            success_count += 1
                except Exception as e:
                    print(f"    Warmup {i}: {e}")

        if success_count >= WARMUP_REQUESTS // 2:
            print(f"  Warmup complete ({success_count}/{WARMUP_REQUESTS} succeeded)")
            return True
        else:
            print(f"  Warmup failed ({success_count}/{WARMUP_REQUESTS} succeeded)")
            return False
    except Exception as e:
        print(f"  Warmup error: {e}")
        return False


async def run_single_turn(
    session: aiohttp.ClientSession,
    conv_id: str,
    turn_num: int,
    human_text: str,
    output_tokens: int,
    context_tokens: int,
    history: str,
) -> Tuple[TurnResult, str]:
    """Run a single turn request using ShareGPT data."""
    # Build prompt
    if turn_num == 1:
        prompt = f"User: {human_text}\nAssistant:"
    else:
        prompt = f"{history}\nUser: {human_text}\nAssistant:"

    request_data = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.8,
        "ignore_eos": True,  # Important: control exact output length
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if turn_num > 1:
        request_data["conv_id"] = conv_id

    start_time = time.perf_counter()
    ttft = 0.0
    completion_tokens = 0
    response_text = ""

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                error_lower = error_text.lower()
                if "out of memory" in error_lower or "oom" in error_lower:
                    error_msg = f"OOM: {error_text[:100]}"
                else:
                    error_msg = f"HTTP {response.status}: {error_text[:100]}"
                return TurnResult(
                    turn=turn_num,
                    input_tokens=len(human_text) // 4,  # Rough estimate
                    output_tokens=output_tokens,
                    context_tokens=context_tokens,
                    ttft_ms=0, tpot_ms=0, e2e_ms=0,
                    success=False, error=error_msg
                ), history

            first_token = False
            buffer = ""

            async for chunk_bytes in response.content.iter_any():
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
                                    ttft = (time.perf_counter() - start_time) * 1000
                                    first_token = True
                                    response_text += text
                            elif chunk.get("choices"):
                                response_text += chunk["choices"][0].get("text", "")
                            if chunk.get("usage"):
                                completion_tokens = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

            end_time = time.perf_counter()
            e2e_ms = (end_time - start_time) * 1000

            # TPOT calculation
            if completion_tokens > 1:
                tpot_ms = (e2e_ms - ttft) / (completion_tokens - 1)
            else:
                tpot_ms = 0

            # Build updated history for next turn
            updated_history = f"{prompt}{response_text}"

            return TurnResult(
                turn=turn_num,
                input_tokens=len(human_text) // 4,
                output_tokens=output_tokens,
                context_tokens=context_tokens,
                ttft_ms=ttft,
                tpot_ms=tpot_ms,
                e2e_ms=e2e_ms,
                success=True,
                completion_tokens=completion_tokens
            ), updated_history

    except asyncio.TimeoutError:
        return TurnResult(
            turn=turn_num,
            input_tokens=len(human_text) // 4,
            output_tokens=output_tokens,
            context_tokens=context_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False, error="Timeout"
        ), history
    except aiohttp.ClientError as e:
        error_msg = str(e)[:100]
        return TurnResult(
            turn=turn_num,
            input_tokens=len(human_text) // 4,
            output_tokens=output_tokens,
            context_tokens=context_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False, error=error_msg
        ), history
    except Exception as e:
        return TurnResult(
            turn=turn_num,
            input_tokens=len(human_text) // 4,
            output_tokens=output_tokens,
            context_tokens=context_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False, error=str(e)[:100]
        ), history


async def run_conversation(
    session: aiohttp.ClientSession,
    conv_data: dict,
    arrival_time: float,
    start_time: float,
) -> ConversationResult:
    """Run a complete multi-turn conversation."""
    # Wait for arrival time
    now = time.perf_counter() - start_time
    if arrival_time > now:
        await asyncio.sleep(arrival_time - now)

    conv_id = conv_data["id"]
    turns_data = conv_data["turns"]

    result = ConversationResult(
        conv_id=conv_id,
        num_turns=len(turns_data),
    )

    history = ""

    for turn_data in turns_data:
        turn_result, history = await run_single_turn(
            session=session,
            conv_id=conv_id,
            turn_num=turn_data["turn"],
            human_text=turn_data["human_text"],
            output_tokens=turn_data["output_tokens"],
            context_tokens=turn_data["context_tokens"],
            history=history,
        )

        result.turns.append(turn_result)

        if not turn_result.success:
            result.success = False
            break  # Stop on failure

    return result


async def run_benchmark(
    conversations: List[dict],
    qps: float,
    duration_sec: int = None,
) -> Tuple[List[ConversationResult], float]:
    """Run benchmark with given conversations and QPS.

    Args:
        conversations: List of conversation data
        qps: Queries (conversations) per second
        duration_sec: Optional fixed duration (if None, uses num conversations)

    Returns:
        (results, duration)
    """
    # Generate Poisson arrivals
    if duration_sec:
        num_conversations = int(qps * duration_sec)
        num_conversations = min(num_conversations, len(conversations))
    else:
        num_conversations = len(conversations)

    arrival_times = []
    current_time = 0
    for i in range(num_conversations):
        interval = random.expovariate(qps) if qps > 0 else 1.0
        current_time += interval
        arrival_times.append(current_time)

    results = []
    start_time = time.perf_counter()

    # Use connection pooling
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, arrival in enumerate(arrival_times):
            conv_data = conversations[i % len(conversations)]
            tasks.append(run_conversation(session, conv_data, arrival, start_time))

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in raw_results:
            if isinstance(r, Exception):
                print(f"    Conversation error: {r}")
                continue
            results.append(r)

    duration = time.perf_counter() - start_time
    return results, duration


def compute_statistics(
    results: List[ConversationResult],
    duration: float,
) -> dict:
    """Compute comprehensive statistics from results.

    Key: Aggregates focus on Turn 2+ (excludes Turn 1).
    """
    all_turns = []
    turn1_turns = []
    turn2plus_turns = []

    for conv in results:
        for turn in conv.turns:
            all_turns.append(turn)
            if turn.turn == 1:
                turn1_turns.append(turn)
            else:
                turn2plus_turns.append(turn)

    # Compute stats
    turn1_stats = LatencyStats.from_turns(turn1_turns)
    turn2plus_stats = LatencyStats.from_turns(turn2plus_turns)
    all_stats = LatencyStats.from_turns(all_turns)

    # Success counts
    successful_turns = sum(1 for t in all_turns if t.success)
    successful_convs = sum(1 for c in results if c.success)
    total_tokens = sum(t.completion_tokens for t in all_turns if t.success)

    # Throughput
    throughput = {
        "conversations_per_sec": len(results) / duration if duration > 0 else 0,
        "requests_per_sec": len(all_turns) / duration if duration > 0 else 0,
        "tokens_per_sec": total_tokens / duration if duration > 0 else 0,
    }

    # SLO attainment (for Turn 2+)
    t2_ttfts = [t.ttft_ms for t in turn2plus_turns if t.success]
    t2_tpots = [t.tpot_ms for t in turn2plus_turns if t.success and t.tpot_ms > 0]
    t2_e2es = [t.e2e_ms for t in turn2plus_turns if t.success]

    slo_attainment = {}
    if t2_ttfts:
        slo_attainment["ttft_100ms"] = sum(1 for t in t2_ttfts if t <= 100) / len(t2_ttfts)
        slo_attainment["ttft_200ms"] = sum(1 for t in t2_ttfts if t <= 200) / len(t2_ttfts)
        slo_attainment["ttft_500ms"] = sum(1 for t in t2_ttfts if t <= 500) / len(t2_ttfts)
    if t2_tpots:
        slo_attainment["tpot_15ms"] = sum(1 for t in t2_tpots if t <= 15) / len(t2_tpots)
        slo_attainment["tpot_30ms"] = sum(1 for t in t2_tpots if t <= 30) / len(t2_tpots)
    if t2_e2es:
        slo_attainment["e2e_5000ms"] = sum(1 for t in t2_e2es if t <= 5000) / len(t2_e2es)
        slo_attainment["e2e_10000ms"] = sum(1 for t in t2_e2es if t <= 10000) / len(t2_e2es)

    return {
        "turn1": asdict(turn1_stats),
        "turn2plus": asdict(turn2plus_stats),  # Main focus
        "all_turns": asdict(all_stats),
        "throughput": throughput,
        "success_rate": {
            "turns": successful_turns / len(all_turns) * 100 if all_turns else 0,
            "conversations": successful_convs / len(results) * 100 if results else 0,
        },
        "counts": {
            "total_conversations": len(results),
            "successful_conversations": successful_convs,
            "total_turns": len(all_turns),
            "turn1_count": len(turn1_turns),
            "turn2plus_count": len(turn2plus_turns),
            "successful_turns": successful_turns,
        },
        "slo_attainment": slo_attainment,
    }


def start_config(config: str, enable_ppd: bool = False, ppd_benchmark_path: str = None,
                 w_ttft: float = 1.0, w_tpot: float = 1.0) -> bool:
    """Start a specific configuration.

    Args:
        config: Configuration name (e.g., "2P_2D")
        enable_ppd: Whether to enable dynamic PPD mode
        ppd_benchmark_path: Path to benchmark data for PPD decision engine
        w_ttft: Weight for TTFT improvement in PPD decision
        w_tpot: Weight for TPOT degradation penalty in PPD decision
    """
    script = Path(PROJECT_DIR) / "scripts" / "server" / f"start_{config}.sh"
    if not script.exists():
        print(f"  ERROR: Script not found: {script}")
        return False

    # Set up environment variables for PPD mode
    env = os.environ.copy()
    if enable_ppd:
        env["ENABLE_PPD_MODE"] = "true"
        env["PPD_BENCHMARK_PATH"] = ppd_benchmark_path or str(Path(PROJECT_DIR) / "results" / "comprehensive")
        env["W_TTFT"] = str(w_ttft)
        env["W_TPOT"] = str(w_tpot)
        print(f"  PPD Mode: ENABLED (w_ttft={w_ttft}, w_tpot={w_tpot})")

    try:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )

        start_time = time.time()
        while time.time() - start_time < 600:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"    {line.rstrip()}")

        if proc.poll() is None:
            proc.kill()
            print("  ERROR: Server startup timed out")
            return False

        return proc.wait() == 0
    except Exception as e:
        print(f"  ERROR: Failed to start server: {e}")
        return False


def stop_config() -> bool:
    """Stop all servers."""
    cleanup_script = Path(PROJECT_DIR) / "scripts" / "server" / "cleanup_all.sh"
    if cleanup_script.exists():
        try:
            result = subprocess.run(
                ["bash", str(cleanup_script)],
                capture_output=True, text=True, timeout=60
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("    WARNING: Cleanup script timed out")
            return False
    return False


async def restart_server(config: str, enable_ppd: bool = False,
                         w_ttft: float = 1.0, w_tpot: float = 1.0) -> bool:
    """Restart server after failure.

    Args:
        config: GPU configuration name
        enable_ppd: Whether to enable PPD mode
        w_ttft: Weight for TTFT improvement in PPD decision
        w_tpot: Weight for TPOT degradation penalty in PPD decision

    Returns:
        True if restart successful
    """
    print(f"\n  Attempting server restart for {config}...")

    # Cleanup first
    stop_config()
    await asyncio.sleep(10)

    # Start server
    if start_config(config, enable_ppd, w_ttft=w_ttft, w_tpot=w_tpot):
        # Wait for server to be ready
        await asyncio.sleep(5)

        # Check health
        healthy, error = await check_server_health()
        if not healthy:
            print(f"  ERROR: Server unhealthy after restart: {error}")
            return False

        # Warmup
        if not await warmup_servers():
            print("  WARNING: Warmup failed after restart, but server is healthy")

        print("  Server restart complete")
        return True

    print("  ERROR: Server restart failed during startup")
    return False


async def run_full_benchmark(
    config: str,
    qps_points: List[float],
    num_conversations: int,
    enable_ppd: bool,
    output_dir: Path,
    skip_startup: bool = False,
    sharegpt_path: Path = None,
    w_ttft: float = 1.0,
    w_tpot: float = 1.0,
):
    """Run full benchmark for a configuration.

    Args:
        config: Configuration name (e.g., "2P_2D")
        qps_points: List of QPS values to test
        num_conversations: Number of conversations to run
        enable_ppd: Whether to enable dynamic PPD mode
        output_dir: Output directory for results
        skip_startup: Skip server startup (assume already running)
        sharegpt_path: Path to ShareGPT/WildChat dataset
        w_ttft: Weight for TTFT improvement in PPD decision
        w_tpot: Weight for TPOT degradation penalty in PPD decision
    """
    print(f"\n{'='*60}")
    print(f"ShareGPT Benchmark: {config}")
    print(f"PPD Mode: {'ENABLED' if enable_ppd else 'DISABLED'}")
    if enable_ppd:
        print(f"PPD Weights: w_ttft={w_ttft}, w_tpot={w_tpot}")
    print(f"{'='*60}")

    # Load ShareGPT data
    data_path = sharegpt_path or SHAREGPT_PATH
    loader = ShareGPTDataLoader(str(data_path))
    all_conversations = loader.load_data(min_turns=2, max_turns=10)

    if len(all_conversations) < num_conversations:
        print(f"Warning: Only {len(all_conversations)} conversations available")
        num_conversations = len(all_conversations)

    conversations = loader.sample_conversations(num_conversations)
    print(f"Sampled {len(conversations)} conversations for testing")

    # Start servers if needed
    if not skip_startup:
        print("\nStarting servers...")
        if not start_config(config, enable_ppd, w_ttft=w_ttft, w_tpot=w_tpot):
            print(f"ERROR: Failed to start {config}")
            return

    # Check health
    healthy, error = await check_server_health()
    if not healthy:
        print(f"ERROR: Server not healthy: {error}")
        return

    # Warmup
    if not await warmup_servers():
        print("WARNING: Warmup had issues, continuing anyway...")

    # Run benchmarks for each QPS with recovery logic
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track failures for recovery
    consecutive_failures = 0
    server_restarts = 0

    for qps in qps_points:
        print(f"\n  Testing QPS={qps}...")

        test_success = False
        try:
            results, duration = await asyncio.wait_for(
                run_benchmark(conversations, qps),
                timeout=TEST_POINT_TIMEOUT_SEC
            )

            stats = compute_statistics(results, duration)

            # Save results
            result_data = {
                "config": config,
                "ppd_enabled": enable_ppd,
                "qps": qps,
                "num_conversations": len(conversations),
                "duration_sec": duration,
                "timestamp": datetime.now().isoformat(),
                **stats,
            }

            filename = f"{config}_{'ppd' if enable_ppd else 'baseline'}_qps{qps}.json"
            filepath = output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)

            # Print summary
            t2_ttft = stats["turn2plus"]["avg_ttft_ms"]
            t2_e2e = stats["turn2plus"]["avg_e2e_ms"]
            t2_count = stats["counts"]["turn2plus_count"]
            success = stats["success_rate"]["turns"]

            print(f"    T2+ TTFT: {t2_ttft:.1f}ms, E2E: {t2_e2e:.1f}ms, "
                  f"Count: {t2_count}, Success: {success:.0f}%")

            # Check if success rate is acceptable (>50% turns succeeded)
            if success >= 50:
                test_success = True
                consecutive_failures = 0
            else:
                print(f"    WARNING: Low success rate ({success:.0f}%)")
                consecutive_failures += 1

            # Force GC
            gc.collect()

        except asyncio.TimeoutError:
            print(f"    TIMEOUT: Test point exceeded {TEST_POINT_TIMEOUT_SEC}s")
            consecutive_failures += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            consecutive_failures += 1

        # Check if server needs restart
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            if server_restarts >= MAX_SERVER_RESTARTS:
                print(f"\n  ERROR: Max server restarts ({MAX_SERVER_RESTARTS}) exceeded")
                print(f"  Skipping remaining QPS points for {config}")
                break

            print(f"\n  Too many consecutive failures ({consecutive_failures}), attempting restart...")
            server_restarts += 1

            if not skip_startup and await restart_server(config, enable_ppd, w_ttft=w_ttft, w_tpot=w_tpot):
                print("  Server restart successful, continuing...")
                consecutive_failures = 0
            else:
                print("  Server restart failed or skipped, stopping benchmark")
                break

        # Health check after each QPS point (if not already restarting)
        if test_success and consecutive_failures == 0:
            healthy, error = await check_server_health()
            if not healthy:
                print(f"  [Health] Server unhealthy after QPS={qps}: {error}")
                if server_restarts >= MAX_SERVER_RESTARTS:
                    print(f"  ERROR: Max server restarts exceeded, stopping")
                    break

                server_restarts += 1
                if not skip_startup and await restart_server(config, enable_ppd, w_ttft=w_ttft, w_tpot=w_tpot):
                    print("  [Health] Server restart successful")
                    consecutive_failures = 0
                else:
                    print("  [Health] Server restart failed, stopping")
                    break

    # Print recovery summary
    if server_restarts > 0:
        print(f"\n  Server restarts during this benchmark: {server_restarts}")

    # Stop servers
    if not skip_startup:
        print("\nStopping servers...")
        stop_config()


async def main():
    parser = argparse.ArgumentParser(description="ShareGPT Multi-turn Benchmark")
    parser.add_argument("--config", type=str, required=True,
                        help="GPU configuration (e.g., 2P_2D, 2P_2pD)")
    parser.add_argument("--num-conversations", type=int, default=1000,
                        help="Number of conversations to test")
    parser.add_argument("--qps", type=float, nargs="+", default=[4.0],
                        help="QPS points to test")
    parser.add_argument("--enable-ppd", action="store_true",
                        help="Enable dynamic PPD mode")
    parser.add_argument("--w-ttft", type=float, default=1.0,
                        help="Weight for TTFT improvement in PPD decision (default: 1.0)")
    parser.add_argument("--w-tpot", type=float, default=1.0,
                        help="Weight for TPOT degradation penalty in PPD decision (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default="results/sharegpt",
                        help="Output directory")
    parser.add_argument("--skip-startup", action="store_true",
                        help="Skip server startup (assume already running)")
    parser.add_argument("--sharegpt-path", type=str, default=None,
                        help="Path to ShareGPT dataset")

    args = parser.parse_args()

    # Use provided path or default
    sharegpt_path = Path(args.sharegpt_path) if args.sharegpt_path else SHAREGPT_PATH

    if not sharegpt_path.exists():
        print(f"ERROR: ShareGPT data not found at {sharegpt_path}")
        return

    output_dir = Path(args.output_dir) / args.config

    print("="*60)
    print("SHAREGPT MULTI-TURN BENCHMARK")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {args.config}")
    print(f"PPD Mode: {'ENABLED' if args.enable_ppd else 'DISABLED'}")
    if args.enable_ppd:
        print(f"PPD Weights: w_ttft={args.w_ttft}, w_tpot={args.w_tpot}")
    print(f"Conversations: {args.num_conversations}")
    print(f"QPS points: {args.qps}")
    print(f"Output: {output_dir}")

    await run_full_benchmark(
        config=args.config,
        qps_points=args.qps,
        num_conversations=args.num_conversations,
        enable_ppd=args.enable_ppd,
        output_dir=output_dir,
        skip_startup=args.skip_startup,
        sharegpt_path=sharegpt_path,
        w_ttft=args.w_ttft,
        w_tpot=args.w_tpot,
    )

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
