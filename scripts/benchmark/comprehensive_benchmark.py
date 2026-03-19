#!/usr/bin/env python3
"""
Comprehensive Benchmark for GPU Configuration Trade-off Analysis

Tests 17 GPU configurations × 18 workloads × 10 QPS points = 3,060 test points.

OOM Resilience Features:
- Per-test-point timeout (5 minutes max)
- Server health check after each test point
- Automatic server restart on failure
- Graceful degradation with error recording
- Connection pooling with proper cleanup

GPU Configurations:
- Pure Replica: 4R
- Pure Disaggregated: 1P_3D, 1P_2D_1pD, 1P_1D_2pD, 1P_3pD, 2P_2D, 2P_1D_1pD, 2P_2pD, 3P_1D, 3P_1pD
- Hybrid: 1R_1P_2D, 1R_1P_1D_1pD, 1R_1P_2pD, 1R_2P_1D, 1R_2P_1pD, 2R_1P_1D, 2R_1P_1pD

Workload Matrix (18 = 2 T1 × 9 T2):
- T1: small (128→128), large (1024→1024)
- T2: tiny, short_gen, long_gen, very_long_gen, small_bal, mid_bal, mid_paste, big_paste, huge_paste

QPS Points: 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20
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
import string
import asyncio
import aiohttp
import argparse
import subprocess
import traceback
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

from ppd.config import MODEL_PATH

PROXY_URL = "http://localhost:10001"

# All 17 GPU configurations
GPU_CONFIGURATIONS = [
    # Pure Replica (1)
    "4R",
    # Pure Disaggregated (9)
    "1P_3D", "1P_2D_1pD", "1P_1D_2pD", "1P_3pD",
    "2P_2D", "2P_1D_1pD", "2P_2pD",
    "3P_1D", "3P_1pD",
    # Hybrid (7)
    "1R_1P_2D", "1R_1P_1D_1pD", "1R_1P_2pD", "1R_2P_1D", "1R_2P_1pD",
    "2R_1P_1D", "2R_1P_1pD",
]

# T1 configurations (first turn context setup)
T1_CONFIGS = {
    "small": {"input": 128, "output": 128},
    "large": {"input": 1024, "output": 1024},
}

# T2 configurations (second turn workload types)
T2_CONFIGS = {
    "tiny":         {"input": 16, "output": 32},      # I/O ratio 0.5
    "short_gen":    {"input": 32, "output": 256},     # I/O ratio 0.125
    "long_gen":     {"input": 32, "output": 512},     # I/O ratio 0.0625
    "very_long_gen":{"input": 64, "output": 1024},    # I/O ratio 0.0625
    "small_bal":    {"input": 64, "output": 64},      # I/O ratio 1.0
    "mid_bal":      {"input": 128, "output": 128},    # I/O ratio 1.0
    "mid_paste":    {"input": 256, "output": 64},     # I/O ratio 4.0
    "big_paste":    {"input": 512, "output": 64},     # I/O ratio 8.0
    "huge_paste":   {"input": 1024, "output": 32},    # I/O ratio 32.0
}

# QPS test points
QPS_POINTS = [0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20]

# Benchmark parameters
DURATION_PER_POINT_SEC = 10
REQUEST_TIMEOUT_SEC = 120
WARMUP_REQUESTS = 5

# OOM Resilience parameters
# Note: QPS=20 + large_very_long_gen takes ~95s, so 600s gives 6x safety margin
TEST_POINT_TIMEOUT_SEC = 600  # 10 minutes max per test point (conservative)
HEALTH_CHECK_TIMEOUT_SEC = 10
SERVER_RESTART_WAIT_SEC = 120
MAX_CONSECUTIVE_FAILURES = 3
MAX_SERVER_RESTARTS = 2

# Inter-workload safety parameters
WORKLOAD_DRAIN_WAIT_SEC = 5    # Wait after each workload to let in-flight requests complete
WORKLOAD_HEALTH_CHECK = True   # Check server health between workloads

# Word pool for realistic text
WORD_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "system", "data", "model", "process", "analysis", "result", "method",
    "research", "information", "development", "performance", "application",
    "technology", "network", "algorithm", "function", "structure", "interface",
    "create", "update", "delete", "execute", "configure", "optimize",
    "implement", "design", "build", "deploy", "monitor", "validate",
    "efficient", "scalable", "robust", "dynamic", "parallel", "distributed",
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TurnResult:
    """Result of a single turn"""
    turn: int
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    success: bool
    completion_tokens: int = 0
    error: Optional[str] = None


@dataclass
class LatencyStats:
    """Latency statistics"""
    count: int = 0
    avg_ttft_ms: float = 0
    avg_tpot_ms: float = 0
    avg_e2e_ms: float = 0
    p50_ttft_ms: float = 0
    p50_tpot_ms: float = 0
    p50_e2e_ms: float = 0
    p99_ttft_ms: float = 0
    p99_tpot_ms: float = 0
    p99_e2e_ms: float = 0

    @classmethod
    def from_turns(cls, turns: List[TurnResult]) -> "LatencyStats":
        """Compute stats from turn results"""
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
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            p99_tpot_ms=float(np.percentile(tpots, 99)) if tpots else 0,
            p99_e2e_ms=float(np.percentile(e2es, 99)),
        )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark point"""
    config: str
    workload: str
    qps: float
    turn1: LatencyStats
    turn2: LatencyStats
    average: LatencyStats
    throughput: Dict[str, float]
    success_rate: float
    total_requests: int
    failed_requests: int
    duration_sec: float
    timestamp: str
    error: Optional[str] = None


# ============================================================================
# Prompt Generation
# ============================================================================

def generate_prompt(num_tokens: int, prefix: str = "", seed: str = "") -> str:
    """Generate a realistic, diverse prompt with approximately num_tokens tokens.

    Uses seed for reproducibility while ensuring diverse content.
    Each prompt has varied structure, content, and vocabulary.
    """
    rng = random.Random(seed) if seed else random.Random()
    target_words = int(num_tokens * 0.75)

    # Expanded vocabulary for diversity
    topics = [
        "coding", "machine learning", "data science", "web development", "cloud computing",
        "cybersecurity", "mobile apps", "database design", "API development", "testing",
        "cooking", "travel", "photography", "music", "sports", "healthcare", "education",
        "finance", "marketing", "legal", "research", "engineering", "design", "writing"
    ]

    # Select topic and question template
    topic = rng.choice(topics)
    templates = [
        f"Explain the concept of {topic} in detail.",
        f"What are the best practices for {topic}?",
        f"Can you provide a comprehensive guide to {topic}?",
        f"I'm working on a {topic} project. Here are the details:",
        f"Help me understand {topic} better.",
        f"What are the latest trends in {topic}?",
    ]

    intro = rng.choice(templates)

    # Generate diverse filler content
    words = []
    remaining_words = target_words - len(intro.split())

    for _ in range(remaining_words):
        if rng.random() < 0.4:
            # Technical/domain words
            words.append(rng.choice(WORD_POOL))
        elif rng.random() < 0.7:
            # Common words
            common = ["with", "from", "about", "through", "after", "before", "when",
                     "where", "what", "how", "why", "can", "will", "should", "would",
                     "also", "just", "now", "then", "here", "there", "make", "get"]
            words.append(rng.choice(common))
        else:
            # Random words for uniqueness
            word_len = rng.randint(3, 10)
            words.append(''.join(rng.choices(string.ascii_lowercase, k=word_len)))

    # Mix sentence structures
    content_parts = [intro]
    chunk_size = rng.randint(8, 15)
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            content_parts.append(chunk + ".")

    content = ' '.join(content_parts)

    # Vary prompt format
    formats = [
        f"User: {prefix}{content}\nAssistant:",
        f"Question: {prefix}{content}\nAnswer:",
        f"<|user|>{prefix}{content}<|assistant|>",
    ]

    return rng.choice(formats)


# ============================================================================
# Server Management
# ============================================================================

def run_cleanup() -> bool:
    """Run comprehensive cleanup"""
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


def start_config(config: str) -> bool:
    """Start a specific configuration"""
    script = Path(PROJECT_DIR) / "scripts" / "server" / f"start_{config}.sh"
    if not script.exists():
        print(f"  ERROR: Script not found: {script}")
        return False

    try:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream output with timeout
        start_time = time.time()
        while time.time() - start_time < 600:  # 10 minute max for startup
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


def stop_config(config: str) -> bool:
    """Stop a specific configuration"""
    return run_cleanup()


async def check_server_health() -> Tuple[bool, Optional[str]]:
    """Check if proxy and backend servers are healthy.

    Returns:
        (is_healthy, error_message)
    """
    try:
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_SEC)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status != 200:
                    return False, f"Proxy returned status {resp.status}"
                data = await resp.json()
                # Check if we have registered instances
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
    """Send warmup requests to trigger NCCL initialization.

    Returns:
        True if warmup succeeded, False otherwise
    """
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


# ============================================================================
# Request Functions
# ============================================================================

async def run_single_turn(
    session: aiohttp.ClientSession,
    conv_id: str,
    turn_num: int,
    input_tokens: int,
    output_tokens: int,
    history: str,
) -> Tuple[TurnResult, str]:
    """Run a single turn request."""
    # Build prompt
    if turn_num == 1:
        prefix = f"CONV_{conv_id[:8]}_T1_"
        prompt = generate_prompt(input_tokens, prefix=prefix, seed=f"{conv_id}_t{turn_num}")
    else:
        prefix = f"T{turn_num}_followup_"
        new_input = generate_prompt(input_tokens, prefix=prefix, seed=f"{conv_id}_t{turn_num}")
        prompt = history + "\n" + new_input

    request_data = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.8,
        "ignore_eos": True,
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
                # Check for OOM indicators
                error_lower = error_text.lower()
                if "out of memory" in error_lower or "oom" in error_lower or "cuda" in error_lower:
                    error_msg = f"OOM: {error_text[:100]}"
                else:
                    error_msg = f"HTTP {response.status}: {error_text[:100]}"
                return TurnResult(
                    turn=turn_num,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=0, tpot_ms=0, e2e_ms=0,
                    success=False,
                    error=error_msg
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

            updated_history = prompt + response_text

            return TurnResult(
                turn=turn_num,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft,
                tpot_ms=tpot_ms,
                e2e_ms=e2e_ms,
                success=True,
                completion_tokens=completion_tokens
            ), updated_history

    except asyncio.TimeoutError:
        return TurnResult(
            turn=turn_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False, error="Timeout"
        ), history
    except aiohttp.ClientError as e:
        error_msg = str(e)[:100]
        if "Connection" in error_msg:
            error_msg = f"ServerDown: {error_msg}"
        return TurnResult(
            turn=turn_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False, error=error_msg
        ), history
    except Exception as e:
        return TurnResult(
            turn=turn_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False, error=str(e)[:100]
        ), history


async def run_conversation(
    session: aiohttp.ClientSession,
    conv_id: str,
    t1_config: Dict,
    t2_config: Dict,
    arrival_time: float,
    start_time: float,
) -> Tuple[List[TurnResult], List[TurnResult]]:
    """Run a complete 2-turn conversation."""
    # Wait for arrival time
    now = time.perf_counter() - start_time
    if arrival_time > now:
        await asyncio.sleep(arrival_time - now)

    t1_results = []
    t2_results = []

    # Turn 1
    t1_result, history = await run_single_turn(
        session, conv_id, 1,
        t1_config["input"], t1_config["output"],
        ""
    )
    t1_results.append(t1_result)

    if not t1_result.success:
        return t1_results, t2_results

    # Turn 2 (burst mode - no delay)
    t2_result, _ = await run_single_turn(
        session, conv_id, 2,
        t2_config["input"], t2_config["output"],
        history
    )
    t2_results.append(t2_result)

    return t1_results, t2_results


async def run_benchmark_point_inner(
    config: str,
    workload: str,
    qps: float,
    t1_config: Dict,
    t2_config: Dict,
) -> BenchmarkResult:
    """Inner function to run a single benchmark point."""
    # Generate Poisson arrivals
    num_conversations = int(qps * DURATION_PER_POINT_SEC)
    arrival_times = []
    current_time = 0
    for _ in range(num_conversations):
        interval = random.expovariate(qps) if qps > 0 else 0.5
        current_time += interval
        if current_time < DURATION_PER_POINT_SEC:
            arrival_times.append(current_time)

    all_t1_results = []
    all_t2_results = []
    start_time = time.perf_counter()

    # Use connection pooling with limits to avoid resource exhaustion
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, arrival in enumerate(arrival_times):
            conv_id = f"{config}_{workload}_{i}_{int(time.time()*1000)}"
            tasks.append(run_conversation(
                session, conv_id, t1_config, t2_config, arrival, start_time
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                # Don't print every exception - could be many
                continue
            t1_list, t2_list = result
            all_t1_results.extend(t1_list)
            all_t2_results.extend(t2_list)

    duration = time.perf_counter() - start_time

    # Compute statistics
    t1_stats = LatencyStats.from_turns(all_t1_results)
    t2_stats = LatencyStats.from_turns(all_t2_results)

    # Combined average
    all_results = all_t1_results + all_t2_results
    avg_stats = LatencyStats.from_turns(all_results)

    # Throughput
    successful_reqs = sum(1 for t in all_results if t.success)
    total_tokens = sum(t.completion_tokens for t in all_results if t.success)
    throughput = {
        "requests_per_sec": successful_reqs / duration if duration > 0 else 0,
        "tokens_per_sec": total_tokens / duration if duration > 0 else 0,
    }

    total_reqs = len(all_t1_results) + len(all_t2_results)
    failed_reqs = total_reqs - successful_reqs

    # Collect error messages
    error_msgs = set()
    for t in all_results:
        if not t.success and t.error:
            error_msgs.add(t.error[:50])
    error_summary = "; ".join(list(error_msgs)[:3]) if error_msgs else None

    return BenchmarkResult(
        config=config,
        workload=workload,
        qps=qps,
        turn1=t1_stats,
        turn2=t2_stats,
        average=avg_stats,
        throughput=throughput,
        success_rate=successful_reqs / total_reqs * 100 if total_reqs > 0 else 0,
        total_requests=total_reqs,
        failed_requests=failed_reqs,
        duration_sec=duration,
        timestamp=datetime.now().isoformat(),
        error=error_summary,
    )


async def run_benchmark_point(
    config: str,
    workload: str,
    qps: float,
    t1_config: Dict,
    t2_config: Dict,
) -> Tuple[BenchmarkResult, bool]:
    """Run a single benchmark point with timeout protection.

    Returns:
        (result, server_healthy) - result and whether server is still healthy
    """
    try:
        # Wrap in timeout
        result = await asyncio.wait_for(
            run_benchmark_point_inner(config, workload, qps, t1_config, t2_config),
            timeout=TEST_POINT_TIMEOUT_SEC
        )

        # Check server health after test
        healthy, _ = await check_server_health()

        # Force garbage collection
        gc.collect()

        return result, healthy

    except asyncio.TimeoutError:
        # Test point timed out - this is bad, likely server is stuck
        print(f"\n    TIMEOUT: Test point exceeded {TEST_POINT_TIMEOUT_SEC}s limit")

        # Create error result
        error_result = BenchmarkResult(
            config=config,
            workload=workload,
            qps=qps,
            turn1=LatencyStats(),
            turn2=LatencyStats(),
            average=LatencyStats(),
            throughput={"requests_per_sec": 0, "tokens_per_sec": 0},
            success_rate=0,
            total_requests=0,
            failed_requests=0,
            duration_sec=TEST_POINT_TIMEOUT_SEC,
            timestamp=datetime.now().isoformat(),
            error="TestPointTimeout",
        )

        # Server is probably unhealthy
        return error_result, False

    except Exception as e:
        print(f"\n    ERROR: {e}")
        traceback.print_exc()

        error_result = BenchmarkResult(
            config=config,
            workload=workload,
            qps=qps,
            turn1=LatencyStats(),
            turn2=LatencyStats(),
            average=LatencyStats(),
            throughput={"requests_per_sec": 0, "tokens_per_sec": 0},
            success_rate=0,
            total_requests=0,
            failed_requests=0,
            duration_sec=0,
            timestamp=datetime.now().isoformat(),
            error=str(e)[:100],
        )

        healthy, _ = await check_server_health()
        return error_result, healthy


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint(checkpoint_file: Path) -> set:
    """Load completed test points from checkpoint"""
    if not checkpoint_file.exists():
        return set()

    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return set(data.get("completed", []))
    except Exception:
        return set()


def save_checkpoint(checkpoint_file: Path, completed: set):
    """Save completed test points to checkpoint"""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump({"completed": list(completed)}, f)


def get_test_point_key(config: str, workload: str, qps: float) -> str:
    """Generate unique key for a test point"""
    return f"{config}_{workload}_{qps}"


# ============================================================================
# Result Storage
# ============================================================================

def save_result(result: BenchmarkResult, output_dir: Path):
    """Save a single benchmark result"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.config}_{result.workload}_{result.qps}.json"
    filepath = output_dir / filename

    # Convert to dict
    result_dict = {
        "config": result.config,
        "workload": result.workload,
        "qps": result.qps,
        "turn1": asdict(result.turn1),
        "turn2": asdict(result.turn2),
        "average": asdict(result.average),
        "throughput": result.throughput,
        "success_rate": result.success_rate,
        "total_requests": result.total_requests,
        "failed_requests": result.failed_requests,
        "duration_sec": result.duration_sec,
        "timestamp": result.timestamp,
        "error": result.error,
    }

    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2)


# ============================================================================
# Main Benchmark Runner
# ============================================================================

async def restart_server(config: str) -> bool:
    """Restart server after failure."""
    print(f"\n  Attempting server restart...")

    # Cleanup first
    run_cleanup()
    await asyncio.sleep(10)

    # Start server
    if start_config(config):
        # Wait for warmup
        await asyncio.sleep(5)
        if await warmup_servers():
            healthy, _ = await check_server_health()
            return healthy

    return False


async def run_config_benchmark(
    config: str,
    workloads: List[str],
    qps_points: List[float],
    output_dir: Path,
    checkpoint_file: Path,
    skip_startup: bool = False,
):
    """Run benchmark for a single configuration with OOM resilience."""
    print(f"\n{'='*60}")
    print(f"Configuration: {config}")
    print(f"{'='*60}")

    # Load checkpoint
    completed = load_checkpoint(checkpoint_file)

    # Start servers if needed
    if not skip_startup:
        print("  Starting servers...")
        if not start_config(config):
            print(f"  ERROR: Failed to start {config}")
            return

    # Check proxy readiness
    healthy, error = await check_server_health()
    if not healthy:
        print(f"  ERROR: Server not healthy: {error}")
        return

    # Warmup
    if not await warmup_servers():
        print("  WARNING: Warmup had issues, continuing anyway...")

    # Track failures for recovery logic
    consecutive_failures = 0
    server_restarts = 0
    workload_count = 0

    # Run all workload × QPS combinations
    for workload in workloads:
        t1_name, t2_name = workload.split("_", 1)
        t1_config = T1_CONFIGS[t1_name]
        t2_config = T2_CONFIGS[t2_name]

        for qps in qps_points:
            key = get_test_point_key(config, workload, qps)
            if key in completed:
                print(f"  [{workload}] QPS={qps}: SKIPPED (already done)")
                continue

            print(f"  [{workload}] QPS={qps}...", end=" ", flush=True)

            # Run benchmark point
            result, server_healthy = await run_benchmark_point(
                config, workload, qps, t1_config, t2_config
            )

            # Save result (even if failed)
            save_result(result, output_dir / config)

            # Update checkpoint
            completed.add(key)
            save_checkpoint(checkpoint_file, completed)

            # Print result
            if result.success_rate > 0:
                print(f"T1={result.turn1.avg_e2e_ms:.1f}ms T2={result.turn2.avg_e2e_ms:.1f}ms "
                      f"({result.success_rate:.0f}% success)")
                consecutive_failures = 0
            else:
                print(f"FAILED: {result.error}")
                consecutive_failures += 1

            # Check if server needs restart
            if not server_healthy or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                if server_restarts >= MAX_SERVER_RESTARTS:
                    print(f"\n  ERROR: Max server restarts ({MAX_SERVER_RESTARTS}) exceeded")
                    print(f"  Skipping remaining tests for {config}")
                    # Mark remaining as completed to avoid retry loops
                    for w in workloads[workloads.index(workload):]:
                        for q in qps_points:
                            remaining_key = get_test_point_key(config, w, q)
                            if remaining_key not in completed:
                                completed.add(remaining_key)
                    save_checkpoint(checkpoint_file, completed)
                    break

                print(f"\n  Server unhealthy or too many failures, restarting...")
                server_restarts += 1

                if await restart_server(config):
                    print("  Server restart successful")
                    consecutive_failures = 0
                else:
                    print("  Server restart failed, skipping config")
                    break
        else:
            # Inner loop (QPS) completed normally for this workload
            workload_count += 1

            # === Inter-workload safety: drain and health check ===
            if WORKLOAD_HEALTH_CHECK and workload_count < len(workloads):
                # 1. Wait for any remaining in-flight requests to complete
                print(f"  [Drain] Waiting {WORKLOAD_DRAIN_WAIT_SEC}s for in-flight requests...")
                await asyncio.sleep(WORKLOAD_DRAIN_WAIT_SEC)

                # 2. Check server health before starting next workload
                healthy, error = await check_server_health()
                if not healthy:
                    print(f"  [Health] Server unhealthy after workload: {error}")
                    if server_restarts >= MAX_SERVER_RESTARTS:
                        print(f"  ERROR: Max server restarts exceeded, stopping")
                        break

                    server_restarts += 1
                    if await restart_server(config):
                        print("  [Health] Server restart successful, continuing")
                        consecutive_failures = 0
                    else:
                        print("  [Health] Server restart failed, stopping")
                        break
                else:
                    print(f"  [Health] Server OK, proceeding to next workload")

            continue
        # Inner loop was broken
        break

    # Stop servers if we started them
    if not skip_startup:
        print("  Stopping servers...")
        stop_config(config)


def build_workload_list() -> List[str]:
    """Build list of all workloads"""
    workloads = []
    for t1_name in T1_CONFIGS:
        for t2_name in T2_CONFIGS:
            workloads.append(f"{t1_name}_{t2_name}")
    return workloads


async def main():
    global TEST_POINT_TIMEOUT_SEC

    parser = argparse.ArgumentParser(description="Comprehensive GPU Configuration Benchmark")
    parser.add_argument("--config", type=str, nargs="+", help="Configs to test (or 'all')")
    parser.add_argument("--workload", type=str, nargs="+", help="Workloads to test (or 'all')")
    parser.add_argument("--qps", type=float, nargs="+", help="QPS points to test")
    parser.add_argument("--output-dir", type=str, default="results/comprehensive")
    parser.add_argument("--skip-startup", action="store_true", help="Skip server startup (assume already running)")
    parser.add_argument("--list-configs", action="store_true", help="List all configurations")
    parser.add_argument("--list-workloads", action="store_true", help="List all workloads")
    parser.add_argument("--test-timeout", type=int, default=TEST_POINT_TIMEOUT_SEC,
                        help=f"Timeout per test point in seconds (default: {TEST_POINT_TIMEOUT_SEC})")
    args = parser.parse_args()

    # Apply configurable timeout
    TEST_POINT_TIMEOUT_SEC = args.test_timeout

    if args.list_configs:
        print("GPU Configurations:")
        for c in GPU_CONFIGURATIONS:
            print(f"  {c}")
        return

    if args.list_workloads:
        print("Workloads (T1_T2):")
        for w in build_workload_list():
            print(f"  {w}")
        return

    # Determine what to test
    if args.config and args.config != ["all"]:
        configs = args.config
    else:
        configs = GPU_CONFIGURATIONS

    if args.workload and args.workload != ["all"]:
        workloads = args.workload
    else:
        workloads = build_workload_list()

    qps_points = args.qps if args.qps else QPS_POINTS

    output_dir = Path(args.output_dir)
    checkpoint_file = output_dir / "checkpoint.json"

    print("="*60)
    print("COMPREHENSIVE BENCHMARK (OOM-Resilient)")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configurations: {len(configs)}")
    print(f"Workloads: {len(workloads)}")
    print(f"QPS points: {qps_points}")
    print(f"Total test points: {len(configs) * len(workloads) * len(qps_points)}")
    print(f"Duration per point: {DURATION_PER_POINT_SEC}s")
    print(f"Test point timeout: {TEST_POINT_TIMEOUT_SEC}s")
    print(f"Output directory: {output_dir}")

    for config in configs:
        await run_config_benchmark(
            config, workloads, qps_points,
            output_dir, checkpoint_file,
            skip_startup=args.skip_startup
        )

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
