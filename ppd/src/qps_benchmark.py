#!/usr/bin/env python3
"""
QPS vs Latency Benchmark for PD vs PPD Mode Comparison (V3)

================================================================================
VERSION 3: 3D Orthogonal Workload Design
================================================================================

Key design principles:
1. Separate T1 (context building) from T2 (incremental request)
2. T2 split into dU2 (new input) and A2 (output length)
3. QPS stratified by workload load characteristics

================================================================================
DIMENSIONS
================================================================================

1. C1 (Context after T1): Controls "return-to-P re-computation cost"
   - S:  512 tokens  (256→256)
   - M:  1024 tokens (512→512)
   - L:  2048 tokens (1024→1024)
   - XL: 4096 tokens (2048→2048)

2. T2 Type (dU2→A2): Controls prefill/decode balance
   - a: 32→64   (tiny followup, short answer)
   - b: 32→512  (short question, long output)
   - c: 256→256 (medium balanced)
   - d: 1024→64 (big paste, short answer)

================================================================================
WORKLOAD MATRIX (16 combinations)
================================================================================

         |  T2_a    |  T2_b     |  T2_c     |  T2_d     |
         | (32→64)  | (32→512)  | (256→256) | (1024→64) |
---------|----------|-----------|-----------|-----------|
T1_S     | S_a      | S_b       | S_c       | S_d       |
(C1=512) |          |           |           |           |
---------|----------|-----------|-----------|-----------|
T1_M     | M_a      | M_b       | M_c       | M_d       |
(C1=1024)|          |           |           |           |
---------|----------|-----------|-----------|-----------|
T1_L     | L_a      | L_b       | L_c       | L_d       |
(C1=2048)|          |           |           |           |
---------|----------|-----------|-----------|-----------|
T1_XL    | XL_a     | XL_b      | XL_c      | XL_d      |
(C1=4096)|          |           |           |           |

================================================================================
T2 TYPE CHARACTERISTICS
================================================================================

| Type | dU2  | A2   | Prefill | Decode | Real-world scenario           |
|------|------|------|---------|--------|-------------------------------|
| a    | 32   | 64   | Light   | Light  | Quick clarification           |
| b    | 32   | 512  | Light   | Heavy  | "Write me a summary"          |
| c    | 256  | 256  | Medium  | Medium | Normal chat continuation      |
| d    | 1024 | 64   | Heavy   | Light  | "Here's a doc, answer briefly"|

================================================================================
QPS STRATIFICATION (by T2 type)
================================================================================

Different T2 types have different sustainable QPS ranges:
- Type a (light):  [0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
- Type b (decode): [0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 12.0]
- Type c (medium): [0.05, 0.1, 1.0, 2.0, 4.0, 8.0]
- Type d (prefill):[0.05, 0.1, 1.0, 2.0, 4.0]

================================================================================
"""

import argparse
import asyncio
import hashlib
import json
import random
import string
import time
import numpy as np
import aiohttp
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

PROXY_URL = "http://localhost:10001"
MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def get_duration_for_qps(base_duration: int, qps: float, is_large_workload: bool = False) -> int:
    """
    Calculate duration based on QPS to ensure sufficient sample count.

    Strategy:
    - Low QPS (< 0.5): 2x base duration (need more time for samples)
    - Medium QPS (0.5 - 2.0): 1.5x base duration
    - High QPS (>= 2.0): base duration (enough samples quickly)

    For L/XL workloads, reduce by 33% to avoid timeout.
    """
    if qps < 0.5:
        duration = base_duration * 2
    elif qps < 2.0:
        duration = int(base_duration * 1.5)
    else:
        duration = base_duration

    # Reduce for large workloads
    if is_large_workload:
        duration = int(duration * 0.67)

    return max(duration, 10)  # Minimum 10s


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    req_id: str
    mode: str
    workload: str
    qps_target: float
    input_len: int
    output_len: int
    start_time: float
    ttft_ms: float
    e2e_latency_ms: float
    output_tokens: int
    throughput_tps: float
    tpot_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    turn: int = 1


@dataclass
class TurnMetrics:
    """Aggregated metrics for a single turn."""
    turn: int = 1
    sample_count: int = 0
    success_count: int = 0

    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p90_ttft: float = 0.0
    p99_ttft: float = 0.0

    avg_tpot: float = 0.0
    p50_tpot: float = 0.0
    p99_tpot: float = 0.0

    avg_e2e: float = 0.0
    p50_e2e: float = 0.0
    p99_e2e: float = 0.0

    avg_throughput_tps: float = 0.0
    total_tokens: int = 0


@dataclass
class TurnConfig:
    """Configuration for a single turn."""
    input_tokens: int
    output_tokens: int


@dataclass
class WorkloadConfig:
    """Configuration for a workload (Turn1 + Turn2 combination)."""
    name: str
    description: str
    turn1: TurnConfig
    turn2: TurnConfig

    @property
    def context_after_turn1(self) -> int:
        """Total context length after Turn 1 completes."""
        return self.turn1.input_tokens + self.turn1.output_tokens


@dataclass
class QPSStepResult:
    """Result for a single QPS step."""
    workload: str
    target_qps: float
    mode: str
    duration_s: int

    # Turn 1 config
    t1_input: int = 0
    t1_output: int = 0

    # Turn 2 config
    t2_input: int = 0
    t2_output: int = 0

    # Turn 1 metrics
    turn1_metrics: Optional[TurnMetrics] = None

    # Turn 2 metrics (main benchmark)
    turn2_metrics: Optional[TurnMetrics] = None

    # Legacy fields for backward compatibility
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    real_qps: float = 0.0

    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p90_ttft: float = 0.0
    p99_ttft: float = 0.0
    min_ttft: float = 0.0
    max_ttft: float = 0.0

    avg_e2e: float = 0.0
    p50_e2e: float = 0.0
    p90_e2e: float = 0.0
    p99_e2e: float = 0.0

    # Total E2E (Turn 1 + Turn 2)
    avg_total_e2e: float = 0.0
    p50_total_e2e: float = 0.0
    p99_total_e2e: float = 0.0

    avg_throughput_tps: float = 0.0
    total_tokens: int = 0

    raw_metrics: list = field(default_factory=list)


# ============================================================================
# Turn Configurations (V3: Orthogonal Design)
# ============================================================================

# T1 Configurations: For building context (C1)
T1_CONFIGS = {
    "S": TurnConfig(input_tokens=256, output_tokens=256),    # C1=512
    "M": TurnConfig(input_tokens=512, output_tokens=512),    # C1=1024
    "L": TurnConfig(input_tokens=1024, output_tokens=1024),  # C1=2048
    "XL": TurnConfig(input_tokens=2048, output_tokens=2048), # C1=4096
}

# T2 Types: Different prefill/decode balance scenarios
T2_TYPES = {
    "a": TurnConfig(input_tokens=32, output_tokens=64),    # Tiny followup
    "b": TurnConfig(input_tokens=32, output_tokens=512),   # Short Q, long output
    "c": TurnConfig(input_tokens=256, output_tokens=256),  # Medium balanced
    "d": TurnConfig(input_tokens=1024, output_tokens=64),  # Big paste, short answer
}

# Base QPS lists by T1 size (for _a, _b, _c workloads)
QPS_BY_T1_BASE = {
    "S":  [0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0],  # 8 points
    "M":  [0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 10.0, 12.0],  # 8 points
    "L":  [0.05, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0],         # 7 points
    "XL": [0.05, 0.1, 1.0, 2.0, 4.0],                   # 5 points
}

# Extended QPS lists for _d workloads (prefill-heavy, where PD TPOT advantage shows)
QPS_BY_T1_EXTENDED = {
    "S":  [0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 17.0, 18.0, 19.0, 20.0],  # 12 points
    "M":  [0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0],  # 12 points
    "L":  [0.05, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0],          # 11 points
    "XL": [0.05, 0.1, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0],                       # 9 points
}

# Combined for backward compatibility (used in output metadata)
QPS_BY_T1 = QPS_BY_T1_EXTENDED


def get_qps_list_for_workload(workload_name: str) -> list[float]:
    """Get QPS list based on workload's T1 size and T2 type.

    _d workloads (prefill-heavy) use extended QPS range to capture PD TPOT advantage.
    Other workloads (_a, _b, _c) use base QPS range.
    """
    parts = workload_name.split("_")
    if len(parts) != 2:
        return QPS_BY_T1_BASE["M"]  # Default to M base

    t1_type, t2_type = parts[0], parts[1]

    # _d workloads get extended QPS range
    if t2_type == "d":
        return QPS_BY_T1_EXTENDED.get(t1_type, QPS_BY_T1_EXTENDED["M"])
    else:
        return QPS_BY_T1_BASE.get(t1_type, QPS_BY_T1_BASE["M"])


def generate_workloads() -> list[WorkloadConfig]:
    """Generate all 16 workload combinations (4 T1 x 4 T2)."""
    workloads = []

    for t1_name, t1_cfg in T1_CONFIGS.items():
        for t2_name, t2_cfg in T2_TYPES.items():
            name = f"{t1_name}_{t2_name}"

            # Generate description with context info
            c1 = t1_cfg.input_tokens + t1_cfg.output_tokens
            desc = (f"C1={c1} ({t1_cfg.input_tokens}→{t1_cfg.output_tokens}) + "
                   f"T2({t2_cfg.input_tokens}→{t2_cfg.output_tokens})")

            workloads.append(WorkloadConfig(
                name=name,
                description=desc,
                turn1=t1_cfg,
                turn2=t2_cfg,
            ))

    return workloads


WORKLOADS = generate_workloads()

# Legacy: Default QPS sweep (used when --qps not specified)
QPS_SWEEP = [0.05, 0.1, 1.0, 2.0, 4.0, 8.0]


# ============================================================================
# Random Text Generation
# ============================================================================

# Word pool for generating realistic-looking random text
WORD_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "system", "data", "model", "process", "analysis", "result", "method", "study",
    "research", "information", "development", "performance", "application", "design",
    "technology", "network", "algorithm", "function", "structure", "interface",
    "parameter", "variable", "implementation", "optimization", "evaluation", "framework",
    "architecture", "component", "module", "service", "database", "server", "client",
    "request", "response", "protocol", "configuration", "deployment", "monitoring",
    "security", "authentication", "authorization", "encryption", "validation", "testing",
]


def generate_random_text(num_tokens: int, seed: str = "") -> str:
    """
    Generate completely random text with approximately the specified number of tokens.

    Args:
        num_tokens: Target number of tokens (approximate)
        seed: Optional seed for reproducibility within a conversation

    Returns:
        Random text string
    """
    if seed:
        # Use seed for reproducibility within same conversation
        rng = random.Random(hashlib.md5(seed.encode()).hexdigest())
    else:
        rng = random.Random()

    words = []
    # Rough estimate: 1 token ≈ 0.75 words for English text
    target_words = int(num_tokens * 0.75)

    for i in range(target_words):
        if rng.random() < 0.7:
            # 70% chance: use common word
            word = rng.choice(WORD_POOL)
        else:
            # 30% chance: generate random word (3-10 chars)
            word_len = rng.randint(3, 10)
            word = ''.join(rng.choices(string.ascii_lowercase, k=word_len))

        # Occasionally add punctuation
        if i > 0 and rng.random() < 0.1:
            words[-1] += rng.choice([',', '.', ';', ':'])

        words.append(word)

    # Join with spaces and capitalize first letter
    text = ' '.join(words)
    if text:
        text = text[0].upper() + text[1:]

    return text


def generate_unique_prompt(num_tokens: int, unique_id: str) -> str:
    """Generate a unique prompt with random text."""
    # Use unique_id as seed to ensure same conversation gets consistent text
    random_text = generate_random_text(num_tokens, seed=unique_id)
    return f"[ID:{unique_id[:16]}] {random_text}"


# ============================================================================
# API Helpers
# ============================================================================

async def set_mode(session: aiohttp.ClientSession, mode: str) -> bool:
    """Set the proxy routing mode."""
    try:
        async with session.post(f"{PROXY_URL}/mode/{mode}") as resp:
            return resp.status == 200
    except Exception:
        return False


async def clear_state(session: aiohttp.ClientSession):
    """Clear proxy state."""
    try:
        await session.post(f"{PROXY_URL}/conversations/clear")
        await session.post(f"{PROXY_URL}/metrics/clear")
    except Exception:
        pass


async def nccl_warmup(session: aiohttp.ClientSession) -> bool:
    """Warmup NCCL connections for both PD and PPD modes."""
    print("\n" + "=" * 50)
    print("NCCL Warmup Phase (Extended)")
    print("=" * 50)
    print("  Sending multiple warmup requests to fully establish NCCL connections...")

    success = True
    WARMUP_ROUNDS = 5  # Multiple rounds to ensure all NCCL paths are hot

    for mode in ["pd", "ppd"]:
        await set_mode(session, mode)
        await asyncio.sleep(0.5)

        # Send multiple concurrent warmup requests
        for round_idx in range(WARMUP_ROUNDS):
            warmup_tasks = []
            for i in range(4):  # 4 concurrent requests per round
                conv_id = f"warmup_{mode}_{round_idx}_{i}_{int(time.time())}"
                warmup_tasks.append(
                    do_turn1(
                        session, conv_id,
                        input_tokens=128, output_tokens=64,
                        mode=mode, collect_metrics=False
                    )
                )

            results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if isinstance(r, tuple) and r[0])
            if success_count < 2:
                print(f"  Warning: {mode.upper()} warmup round {round_idx+1} had low success rate")

            await asyncio.sleep(0.3)

        print(f"  {mode.upper()}: Warmup complete ({WARMUP_ROUNDS} rounds x 4 requests)")

        # Also do Turn 2 warmup for one conversation
        conv_id = f"warmup_{mode}_t2_{int(time.time())}"
        history, _ = await do_turn1(
            session, conv_id,
            input_tokens=128, output_tokens=64,
            mode=mode, collect_metrics=False
        )
        if not history:
            print(f"  Warning: {mode.upper()} Turn 1 warmup failed")
            success = False
            continue

        # Turn 2 warmup
        prompt = generate_unique_prompt(128, f"{conv_id}_t2")
        full_prompt = f"{history}\nUser: {prompt}\nAssistant:"

        try:
            async with session.post(
                f"{PROXY_URL}/v1/completions",
                json={
                    "model": MODEL_PATH,
                    "prompt": full_prompt,
                    "max_tokens": 64,
                    "temperature": 0.8,
                    "stream": False,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 200:
                    await resp.json()
                    print(f"  {mode.upper()}: NCCL warmup complete")
                else:
                    print(f"  Warning: {mode.upper()} Turn 2 warmup failed (status {resp.status})")
                    success = False
        except Exception as e:
            print(f"  Warning: {mode.upper()} warmup error: {e}")
            success = False

        await asyncio.sleep(1)

    await clear_state(session)
    print("  Warmup complete. Cleared state for fresh benchmark start.")
    print("=" * 50 + "\n")

    return success


# ============================================================================
# Turn Execution
# ============================================================================

# Timeout for individual requests - 5 minutes should be enough for any single request
# XL workloads with 1024 output tokens at ~30 tok/s = ~35s, add safety margin
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=5 * 60)


async def check_server_alive(session: aiohttp.ClientSession) -> bool:
    """Quick health check - returns True if server is responding."""
    try:
        async with session.get(
            f"{PROXY_URL}/status",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def restart_servers_sync() -> bool:
    """Restart servers synchronously. Returns True if successful."""
    import subprocess
    import os

    project_dir = Path(__file__).parent.parent
    stop_script = project_dir / "scripts" / "stop_servers_4gpu.sh"
    start_script = project_dir / "scripts" / "start_servers_4gpu.sh"

    print("      Restarting servers...")

    try:
        # Stop servers
        print("        Stopping servers...")
        result = subprocess.run(
            [str(stop_script)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"        Warning: stop script returned {result.returncode}")

        time.sleep(5)

        # Start servers
        print("        Starting servers (this may take 2-3 minutes)...")
        env = os.environ.copy()
        result = subprocess.run(
            [str(start_script), "ppd"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )

        if result.returncode != 0:
            print(f"        Error: start script returned {result.returncode}")
            print(f"        stderr: {result.stderr[:500]}")
            return False

        # Wait for server to be ready
        print("        Waiting for server to be ready...")
        time.sleep(120)

        return True

    except subprocess.TimeoutExpired:
        print("        Error: Server restart timed out")
        return False
    except Exception as e:
        print(f"        Error restarting servers: {e}")
        return False


async def ensure_server_healthy(session: aiohttp.ClientSession, mode_name: str) -> bool:
    """Check server health and restart if needed. Returns True if server is healthy."""
    if await check_server_alive(session):
        return True

    print(f"\n      Server appears dead before {mode_name} mode. Attempting restart...")

    # Run restart in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, restart_servers_sync)

    if success:
        # Verify server is responding
        await asyncio.sleep(10)
        if await check_server_alive(session):
            print(f"      Server restarted successfully!")
            return True
        else:
            print(f"      Server still not responding after restart")
            return False
    else:
        print(f"      Failed to restart server")
        return False


async def do_turn1(
    session: aiohttp.ClientSession,
    conv_id: str,
    input_tokens: int,
    output_tokens: int,
    mode: str = "ppd",
    workload: str = "",
    qps: float = 0.0,
    collect_metrics: bool = False
) -> tuple[str, Optional[RequestMetrics]]:
    """
    Execute Turn 1 with streaming to get proper TTFT measurement.

    Returns:
        tuple: (history_string, metrics or None)
    """
    # Generate random prompt for Turn 1
    prompt = generate_unique_prompt(input_tokens, f"{conv_id}_t1")
    full_prompt = f"User: {prompt}\nAssistant:"

    start_t = time.perf_counter()
    ttft = 0.0
    first_token = False
    tokens_received = 0
    response_text = ""

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": full_prompt,
                "max_tokens": output_tokens,
                "temperature": 0.8,
                "stream": True,  # Use streaming for proper TTFT
                "stream_options": {"include_usage": True},
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                return "", None

            buffer = ""
            async for chunk_bytes in resp.content.iter_any():
                if not chunk_bytes:
                    continue
                buffer += chunk_bytes.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            # Detect first token for TTFT
                            if not first_token and chunk.get("choices"):
                                text = chunk["choices"][0].get("text", "")
                                if text:
                                    ttft = (time.perf_counter() - start_t) * 1000
                                    first_token = True
                            # Accumulate response text
                            if chunk.get("choices"):
                                text = chunk["choices"][0].get("text", "")
                                if text:
                                    response_text += text
                            # Get token count from usage (final chunk)
                            if chunk.get("usage"):
                                tokens_received = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

            end_t = time.perf_counter()
            latency = (end_t - start_t) * 1000

            history = f"{full_prompt}{response_text}"

            if collect_metrics:
                tps = (tokens_received / (latency / 1000)) if (latency > 0 and tokens_received > 0) else 0
                tpot = (latency - ttft) / max(tokens_received - 1, 1) if tokens_received > 1 else 0

                metrics = RequestMetrics(
                    req_id=f"{conv_id}_t1",
                    mode=mode,
                    workload=workload,
                    qps_target=qps,
                    input_len=input_tokens,
                    output_len=output_tokens,
                    start_time=start_t,
                    ttft_ms=ttft,
                    e2e_latency_ms=latency,
                    output_tokens=tokens_received,
                    throughput_tps=tps,
                    tpot_ms=tpot,
                    success=True,
                    turn=1
                )
                return history, metrics

            return history, None

    except Exception as e:
        return "", None


async def do_turn2(
    session: aiohttp.ClientSession,
    req_id: str,
    mode: str,
    workload: str,
    qps: float,
    input_tokens: int,
    output_tokens: int,
    history: str,
) -> RequestMetrics:
    """
    Execute Turn 2 with streaming and measure metrics.

    Token counting: ONLY use usage.completion_tokens from the final chunk.
    Do NOT count streaming chunks as tokens (chunks != tokens).
    """
    # Generate random prompt for Turn 2
    prompt = generate_unique_prompt(input_tokens, req_id)
    full_prompt = f"{history}\nUser: {prompt}\nAssistant:"

    start_t = time.perf_counter()
    ttft = 0.0
    first_token = False
    tokens_received = 0  # Will be set from usage.completion_tokens only

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": full_prompt,
                "max_tokens": output_tokens,
                "temperature": 0.8,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=AIOHTTP_TIMEOUT,
        ) as resp:
            if resp.status != 200:
                return RequestMetrics(
                    req_id=req_id, mode=mode, workload=workload, qps_target=qps,
                    input_len=input_tokens, output_len=output_tokens, start_time=start_t,
                    ttft_ms=0, e2e_latency_ms=0, output_tokens=0, throughput_tps=0,
                    success=False, error=f"HTTP {resp.status}", turn=2
                )

            buffer = ""
            async for chunk_bytes in resp.content.iter_any():
                if not chunk_bytes:
                    continue
                buffer += chunk_bytes.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            # Detect first token for TTFT
                            if not first_token and chunk.get("choices"):
                                if chunk["choices"][0].get("text"):
                                    ttft = (time.perf_counter() - start_t) * 1000
                                    first_token = True
                            # Get token count ONLY from usage (final chunk)
                            # Do NOT increment for each chunk - that counts chunks, not tokens!
                            if chunk.get("usage"):
                                tokens_received = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

    except asyncio.TimeoutError:
        return RequestMetrics(
            req_id=req_id, mode=mode, workload=workload, qps_target=qps,
            input_len=input_tokens, output_len=output_tokens, start_time=start_t,
            ttft_ms=0, e2e_latency_ms=0, output_tokens=0, throughput_tps=0,
            success=False, error="Timeout", turn=2
        )
    except Exception as e:
        return RequestMetrics(
            req_id=req_id, mode=mode, workload=workload, qps_target=qps,
            input_len=input_tokens, output_len=output_tokens, start_time=start_t,
            ttft_ms=0, e2e_latency_ms=0, output_tokens=0, throughput_tps=0,
            success=False, error=str(e), turn=2
        )

    end_t = time.perf_counter()
    latency = (end_t - start_t) * 1000

    # Calculate throughput and TPOT only if we have valid token count
    tps = (tokens_received / (latency / 1000)) if (latency > 0 and tokens_received > 0) else 0
    tpot = (latency - ttft) / max(tokens_received - 1, 1) if tokens_received > 1 else 0

    return RequestMetrics(
        req_id=req_id, mode=mode, workload=workload, qps_target=qps,
        input_len=input_tokens, output_len=output_tokens, start_time=start_t,
        ttft_ms=ttft, e2e_latency_ms=latency, output_tokens=tokens_received,
        throughput_tps=tps, tpot_ms=tpot, success=True, turn=2
    )


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_turn_metrics(metrics_list: list[RequestMetrics], turn: int) -> TurnMetrics:
    """Compute aggregated metrics for a specific turn."""
    result = TurnMetrics(turn=turn)
    turn_metrics = [m for m in metrics_list if m.turn == turn]
    successful = [m for m in turn_metrics if m.success]

    result.sample_count = len(turn_metrics)
    result.success_count = len(successful)

    if not successful:
        return result

    ttfts = [m.ttft_ms for m in successful if m.ttft_ms > 0]
    tpots = [m.tpot_ms for m in successful if m.tpot_ms > 0]
    e2es = [m.e2e_latency_ms for m in successful]
    throughputs = [m.throughput_tps for m in successful]

    if ttfts:
        result.avg_ttft = np.mean(ttfts)
        result.p50_ttft = np.percentile(ttfts, 50)
        result.p90_ttft = np.percentile(ttfts, 90)
        result.p99_ttft = np.percentile(ttfts, 99)

    if tpots:
        result.avg_tpot = np.mean(tpots)
        result.p50_tpot = np.percentile(tpots, 50)
        result.p99_tpot = np.percentile(tpots, 99)

    if e2es:
        result.avg_e2e = np.mean(e2es)
        result.p50_e2e = np.percentile(e2es, 50)
        result.p99_e2e = np.percentile(e2es, 99)

    if throughputs:
        result.avg_throughput_tps = np.mean(throughputs)

    result.total_tokens = sum(m.output_tokens for m in successful)

    return result


# ============================================================================
# Benchmark Execution (with Conversation Reuse)
# ============================================================================

async def establish_conversations(
    session: aiohttp.ClientSession,
    workload: WorkloadConfig,
    num_conversations: int,
    mode: str,
) -> tuple[dict[str, str], list[RequestMetrics], dict[str, float]]:
    """
    Establish conversations (Turn 1) for later reuse.
    Returns: (histories dict, turn1 metrics list, turn1 e2e dict per conversation)
    """
    histories = {}
    turn1_metrics = []
    turn1_e2e_per_conv = {}  # Store Turn 1 E2E for each conversation

    batch_size = 10
    for batch_start in range(0, num_conversations, batch_size):
        batch_end = min(batch_start + batch_size, num_conversations)
        batch_tasks = []
        batch_conv_ids = []

        for i in range(batch_start, batch_end):
            conv_id = f"{workload.name}_{mode}_{i}_{int(time.time())}"
            batch_conv_ids.append(conv_id)
            batch_tasks.append(
                do_turn1(
                    session, conv_id,
                    input_tokens=workload.turn1.input_tokens,
                    output_tokens=workload.turn1.output_tokens,
                    mode=mode,
                    workload=workload.name,
                    qps=0.0,
                    collect_metrics=True
                )
            )

        results_batch = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for conv_id, res in zip(batch_conv_ids, results_batch):
            if isinstance(res, tuple) and len(res) == 2:
                hist, metrics = res
                if hist:
                    histories[conv_id] = hist
                if metrics:
                    turn1_metrics.append(metrics)
                    turn1_e2e_per_conv[conv_id] = metrics.e2e_latency_ms

        await asyncio.sleep(0.3)

    return histories, turn1_metrics, turn1_e2e_per_conv


async def run_turn2_benchmark(
    session: aiohttp.ClientSession,
    workload: WorkloadConfig,
    qps: float,
    duration_s: int,
    mode: str,
    histories: dict[str, str],
    turn1_metrics: TurnMetrics,
    turn1_e2e_per_conv: dict[str, float],
) -> QPSStepResult:
    """Run Turn 2 benchmark using pre-established conversations."""
    mode_str = mode.upper()
    print(f"    [{mode_str}] Target QPS: {qps}, Duration: {duration_s}s...")

    result = QPSStepResult(
        workload=workload.name,
        target_qps=qps,
        mode=mode,
        duration_s=duration_s,
        t1_input=workload.turn1.input_tokens,
        t1_output=workload.turn1.output_tokens,
        t2_input=workload.turn2.input_tokens,
        t2_output=workload.turn2.output_tokens,
    )

    # Store Turn 1 metrics (shared across all QPS points)
    result.turn1_metrics = turn1_metrics

    # Generate Poisson arrival times
    expected_requests = max(int(qps * duration_s), 1)
    max_attempts = max(expected_requests * 3, 10)
    arrival_intervals = np.random.exponential(1.0 / qps, max_attempts)
    arrival_times = np.cumsum(arrival_intervals)
    arrival_times = arrival_times[arrival_times < duration_s]

    if len(arrival_times) == 0:
        arrival_times = np.array([0.0])

    actual_requests = min(len(arrival_times), len(histories))
    conv_ids = list(histories.keys())[:actual_requests]
    turn2_metrics = []
    tasks = []

    start_benchmark = time.perf_counter()

    for i, (delay, conv_id) in enumerate(zip(arrival_times, conv_ids)):
        now = time.perf_counter() - start_benchmark
        wait = delay - now
        if wait > 0:
            await asyncio.sleep(wait)

        req_id = f"{conv_id}_t2"
        task = asyncio.create_task(
            do_turn2(
                session, req_id, mode, workload.name, qps,
                workload.turn2.input_tokens, workload.turn2.output_tokens,
                histories.get(conv_id, "")
            )
        )
        tasks.append((conv_id, task))

    # Gather results and compute total E2E per conversation
    total_e2es = []
    for conv_id, task in tasks:
        try:
            r = await task
            if isinstance(r, RequestMetrics):
                turn2_metrics.append(r)
                # Calculate total E2E = Turn 1 E2E + Turn 2 E2E
                t1_e2e = turn1_e2e_per_conv.get(conv_id, 0)
                total_e2e = t1_e2e + r.e2e_latency_ms
                total_e2es.append(total_e2e)
        except Exception:
            pass

    # Compute Turn 2 metrics
    result.turn2_metrics = compute_turn_metrics(turn2_metrics, turn=2)

    # Fill legacy fields from Turn 2
    successful = [m for m in turn2_metrics if m.success]
    failed = [m for m in turn2_metrics if not m.success]

    result.sample_count = len(turn2_metrics)
    result.success_count = len(successful)
    result.failure_count = len(failed)
    result.real_qps = len(successful) / duration_s if duration_s > 0 else 0

    if successful:
        ttfts = [m.ttft_ms for m in successful if m.ttft_ms > 0]
        e2es = [m.e2e_latency_ms for m in successful]
        throughputs = [m.throughput_tps for m in successful]

        if ttfts:
            result.avg_ttft = np.mean(ttfts)
            result.p50_ttft = np.percentile(ttfts, 50)
            result.p90_ttft = np.percentile(ttfts, 90)
            result.p99_ttft = np.percentile(ttfts, 99)
            result.min_ttft = np.min(ttfts)
            result.max_ttft = np.max(ttfts)

        if e2es:
            result.avg_e2e = np.mean(e2es)
            result.p50_e2e = np.percentile(e2es, 50)
            result.p90_e2e = np.percentile(e2es, 90)
            result.p99_e2e = np.percentile(e2es, 99)

        if throughputs:
            result.avg_throughput_tps = np.mean(throughputs)

        result.total_tokens = sum(m.output_tokens for m in successful)
        result.raw_metrics = [asdict(m) for m in successful[:10]]

        # Compute Total E2E (Turn 1 + Turn 2)
        if total_e2es:
            result.avg_total_e2e = np.mean(total_e2es)
            result.p50_total_e2e = np.percentile(total_e2es, 50)
            result.p99_total_e2e = np.percentile(total_e2es, 99)

    # Mark unreliable results (survivorship bias when success rate < 50%)
    success_rate = result.success_count / result.sample_count if result.sample_count > 0 else 0
    unreliable = success_rate < 0.5 and result.sample_count >= 10
    unreliable_mark = " [UNRELIABLE - survivorship bias]" if unreliable else ""

    print(f"      Complete. Success: {result.success_count}/{result.sample_count} ({success_rate:.0%}), "
          f"P99 TTFT: {result.p99_ttft:.2f}ms, Avg E2E: {result.avg_e2e:.2f}ms{unreliable_mark}")

    return result


async def run_workload_sweep(
    workload: WorkloadConfig,
    qps_list: list[float],
    base_duration_s: int = 30,
) -> list[QPSStepResult]:
    """Run a complete QPS sweep for a workload (both PD and PPD) with conversation reuse."""
    print(f"\n{'#'*70}")
    print(f"WORKLOAD: {workload.name}")
    print(f"  {workload.description}")
    print(f"  Context after T1: {workload.context_after_turn1} tokens")
    print(f"{'#'*70}")

    results = []

    # Check if this is a large workload (L or XL)
    is_large = "L" in workload.name or "XL" in workload.name

    # Calculate max conversations needed (consider varying durations per QPS)
    max_conversations = 0
    for qps in qps_list:
        duration = get_duration_for_qps(base_duration_s, qps, is_large)
        conv_needed = int(qps * duration)
        max_conversations = max(max_conversations, conv_needed)
    max_conversations = max(max_conversations, 10)

    # Print duration plan
    print(f"\n  Duration plan (base={base_duration_s}s, is_large={is_large}):")
    for qps in qps_list:
        d = get_duration_for_qps(base_duration_s, qps, is_large)
        samples = int(qps * d)
        print(f"    QPS {qps:>5.2f}: {d:>3}s → ~{samples:>4} samples")

    print(f"\n  Pre-establishing {max_conversations} conversations for reuse...")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        for mode in ["pd", "ppd"]:
            mode_str = mode.upper()
            print(f"\n  === {mode_str} Mode ===")

            # Check server health before starting (and restart if needed)
            if not await ensure_server_healthy(session, mode_str):
                print(f"    ERROR: Server not available for {mode_str} mode, skipping...")
                continue

            # Set mode and clear state
            await set_mode(session, mode)
            await clear_state(session)
            await asyncio.sleep(1)

            # Phase 1: Establish all conversations once
            print(f"    Phase 1: Establishing {max_conversations} conversations "
                  f"(Turn 1: {workload.turn1.input_tokens}→{workload.turn1.output_tokens})...")

            histories, turn1_metrics_list, turn1_e2e_per_conv = await establish_conversations(
                session, workload, max_conversations, mode
            )

            print(f"    Phase 1 complete: {len(histories)}/{max_conversations} conversations established")

            if not histories:
                print(f"    ERROR: No conversations established for {mode_str}, skipping...")
                continue

            # Compute Turn 1 aggregated metrics
            turn1_aggregated = compute_turn_metrics(turn1_metrics_list, turn=1)
            print(f"    Turn 1 metrics: Avg TTFT={turn1_aggregated.avg_ttft:.1f}ms, "
                  f"Avg E2E={turn1_aggregated.avg_e2e:.1f}ms")

            await asyncio.sleep(2)

            # Phase 2: Run Turn 2 benchmark for each QPS point
            print(f"\n    Phase 2: Running Turn 2 benchmarks across {len(qps_list)} QPS points...")

            server_dead = False
            for qps in qps_list:
                if server_dead:
                    print(f"\n    --- QPS: {qps} --- SKIPPED (server crashed)")
                    continue

                # Get dynamic duration for this QPS
                qps_duration = get_duration_for_qps(base_duration_s, qps, is_large)
                print(f"\n    --- QPS: {qps} (duration: {qps_duration}s) ---")

                step_result = await run_turn2_benchmark(
                    session, workload, qps, qps_duration, mode,
                    histories, turn1_aggregated, turn1_e2e_per_conv
                )
                results.append(step_result)

                # Check if server crashed (success rate < 30% with enough samples)
                if step_result.sample_count >= 10:
                    success_rate = step_result.success_count / step_result.sample_count
                    if success_rate < 0.3:
                        print(f"      WARNING: Low success rate ({success_rate:.1%}), checking server health...")
                        if not await check_server_alive(session):
                            print(f"      ERROR: Server crashed! Skipping remaining QPS points for {mode_str}.")
                            server_dead = True

                await asyncio.sleep(2)

            await asyncio.sleep(3)

    return results


# ============================================================================
# Output
# ============================================================================

def print_summary(all_results: list[QPSStepResult]):
    """Print summary with both Turn 1 and Turn 2 metrics."""
    print("\n" + "=" * 120)
    print("QPS BENCHMARK SUMMARY (V2 - Per-Turn Metrics)")
    print("=" * 120)

    # Group by workload
    workloads = {}
    for r in all_results:
        if r.workload not in workloads:
            workloads[r.workload] = []
        workloads[r.workload].append(r)

    for wk_name, results in workloads.items():
        # Find workload config
        wk_cfg = next((w for w in WORKLOADS if w.name == wk_name), None)

        print(f"\n{'='*60}")
        print(f"{wk_name}")
        if wk_cfg:
            print(f"  T1: {wk_cfg.turn1.input_tokens} in → {wk_cfg.turn1.output_tokens} out")
            print(f"  T2: {wk_cfg.turn2.input_tokens} in → {wk_cfg.turn2.output_tokens} out")
        print(f"{'='*60}")

        # Turn 1 metrics
        print(f"\n  TURN 1 (Context Establishment)")
        print(f"  {'-'*100}")
        print(f"  {'QPS':<6} {'Mode':<5} {'Count':<8} {'Avg TTFT':<12} {'P99 TTFT':<12} {'Avg E2E':<12}")
        print(f"  {'-'*100}")

        for r in sorted(results, key=lambda x: (x.target_qps, x.mode)):
            if r.turn1_metrics:
                t1 = r.turn1_metrics
                print(f"  {r.target_qps:<6.1f} {r.mode.upper():<5} {t1.success_count:<8} "
                      f"{t1.avg_ttft:<12.1f} {t1.p99_ttft:<12.1f} {t1.avg_e2e:<12.1f}")

        # Turn 2 metrics
        print(f"\n  TURN 2 (Main Benchmark - Poisson Arrival)")
        print(f"  {'-'*100}")
        print(f"  {'QPS':<6} {'Mode':<5} {'Success':<10} {'Avg TTFT':<12} {'P99 TTFT':<12} {'Avg TPOT':<12} {'Avg E2E':<12}")
        print(f"  {'-'*100}")

        for r in sorted(results, key=lambda x: (x.target_qps, x.mode)):
            if r.turn2_metrics:
                t2 = r.turn2_metrics
                print(f"  {r.target_qps:<6.1f} {r.mode.upper():<5} "
                      f"{t2.success_count}/{r.sample_count:<6} "
                      f"{t2.avg_ttft:<12.1f} {t2.p99_ttft:<12.1f} "
                      f"{t2.avg_tpot:<12.1f} {t2.avg_e2e:<12.1f}")

        # Total E2E (Turn 1 + Turn 2)
        print(f"\n  TOTAL E2E (Turn 1 + Turn 2)")
        print(f"  {'-'*100}")
        print(f"  {'QPS':<6} {'Mode':<5} {'Avg Total':<14} {'P50 Total':<14} {'P99 Total':<14}")
        print(f"  {'-'*100}")

        for r in sorted(results, key=lambda x: (x.target_qps, x.mode)):
            if r.avg_total_e2e > 0:
                print(f"  {r.target_qps:<6.1f} {r.mode.upper():<5} "
                      f"{r.avg_total_e2e:<14.1f} {r.p50_total_e2e:<14.1f} {r.p99_total_e2e:<14.1f}")

    # Key observations
    print("\n" + "=" * 120)
    print("KEY OBSERVATIONS")
    print("=" * 120)

    for wk_name, results in workloads.items():
        print(f"\n{wk_name}:")
        for qps in sorted(set(r.target_qps for r in results)):
            pd_result = next((r for r in results if r.target_qps == qps and r.mode == "pd"), None)
            ppd_result = next((r for r in results if r.target_qps == qps and r.mode == "ppd"), None)

            if pd_result and ppd_result and pd_result.p99_ttft > 0 and ppd_result.p99_ttft > 0:
                ratio = ppd_result.p99_ttft / pd_result.p99_ttft
                winner = "PPD" if ratio < 1 else "PD"
                print(f"  QPS {qps}: PPD T2_P99={ppd_result.p99_ttft:.1f}ms, "
                      f"PD T2_P99={pd_result.p99_ttft:.1f}ms -> {winner} wins ({ratio:.2f}x)")


async def main():
    parser = argparse.ArgumentParser(description="QPS Benchmark V3 (3D Orthogonal Workload Design)")
    parser.add_argument("--duration", type=int, default=30,
                       help="Base duration in seconds. Actual duration varies by QPS: "
                            "low QPS (<0.5) uses 2x, medium (0.5-2.0) uses 1.5x, high (>=2.0) uses 1x. "
                            "L/XL workloads reduce by 33%%. (default: 30)")
    parser.add_argument("--workload", type=str, default="all",
                       help="Workload to run: 'all', 'S_a', 'M_b', etc., or 'quick' for subset")
    parser.add_argument("--qps", type=str, default="auto",
                       help="Comma-separated QPS values, or 'auto' to use per-workload QPS (default: auto)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs to average (default: 1)")
    parser.add_argument("--list", action="store_true", help="List available workloads")

    args = parser.parse_args()

    if args.list:
        print("=" * 70)
        print("QPS Benchmark V3: Available Workloads (16 combinations)")
        print("=" * 70)
        print("\nT1 Configurations (Context Building):")
        for name, cfg in T1_CONFIGS.items():
            c1 = cfg.input_tokens + cfg.output_tokens
            print(f"  {name}: {cfg.input_tokens}→{cfg.output_tokens} (C1={c1})")
        print("\nT2 Types (Incremental Request):")
        for name, cfg in T2_TYPES.items():
            print(f"  {name}: {cfg.input_tokens}→{cfg.output_tokens}")
        print("\nQPS Lists by T1 Size:")
        for t1_type, qps_list in QPS_BY_T1.items():
            print(f"  {t1_type}: {qps_list} ({len(qps_list)} points)")
        print("\nWorkload Matrix (T1_T2):")
        for wk in WORKLOADS:
            qps = get_qps_list_for_workload(wk.name)
            print(f"  {wk.name}: {wk.description} | QPS: {len(qps)} points")
        return

    # Select workloads
    if args.workload == "all":
        workloads = WORKLOADS
    elif args.workload == "quick":
        # Quick subset: one from each T2 type
        quick_names = ["S_a", "M_b", "L_c", "XL_d"]
        workloads = [w for w in WORKLOADS if w.name in quick_names]
    elif args.workload == "diagonal":
        # Diagonal: same T1 size progression with varied T2
        diagonal_names = ["S_a", "M_b", "L_c", "XL_d"]
        workloads = [w for w in WORKLOADS if w.name in diagonal_names]
    else:
        # Specific workload(s)
        names = [n.strip() for n in args.workload.split(",")]
        workloads = [w for w in WORKLOADS if w.name in names]

    if not workloads:
        print(f"No matching workloads found for: {args.workload}")
        return

    # Determine QPS mode
    use_auto_qps = (args.qps == "auto")
    if not use_auto_qps:
        fixed_qps_list = [float(q.strip()) for q in args.qps.split(",")]

    print("=" * 70)
    print("QPS vs Latency Benchmark V3: PD vs PPD (3D Orthogonal Design)")
    print("=" * 70)
    print(f"Workloads: {len(workloads)}")
    for wk in workloads:
        qps = get_qps_list_for_workload(wk.name) if use_auto_qps else fixed_qps_list
        print(f"  - {wk.name}: {wk.description} | QPS: {qps}")
    if use_auto_qps:
        print(f"QPS mode: auto (per-workload stratification)")
    else:
        print(f"QPS sweep: {fixed_qps_list}")
    print(f"Base duration: {args.duration}s (dynamic per QPS)")
    print(f"  - Low QPS (<0.5): {args.duration * 2}s")
    print(f"  - Medium QPS (0.5-2.0): {int(args.duration * 1.5)}s")
    print(f"  - High QPS (>=2.0): {args.duration}s")
    print(f"  - L/XL workloads: reduced by 33%")
    print(f"Runs per config: {args.runs}")

    # Calculate total experiments
    total_exp = 0
    for wk in workloads:
        qps = get_qps_list_for_workload(wk.name) if use_auto_qps else fixed_qps_list
        total_exp += len(qps) * 2 * args.runs
    print(f"Total experiments: {total_exp}")

    # Better time estimate with dynamic duration
    avg_duration = (args.duration * 2 + args.duration * 1.5 + args.duration) / 3
    avg_qps_points = sum(len(get_qps_list_for_workload(wk.name)) for wk in workloads) / len(workloads)
    est_time = len(workloads) * avg_qps_points * 2 * args.runs * (avg_duration + 10) / 60
    print(f"Estimated time: ~{est_time:.0f} minutes")
    print("=" * 70)

    # Check connection
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status == 200:
                    status = await resp.json()
                    print(f"Connected to proxy at {PROXY_URL} (mode: {status.get('mode', 'unknown')})")
                else:
                    print(f"Proxy returned status {resp.status}")
                    return

            # NCCL warmup
            await nccl_warmup(session)

    except Exception as e:
        print(f"Cannot connect to proxy at {PROXY_URL}: {e}")
        print("Start servers with: ./scripts/start_servers_4gpu.sh ppd")
        return

    # Run benchmark
    all_results = []
    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n{'#'*70}")
            print(f"RUN {run_idx + 1} of {args.runs}")
            print(f"{'#'*70}")

        for workload in workloads:
            # Get QPS list for this workload (auto or fixed)
            qps_list = get_qps_list_for_workload(workload.name) if use_auto_qps else fixed_qps_list
            results = await run_workload_sweep(workload, qps_list, args.duration)
            for r in results:
                r.raw_metrics.insert(0, {"run": run_idx + 1})
            all_results.extend(results)

        if run_idx < args.runs - 1:
            print("\n  Cooling down for 10 seconds before next run...")
            await asyncio.sleep(10)

    # Print summary
    print_summary(all_results)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or str(RESULTS_DIR / f"qps_benchmark_v3_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump({
            "version": "3.0",
            "timestamp": timestamp,
            "config": {
                "t1_configs": {k: asdict(v) for k, v in T1_CONFIGS.items()},
                "t2_types": {k: asdict(v) for k, v in T2_TYPES.items()},
                "qps_by_t1": QPS_BY_T1,
                "workloads": [asdict(wk) for wk in workloads],
                "duration_s": args.duration,
                "runs": args.runs,
                "qps_mode": "auto" if use_auto_qps else "fixed",
            },
            "results": [asdict(r) for r in all_results],
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
