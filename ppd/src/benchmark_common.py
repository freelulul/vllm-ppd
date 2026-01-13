#!/usr/bin/env python3
"""
Common utilities for PD/PPD/Replica benchmarks.
Shared data structures, request functions, and metrics computation.
"""

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
from typing import Optional, List, Dict, Tuple

# ============================================================================
# Constants
# ============================================================================

MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

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
    # Cache metrics
    local_cache_hits: int = 0
    external_cache_hits: int = 0


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

    # Cache metrics
    total_local_cache_hits: int = 0
    total_external_cache_hits: int = 0
    total_cache_queries: int = 0
    local_hit_rate: float = 0.0
    external_hit_rate: float = 0.0


@dataclass
class CacheSnapshot:
    """Snapshot of cache metrics from a server."""
    local_queries: int = 0
    local_hits: int = 0
    external_queries: int = 0
    external_hits: int = 0
    kv_usage_pct: float = 0.0


@dataclass
class TurnConfig:
    """Configuration for a single turn."""
    input_tokens: int
    output_tokens: int


@dataclass
class WorkloadConfig:
    """Configuration for a workload."""
    name: str
    description: str
    turn1: TurnConfig
    turn2: TurnConfig
    num_turns: int = 2  # Support 2-5 turns

    @property
    def context_after_turn1(self) -> int:
        return self.turn1.input_tokens + self.turn1.output_tokens


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    mode: str
    workload: str
    target_qps: float
    duration_s: int
    run_id: int
    timestamp: str

    # Turn configs
    t1_input: int = 0
    t1_output: int = 0
    t2_input: int = 0
    t2_output: int = 0
    num_turns: int = 2

    # Turn metrics (turn1_metrics, turn2_metrics format)
    turn1_metrics: Dict = field(default_factory=dict)
    turn2_metrics: Dict = field(default_factory=dict)

    # Aggregate metrics (matching previous format)
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    real_qps: float = 0.0

    # TTFT aggregates
    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p90_ttft: float = 0.0
    p99_ttft: float = 0.0
    min_ttft: float = 0.0
    max_ttft: float = 0.0

    # E2E aggregates
    avg_e2e: float = 0.0
    p50_e2e: float = 0.0
    p90_e2e: float = 0.0
    p99_e2e: float = 0.0
    avg_total_e2e: float = 0.0
    p50_total_e2e: float = 0.0
    p99_total_e2e: float = 0.0

    # Throughput
    avg_throughput_tps: float = 0.0
    total_tokens: int = 0

    # Cache metrics (from vLLM servers)
    cache_stats: Dict = field(default_factory=dict)

    # Cache affinity stats (from proxy - tracks same-conversation D routing)
    cache_affinity_stats: Dict = field(default_factory=dict)

    # Raw metrics for detailed analysis
    raw_metrics: List[Dict] = field(default_factory=list)


# ============================================================================
# Workload Definitions (V5)
# ============================================================================

# Context size configs
CONTEXT_CONFIGS = {
    "XS": TurnConfig(128, 128),    # C1 = 256
    "S": TurnConfig(256, 256),     # C1 = 512
    "M": TurnConfig(512, 512),     # C1 = 1024
    "L": TurnConfig(1024, 1024),   # C1 = 2048
    "XL": TurnConfig(2048, 1024),  # C1 = 3072
    "XXL": TurnConfig(3072, 1024), # C1 = 4096
}

# T2 type configs
T2_CONFIGS = {
    "a": TurnConfig(16, 32),    # Tiny followup
    "b": TurnConfig(32, 256),   # Short Q, long output
    "c": TurnConfig(128, 128),  # Balanced
    "d": TurnConfig(512, 64),   # Big paste, short answer
    "e": TurnConfig(1024, 32),  # Very large increment
}

# ============================================================================
# UNIFIED QPS Configuration for Fair Comparison
# ============================================================================
# All three modes (PPD, PD, Replica) MUST use this same configuration
# to ensure equal data points for fair comparison.
#
# QPS Boundaries tested on 2026-01-12 (all modes achieved 100% success):
#   XS: max 12 QPS (all modes)
#   S:  max 8 QPS (all modes)
#   M:  max 6 QPS (all modes)
#   L:  max 4 QPS (all modes)
#   XL: max 2-3 QPS (PPD XL_c only 2, others 3)
#
# Test points range from 1 to max boundary for load scaling analysis.

QPS_UNIFIED = {
    "XS": [1, 2, 4, 6, 8, 10, 12],    # 7 points, max 12 QPS
    "S":  [1, 2, 3, 4, 6, 8],          # 6 points, max 8 QPS
    "M":  [1, 2, 3, 4, 5, 6],          # 6 points, max 6 QPS
    "L":  [1, 2, 3, 4],                # 4 points, max 4 QPS
    "XL": [0.5, 1, 1.5, 2],            # 4 points, max 2 QPS (conservative)
    "XXL": [0.5, 1],                   # 2 points - boundary
}

# Legacy configs - kept for backward compatibility, but use QPS_UNIFIED instead
QPS_BY_CONTEXT = QPS_UNIFIED
QPS_BY_CONTEXT_PD = QPS_UNIFIED

def create_workload(context_size: str, t2_type: str, num_turns: int = 2) -> WorkloadConfig:
    """Create a workload from context size and T2 type."""
    return WorkloadConfig(
        name=f"{context_size}_{t2_type}",
        description=f"Context {context_size}, T2 type {t2_type}, {num_turns} turns",
        turn1=CONTEXT_CONFIGS[context_size],
        turn2=T2_CONFIGS[t2_type],
        num_turns=num_turns,
    )

# ============================================================================
# Complete Workload Matrix: 5 Context Sizes × 4 T2 Types = 20 workloads
# ============================================================================
# Each context size tested with all T2 types for comprehensive analysis
#
# T2 Types:
#   a: Tiny followup (16 in, 32 out) - minimal T2 overhead
#   b: Short Q, long output (32 in, 256 out) - generation heavy
#   c: Balanced (128 in, 128 out) - equal I/O
#   d: Big paste (512 in, 64 out) - prefill heavy

CORE_WORKLOADS = [
    # XS context (128 tokens) - smallest, should handle highest QPS
    ("XS", "a", 2),
    ("XS", "b", 2),
    ("XS", "c", 2),
    ("XS", "d", 2),

    # S context (256 tokens) - small
    ("S", "a", 2),
    ("S", "b", 2),
    ("S", "c", 2),
    ("S", "d", 2),

    # M context (512 tokens) - medium
    ("M", "a", 2),
    ("M", "b", 2),
    ("M", "c", 2),
    ("M", "d", 2),

    # L context (1024 tokens) - large
    ("L", "a", 2),
    ("L", "b", 2),
    ("L", "c", 2),
    ("L", "d", 2),

    # XL context (2048 tokens) - extra large, near memory boundary
    ("XL", "a", 2),
    ("XL", "b", 2),
    ("XL", "c", 2),
    ("XL", "d", 2),
]

# Default server URLs
DECODE_URLS = ["http://localhost:8200", "http://localhost:8201"]
PREFILL_URLS = ["http://localhost:8100", "http://localhost:8101"]

# Success rate threshold for adaptive stopping
# Set to 0 to disable adaptive stopping (ensure all test points are collected)
# For fair comparison, we want ALL data points even if some have lower success rate
MIN_SUCCESS_RATE = 0.0  # Disabled - collect all data points for analysis


# ============================================================================
# Server Health Check Functions
# ============================================================================

async def check_server_health(url: str, timeout: float = 5.0) -> bool:
    """Check if a server is healthy and responding."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/health",
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def check_proxy_health(proxy_url: str = "http://localhost:10001") -> bool:
    """Check if proxy is healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{proxy_url}/status",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def check_all_servers_health(
    proxy_url: str = "http://localhost:10001",
    decode_urls: List[str] = None,
    prefill_urls: List[str] = None,
) -> Tuple[bool, str]:
    """
    Check health of all servers.
    Returns (healthy, message).
    """
    if decode_urls is None:
        decode_urls = DECODE_URLS
    if prefill_urls is None:
        prefill_urls = PREFILL_URLS

    # Check proxy
    if not await check_proxy_health(proxy_url):
        return False, "Proxy not responding"

    # Check decode servers
    for i, url in enumerate(decode_urls):
        if not await check_server_health(url):
            return False, f"Decode server {i} ({url}) not responding"

    # Check prefill servers
    for i, url in enumerate(prefill_urls):
        if not await check_server_health(url):
            return False, f"Prefill server {i} ({url}) not responding"

    return True, "All servers healthy"


async def wait_for_servers_recovery(
    proxy_url: str = "http://localhost:10001",
    max_wait_s: int = 60,
    check_interval_s: int = 5,
) -> bool:
    """
    Wait for servers to recover after potential crash.
    Returns True if servers recovered, False if timeout.
    """
    start = time.time()
    while time.time() - start < max_wait_s:
        healthy, msg = await check_all_servers_health(proxy_url)
        if healthy:
            return True
        print(f"    Waiting for server recovery... ({msg})")
        await asyncio.sleep(check_interval_s)
    return False


# ============================================================================
# Cache Affinity Statistics (from Proxy)
# ============================================================================

async def get_proxy_cache_stats(proxy_url: str = "http://localhost:10001") -> Dict:
    """Get cache affinity statistics from proxy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{proxy_url}/cache_stats",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return {}


async def reset_proxy_cache_stats(proxy_url: str = "http://localhost:10001") -> bool:
    """Reset cache affinity statistics on proxy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{proxy_url}/cache_stats/reset",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


# ============================================================================
# Cache Metrics Functions (from vLLM servers)
# ============================================================================

async def get_cache_snapshot(url: str) -> CacheSnapshot:
    """Get cache metrics snapshot from a vLLM server.

    Handles multiple metric name formats:
    - vllm:prefix_cache_queries_total (vLLM v1 with colon)
    - vllm_prefix_cache_queries_total (vLLM with underscore)
    - prefix_cache_queries_total (legacy format)
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                text = await resp.text()
                snapshot = CacheSnapshot()
                for line in text.split('\n'):
                    # Skip comments
                    if line.startswith('#'):
                        continue
                    # Local prefix cache queries (match various formats)
                    if 'prefix_cache_queries_total{' in line and 'external' not in line:
                        try:
                            snapshot.local_queries = int(float(line.split('}')[1].strip()))
                        except (IndexError, ValueError):
                            pass
                    # Local prefix cache hits
                    elif 'prefix_cache_hits_total{' in line and 'external' not in line:
                        try:
                            snapshot.local_hits = int(float(line.split('}')[1].strip()))
                        except (IndexError, ValueError):
                            pass
                    # External prefix cache queries (KV connector)
                    elif 'external_prefix_cache_queries_total{' in line:
                        try:
                            snapshot.external_queries = int(float(line.split('}')[1].strip()))
                        except (IndexError, ValueError):
                            pass
                    # External prefix cache hits
                    elif 'external_prefix_cache_hits_total{' in line:
                        try:
                            snapshot.external_hits = int(float(line.split('}')[1].strip()))
                        except (IndexError, ValueError):
                            pass
                    # KV cache usage percentage
                    elif 'kv_cache_usage_perc{' in line:
                        try:
                            snapshot.kv_usage_pct = float(line.split('}')[1].strip()) * 100
                        except (IndexError, ValueError):
                            pass
                return snapshot
    except Exception:
        return CacheSnapshot()


async def get_all_cache_snapshots(decode_urls: List[str] = None) -> Dict[str, CacheSnapshot]:
    """Get cache snapshots from all decode servers."""
    if decode_urls is None:
        decode_urls = DECODE_URLS

    snapshots = {}
    for i, url in enumerate(decode_urls):
        snapshots[f"D{i}"] = await get_cache_snapshot(url)
    return snapshots


def compute_cache_delta(before: Dict[str, CacheSnapshot], after: Dict[str, CacheSnapshot]) -> Dict:
    """Compute cache metrics delta between two snapshots."""
    total_local_queries = 0
    total_local_hits = 0
    total_external_queries = 0
    total_external_hits = 0
    max_kv_usage = 0.0

    for key in after:
        if key in before:
            total_local_queries += after[key].local_queries - before[key].local_queries
            total_local_hits += after[key].local_hits - before[key].local_hits
            total_external_queries += after[key].external_queries - before[key].external_queries
            total_external_hits += after[key].external_hits - before[key].external_hits
        max_kv_usage = max(max_kv_usage, after[key].kv_usage_pct)

    local_hit_rate = (total_local_hits / total_local_queries * 100) if total_local_queries > 0 else 0
    external_hit_rate = (total_external_hits / total_external_queries * 100) if total_external_queries > 0 else 0

    return {
        "local_cache_queries": total_local_queries,
        "local_cache_hits": total_local_hits,
        "local_hit_rate_pct": local_hit_rate,
        "external_cache_queries": total_external_queries,
        "external_cache_hits": total_external_hits,
        "external_hit_rate_pct": external_hit_rate,
        "max_kv_usage_pct": max_kv_usage,
    }


# ============================================================================
# Request Functions
# ============================================================================

def generate_prompt(num_tokens: int, prefix: str = "", seed: str = "") -> str:
    """Generate a prompt with approximately num_tokens tokens.

    Args:
        num_tokens: Target number of tokens
        prefix: Prefix to add (e.g., conversation ID)
        seed: Optional seed for reproducibility. If empty, uses random content.
    """
    import random
    import string

    # Word pool for realistic text (~100 words to reduce accidental prefix matches)
    WORD_POOL = [
        # Common words (40)
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        # Technical terms (30)
        "system", "data", "model", "process", "analysis", "result", "method",
        "research", "information", "development", "performance", "application",
        "technology", "network", "algorithm", "function", "structure", "interface",
        "database", "server", "client", "protocol", "framework", "module",
        "component", "service", "request", "response", "memory", "storage",
        # Action words (15)
        "create", "update", "delete", "execute", "configure", "optimize",
        "implement", "design", "build", "deploy", "monitor", "validate",
        "transform", "generate", "compute",
        # Descriptive words (15)
        "efficient", "scalable", "robust", "dynamic", "parallel", "distributed",
        "synchronous", "asynchronous", "concurrent", "sequential", "recursive",
        "iterative", "adaptive", "modular", "flexible",
    ]

    if seed:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # ~0.75 words per token for English
    target_words = int(num_tokens * 0.75)
    words = []

    for _ in range(target_words):
        if rng.random() < 0.7:
            words.append(rng.choice(WORD_POOL))
        else:
            # Random word 3-8 chars
            word_len = rng.randint(3, 8)
            words.append(''.join(rng.choices(string.ascii_lowercase, k=word_len)))

    content = ' '.join(words)
    return f"User: {prefix}{content}\nAssistant:"


async def send_turn1(
    session: aiohttp.ClientSession,
    proxy_url: str,
    conv_id: str,
    workload: WorkloadConfig,
    mode: str,
    qps: float = 0.0,
) -> Tuple[bool, str, RequestMetrics]:
    """Send Turn 1 request. Returns (success, history, metrics)."""
    prefix = f"CONV_{conv_id[:8]}_T1_"
    prompt = generate_prompt(workload.turn1.input_tokens, prefix)

    start_t = time.perf_counter()
    ttft = 0.0
    tokens_received = 0

    try:
        async with session.post(
            f"{proxy_url}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": prompt,
                "max_tokens": workload.turn1.output_tokens,
                "temperature": 0.8,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                return False, "", RequestMetrics(
                    req_id=conv_id, mode=mode, workload=workload.name, qps_target=qps,
                    input_len=workload.turn1.input_tokens, output_len=workload.turn1.output_tokens,
                    start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
                    throughput_tps=0, success=False, error=f"HTTP {resp.status}", turn=1
                )

            first_token = False
            buffer = ""
            response_text = ""

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
                            if not first_token and chunk.get("choices"):
                                text = chunk["choices"][0].get("text", "")
                                if text:
                                    ttft = (time.perf_counter() - start_t) * 1000
                                    first_token = True
                                    response_text += text
                            elif chunk.get("choices"):
                                response_text += chunk["choices"][0].get("text", "")
                            if chunk.get("usage"):
                                tokens_received = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

            end_t = time.perf_counter()
            latency = (end_t - start_t) * 1000
            tps = (tokens_received / (latency / 1000)) if latency > 0 and tokens_received > 0 else 0
            tpot = (latency - ttft) / max(tokens_received - 1, 1) if tokens_received > 1 else 0

            # Build history for Turn 2
            history = prompt + response_text

            return True, history, RequestMetrics(
                req_id=conv_id, mode=mode, workload=workload.name, qps_target=qps,
                input_len=workload.turn1.input_tokens, output_len=workload.turn1.output_tokens,
                start_time=start_t, ttft_ms=ttft, e2e_latency_ms=latency, output_tokens=tokens_received,
                throughput_tps=tps, tpot_ms=tpot, success=True, turn=1
            )

    except asyncio.TimeoutError:
        return False, "", RequestMetrics(
            req_id=conv_id, mode=mode, workload=workload.name, qps_target=qps,
            input_len=workload.turn1.input_tokens, output_len=workload.turn1.output_tokens,
            start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
            throughput_tps=0, success=False, error="Timeout", turn=1
        )
    except Exception as e:
        return False, "", RequestMetrics(
            req_id=conv_id, mode=mode, workload=workload.name, qps_target=qps,
            input_len=workload.turn1.input_tokens, output_len=workload.turn1.output_tokens,
            start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
            throughput_tps=0, success=False, error=str(e)[:100], turn=1
        )


async def send_turn2(
    session: aiohttp.ClientSession,
    proxy_url: str,
    conv_id: str,
    history: str,
    workload: WorkloadConfig,
    mode: str,
    qps: float = 0.0,
    turn_num: int = 2,
) -> Tuple[bool, str, RequestMetrics]:
    """Send Turn 2+ request. Returns (success, updated_history, metrics)."""
    prefix = f"T{turn_num}_followup_"
    new_input = generate_prompt(workload.turn2.input_tokens, prefix)
    prompt = history + "\n" + new_input

    start_t = time.perf_counter()
    ttft = 0.0
    tokens_received = 0

    try:
        async with session.post(
            f"{proxy_url}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": prompt,
                "max_tokens": workload.turn2.output_tokens,
                "temperature": 0.8,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                return False, history, RequestMetrics(
                    req_id=conv_id, mode=mode, workload=workload.name, qps_target=qps,
                    input_len=workload.turn2.input_tokens, output_len=workload.turn2.output_tokens,
                    start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
                    throughput_tps=0, success=False, error=f"HTTP {resp.status}", turn=turn_num
                )

            first_token = False
            buffer = ""
            response_text = ""

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
                            if not first_token and chunk.get("choices"):
                                text = chunk["choices"][0].get("text", "")
                                if text:
                                    ttft = (time.perf_counter() - start_t) * 1000
                                    first_token = True
                                    response_text += text
                            elif chunk.get("choices"):
                                response_text += chunk["choices"][0].get("text", "")
                            if chunk.get("usage"):
                                tokens_received = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

            end_t = time.perf_counter()
            latency = (end_t - start_t) * 1000
            tps = (tokens_received / (latency / 1000)) if latency > 0 and tokens_received > 0 else 0
            tpot = (latency - ttft) / max(tokens_received - 1, 1) if tokens_received > 1 else 0

            updated_history = prompt + response_text

            return True, updated_history, RequestMetrics(
                req_id=conv_id, mode=mode, workload=workload.name, qps_target=qps,
                input_len=workload.turn2.input_tokens, output_len=workload.turn2.output_tokens,
                start_time=start_t, ttft_ms=ttft, e2e_latency_ms=latency, output_tokens=tokens_received,
                throughput_tps=tps, tpot_ms=tpot, success=True, turn=turn_num
            )

    except asyncio.TimeoutError:
        return False, history, RequestMetrics(
            req_id=conv_id, mode=mode, workload=workload.name, qps_target=qps,
            input_len=workload.turn2.input_tokens, output_len=workload.turn2.output_tokens,
            start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
            throughput_tps=0, success=False, error="Timeout", turn=turn_num
        )
    except Exception as e:
        return False, history, RequestMetrics(
            req_id=conv_id, mode=mode, workload=workload.name, qps_target=qps,
            input_len=workload.turn2.input_tokens, output_len=workload.turn2.output_tokens,
            start_time=start_t, ttft_ms=0, e2e_latency_ms=0, output_tokens=0,
            throughput_tps=0, success=False, error=str(e)[:100], turn=turn_num
        )


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_turn_metrics(metrics_list: List[RequestMetrics], turn: int) -> TurnMetrics:
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


def save_results(results: List[BenchmarkResult], output_dir: Path, mode: str, run_id: int):
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{mode}_run{run_id}_{timestamp}.json"

    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path
