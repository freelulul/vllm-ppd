#!/usr/bin/env python3
"""
Mini Test: Verify benchmark script works correctly with minimal requests.

This runs just 3 requests per mode to verify:
1. Server connectivity
2. Request/response flow
3. Latency measurement correctness
4. SLO computation logic

Usage:
    python scripts/tests/mini_test_benchmark.py
"""

import os
import sys
import json
import time
import random
import asyncio
import aiohttp
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional

# Server endpoints
PD_PROXY_URL = "http://localhost:10001"
REPLICA_PROXY_URL = "http://localhost:10002"
MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"


@dataclass
class RequestResult:
    conv_id: str
    turn: int
    mode: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    success: bool
    error: Optional[str] = None


def generate_sensible_prompt(num_tokens: int, seed: str = "") -> str:
    """Generate a sensible prompt that the model will actually respond to."""
    if seed:
        rng = random.Random(hashlib.md5(seed.encode()).hexdigest())
    else:
        rng = random.Random()

    # Use meaningful sentence templates instead of random words
    templates = [
        "The quick brown fox jumps over the lazy dog near the old oak tree.",
        "In a distant galaxy, scientists discovered a new planet with rings.",
        "The weather forecast predicts sunny skies and warm temperatures today.",
        "A group of researchers published findings about climate patterns.",
        "The ancient library contained manuscripts from civilizations long gone.",
        "Modern technology enables communication across vast distances instantly.",
        "The mountain expedition team reached the summit after days of climbing.",
        "Ocean currents influence weather patterns around the entire globe.",
        "The museum exhibit showcased artifacts from the medieval period.",
        "Renewable energy sources are becoming increasingly cost effective.",
    ]

    # Build prompt by repeating templates until we reach desired length
    sentences = []
    current_tokens = 0
    while current_tokens < num_tokens:
        sentence = rng.choice(templates)
        sentences.append(sentence)
        current_tokens += len(sentence.split())

    return " ".join(sentences)


async def send_request(session, url, conv_id, turn, prompt, max_tokens, mode, debug=False):
    request_data = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    tokens_received = 0
    token_times = []

    try:
        async with session.post(f"{url}/v1/completions", json=request_data) as response:
            if response.status != 200:
                error_text = await response.text()
                if debug:
                    print(f"      [DEBUG] HTTP {response.status}: {error_text[:200]}")
                return RequestResult(
                    conv_id=conv_id, turn=turn, mode=mode,
                    input_tokens=len(prompt.split()), output_tokens=0,
                    ttft_ms=0, tpot_ms=0, e2e_ms=0,
                    success=False, error=f"HTTP {response.status}"
                )

            buffer = ""
            chunk_count = 0
            async for chunk in response.content.iter_any():
                chunk_count += 1
                chunk_text = chunk.decode('utf-8')
                buffer += chunk_text
                if debug and chunk_count <= 3:
                    print(f"      [DEBUG] Chunk {chunk_count} (len={len(chunk_text)}): {repr(chunk_text[:500])}")
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                text = data["choices"][0].get("text", "")
                                if text:
                                    current_time = time.perf_counter()
                                    if first_token_time is None:
                                        first_token_time = current_time
                                        if debug:
                                            print(f"      [DEBUG] First token: {repr(text)}")
                                    tokens_received += 1
                                    token_times.append(current_time)
                        except json.JSONDecodeError:
                            if debug:
                                print(f"      [DEBUG] JSON decode error: {data_str[:50]}")
                            continue

            if debug:
                print(f"      [DEBUG] Total chunks: {chunk_count}, tokens: {tokens_received}")
                if buffer:
                    print(f"      [DEBUG] Remaining buffer (len={len(buffer)}): {repr(buffer[:200])}")

        end_time = time.perf_counter()
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        e2e_ms = (end_time - start_time) * 1000

        if len(token_times) > 1:
            inter_token_times = [(token_times[i] - token_times[i-1]) * 1000
                                for i in range(1, len(token_times))]
            tpot_ms = sum(inter_token_times) / len(inter_token_times)
        else:
            tpot_ms = 0

        return RequestResult(
            conv_id=conv_id, turn=turn, mode=mode,
            input_tokens=len(prompt.split()), output_tokens=tokens_received,
            ttft_ms=ttft_ms, tpot_ms=tpot_ms, e2e_ms=e2e_ms,
            success=True
        )

    except Exception as e:
        return RequestResult(
            conv_id=conv_id, turn=turn, mode=mode,
            input_tokens=len(prompt.split()), output_tokens=0,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False, error=str(e)
        )


async def test_mode(mode: str, num_requests: int = 3):
    """Test a single mode with minimal requests"""
    print(f"\n{'='*50}")
    print(f"Testing mode: {mode.upper()}")
    print(f"{'='*50}")

    if mode in ["pd", "ppd"]:
        url = PD_PROXY_URL
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(f"{url}/mode/{mode}")
                print(f"  Set proxy mode to: {mode}")
            except Exception as e:
                print(f"  ERROR setting mode: {e}")
                return []
    else:
        url = REPLICA_PROXY_URL

    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        for i in range(num_requests):
            conv_id = f"test_{mode}_{i}_{int(time.time())}"
            input_tokens = 256  # Fixed for test
            output_tokens = 64

            print(f"\n  Request {i+1}/{num_requests}:")
            print(f"    Input tokens: {input_tokens}")
            print(f"    Output tokens: {output_tokens}")

            # Turn 1
            prompt = generate_sensible_prompt(input_tokens, seed=f"{conv_id}_t1")
            result1 = await send_request(session, url, conv_id, 1, prompt, 1, mode, debug=False)

            if not result1.success:
                print(f"    Turn 1 FAILED: {result1.error}")
                continue

            print(f"    Turn 1: TTFT={result1.ttft_ms:.1f}ms")

            await asyncio.sleep(0.1)

            # Turn 2
            history = prompt + "\n\nAssistant: " + generate_sensible_prompt(20, f"{conv_id}_resp")
            follow_up = history + "\n\nUser: Please continue with more details about this topic.\n\nAssistant:"
            result2 = await send_request(session, url, conv_id, 2, follow_up, output_tokens, mode, debug=False)

            if result2.success:
                print(f"    Turn 2: TTFT={result2.ttft_ms:.1f}ms, TPOT={result2.tpot_ms:.2f}ms, E2E={result2.e2e_ms:.1f}ms")
                print(f"    Output tokens received: {result2.output_tokens}")
                results.append(result2)
            else:
                print(f"    Turn 2 FAILED: {result2.error}")

    return results


async def main():
    print("="*70)
    print("MINI TEST: Verify Benchmark Script Correctness")
    print("="*70)

    # Check connectivity
    print("\nChecking server connectivity...")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{PD_PROXY_URL}/status") as resp:
                if resp.status == 200:
                    print(f"  PD/PPD Proxy: OK")
                else:
                    print(f"  PD/PPD Proxy: FAILED")
                    return

            async with session.get(f"{REPLICA_PROXY_URL}/status") as resp:
                if resp.status == 200:
                    print(f"  Replica Proxy: OK")
                else:
                    print(f"  Replica Proxy: FAILED")
                    return
    except Exception as e:
        print(f"  Connection failed: {e}")
        print("\nPlease start servers first:")
        print("  ./scripts/server/start_optimizer_servers.sh")
        return

    # Test each mode
    all_results = {}
    for mode in ["pd", "ppd", "replica"]:
        results = await test_mode(mode, num_requests=3)
        all_results[mode] = results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    slo = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}
    print(f"\nSLO Thresholds: TTFT≤{slo['ttft_ms']}ms, TPOT≤{slo['tpot_ms']}ms, E2E≤{slo['e2e_ms']}ms")

    for mode, results in all_results.items():
        if not results:
            print(f"\n{mode.upper()}: No successful requests")
            continue

        avg_ttft = sum(r.ttft_ms for r in results) / len(results)
        avg_tpot = sum(r.tpot_ms for r in results) / len(results)
        avg_e2e = sum(r.e2e_ms for r in results) / len(results)

        ttft_met = sum(1 for r in results if r.ttft_ms <= slo["ttft_ms"])
        tpot_met = sum(1 for r in results if r.tpot_ms <= slo["tpot_ms"])
        e2e_met = sum(1 for r in results if r.e2e_ms <= slo["e2e_ms"])

        print(f"\n{mode.upper()}:")
        print(f"  Avg TTFT: {avg_ttft:.1f}ms (SLO met: {ttft_met}/{len(results)})")
        print(f"  Avg TPOT: {avg_tpot:.2f}ms (SLO met: {tpot_met}/{len(results)})")
        print(f"  Avg E2E:  {avg_e2e:.1f}ms (SLO met: {e2e_met}/{len(results)})")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    # Verify measurement correctness
    print("\nMeasurement Verification:")
    for mode, results in all_results.items():
        for r in results:
            if r.success:
                expected_e2e_approx = r.ttft_ms + r.output_tokens * r.tpot_ms
                diff = abs(r.e2e_ms - expected_e2e_approx)
                status = "OK" if diff < r.e2e_ms * 0.2 else "WARN"
                print(f"  {mode} request: E2E={r.e2e_ms:.1f}ms, TTFT+tokens*TPOT≈{expected_e2e_approx:.1f}ms [{status}]")


if __name__ == "__main__":
    asyncio.run(main())
