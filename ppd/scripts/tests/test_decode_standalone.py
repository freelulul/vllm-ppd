#!/usr/bin/env python3
"""
Test if a Decode server (kv_consumer) can handle standalone requests.

This tests whether we can send requests directly to a Decode server
WITHOUT going through Prefill first - i.e., can D act as a Replica?

Test scenarios:
1. Send request directly to Decode server (bypass proxy)
2. Compare with sending to Replica server
3. Check if kv_consumer config blocks standalone operation
"""

import asyncio
import json
import time
import aiohttp
import argparse


async def send_request(url: str, prompt: str, max_tokens: int = 32) -> dict:
    """Send a completion request and measure metrics."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    ttft = None
    start_time = time.perf_counter()
    output_tokens = 0
    response_text = ""
    error = None

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
                if resp.status != 200:
                    error = f"HTTP {resp.status}: {await resp.text()}"
                else:
                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        if not line or not line.startswith('data:'):
                            continue
                        if line == 'data: [DONE]':
                            break
                        try:
                            data = json.loads(line[5:].strip())
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    if ttft is None:
                                        ttft = (time.perf_counter() - start_time) * 1000
                                    response_text += delta['content']
                                    output_tokens += 1
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        error = str(e)

    e2e = (time.perf_counter() - start_time) * 1000

    return {
        "success": error is None,
        "error": error,
        "ttft_ms": ttft or e2e,
        "e2e_ms": e2e,
        "output_tokens": output_tokens,
        "response": response_text[:100] + "..." if len(response_text) > 100 else response_text,
    }


async def check_server_health(url: str) -> bool:
    """Check if server is responding."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{url}/v1/models") as resp:
                return resp.status == 200
    except:
        return False


async def main():
    parser = argparse.ArgumentParser(description="Test Decode server standalone capability")
    parser.add_argument("--decode-url", default="http://localhost:8200",
                        help="Decode server URL (kv_consumer)")
    parser.add_argument("--replica-url", default="http://localhost:8300",
                        help="Replica server URL (standalone)")
    parser.add_argument("--prefill-url", default="http://localhost:8100",
                        help="Prefill server URL (kv_producer)")
    args = parser.parse_args()

    print("=" * 70)
    print("DECODE SERVER STANDALONE CAPABILITY TEST")
    print("=" * 70)
    print(f"Decode URL (kv_consumer): {args.decode_url}")
    print(f"Replica URL (standalone): {args.replica_url}")
    print(f"Prefill URL (kv_producer): {args.prefill_url}")
    print()

    # Test prompts
    short_prompt = "Say hello in one word."
    medium_prompt = "A" * 256 + " Summarize this in one sentence."

    results = {}

    # Test 1: Check server health
    print("[Test 1] Checking server health...")
    decode_healthy = await check_server_health(args.decode_url)
    replica_healthy = await check_server_health(args.replica_url)
    prefill_healthy = await check_server_health(args.prefill_url)

    print(f"  Decode server:  {'✓ UP' if decode_healthy else '✗ DOWN'}")
    print(f"  Replica server: {'✓ UP' if replica_healthy else '✗ DOWN'}")
    print(f"  Prefill server: {'✓ UP' if prefill_healthy else '✗ DOWN'}")

    if not decode_healthy:
        print("\n⚠️  Decode server not available. Skipping decode tests.")
        print("   Start PD/PPD servers first: ./scripts/server/start_servers_4gpu.sh ppd")
        return

    # Test 2: Send request directly to Decode server (bypass proxy)
    print("\n[Test 2] Direct request to Decode server (kv_consumer)...")
    print("  This tests if D can work as standalone Replica")

    print("\n  2a. Short prompt:")
    result = await send_request(args.decode_url, short_prompt, max_tokens=16)
    results["decode_short"] = result
    if result["success"]:
        print(f"      ✓ SUCCESS! TTFT: {result['ttft_ms']:.1f}ms, Tokens: {result['output_tokens']}")
        print(f"      Response: {result['response']}")
    else:
        print(f"      ✗ FAILED: {result['error']}")

    print("\n  2b. Medium prompt (256 tokens context):")
    result = await send_request(args.decode_url, medium_prompt, max_tokens=32)
    results["decode_medium"] = result
    if result["success"]:
        print(f"      ✓ SUCCESS! TTFT: {result['ttft_ms']:.1f}ms, Tokens: {result['output_tokens']}")
        print(f"      Response: {result['response']}")
    else:
        print(f"      ✗ FAILED: {result['error']}")

    # Test 3: Compare with Replica server (if available)
    if replica_healthy:
        print("\n[Test 3] Direct request to Replica server (baseline)...")

        print("\n  3a. Short prompt:")
        result = await send_request(args.replica_url, short_prompt, max_tokens=16)
        results["replica_short"] = result
        if result["success"]:
            print(f"      ✓ SUCCESS! TTFT: {result['ttft_ms']:.1f}ms, Tokens: {result['output_tokens']}")
        else:
            print(f"      ✗ FAILED: {result['error']}")

        print("\n  3b. Medium prompt:")
        result = await send_request(args.replica_url, medium_prompt, max_tokens=32)
        results["replica_medium"] = result
        if result["success"]:
            print(f"      ✓ SUCCESS! TTFT: {result['ttft_ms']:.1f}ms, Tokens: {result['output_tokens']}")
        else:
            print(f"      ✗ FAILED: {result['error']}")
    else:
        print("\n[Test 3] Replica server not available, skipping comparison.")

    # Test 4: Test Prefill server standalone (for completeness)
    if prefill_healthy:
        print("\n[Test 4] Direct request to Prefill server (kv_producer)...")
        print("  This tests if P can also work standalone")

        result = await send_request(args.prefill_url, short_prompt, max_tokens=16)
        results["prefill_short"] = result
        if result["success"]:
            print(f"      ✓ SUCCESS! TTFT: {result['ttft_ms']:.1f}ms, Tokens: {result['output_tokens']}")
        else:
            print(f"      ✗ FAILED: {result['error']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    decode_works = results.get("decode_short", {}).get("success", False)
    prefill_works = results.get("prefill_short", {}).get("success", False)

    if decode_works:
        print("✅ DECODE SERVER CAN WORK STANDALONE!")
        print("   kv_consumer configuration does NOT block standalone requests.")
        print("   This means pD servers CAN potentially act as Replica.")

        if "replica_short" in results and results["replica_short"]["success"]:
            decode_ttft = results["decode_short"]["ttft_ms"]
            replica_ttft = results["replica_short"]["ttft_ms"]
            overhead = decode_ttft - replica_ttft
            print(f"\n   Performance comparison (short prompt):")
            print(f"     Decode TTFT:  {decode_ttft:.1f}ms")
            print(f"     Replica TTFT: {replica_ttft:.1f}ms")
            print(f"     Overhead:     {overhead:+.1f}ms")

            if abs(overhead) < 50:
                print("\n   ✅ Overhead is minimal - pD can effectively replace Replica!")
            else:
                print(f"\n   ⚠️  Notable overhead ({overhead:.1f}ms) - consider implications.")
    else:
        print("❌ DECODE SERVER CANNOT WORK STANDALONE")
        print("   kv_consumer configuration may block standalone requests.")
        print("   Must use separate Replica servers (1P+1pD+2Replica setup).")

    if prefill_works:
        print("\n✅ PREFILL SERVER CAN ALSO WORK STANDALONE")
        print("   kv_producer servers can handle full requests if needed.")

    # Save results
    with open("results/decode_standalone_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: results/decode_standalone_test.json")


if __name__ == "__main__":
    asyncio.run(main())
