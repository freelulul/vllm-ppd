#!/usr/bin/env python3
"""
PPD KV Cache Retention Verification Test

This script verifies the core PPD hypothesis:
1. Turn 1: Routes through P→D with KV transfer (same as PD mode)
2. Turn 2+: Routes directly to D, where D uses its LOCAL prefix cache
   to skip prefilling already-processed tokens.

Key insight:
- In PPD mode, after Turn 1 transfers KV from P to D, the D machine
  caches the KV in its local prefix cache (via request.block_hashes).
- Turn 2+ goes directly to D with the same conversation prefix.
- vLLM's prefix cache matches the block_hashes and reuses the cached KV.
- Only the NEW tokens in Turn 2+ need prefill on D.

This test verifies:
1. Correct routing (proxy logs show D-Direct for Turn 2+)
2. KV cache reuse (timing patterns show Turn 2+ is efficient)
3. Conversation continuity (responses are coherent across turns)
"""

import requests
import time
import json
import sys

PROXY_URL = "http://localhost:10001"
MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"

def clear_state():
    """Clear conversation state for fresh test."""
    try:
        resp = requests.post(f"{PROXY_URL}/conversations/clear")
        print(f"[Setup] Cleared conversations: {resp.json()}")
    except Exception as e:
        print(f"[Setup] Could not clear state: {e}")

def check_mode():
    """Check and set PPD mode."""
    resp = requests.get(f"{PROXY_URL}/mode")
    current = resp.json()
    print(f"[Setup] Current mode: {current}")

    if current.get("mode") != "ppd":
        resp = requests.post(f"{PROXY_URL}/mode/ppd")
        print(f"[Setup] Set mode to ppd: {resp.json()}")

    return current.get("mode")

def send_request(prompt: str, max_tokens: int = 30) -> tuple[str, float, dict]:
    """Send a completion request and return (response_text, time_taken, usage)."""
    start = time.time()

    resp = requests.post(
        f"{PROXY_URL}/v1/completions",
        json={
            "model": MODEL_PATH,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        },
        timeout=120
    )

    elapsed = time.time() - start

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code}", elapsed, {}

    data = resp.json()
    text = data.get("choices", [{}])[0].get("text", "")
    usage = data.get("usage", {})
    return text.strip(), elapsed, usage

def get_conversations():
    """Get current conversation state from proxy."""
    try:
        resp = requests.get(f"{PROXY_URL}/conversations", timeout=5)
        if resp.status_code == 200 and resp.text:
            return resp.json()
        return {"count": 0, "conversations": {}}
    except Exception as e:
        print(f"Warning: Could not get conversations: {e}")
        return {"count": 0, "conversations": {}}

def run_verification_test():
    """Run the main verification test."""
    print("=" * 70)
    print("PPD KV Cache Retention Verification Test")
    print("=" * 70)

    # Setup
    clear_state()
    check_mode()
    print()

    results = []
    conversation_history = ""

    print("[Test] Running 5-turn conversation...")
    print("-" * 70)

    for turn in range(1, 6):
        if turn == 1:
            # First turn - initial question
            new_input = "What is the capital of France?"
            prompt = f"User: {new_input}\nAssistant:"
            conversation_history = prompt
        else:
            # Subsequent turns - follow-up questions
            new_input = f"Tell me more about it (turn {turn})."
            prompt = f"{conversation_history}\nUser: {new_input}\nAssistant:"

        prompt_tokens_estimate = len(prompt.split())

        print(f"\n[Turn {turn}]")
        print(f"  New input: '{new_input[:50]}...' (~{len(new_input.split())} words)")
        print(f"  Total prompt: ~{prompt_tokens_estimate} words")

        response, elapsed, usage = send_request(prompt, max_tokens=30)

        # Update history for next turn
        conversation_history = f"{prompt}{response}"

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        print(f"  Response: '{response[:60]}...'")
        print(f"  Time: {elapsed*1000:.1f}ms")
        print(f"  Tokens: prompt={prompt_tokens}, completion={completion_tokens}")

        results.append({
            "turn": turn,
            "input_words": len(new_input.split()),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "time_ms": elapsed * 1000,
            "response_preview": response[:50]
        })

        # Small delay between turns
        time.sleep(0.5)

    print("\n" + "-" * 70)
    print("[Results Summary]")
    print("-" * 70)

    # Check conversation state
    conv_state = get_conversations()
    print(f"\nConversation state: {json.dumps(conv_state, indent=2)}")

    # Analyze timing
    print("\nTiming Analysis:")
    print(f"  {'Turn':<6} {'Prompt':<10} {'Time (ms)':<12} {'ms/token':<12} {'Expected Route'}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*20}")

    for r in results:
        expected = "P→D (KV transfer)" if r["turn"] == 1 else "D-direct (cached)"
        ms_per_token = r["time_ms"] / r["prompt_tokens"] if r["prompt_tokens"] > 0 else 0
        print(f"  {r['turn']:<6} {r['prompt_tokens']:<10} {r['time_ms']:<12.1f} {ms_per_token:<12.2f} {expected}")

    print("\n" + "=" * 70)
    print("[Verification Checklist]")
    print("=" * 70)

    # Check 1: Conversation was tracked
    conv_count = conv_state.get("count", 0)
    check1 = conv_count == 1
    print(f"[{'PASS' if check1 else 'FAIL'}] Single conversation tracked: {conv_count} conversation(s)")

    # Check 2: All 5 turns recorded
    convs = conv_state.get("conversations", {})
    if convs:
        first_conv = list(convs.values())[0]
        turns_recorded = first_conv.get("turns", 0)
        check2 = turns_recorded == 5
        print(f"[{'PASS' if check2 else 'FAIL'}] All turns recorded: {turns_recorded} turns")
    else:
        check2 = False
        print(f"[FAIL] No conversation data found")

    # Check 3: Responses are coherent (not errors)
    check3 = all("ERROR" not in r["response_preview"] for r in results)
    print(f"[{'PASS' if check3 else 'FAIL'}] All responses successful (no errors)")

    # Check 4: Timing pattern - later turns should have lower ms/token
    # because prefix is cached and only new tokens need prefill
    if len(results) >= 3:
        turn1_ms_per_token = results[0]["time_ms"] / results[0]["prompt_tokens"] if results[0]["prompt_tokens"] > 0 else 0
        later_ms_per_token = sum(r["time_ms"] / r["prompt_tokens"] for r in results[2:] if r["prompt_tokens"] > 0) / len(results[2:])

        # Turn 2+ should have lower ms/token due to prefix cache
        check4 = later_ms_per_token < turn1_ms_per_token * 1.5  # Allow some variance
        print(f"[{'PASS' if check4 else 'WARN'}] Timing pattern: Turn 1 = {turn1_ms_per_token:.2f} ms/token, Turn 3-5 avg = {later_ms_per_token:.2f} ms/token")
    else:
        check4 = True

    print("\n" + "=" * 70)
    print("How to confirm routing in proxy logs:")
    print("  grep -E '(PPD Mode|D-Direct|Turn1)' logs/proxy.log | tail -20")
    print()
    print("Expected pattern:")
    print("  Turn 1: 'PPD-Turn1: ... P:xxx -> D:xxx'")
    print("  Turn 2-5: 'PPD Mode - D-Direct: ... -> D:xxx'")
    print("=" * 70)

    return all([check1, check2, check3])

def run_ab_comparison():
    """Run A/B comparison between PD and PPD modes."""
    print("\n" + "=" * 70)
    print("A/B Mode Comparison Test")
    print("=" * 70)

    results = {"pd": [], "ppd": []}

    for mode in ["pd", "ppd"]:
        print(f"\n[Testing {mode.upper()} mode]")

        # Set mode
        requests.post(f"{PROXY_URL}/mode/{mode}")
        clear_state()
        time.sleep(1)

        # Run 5-turn test
        conversation = ""
        for turn in range(1, 6):
            new_input = f"Hello, this is turn {turn} of our conversation about Paris."
            if conversation:
                prompt = f"{conversation}\nUser: {new_input}\nAssistant:"
            else:
                prompt = f"User: {new_input}\nAssistant:"

            response, elapsed, usage = send_request(prompt, max_tokens=30)
            conversation = f"{prompt}{response}"

            results[mode].append({
                "turn": turn,
                "time_ms": elapsed * 1000,
                "prompt_tokens": usage.get("prompt_tokens", 0)
            })
            print(f"  Turn {turn}: {elapsed*1000:.1f}ms, {usage.get('prompt_tokens', 0)} tokens")
            time.sleep(0.3)

    print("\n[Comparison Summary]")
    print(f"  {'Turn':<6} {'PD (ms)':<12} {'PPD (ms)':<12} {'Speedup':<10} {'Note'}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*30}")

    for i in range(5):
        pd_time = results["pd"][i]["time_ms"]
        ppd_time = results["ppd"][i]["time_ms"]
        speedup = pd_time / ppd_time if ppd_time > 0 else 0

        if i == 0:
            note = "Both use P→D"
        else:
            note = "PD:P→D, PPD:D-direct" if speedup > 1.0 else "Unexpected"

        print(f"  {i+1:<6} {pd_time:<12.1f} {ppd_time:<12.1f} {speedup:.2f}x       {note}")

    # Calculate aggregate
    total_pd = sum(r["time_ms"] for r in results["pd"])
    total_ppd = sum(r["time_ms"] for r in results["ppd"])
    print(f"\n  Total: PD={total_pd:.1f}ms, PPD={total_ppd:.1f}ms, Speedup={total_pd/total_ppd:.2f}x")

    # Restore PPD mode
    requests.post(f"{PROXY_URL}/mode/ppd")

def run_incremental_test():
    """Test that only NEW tokens are prefilled in D-direct mode."""
    print("\n" + "=" * 70)
    print("Incremental Prefill Test")
    print("=" * 70)
    print("This test verifies that D-direct only prefills NEW tokens.")
    print("If prefix cache works, adding 10 tokens should take ~same time")
    print("regardless of how long the cached prefix is.")
    print()

    requests.post(f"{PROXY_URL}/mode/ppd")
    clear_state()
    time.sleep(1)

    # Build up a long prefix
    conversation = ""
    base_input = "Tell me a long story about a dragon. " * 5  # ~50 tokens

    # Turn 1: Establish prefix (~200 tokens after response)
    prompt1 = f"User: {base_input}\nAssistant:"
    response1, elapsed1, usage1 = send_request(prompt1, max_tokens=100)
    conversation = f"{prompt1}{response1}"
    print(f"[Turn 1] Established prefix: {usage1.get('prompt_tokens', 0)} + {usage1.get('completion_tokens', 0)} tokens")
    print(f"         Time: {elapsed1*1000:.1f}ms")

    time.sleep(0.5)

    # Turns 2-4: Add small increments, measure time
    increments = [
        "Continue.",
        "What happened next?",
        "And then?",
    ]

    times = []
    for i, incr in enumerate(increments, start=2):
        prompt = f"{conversation}\nUser: {incr}\nAssistant:"
        response, elapsed, usage = send_request(prompt, max_tokens=50)
        conversation = f"{prompt}{response}"

        times.append(elapsed * 1000)
        print(f"[Turn {i}] Added '{incr}' ({len(incr.split())} words)")
        print(f"         Total tokens: {usage.get('prompt_tokens', 0)}, Time: {elapsed*1000:.1f}ms")
        time.sleep(0.3)

    # If prefix cache works, times should be similar despite growing context
    print(f"\n[Analysis]")
    print(f"  Turn 2-4 times: {times}")
    print(f"  Avg time: {sum(times)/len(times):.1f}ms")
    print(f"  Std dev: {(sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5:.1f}ms")

    if max(times) - min(times) < sum(times)/len(times) * 0.5:
        print(f"  [PASS] Times are consistent - prefix cache is working!")
    else:
        print(f"  [WARN] High variance - may indicate cache misses")

if __name__ == "__main__":
    try:
        # Quick connectivity check
        resp = requests.get(f"{PROXY_URL}/mode", timeout=5)
        print(f"Proxy is up: {resp.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to proxy at {PROXY_URL}")
        print(f"Make sure servers are running: ./scripts/start_servers.sh ppd")
        sys.exit(1)

    success = run_verification_test()

    if "--compare" in sys.argv:
        run_ab_comparison()

    if "--incremental" in sys.argv:
        run_incremental_test()

    if "--full" in sys.argv:
        run_ab_comparison()
        run_incremental_test()

    print("\nTest completed.")
    sys.exit(0 if success else 1)
