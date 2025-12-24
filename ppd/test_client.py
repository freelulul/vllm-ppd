#!/usr/bin/env python3
"""
Test client for vLLM PD Separation
Sends requests and collects timing metrics
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_time: float  # Total request time (client-side)
    ttft: Optional[float] = None  # Time to first token (if streaming)
    response_text: str = ""
    error: Optional[str] = None


def send_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    stream: bool = False,
) -> RequestMetrics:
    """Send a completion request and measure timing"""

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    start_time = time.perf_counter()
    first_token_time = None
    response_text = ""

    try:
        if stream:
            # Streaming request
            response = requests.post(
                f"{url}/v1/completions",
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            if chunk.get('choices'):
                                response_text += chunk['choices'][0].get('text', '')
                        except json.JSONDecodeError:
                            pass
        else:
            # Non-streaming request
            response = requests.post(
                f"{url}/v1/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            if 'choices' in result and result['choices']:
                response_text = result['choices'][0].get('text', '')

            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)

    except Exception as e:
        end_time = time.perf_counter()
        return RequestMetrics(
            prompt=prompt,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            request_time=end_time - start_time,
            error=str(e),
        )

    end_time = time.perf_counter()

    # Estimate tokens if not provided
    if 'usage' not in result:
        # Rough estimation: 4 chars per token
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response_text) // 4
        total_tokens = prompt_tokens + completion_tokens

    return RequestMetrics(
        prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        request_time=end_time - start_time,
        ttft=first_token_time - start_time if first_token_time else None,
        response_text=response_text[:100] + "..." if len(response_text) > 100 else response_text,
    )


def run_test_suite(url: str, model: str):
    """Run a suite of test requests"""

    test_cases = [
        # (prompt, max_tokens, description)
        ("Hello, world!", 10, "Very short prompt"),
        ("San Francisco is a", 20, "Short prompt"),
        ("The capital of France is", 20, "Simple question"),
        ("In the field of artificial intelligence, " * 10, 50, "Medium prompt (~400 chars)"),
        ("Explain the concept of machine learning in simple terms. " * 5, 100, "Longer prompt with more output"),
    ]

    print("\n" + "=" * 80)
    print("vLLM PD Separation Test Suite")
    print("=" * 80)
    print(f"Target URL: {url}")
    print(f"Model: {model}")
    print("=" * 80 + "\n")

    results = []

    for i, (prompt, max_tokens, description) in enumerate(test_cases, 1):
        print(f"[Test {i}/{len(test_cases)}] {description}")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Max tokens: {max_tokens}")

        metrics = send_request(url, model, prompt, max_tokens)
        results.append((description, metrics))

        if metrics.error:
            print(f"  ERROR: {metrics.error}")
        else:
            print(f"  Prompt tokens: {metrics.prompt_tokens}")
            print(f"  Completion tokens: {metrics.completion_tokens}")
            print(f"  Request time: {metrics.request_time:.3f}s")
            if metrics.ttft:
                print(f"  TTFT: {metrics.ttft:.3f}s")
            print(f"  Response: {metrics.response_text}")

        print()
        time.sleep(0.5)  # Small delay between requests

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Test':<40} {'Prompt Tok':<12} {'Comp Tok':<10} {'Time (s)':<10}")
    print("-" * 80)

    total_time = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for description, metrics in results:
        if not metrics.error:
            print(f"{description:<40} {metrics.prompt_tokens:<12} {metrics.completion_tokens:<10} {metrics.request_time:<10.3f}")
            total_time += metrics.request_time
            total_prompt_tokens += metrics.prompt_tokens
            total_completion_tokens += metrics.completion_tokens

    print("-" * 80)
    print(f"{'Total':<40} {total_prompt_tokens:<12} {total_completion_tokens:<10} {total_time:<10.3f}")
    print(f"\nAverage request time: {total_time / len(results):.3f}s")
    print(f"Total throughput: {(total_prompt_tokens + total_completion_tokens) / total_time:.1f} tokens/s")


def main():
    parser = argparse.ArgumentParser(description="Test client for vLLM PD Separation")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the proxy server")
    parser.add_argument("--model", default="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
                        help="Model name/path")
    parser.add_argument("--prompt", help="Single prompt to test (optional)")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens for single prompt")

    args = parser.parse_args()

    if args.prompt:
        # Single request mode
        print(f"Sending single request to {args.url}")
        metrics = send_request(args.url, args.model, args.prompt, args.max_tokens)
        print(f"Response: {metrics.response_text}")
        print(f"Request time: {metrics.request_time:.3f}s")
        if metrics.error:
            print(f"Error: {metrics.error}")
    else:
        # Run full test suite
        run_test_suite(args.url, args.model)


if __name__ == "__main__":
    main()
