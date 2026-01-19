#!/usr/bin/env python3
"""
OOM Stress Test for 1R_1P_1D_1pD Configuration

High-load test with large context (3000 input + 3000 output = 6000 tokens)
to observe memory behavior differences across machine types.
"""

import asyncio
import aiohttp
import time
import random

MODEL_PATH = '/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B'

async def oom_stress_test():
    """
    High-load OOM test - 1R_1P_1D_1pD configuration
    Observe OOM behavior differences across different machine types
    """
    url = 'http://localhost:10001/v1/completions'

    QPS = 16  # Slightly reduced from 20 to 16
    DURATION = 20
    INPUT_TOKENS = 3000
    OUTPUT_TOKENS = 3000

    print(f'=== OOM Stress Test: 1R_1P_1D_1pD ===')
    print(f'QPS: {QPS}, Duration: {DURATION}s')
    print(f'Input: ~{INPUT_TOKENS} tokens, Output: {OUTPUT_TOKENS} tokens')
    print(f'Total: ~{INPUT_TOKENS + OUTPUT_TOKENS} tokens per request')
    print(f'Expected total requests: {QPS * DURATION} = {QPS * DURATION}')
    print()

    stats = {'success': 0, 'fail': 0, 'timeout': 0, 'errors': [], 'latencies': []}

    def make_prompt():
        # ~3000 tokens (4 words repeated 750 times)
        words = ['hello', 'world', 'test', 'data'] * 750
        return ' '.join(words)

    async def single_request(i):
        payload = {
            'model': MODEL_PATH,
            'prompt': make_prompt(),
            'max_tokens': OUTPUT_TOKENS,
            'temperature': 0.7,
            'ignore_eos': True,  # ← CRITICAL: Force generation to max_tokens
            'stream': False
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                start = time.time()
                async with session.post(url, json=payload) as resp:
                    result = await resp.json()
                    elapsed = time.time() - start

                    if 'error' in result:
                        stats['fail'] += 1
                        err = result.get('error', {}).get('message', '')[:100]
                        if err not in stats['errors']:
                            stats['errors'].append(err)
                        print(f'  Req {i}: ✗ Error - {err[:50]}')
                        return f'Req {i}: Error'
                    else:
                        stats['success'] += 1
                        stats['latencies'].append(elapsed)
                        if i % 10 == 0:
                            print(f'  Req {i}: ✓ OK ({elapsed:.1f}s)')
                        return f'Req {i}: OK {elapsed:.1f}s'
        except asyncio.TimeoutError:
            stats['timeout'] += 1
            print(f'  Req {i}: ⏱ Timeout (>180s)')
            return f'Req {i}: Timeout'
        except Exception as e:
            stats['fail'] += 1
            err_msg = f'{type(e).__name__}: {str(e)[:50]}'
            if err_msg not in stats['errors']:
                stats['errors'].append(err_msg)
            print(f'  Req {i}: ✗ {type(e).__name__}')
            return f'Req {i}: {type(e).__name__}'

    tasks = []
    start_time = time.time()
    req_count = 0

    print(f'Sending requests...')
    print()

    # Poisson arrival process
    while time.time() - start_time < DURATION:
        task = asyncio.create_task(single_request(req_count))
        tasks.append(task)
        req_count += 1

        # Exponential inter-arrival time
        interval = random.expovariate(QPS)
        await asyncio.sleep(interval)

    print()
    print(f'Sent {req_count} requests, waiting for completion...')
    print()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time

    print()
    print(f'=== Results ===')
    print(f'Total requests: {req_count}')
    print(f'Success: {stats["success"]}')
    print(f'Failed: {stats["fail"]}')
    print(f'Timeout: {stats["timeout"]}')
    print(f'Success rate: {stats["success"]/req_count*100:.1f}%')
    print(f'Total duration: {total_time:.1f}s')

    if stats['latencies']:
        import statistics
        print(f'\nLatency stats:')
        print(f'  Avg: {statistics.mean(stats["latencies"]):.1f}s')
        print(f'  P50: {statistics.median(stats["latencies"]):.1f}s')
        if len(stats['latencies']) > 1:
            p99_idx = int(len(stats['latencies']) * 0.99)
            sorted_lat = sorted(stats['latencies'])
            print(f'  P99: {sorted_lat[p99_idx]:.1f}s')

    if stats['errors']:
        print(f'\nError types:')
        for err in stats['errors'][:5]:
            print(f'  - {err}')

    print()
    print('=== Test Complete ===')

if __name__ == '__main__':
    asyncio.run(oom_stress_test())
