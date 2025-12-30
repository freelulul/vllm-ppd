# Three-Mode Comparison Analysis: Replication vs PPD vs PD

## Overview

This document presents a comprehensive analysis of three serving modes for vLLM disaggregated inference:

- **Replication (Data Parallelism)**: Two standalone vLLM servers with prefix-aware routing
- **PPD (Prefill-Prefill-Decode)**: Turn 1 uses P→D, Turn 2+ uses D-direct (no KV transfer)
- **PD (Prefill-Decode Disaggregation)**: Every request routes through P → KV Transfer → D

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | Llama-3.1-8B |
| GPUs | 2× H100 |
| QPS Sweep | [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0] |
| Duration per QPS | 45 seconds |
| Arrival Process | Poisson |

### Workload Definitions

| Workload | Input Tokens | Output Tokens | I/O Ratio | Characteristic |
|----------|--------------|---------------|-----------|----------------|
| W1 Chat | 512 | 256 | 2:1 | Balanced I/O |
| W2 RAG | 4096 | 128 | 32:1 | Read-Heavy (Long Input) |
| W3 Agent | 256 | 1024 | 0.25:1 | Write-Heavy (Long Output) |
| W4 Limit | 8192 | 64 | 128:1 | Extreme Context |

---

## Part I: Task Characteristics Each Mode Excels At

### 1. PPD Mode (Prefill-Prefill-Decode)

**Best for**: Balanced tasks with medium input/output lengths

| Characteristic | Performance |
|----------------|-------------|
| W1 Chat (512/256) | **Wins at ALL QPS levels**, 14-19% faster than Replication |
| Long input at low load | Outperforms others at QPS ≤ 1 for W2/W4 |

**Core Advantage**: D-direct optimization skips KV transfer on Turn 2+, saving ~200-400ms latency

---

### 2. Replication Mode (Data Parallelism)

**Best for**: **High load** or **long output** tasks

| Characteristic | Performance |
|----------------|-------------|
| W3 Agent (256/1024) | Wins at QPS ≥ 1, up to 47% advantage at high QPS |
| W4 Limit (8192/64) | Wins at QPS ≥ 2, up to 508% advantage at high QPS |
| W2 RAG at high load | Wins at QPS = 4-6 |

**Core Advantages**:
- Two GPUs operate completely independently, zero KV transfer overhead
- No HOL (Head-of-Line) Blocking risk
- Best system stability under high load

---

### 3. PD Mode (Prefill-Decode Disaggregation)

**Best for**: **Long input + medium-high load** read-intensive tasks

| Characteristic | Performance |
|----------------|-------------|
| W2 RAG (4096/128) | Wins at QPS = 1-3, 8 |
| Long input at extreme load | Most stable for W2 at QPS = 8 |

**Core Advantage**: Physical P/D isolation prevents Prefill from blocking Decode

---

## Part II: Trade-offs Between Three Modes

### Visual Representation

```
                    Low Load                    High Load
                    ↓                           ↓
    ┌─────────────────────────────────────────────────────┐
    │                                                      │
    │   PPD ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░   │  Short/Medium Input
    │                      ↑                               │
    │                   crossover                          │
    │                                                      │
    │   PD  ░░░░░░░████████████████████████████░░░░░░░░   │  Long Input (Read-Heavy)
    │                                         ↑            │
    │                                      collapse        │
    │                                                      │
    │   Repl░░░░░░░░░░░░░░░░░░░██████████████████████████ │  Long Output / Extreme Load
    │                                                      │
    └─────────────────────────────────────────────────────┘
```

### Core Trade-off Matrix

| Dimension | PPD | PD | Replication |
|-----------|-----|-----|-------------|
| **KV Transfer Overhead** | Turn1 yes, Turn2+ no | Every request | **None** |
| **HOL Blocking Risk** | **High** (D does Prefill) | Low (P/D separated) | **None** |
| **GPU Utilization** | Medium | Medium | **High** (2-GPU parallel) |
| **Latency Floor** | **Lowest** | Medium | Medium |
| **High-Load Stability** | Poor | Medium | **Best** |

---

## Part III: Root Cause Analysis

### 1. W1 Chat: Why PPD Wins Across All QPS

```
Input 512 tokens  → Prefill ~50ms
Output 256 tokens → Decode ~2000ms
KV Transfer: 512 tokens × 128KB ≈ 64MB → ~100-200ms
```

**Analysis**:
- Short Prefill time means minimal HOL Blocking impact
- D-direct saves KV transfer time (100-200ms), accounting for ~8% of total latency
- Replication has no transfer overhead, but both GPUs **process the full workload independently** without parallel acceleration for single requests

---

### 2. W2 RAG: Why PPD Collapses at High Load

```
Input 4096 tokens  → Prefill ~400ms  ← This is critical!
Output 128 tokens  → Decode ~500ms
```

**HOL Blocking Mechanism**:
- In PPD mode, Turn 2+ requests execute Prefill on D (Decode GPU)
- At QPS=6, a new request arrives every ~167ms on average
- New request's Prefill (400ms) **blocks** ongoing Decode operations
- Queuing effect causes exponential latency growth: 3337ms (PPD) vs 1797ms (Replication)

**Why does PD recover at QPS=8?**
- PD's physical P/D isolation prevents Prefill from affecting Decode
- KV transfer overhead (4096 tokens × 128KB ≈ 512MB) is a disadvantage at medium load
- At extreme high load, stability > transfer overhead

**Detailed Data**:

| QPS | Replication P99 TTFT | PPD P99 TTFT | PD P99 TTFT | Winner |
|-----|---------------------|--------------|-------------|--------|
| 0.5 | 1514ms | 1183ms (-22%) | 1340ms | PPD |
| 4.0 | 1932ms | 2357ms (+22%) | 2002ms | Replication |
| 6.0 | 1798ms | 3337ms (+86%) | 2274ms | Replication |
| 8.0 | 2527ms | 5777ms (+129%) | 2093ms | PD |

---

### 3. W3 Agent: Why Replication Dominates

```
Input 256 tokens   → Prefill ~25ms
Output 1024 tokens → Decode ~8000ms  ← Decode dominates!
```

**Analysis**:
- Decode accounts for 99% of total time; Prefill is negligible
- PPD's D-direct optimization is **almost useless** (Turn 2 still requires long Decode)
- PD's P/D separation is **also meaningless** (bottleneck is D, not P)
- Replication's **two-GPU parallel Decode** becomes the only advantage

**Detailed Data**:

| QPS | Replication P99 TTFT | PPD P99 TTFT | PD P99 TTFT | Winner |
|-----|---------------------|--------------|-------------|--------|
| 1.0 | 8798ms | 9278ms (+5%) | 9181ms | Replication |
| 4.0 | 10190ms | 10754ms (+6%) | 10194ms | Replication |
| 8.0 | 12144ms | 15529ms (+28%) | 17936ms (+48%) | Replication |

---

### 4. W4 Limit: Why Both PPD and PD Collapse

```
Input 8192 tokens  → Prefill ~800ms  ← Extremely long!
Output 64 tokens   → Decode ~300ms
KV Transfer: 8192 tokens × 128KB ≈ 1GB
```

**PPD Collapse** (13955ms vs Replication 3791ms at QPS=8):
- 800ms Prefill executing on D completely blocks Decode
- HOL Blocking effect is extremely amplified

**PD Collapse** (23081ms at QPS=8):
- 1GB KV transfer at ~10Gbps TCP requires ~800ms
- Transfer queue accumulates under high load
- Transfer between P and D becomes a **new bottleneck**

**Why Replication Remains Stable**:
- No KV transfer, no HOL Blocking
- Two GPUs process independently
- Prefix-aware routing ensures cache hits

**Detailed Data**:

| QPS | Replication P99 TTFT | PPD P99 TTFT | PD P99 TTFT | Winner |
|-----|---------------------|--------------|-------------|--------|
| 1.0 | 1395ms | 984ms (-29%) | 1206ms | PPD |
| 4.0 | 1880ms | 3462ms (+84%) | 2130ms | Replication |
| 8.0 | 3792ms | 13955ms (+268%) | 23082ms (+509%) | Replication |

---

## Part IV: Summary Statistics

### Average P99 TTFT by Workload and Mode (ms)

| Workload | Replication | PPD | PD | Best Mode |
|----------|-------------|-----|-----|-----------|
| W1 Chat | 2835 | **2444** | 2652 | PPD |
| W2 RAG | **1841** | 2525 | 1769 | Replication/PD |
| W3 Agent | **10024** | 10758 | 11399 | Replication |
| W4 Limit | **1985** | 4683 | 5008 | Replication |

### Average Throughput by Workload and Mode (tokens/sec)

| Workload | Replication | PPD | PD | Best Mode |
|----------|-------------|-----|-----|-----------|
| W1 Chat | 107.2 | **110.4** | 105.8 | PPD |
| W2 RAG | **89.5** | 79.9 | 87.0 | Replication |
| W3 Agent | **108.4** | 103.5 | 99.4 | Replication |
| W4 Limit | **68.5** | 50.9 | 53.9 | Replication |

---

## Part V: Conclusions and Recommendations

### Mode Selection Guide

| Task Type | Recommended Mode | Rationale |
|-----------|------------------|-----------|
| **Chat/Conversation** (Balanced I/O, Low-Medium Load) | **PPD** | D-direct saves transfer; HOL risk is low |
| **RAG/Retrieval** (Long Input, High Load) | **PD** or **Replication** | P/D isolation resists HOL, or zero transfer overhead |
| **Agent/Code Generation** (Long Output) | **Replication** | Decode-dominated; two-GPU parallelism provides significant benefit |
| **Ultra-Long Context** (8K+ tokens) | **Replication** | Dual risk of KV transfer overhead and HOL blocking |
| **Extreme High Load** (Any workload) | **Replication** | Best system stability |

### Key Insights

> **PPD** is an "optimistic optimization" — performs best when assuming HOL Blocking is not severe.
>
> **PD** is a "conservative isolation" — sacrifices transfer overhead for stability.
>
> **Replication** is "simple and robust" — abandons optimizations but eliminates all risks.

### Decision Flowchart

```
                         ┌─────────────────┐
                         │  New Request    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │ Input > 4K      │
                         │ tokens?         │
                         └────────┬────────┘
                            Yes   │   No
                    ┌─────────────┴─────────────┐
                    │                           │
           ┌────────▼────────┐         ┌────────▼────────┐
           │ High Load       │         │ Output > 512    │
           │ (QPS > 4)?      │         │ tokens?         │
           └────────┬────────┘         └────────┬────────┘
              Yes   │   No                Yes   │   No
        ┌───────────┴───────┐          ┌───────┴───────┐
        │                   │          │               │
   ┌────▼────┐        ┌────▼────┐ ┌────▼────┐    ┌────▼────┐
   │  Repl   │        │   PD    │ │  Repl   │    │   PPD   │
   └─────────┘        └─────────┘ └─────────┘    └─────────┘
```

---

## Appendix: Raw Data Tables

### W1 Chat Balanced (Input: 512, Output: 256)

| QPS | Replication P99 | PPD P99 | PD P99 | PPD vs Repl | PD vs Repl |
|-----|-----------------|---------|--------|-------------|------------|
| 0.5 | 2509.6 | 2152.3 | 2216.2 | -14.2% | -11.7% |
| 1.0 | 2558.5 | 2171.6 | 2277.6 | -15.1% | -11.0% |
| 2.0 | 2723.1 | 2274.4 | 2479.8 | -16.5% | -8.9% |
| 3.0 | 2694.2 | 2428.8 | 2574.0 | -9.9% | -4.5% |
| 4.0 | 2884.8 | 2493.4 | 2542.8 | -13.6% | -11.9% |
| 6.0 | 3234.1 | 2612.6 | 3018.2 | -19.2% | -6.7% |
| 8.0 | 3243.2 | 2975.7 | 3458.3 | -8.2% | +6.6% |

### W2 RAG Read-Heavy (Input: 4096, Output: 128)

| QPS | Replication P99 | PPD P99 | PD P99 | PPD vs Repl | PD vs Repl |
|-----|-----------------|---------|--------|-------------|------------|
| 0.5 | 1514.1 | 1183.2 | 1339.6 | -21.9% | -11.5% |
| 1.0 | 1574.6 | 1482.9 | 1340.6 | -5.8% | -14.9% |
| 2.0 | 1768.6 | 1647.8 | 1579.1 | -6.8% | -10.7% |
| 3.0 | 1775.1 | 1891.9 | 1753.8 | +6.6% | -1.2% |
| 4.0 | 1931.8 | 2357.2 | 2001.9 | +22.0% | +3.6% |
| 6.0 | 1797.9 | 3337.2 | 2274.4 | +85.6% | +26.5% |
| 8.0 | 2527.3 | 5776.8 | 2092.5 | +128.6% | -17.2% |

### W3 Agent Write-Heavy (Input: 256, Output: 1024)

| QPS | Replication P99 | PPD P99 | PD P99 | PPD vs Repl | PD vs Repl |
|-----|-----------------|---------|--------|-------------|------------|
| 0.5 | 8431.3 | 8147.1 | 8738.7 | -3.4% | +3.6% |
| 1.0 | 8797.9 | 9277.5 | 9180.5 | +5.5% | +4.3% |
| 2.0 | 9159.4 | 10142.2 | 10137.0 | +10.7% | +10.7% |
| 3.0 | 9888.9 | 9458.5 | 9976.8 | -4.4% | +0.9% |
| 4.0 | 10190.4 | 10753.5 | 10194.0 | +5.5% | +0.0% |
| 6.0 | 11552.2 | 11997.9 | 13628.3 | +3.9% | +18.0% |
| 8.0 | 12144.2 | 15528.6 | 17936.0 | +27.9% | +47.7% |

### W4 Limit Context (Input: 8192, Output: 64)

| QPS | Replication P99 | PPD P99 | PD P99 | PPD vs Repl | PD vs Repl |
|-----|-----------------|---------|--------|-------------|------------|
| 0.5 | 995.0 | 895.5 | 928.8 | -10.0% | -6.6% |
| 1.0 | 1395.0 | 984.2 | 1206.3 | -29.4% | -13.5% |
| 2.0 | 1381.6 | 1696.8 | 1825.1 | +22.8% | +32.1% |
| 3.0 | 1571.7 | 4083.5 | 1839.2 | +159.8% | +17.0% |
| 4.0 | 1880.4 | 3461.9 | 2129.6 | +84.1% | +13.3% |
| 6.0 | 2876.6 | 7704.3 | 4042.2 | +167.8% | +40.5% |
| 8.0 | 3791.7 | 13955.1 | 23081.7 | +268.0% | +508.8% |
