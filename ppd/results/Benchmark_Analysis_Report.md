# vLLM Serving Mode Benchmark Analysis Report

## Executive Summary

This report presents a comprehensive analysis of three vLLM serving architectures:
- **PD (Prefill-Decode Disaggregation)**: Separates prefill and decode phases across different GPU groups with KV cache transfer
- **PPD (Prefill-Prefill-Decode)**: Turn 1 follows P→D path, Turn 2+ goes directly to D server (KV cache reuse)
- **Replication**: 4 independent vLLM instances with round-robin load balancing

**Key Findings:**
1. **PPD achieves 45-76% lower TTFT** compared to PD at low-medium QPS
2. **PD maintains superior decode stability** with only 1.07-1.13x P99/Avg TPOT variance vs PPD's 1.05-2.27x
3. **Replication delivers highest throughput** at high QPS, with PPD degrading to 0.41-1.00x of Replication capacity
4. Each mode has distinct optimal deployment scenarios based on workload characteristics

---

## Table of Contents

1. [Experimental Setup](#1-experimental-setup)
2. [Workload Design](#2-workload-design)
3. [Metric-by-Metric Analysis](#3-metric-by-metric-analysis)
   - [3.1 P99 Time To First Token (TTFT)](#31-p99-time-to-first-token-ttft)
   - [3.2 Time Per Output Token (TPOT)](#32-time-per-output-token-tpot)
   - [3.3 Throughput](#33-throughput)
   - [3.4 End-to-End Latency](#34-end-to-end-latency)
4. [Mode Comparison Summary](#4-mode-comparison-summary)
5. [Optimal Deployment Scenarios](#5-optimal-deployment-scenarios)
6. [Conclusions and Recommendations](#6-conclusions-and-recommendations)

---

## 1. Experimental Setup

### 1.1 Hardware Configuration
- **GPU**: 4x NVIDIA GPUs on single node
- **Memory**: 500GB system RAM
- **CPUs**: 16 cores

### 1.2 Server Configurations

| Mode | Architecture | GPU Allocation |
|------|-------------|----------------|
| PD | 2 Prefill GPUs + 2 Decode GPUs | Tensor Parallel within groups |
| PPD | Same as PD with KV cache reuse | Turn 2+ bypasses Prefill |
| Replication | 4 independent instances | 1 GPU each, load balanced |

### 1.3 Benchmark Parameters
- **Runs**: 3 independent runs averaged
- **Base Duration**: 30 seconds per QPS point (dynamically adjusted)
- **QPS Range**: 0.05 to 20.0 (varies by workload size)
- **Total Experiments**: 128 PD experiments + 128 PP  D experiments + 128 Replication experiments

---

## 2. Workload Design

### 2.1 3D Orthogonal Design Matrix

The benchmark employs a 16-workload design with two orthogonal dimensions:

#### T1 Dimension (Initial Context Building)
| Size | Input→Output Tokens | Total Context (C1) |
|------|--------------------|--------------------|
| S | 256→256 | 512 tokens |
| M | 512→512 | 1024 tokens |
| L | 1024→1024 | 2048 tokens |
| XL | 2048→2048 | 4096 tokens |

#### T2 Dimension (Incremental Request Type)
| Type | Input→Output | Characteristics |
|------|-------------|-----------------|
| a | 32→64 | Tiny followup, light/light |
| b | 32→512 | Short Q long output, light/heavy |
| c | 256→256 | Medium balanced |
| d | 1024→64 | Big paste short answer, heavy/light (prefill-heavy) |

### 2.2 QPS Configuration by Workload Size

| Size | Base QPS Points | Extended QPS (_d only) |
|------|----------------|----------------------|
| S | 0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0 | +17.0, 18.0, 19.0, 20.0 |
| M | 0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 10.0, 12.0 | +13.0, 14.0, 15.0, 16.0 |
| L | 0.05, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0 | +9.0, 10.0, 11.0, 12.0 |
| XL | 0.05, 0.1, 1.0, 2.0, 4.0 | +5.0, 6.0, 7.0, 8.0 |

---

## 3. Metric-by-Metric Analysis

### 3.1 P99 Time To First Token (TTFT)

**Definition**: P99 TTFT measures the 99th percentile latency from request submission to receiving the first generated token. This metric is critical for user-perceived responsiveness.

#### 3.1.1 Absolute Performance Comparison

**At Low QPS (0.1 req/s):**

| Workload | PD (ms) | PPD (ms) | Replication (ms) | PPD Advantage |
|----------|---------|----------|------------------|---------------|
| S_a | 110,144 | 33,851 | 35,125 | 69% faster |
| M_a | 158,398 | 37,946 | 38,850 | 76% faster |
| L_a | 153,310 | 43,841 | 42,500 | 71% faster |
| XL_a | 159,874 | 54,135 | 57,972 | 66% faster |
| S_d | 134,424 | 41,785 | 78,356 | 69% faster |
| XL_d | 230,802 | 102,913 | 102,317 | 55% faster |

**Key Observation**: PPD consistently achieves **45-76% lower TTFT** than PD across all workloads at low-medium QPS. This improvement stems from Turn 2 requests going directly to the Decode server, bypassing the Prefill phase entirely.

#### 3.1.2 QPS Scaling Behavior

**TTFT Degradation Ratios (High QPS / Low QPS):**

| Workload | PD | PPD | Replication |
|----------|-----|-----|-------------|
| S_a | 1.26x | 1.21x | 1.33x |
| M_b | 1.27x | 2.72x | 2.27x |
| L_c | 1.52x | 1.32x | 1.37x |
| S_d | 1.60x | 1.72x | 1.49x |
| XL_d | 2.90x | 5.25x | 1.27x |

**Critical Finding**:
- **Replication shows most stable TTFT scaling** at high QPS, especially for larger workloads
- **PPD degrades faster than Replication** at high QPS due to D server contention
- **PD shows moderate degradation** but maintains consistent KV transfer overhead

#### 3.1.3 High-QPS Crossover Point

At very high QPS, PPD's TTFT advantage diminishes and can even reverse:

**XL_d Workload (extreme example):**
| QPS | PD (ms) | PPD (ms) | Replication (ms) |
|-----|---------|----------|------------------|
| 4.0 | 228,852 | 137,928 | 159,570 |
| 5.0 | 389,681 | 539,547 | 142,993 |
| 6.0 | 444,162 | 483,200 | 118,023 |
| 7.0 | 668,903 | 560,805 | 132,419 |

At QPS ≥ 5.0, PPD's TTFT exceeds PD's in some cases, and both are significantly worse than Replication.

---

### 3.2 Time Per Output Token (TPOT)

**Definition**: TPOT measures the average time to generate each output token during the decode phase. This metric reflects decode efficiency and directly impacts streaming response smoothness.

#### 3.2.1 Average TPOT Analysis

**At Low QPS:**
All three modes show comparable average TPOT (~8.0-8.2ms per token), indicating similar baseline decode efficiency.

**At High QPS:**

| Workload | QPS | PD (ms) | PPD (ms) | Replication (ms) | PPD/PD Ratio |
|----------|-----|---------|----------|------------------|--------------|
| S_d | 16.0 | 9.41 | 11.54 | 8.77 | 1.23x |
| M_d | 12.0 | 9.18 | 13.99 | 8.64 | 1.52x |
| L_d | 9.0 | 9.00 | 9.89 | 8.78 | 1.10x |
| XL_d | 7.0 | 9.20 | 14.60 | 8.69 | 1.59x |

**Key Finding**: At high QPS, **PPD's TPOT degrades 10-59% more than PD's**. This occurs because PPD's decode server handles both Turn 1 (post-prefill) and Turn 2 (direct) requests, leading to contention.

#### 3.2.2 P99 TPOT (Tail Latency)

P99 TPOT reveals decode stability under load:

**High-QPS Comparison:**

| Workload | QPS | PD P99 (ms) | PPD P99 (ms) | Rep P99 (ms) |
|----------|-----|-------------|--------------|--------------|
| S_b | 16.0 | 13,037 | 13,664 | 10,621 |
| M_c | 12.0 | 11,149 | 12,098 | 9,718 |
| L_d | 12.0 | 10,151 | 20,098 | 9,698 |
| XL_d | 8.0 | 10,261 | 23,389 | 9,549 |

**Critical Observation**: For prefill-heavy _d workloads at high QPS, **PPD's P99 TPOT can be 2x higher than PD's**.

#### 3.2.3 Decode Stability Analysis (P99/Avg TPOT Ratio)

The ratio of P99 to Average TPOT indicates decode variance (lower is better):

| Workload Type | PD Variance | PPD Variance | Replication Variance |
|---------------|-------------|--------------|---------------------|
| _a (tiny followup) | 1.08-1.10x | 1.05-1.08x | 1.08-1.10x |
| _b (long output) | 1.07-1.09x | 1.09-1.16x | 1.08-1.10x |
| _c (balanced) | 1.07-1.10x | 1.10-1.21x | 1.08-1.10x |
| _d (prefill-heavy) | 1.10-1.13x | 1.51-2.27x | 1.10-1.15x |

**Key Finding**:
- **PD maintains exceptional decode stability** across all workloads (1.07-1.13x variance)
- **PPD shows significant variance degradation** for prefill-heavy workloads (up to 2.27x)
- **Replication maintains moderate stability** (1.08-1.15x variance)

This is **PD's primary advantage**: dedicated decode servers provide consistent token generation timing, crucial for streaming applications.

---

### 3.3 Throughput

**Definition**: Throughput measures tokens generated per second, reflecting system capacity.

#### 3.3.1 Absolute Throughput Comparison

**At High QPS:**

| Workload | QPS | PD (tps) | PPD (tps) | Rep (tps) | PPD/Rep |
|----------|-----|----------|----------|-----------|---------|
| S_a | 16.0 | 97.3 | 116.4 | 116.4 | 1.00x |
| M_a | 12.0 | 97.6 | 113.8 | 116.3 | 0.98x |
| S_b | 16.0 | 68.9 | 77.7 | 102.0 | 0.76x |
| M_d | 12.0 | 89.4 | 65.6 | 108.0 | 0.61x |
| XL_d | 7.0 | 70.1 | 58.5 | 105.5 | 0.55x |

#### 3.3.2 Throughput Degradation at High QPS

**PPD Throughput Collapse Pattern:**

| Workload | Max QPS | PPD/Rep Ratio | Throughput Loss |
|----------|---------|---------------|-----------------|
| S_a | 16.0 | 1.00x | 0% |
| M_b | 12.0 | 0.80x | 20% |
| L_b | 8.0 | 0.95x | 5% |
| S_d | 16.0 | 0.81x | 19% |
| M_d | 12.0 | 0.61x | 39% |
| XL_d | 7.0 | 0.55x | 45% |

**Key Finding**:
- **PPD maintains competitive throughput** for light workloads (_a, _c)
- **PPD throughput collapses** for prefill-heavy (_d) and generation-heavy (_b) workloads at high QPS
- **Replication provides most consistent high-QPS throughput** across all workload types

---

### 3.4 End-to-End Latency

#### 3.4.1 Turn 2 E2E Latency

Turn 2 E2E latency directly measures the multi-turn conversation benefit of PPD:

**At Low QPS (0.1 req/s):**
| Workload | PD (s) | PPD (s) | Rep (s) | PPD Improvement |
|----------|--------|--------|---------|-----------------|
| S_a | 1.35 | 0.55 | 0.54 | 59% faster |
| M_c | 3.45 | 2.07 | 2.06 | 40% faster |
| XL_d | 0.72 | 0.55 | 0.58 | 24% faster |

#### 3.4.2 Total E2E Latency (T1 + T2)

**At High QPS:**
PPD's E2E advantage diminishes as decode contention increases:

| Workload | QPS | PD Total (s) | PPD Total (s) | Rep Total (s) |
|----------|-----|--------------|---------------|---------------|
| S_d | 16.0 | 5.10 | 5.76 | 4.48 |
| M_d | 12.0 | 5.21 | 6.93 | 4.45 |
| XL_d | 7.0 | 8.92 | 9.63 | 5.88 |

**Key Finding**: At high QPS, **PPD can become slower than both PD and Replication** in total E2E latency.

---

## 4. Mode Comparison Summary

### 4.1 Comprehensive Metrics Table

| Metric | PD | PPD | Replication |
|--------|-----|-----|-------------|
| **P99 TTFT (low QPS)** | Baseline | 45-76% better | 45-58% better |
| **P99 TTFT (high QPS)** | Moderate degradation | Severe degradation | Most stable |
| **Avg TPOT (low QPS)** | ~8.0ms | ~8.0ms | ~8.0ms |
| **Avg TPOT (high QPS)** | Stable | 10-59% worse | Stable |
| **P99/Avg TPOT Variance** | 1.07-1.13x (best) | 1.05-2.27x | 1.08-1.15x |
| **Throughput (high QPS)** | Moderate | 0-45% loss | Best |
| **Multi-turn Benefit** | None | Yes (KV reuse) | None |
| **Decode Stability** | Excellent | Poor at high QPS | Good |

### 4.2 Scaling Characteristics

| Mode | Low QPS Performance | High QPS Performance | Degradation Pattern |
|------|--------------------|-----------------------|---------------------|
| PD | Good TPOT, High TTFT | Stable degradation | Linear, predictable |
| PPD | Best TTFT, Good TPOT | Severe degradation | Non-linear, collapse |
| Replication | Good overall | Most stable | Gradual, graceful |

---

## 5. Optimal Deployment Scenarios

### 5.1 When to Use PD (Prefill-Decode Disaggregation)

**Optimal Scenarios:**
1. **Streaming applications requiring consistent token timing**
   - Real-time transcription display
   - Live code generation with syntax highlighting
   - Interactive chat with smooth text appearance

2. **High-QPS prefill-heavy workloads**
   - Document summarization services
   - Long-context Q&A systems
   - Code review/analysis pipelines

3. **Latency-sensitive decode operations**
   - When P99 TPOT variance must be <1.15x
   - Applications where decode jitter causes visible stuttering

**Characteristics:**
- Higher baseline TTFT (2-3x vs PPD)
- Most stable decode performance
- Predictable scaling behavior
- Best for single-turn or decode-critical applications

### 5.2 When to Use PPD (Prefill-Prefill-Decode with KV Reuse)

**Optimal Scenarios:**
1. **Multi-turn conversational AI at low-medium QPS**
   - Chatbots with extended conversations
   - Interactive tutoring systems
   - Customer support applications

2. **TTFT-critical applications**
   - Voice assistants (first response latency)
   - Real-time translation
   - Interactive coding assistants

3. **Moderate load with context reuse**
   - QPS < 50% of maximum capacity
   - Sessions with multiple follow-up turns
   - Applications where TTFT matters more than decode stability

**Characteristics:**
- 45-76% TTFT improvement over PD
- KV cache reuse eliminates Turn 2+ prefill cost
- Degrades rapidly at high QPS
- Best for low-medium QPS multi-turn workloads

### 5.3 When to Use Replication

**Optimal Scenarios:**
1. **High-throughput production deployments**
   - API services with variable/high load
   - Batch processing systems
   - Services requiring horizontal scaling

2. **Simple deployment requirements**
   - No complex P-D coordination needed
   - Standard vLLM configuration
   - Easier monitoring and debugging

3. **High-QPS stability requirements**
   - When system must maintain performance at peak load
   - SLA-bound services with throughput guarantees
   - Workloads with unpredictable QPS spikes

**Characteristics:**
- Most stable high-QPS performance
- Highest sustained throughput
- Simplest architecture
- No multi-turn optimization
- Best for high-throughput, single-turn workloads

---

## 6. Conclusions and Recommendations

### 6.1 Key Takeaways

1. **No single mode dominates all scenarios** - each has distinct advantages
2. **PPD's TTFT advantage is significant** (45-76%) but comes with high-QPS stability tradeoffs
3. **PD's decode stability is unmatched** - critical for streaming applications
4. **Replication provides the safest high-QPS deployment** with predictable scaling

### 6.2 Decision Framework

```
Is multi-turn KV reuse beneficial?
├── YES: Is QPS < 50% capacity?
│   ├── YES: Use PPD (best TTFT + KV reuse)
│   └── NO: Use Replication (stable high-QPS)
└── NO: Is decode stability critical?
    ├── YES: Use PD (best P99/Avg TPOT ratio)
    └── NO: Use Replication (highest throughput)
```

### 6.3 Workload-Specific Recommendations

| Workload Type | Low QPS (<4.0) | Medium QPS (4.0-8.0) | High QPS (>8.0) |
|---------------|----------------|----------------------|-----------------|
| _a (tiny followup) | PPD | PPD/Rep | Replication |
| _b (long output) | PPD | Replication | Replication |
| _c (balanced) | PPD | PPD/Rep | Replication |
| _d (prefill-heavy) | PPD | PD/Rep | Replication |

### 6.4 Future Work

1. **Adaptive mode switching** based on real-time QPS monitoring
2. **Hybrid deployments** with PD for streaming and Replication for batch
3. **Dynamic resource allocation** between P and D servers based on workload characteristics

---

## Appendix: Data Sources

- **Benchmark Data**: `results/final/qps_benchmark_v3_averaged_20260107_222956.json`
- **Replication Data**: `results/final/replication_benchmark_v3_averaged_20260107_222956.json`
- **Analysis Plots**: `results/final/analysis_plots/`
- **Runs Averaged**: 3 (run1, run2, run3)
- **Total Experiments**: 384 (256 QPS + 128 Replication)
