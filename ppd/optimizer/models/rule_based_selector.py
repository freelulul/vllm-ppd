#!/usr/bin/env python3
"""
Rule-Based Mode Selector

Interpretable decision tree based on benchmark analysis.
Rules are derived from the trade-off analysis document.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple


class OptimizationObjective(Enum):
    # Standard objectives (avg metrics)
    TTFT = "ttft"           # Avg TTFT optimization
    TPOT = "tpot"           # Avg TPOT optimization
    THROUGHPUT = "throughput"
    E2E = "e2e"
    # Fine-grained objectives (p99 metrics)
    P99_TTFT = "p99_ttft"   # P99 TTFT optimization (stricter latency SLO)
    P99_TPOT = "p99_tpot"   # P99 TPOT optimization (stricter generation SLO)


class Mode(Enum):
    PD = "pd"
    PPD = "ppd"
    REPLICA = "replica"


@dataclass
class RequestFeatures:
    """Features for a single request/turn"""
    input_length: int
    output_length: int
    turn_number: int
    has_cache: bool
    cached_gpu: Optional[str]  # e.g., "gpu1", "gpu2", "gpu3"
    queue_depths: Dict[str, int]  # gpu_id -> queue_depth
    objective: OptimizationObjective
    current_qps: float = 1.0  # Current system QPS level for load-aware routing

    @property
    def input_output_ratio(self) -> float:
        return self.input_length / max(self.output_length, 1)

    @property
    def is_big_paste(self) -> bool:
        """
        Big-Paste pattern: Long input, short output.

        Based on merged_results.json _d workloads:
        - T2 pattern: 512 input tokens, 64 output tokens
        - High input/output ratio (8:1)

        Detection: input > 256 AND output < 128 AND ratio > 4
        """
        return (
            self.input_length > 256 and
            self.output_length < 128 and
            self.input_output_ratio > 4.0
        )

    @property
    def is_long_generation(self) -> bool:
        """Long output generation"""
        return self.output_length > 512

    @property
    def total_tokens(self) -> int:
        return self.input_length + self.output_length


@dataclass
class RoutingDecision:
    """Result of mode selection"""
    mode: Mode
    target_gpu: str
    reason: str
    estimated_latency: Optional[float] = None
    cache_hit: bool = False


class RuleBasedSelector:
    """
    Rule-based mode selector with interpretable decision logic.

    Decision Tree Structure:
    1. First check optimization objective
    2. Then check workload characteristics
    3. Finally consider queue depths for load balancing
    """

    # GPU configuration for Optimizer V2: 1P + 1D_pure + 1pD + 1R
    # GPU0: P (prefill only, kv_producer)
    # GPU1: D_pure (decode only, kv_consumer) - for PD mode
    # GPU2: pD (decode + append-prefill, kv_both) - for PPD mode
    # GPU3: Replica (standalone, no KV transfer)
    GPU_CONFIG = {
        "pd": {"prefill_gpu": "gpu0", "decode_gpu": "gpu1"},   # P→D_pure
        "ppd": {"prefill_gpu": "gpu0", "decode_gpu": "gpu2"},  # P→pD
        "replica": {"gpu": "gpu3"},
    }

    # Estimated processing times (ms per token) from benchmark
    PREFILL_MS_PER_TOKEN = 0.05   # Prefill speed
    DECODE_MS_PER_TOKEN = 8.5     # Decode speed (TPOT)
    KV_TRANSFER_OVERHEAD_MS = 100  # P→D KV transfer overhead

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.decision_log = []

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Rule] {msg}")
        self.decision_log.append(msg)

    def _estimate_cache_miss_cost(self, input_length: int) -> float:
        """Estimate the cost of cache miss (re-prefill)"""
        return input_length * self.PREFILL_MS_PER_TOKEN

    def _estimate_queue_wait(self, queue_depth: int, avg_request_time: float = 500) -> float:
        """Estimate queue waiting time"""
        return queue_depth * avg_request_time

    def _get_min_queue_gpu(self, features: RequestFeatures, mode_gpus: List[str]) -> Tuple[str, int]:
        """Get the GPU with minimum queue depth from a set of GPUs"""
        min_gpu = mode_gpus[0]
        min_depth = features.queue_depths.get(min_gpu, 0)
        for gpu in mode_gpus[1:]:
            depth = features.queue_depths.get(gpu, 0)
            if depth < min_depth:
                min_depth = depth
                min_gpu = gpu
        return min_gpu, min_depth

    def _check_queue_overflow(self, decision: RoutingDecision, features: RequestFeatures) -> RoutingDecision:
        """
        Check if the selected GPU is overloaded and redirect to a less busy GPU.

        This enables cross-mode routing: when any mode's target GPU is congested,
        requests can overflow to other available GPUs for better load balancing.

        V2 Architecture GPU mapping:
        - gpu1: D_pure (PD mode decode)
        - gpu2: pD (PPD mode decode)
        - gpu3: Replica (standalone)
        """
        target_queue = features.queue_depths.get(decision.target_gpu, 0)

        # Check if target GPU is overloaded
        if target_queue <= self.QUEUE_OVERFLOW_THRESHOLD:
            return decision  # No overflow needed

        # Find the least loaded GPU among all options
        all_gpus = ["gpu1", "gpu2", "gpu3"]
        min_gpu, min_queue = self._get_min_queue_gpu(features, all_gpus)

        # Only overflow if the alternative is significantly less loaded
        if min_queue < target_queue - self.QUEUE_OVERFLOW_MARGIN:
            # Determine mode based on target GPU
            if min_gpu == "gpu1":
                overflow_mode = Mode.PD
            elif min_gpu == "gpu2":
                overflow_mode = Mode.PPD
            else:  # gpu3
                overflow_mode = Mode.REPLICA

            self._log(f"★ Queue overflow: {decision.target_gpu}({target_queue}) → {min_gpu}({min_queue})")

            return RoutingDecision(
                mode=overflow_mode,
                target_gpu=min_gpu,
                reason=f"Queue overflow: {decision.mode.value}→{overflow_mode.value} ({decision.target_gpu}:{target_queue} → {min_gpu}:{min_queue})",
                cache_hit=False  # Overflow loses cache benefit
            )

        return decision  # No better alternative found

    def select_mode(self, features: RequestFeatures) -> RoutingDecision:
        """
        Main entry point: Select the best mode based on features and objective.

        Decision flow:
        1. Apply objective-specific rules (throughput always uses replica)
        2. Check cache affinity for TTFT/TPOT/E2E if cache exists
        3. Load balance among candidates
        4. ★ NEW: Check queue overflow for cross-mode routing
        """
        self.decision_log = []
        self._log(f"Input: {features.input_length} tokens, Output: {features.output_length} tokens")
        self._log(f"Turn: {features.turn_number}, Objective: {features.objective.value}")
        self._log(f"Has cache: {features.has_cache}, Cached GPU: {features.cached_gpu}")
        self._log(f"Queue depths: {features.queue_depths}")

        decision = None

        # Step 1: Throughput always uses Replica (84% win rate, ignores cache)
        if features.objective == OptimizationObjective.THROUGHPUT:
            decision = self._select_for_throughput(features)

        # Step 2: Check cache affinity for Turn 2+ (except at high QPS or special cases)
        # At high QPS, capacity may matter more than cache
        elif features.has_cache and features.cached_gpu:
            # Skip cache affinity at very high QPS where capacity matters
            if features.current_qps <= self.QPS_THRESHOLD_HIGH * 1.5:
                # Skip cache affinity for E2E with very long output (TPOT dominates)
                # In this case, PPD's better TPOT is more important than cache reuse
                skip_cache = (
                    features.objective == OptimizationObjective.E2E and
                    features.output_length > 600
                )
                if not skip_cache:
                    cache_decision = self._check_cache_affinity(features)
                    if cache_decision:
                        decision = cache_decision
                else:
                    self._log("Skipping cache affinity: E2E + very long output → PPD TPOT benefit")

        # Step 3: Apply objective-specific rules if no decision yet
        if decision is None:
            if features.objective == OptimizationObjective.TTFT:
                decision = self._select_for_ttft(features)
            elif features.objective == OptimizationObjective.P99_TTFT:
                decision = self._select_for_p99_ttft(features)
            elif features.objective == OptimizationObjective.TPOT:
                decision = self._select_for_tpot(features)
            elif features.objective == OptimizationObjective.P99_TPOT:
                decision = self._select_for_p99_tpot(features)
            elif features.objective == OptimizationObjective.E2E:
                decision = self._select_for_e2e(features)
            elif features.objective == OptimizationObjective.THROUGHPUT:
                decision = self._select_for_throughput(features)
            else:
                # Default to PPD
                decision = self._default_decision(features)

        # Step 4: ★ Check queue overflow - allow cross-mode routing if target GPU is congested
        # This enables load balancing across all GPUs regardless of original objective
        final_decision = self._check_queue_overflow(decision, features)

        if final_decision.target_gpu != decision.target_gpu:
            self._log(f"★ Cross-mode routing activated: {decision.mode.value}→{final_decision.mode.value}")

        return final_decision

    def _check_cache_affinity(self, features: RequestFeatures) -> Optional[RoutingDecision]:
        """
        Check if we should use cached GPU due to affinity.

        Returns decision if cache affinity wins, None if we should consider other options.
        """
        cached_gpu = features.cached_gpu
        cache_miss_cost = self._estimate_cache_miss_cost(features.input_length)

        # Get queue depth on cached GPU
        cached_queue = features.queue_depths.get(cached_gpu, 0)
        cached_wait = self._estimate_queue_wait(cached_queue)

        # Find minimum queue among all GPUs
        # V2 architecture: gpu1=D_pure (PD), gpu2=pD (PPD), gpu3=Replica
        all_gpus = list(features.queue_depths.keys())
        if not all_gpus:
            all_gpus = ["gpu1", "gpu2", "gpu3"]  # D_pure, pD, Replica

        min_gpu, min_queue = self._get_min_queue_gpu(features, all_gpus)
        min_wait = self._estimate_queue_wait(min_queue)

        # Decision: Use cache if cache_miss_cost > (cached_wait - min_wait)
        # i.e., the cost of missing cache is higher than extra queue wait
        benefit_of_switching = cached_wait - min_wait
        self._log(f"Cache miss cost: {cache_miss_cost:.0f}ms, Queue wait diff: {benefit_of_switching:.0f}ms")

        if cache_miss_cost > benefit_of_switching:
            self._log(f"→ Cache affinity wins: stay on {cached_gpu}")
            # Determine mode based on cached GPU (V2 architecture)
            # gpu1 = D_pure (PD mode), gpu2 = pD (PPD mode), gpu3 = Replica
            if cached_gpu == "gpu1":
                mode = Mode.PD    # GPU1 is D_pure, use PD for cache reuse
            elif cached_gpu == "gpu2":
                mode = Mode.PPD   # GPU2 is pD, use PPD for cache reuse
            elif cached_gpu == "gpu3":
                mode = Mode.REPLICA
            else:
                mode = Mode.REPLICA  # Default

            return RoutingDecision(
                mode=mode,
                target_gpu=cached_gpu,
                reason=f"Cache affinity: miss cost ({cache_miss_cost:.0f}ms) > switch benefit ({benefit_of_switching:.0f}ms)",
                cache_hit=True
            )

        self._log(f"→ Cache affinity loses: considering other options")
        return None

    # QPS thresholds (refined based on benchmark data 2026-01-13)
    # Data shows: Replica competitive at QPS>=4, clearly better at QPS>=8
    QPS_THRESHOLD_LOW = 4   # Below this, PPD is often better
    QPS_THRESHOLD_HIGH = 8  # Above this, Replica has capacity advantage

    # Input length thresholds
    INPUT_SMALL = 300       # Small input: Replica more efficient
    INPUT_LARGE = 1000      # Large input: PPD prefix caching helps

    # Queue overflow thresholds for cross-mode routing
    # When a GPU's queue exceeds this, allow overflow to other GPUs
    QUEUE_OVERFLOW_THRESHOLD = 8   # Trigger overflow when queue > this
    QUEUE_OVERFLOW_MARGIN = 4      # Target GPU must be this much less loaded

    def _select_for_ttft(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode optimizing for TTFT (Time To First Token).

        Refined based on REAL benchmark data (2026-01-13):
        - Turn 1: Replica is 2-3x faster (30-120ms vs PPD 90-160ms)
        - Turn 2+: PPD wins due to prefix caching
        - Small input (<300): Replica more efficient
        - Large input (>1000): PPD prefix caching helps
        """
        self._log(f"Objective: Minimize TTFT (Turn={features.turn_number}, QPS={features.current_qps})")

        # Rule 1: XL Big-Paste (>3000 input, <100 output) → PD
        if features.input_length > 3000 and features.output_length < 100:
            self._log("XL Big-Paste: PD for long prefill efficiency")
            return RoutingDecision(
                mode=Mode.PD,
                target_gpu="gpu1",
                reason="XL Big-Paste + TTFT: PD prefill parallelization"
            )

        # Rule 2: Turn 1 - Replica is consistently faster (no P→D overhead)
        # Benchmark shows: Replica T1_TTFT 30-120ms vs PPD 90-160ms
        if features.turn_number == 1:
            # Exception: Very large input benefits from PPD's prefill optimization
            if features.input_length > self.INPUT_LARGE:
                self._log(f"Turn 1 but large input ({features.input_length}): PPD for prefill efficiency")
                return RoutingDecision(
                    mode=Mode.PPD,
                    target_gpu="gpu2",  # pD for PPD mode
                    reason="Turn 1 + Large input: PPD prefill efficiency"
                )
            # Default Turn 1: Replica is 2-3x faster
            gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
            self._log(f"Turn 1: Replica for fast first response")
            return RoutingDecision(
                mode=Mode.REPLICA,
                target_gpu=gpu,
                reason="Turn 1 + TTFT: Replica 2-3x faster (no P→D overhead)"
            )

        # Rule 3: Turn 2+ with small input at high QPS - Replica
        if features.input_length < self.INPUT_SMALL and features.current_qps >= self.QPS_THRESHOLD_HIGH:
            gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
            self._log(f"Turn 2+ small input + high QPS: Replica for capacity")
            return RoutingDecision(
                mode=Mode.REPLICA,
                target_gpu=gpu,
                reason="Turn 2+ small input + high QPS: Replica capacity"
            )

        # Rule 4: Turn 2+ - PPD wins due to prefix caching
        # PPD T2_TTFT is 2-4x better than PD due to cache
        if features.current_qps < self.QPS_THRESHOLD_LOW:
            self._log(f"Turn 2+ low QPS: PPD for prefix caching")
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu2",  # pD for PPD mode
                reason="Turn 2+ + Low QPS: PPD prefix caching efficient"
            )
        elif features.current_qps > self.QPS_THRESHOLD_HIGH:
            # High QPS: Consider both cache benefit and capacity
            if features.input_length > self.INPUT_SMALL:
                # Larger input: cache benefit outweighs
                self._log(f"Turn 2+ high QPS but larger input: PPD for cache")
                return RoutingDecision(
                    mode=Mode.PPD,
                    target_gpu="gpu2",  # pD for PPD mode
                    reason="Turn 2+ high QPS + larger input: PPD cache benefit"
                )
            gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
            self._log(f"Turn 2+ high QPS small input: Replica for capacity")
            return RoutingDecision(
                mode=Mode.REPLICA,
                target_gpu=gpu,
                reason="Turn 2+ high QPS: Replica parallel capacity"
            )
        else:
            # Medium QPS (4-8): Prefer PPD for cache, but consider queue
            ppd_queue = features.queue_depths.get("gpu2", 0)  # pD is gpu2
            replica_queue = features.queue_depths.get("gpu3", 0)  # Replica is gpu3
            # PPD has cache advantage, so give it slight preference
            if ppd_queue <= replica_queue + 2:
                self._log(f"Medium QPS, PPD cache benefit (queue {ppd_queue})")
                return RoutingDecision(
                    mode=Mode.PPD,
                    target_gpu="gpu2",  # pD for PPD mode
                    reason="Turn 2+ Medium QPS: PPD cache efficient"
                )
            else:
                gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
                self._log(f"Medium QPS, Replica less loaded: {gpu}")
                return RoutingDecision(
                    mode=Mode.REPLICA,
                    target_gpu=gpu,
                    reason="Turn 2+ Medium QPS: Replica less loaded"
                )

    def _select_for_tpot(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode optimizing for TPOT (Time Per Output Token).

        Based on merged_results.json benchmark data:
        - T2 Avg TPOT winner distribution: Replica 61.1%, PD 30.6%, PPD 8.3%
        - PD wins specifically in _d (Big Paste) workloads: large T2 prefill + small decode
        - Replica wins in most other scenarios due to capacity advantage

        Decision matrix from analysis_report.txt:
        - Big Paste (_d): PD (isolation prevents decode interference)
        - High QPS + _d: PD (significant advantage, up to 50% better)
        - Other workloads: Replica slightly better at high QPS, PPD/Replica similar otherwise
        """
        self._log("Objective: Minimize TPOT (avg)")

        # Rule 1: Big-Paste pattern (_d workload) → PD
        # From benchmark: PD wins TPOT in 19 _d scenarios (L_d, M_d, S_d, XL_d, XS_d)
        # _d pattern: T2 has 512 input tokens, 64 output tokens
        if features.is_big_paste:
            self._log("Big-Paste detected: PD for prefill isolation (benchmark: PD wins _d)")
            return RoutingDecision(
                mode=Mode.PD,
                target_gpu="gpu1",  # D_pure for PD mode
                reason="Big-Paste + TPOT: PD prefill isolation"
            )

        # Rule 2: Turn 2+ → PPD (has cache, slightly better than PD)
        if features.turn_number >= 2:
            self._log("Turn 2+: PPD for TPOT with cache benefit")
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu2",  # pD for PPD mode
                reason="Turn 2+ + TPOT: PPD has dedicated decode GPU with cache"
            )

        # Rule 3: Turn 1 with long output → PPD (TPOT matters more)
        if features.output_length > 256:
            self._log("Long output: PPD for better TPOT")
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu2",  # pD for PPD mode
                reason="Long output + TPOT: PPD dedicated decode GPU"
            )

        # Rule 4: Turn 1 default to PD for pure decode (D_pure has no prefill interference)
        return RoutingDecision(
            mode=Mode.PD,
            target_gpu="gpu1",  # D_pure for PD mode
            reason="Turn 1 + TPOT: PD has pure decode GPU (no prefill interference)"
        )

    def _select_for_p99_ttft(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode optimizing for P99 TTFT (stricter latency SLO).

        Based on merged_results.json (T2 P99 TTFT from turn2_metrics.p99_ttft):
        - PPD wins 65.7% (71/108 scenarios) - prefix cache reduces tail latency
        - Replica wins 34.3% (37/108 scenarios) - competitive for small contexts, tiny T2
        - PD wins 0% (0/108 scenarios) - never best for P99 TTFT

        From analysis_report.txt Section 2.5 T2 Type Impact:
        - Tiny (16→32): Replica wins most (14/27)
        - Other types: PPD wins most

        From Section 2.4 Context Size Impact:
        - Small (256+256): Replica wins most (13/24)
        - Medium/Large/XLarge: PPD wins most
        """
        self._log("Objective: Minimize P99 TTFT (stricter SLO)")

        # Rule 1: Tiny T2 output (type 'a': 16→32) → Replica
        # From benchmark: Replica wins 14/27 for tiny T2
        if features.output_length <= 64:
            gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
            self._log("Tiny T2 output: Replica (benchmark: wins 14/27 for type 'a')")
            return RoutingDecision(
                mode=Mode.REPLICA,
                target_gpu=gpu,
                reason="P99 TTFT + Tiny T2: Replica (benchmark data)"
            )

        # Rule 2: Small context (S: 256+256) → Replica competitive
        # From benchmark: Replica wins 13/24 for Small context
        if features.total_tokens < 600:
            gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
            self._log("Small context: Replica (benchmark: wins 13/24 for S)")
            return RoutingDecision(
                mode=Mode.REPLICA,
                target_gpu=gpu,
                reason="P99 TTFT + Small context: Replica (benchmark data)"
            )

        # Rule 3: Default to PPD (wins 65.7% overall for P99 TTFT)
        # PPD's prefix cache reduces P99 latency by avoiding re-prefill
        self._log("PPD for P99 TTFT (benchmark: wins 65.7% overall)")
        return RoutingDecision(
            mode=Mode.PPD,
            target_gpu="gpu2",
            reason="P99 TTFT: PPD cache reduces tail latency (65.7% win rate)"
        )

    def _select_for_p99_tpot(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode optimizing for P99 TPOT (stricter generation SLO).

        Based on merged_results.json (T2 P99 TPOT from turn2_metrics.p99_tpot):
        - Similar pattern to T2 Avg TPOT
        - T2 Avg TPOT distribution: Replica 61.1%, PD 30.6%, PPD 8.3%

        From analysis_report.txt Section 2.2:
        - PD wins TPOT in _d workloads: 19 scenarios
        - _d pattern: large T2 prefill (512 tokens) + small decode (64 tokens)
        - PD isolates prefill on P-machines, D-machines only do decode

        From Section 3.2:
        - Big Paste (_d): Use PD - isolation prevents decode interference
        - Other workloads: PPD/Replica similar, Replica slightly better at high QPS
        """
        self._log("Objective: Minimize P99 TPOT (stricter SLO)")

        # Rule 1: Big-Paste pattern (_d) → PD
        # From benchmark: PD wins TPOT in 19 _d scenarios
        if features.is_big_paste:
            self._log("Big-Paste: PD for decode isolation (benchmark: PD wins 19 _d scenarios)")
            return RoutingDecision(
                mode=Mode.PD,
                target_gpu="gpu1",
                reason="P99 TPOT + Big-Paste: PD decode isolation (benchmark data)"
            )

        # Rule 2: Turn 2+ → PPD (has cache benefit)
        if features.turn_number >= 2:
            self._log("Turn 2+: PPD for P99 TPOT with cache")
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu2",
                reason="P99 TPOT + Turn 2+: PPD cache benefit"
            )

        # Rule 3: Turn 1 default → PD (dedicated decode)
        self._log("Turn 1: PD for P99 TPOT (dedicated decode)")
        return RoutingDecision(
            mode=Mode.PD,
            target_gpu="gpu1",
            reason="P99 TPOT + Turn 1: PD dedicated decode"
        )

    def _select_for_throughput(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode optimizing for Throughput (tokens/second).

        From benchmark analysis:
        - Replica wins ~84% for throughput (more parallel capacity)
        - PPD wins ~16%
        """
        self._log("Objective: Maximize Throughput")

        # Rule 1: Throughput almost always favors Replica
        # Select least loaded replica
        gpu, queue = self._get_min_queue_gpu(features, ["gpu3"])
        self._log(f"Throughput: Replica {gpu} (queue: {queue})")

        return RoutingDecision(
            mode=Mode.REPLICA,
            target_gpu=gpu,
            reason=f"Throughput: Replica provides best batch efficiency (queue: {queue})"
        )

    def _select_for_e2e(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode optimizing for E2E latency.

        E2E = TTFT + (output_tokens * TPOT)

        Refined based on REAL benchmark data (2026-01-13):
        - Turn 1: Replica often wins (lower TTFT compensates)
        - Turn 2+: PPD wins due to prefix caching reducing E2E
        - Very long output: PPD (TPOT dominates)
        """
        self._log(f"Objective: Minimize E2E (Turn={features.turn_number}, QPS={features.current_qps})")

        # Rule 1: Very long generation → PPD (TPOT dominates E2E)
        if features.output_length > 600:
            self._log("Very long output: PPD for better TPOT")
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu2",  # pD for PPD mode
                reason="Very long output + E2E: PPD better TPOT"
            )

        # Rule 2: Turn 1 - Replica's TTFT advantage often wins E2E
        if features.turn_number == 1:
            # Exception: Large input or long output favors PPD
            if features.input_length > self.INPUT_LARGE or features.output_length > 400:
                self._log("Turn 1 but large workload: PPD")
                return RoutingDecision(
                    mode=Mode.PPD,
                    target_gpu="gpu2",  # pD for PPD mode
                    reason="Turn 1 + Large workload: PPD efficiency"
                )
            gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
            self._log("Turn 1: Replica for fast E2E")
            return RoutingDecision(
                mode=Mode.REPLICA,
                target_gpu=gpu,
                reason="Turn 1 + E2E: Replica faster TTFT"
            )

        # Rule 3: Turn 2+ - Consider QPS and workload
        if features.current_qps < self.QPS_THRESHOLD_LOW:
            self._log(f"Turn 2+ low QPS: PPD for cache efficiency")
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu2",  # pD for PPD mode
                reason="Turn 2+ + Low QPS: PPD cache efficient"
            )
        elif features.current_qps > self.QPS_THRESHOLD_HIGH:
            # High QPS: Capacity matters
            if features.input_length > self.INPUT_SMALL:
                self._log("Turn 2+ high QPS larger input: PPD cache")
                return RoutingDecision(
                    mode=Mode.PPD,
                    target_gpu="gpu2",  # pD for PPD mode
                    reason="Turn 2+ high QPS + larger input: PPD cache"
                )
            gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
            self._log(f"Turn 2+ high QPS small input: Replica capacity")
            return RoutingDecision(
                mode=Mode.REPLICA,
                target_gpu=gpu,
                reason="Turn 2+ high QPS: Replica parallel capacity"
            )
        else:
            # Medium QPS: Use queue depths with PPD preference
            # V2 architecture: pD is gpu2, Replica is gpu3
            ppd_queue = features.queue_depths.get("gpu2", 0)  # pD for PPD mode
            replica_queue = features.queue_depths.get("gpu3", 0)  # Only gpu3 is Replica now
            if ppd_queue <= replica_queue + 2:
                self._log(f"Medium QPS, PPD preferred (queue {ppd_queue})")
                return RoutingDecision(
                    mode=Mode.PPD,
                    target_gpu="gpu2",  # pD for PPD mode
                    reason="Turn 2+ Medium QPS: PPD cache efficient"
                )
            else:
                gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
                self._log(f"Medium QPS, Replica less loaded: {gpu}")
                return RoutingDecision(
                    mode=Mode.REPLICA,
                    target_gpu=gpu,
                    reason="Turn 2+ Medium QPS: Replica less loaded"
                )

    def _default_decision(self, features: RequestFeatures) -> RoutingDecision:
        """Default decision when no specific rule applies"""
        gpu, _ = self._get_min_queue_gpu(features, ["gpu3"])
        return RoutingDecision(
            mode=Mode.REPLICA,
            target_gpu=gpu,
            reason="Default: Replica balanced performance"
        )


def evaluate_on_training_data(selector: RuleBasedSelector, data_path: str) -> Dict:
    """Evaluate rule-based selector on training data"""
    import json

    with open(data_path, 'r') as f:
        data = json.load(f)

    correct = 0
    total = 0
    confusion = {}

    for row in data:
        features = RequestFeatures(
            input_length=row['t1_input'],
            output_length=row['t1_output'],
            turn_number=2,  # Training data focuses on Turn 2
            has_cache=False,  # No cache info in static training data
            cached_gpu=None,
            queue_depths={"gpu1": 0, "gpu2": 0, "gpu3": 0},
            objective=OptimizationObjective(row['objective'])
        )

        decision = selector.select_mode(features)
        predicted = decision.mode.value
        actual = row['best_mode']

        total += 1
        if predicted == actual:
            correct += 1

        # Confusion matrix
        key = (actual, predicted)
        confusion[key] = confusion.get(key, 0) + 1

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'confusion_matrix': confusion
    }


if __name__ == "__main__":
    # Demo usage
    # V2 Architecture:
    # - gpu0: P (pure prefill, kv_producer) - not tracked in queue_depths
    # - gpu1: D_pure (decode only, kv_consumer) - for PD mode
    # - gpu2: pD (decode + append-prefill, kv_both) - for PPD mode
    # - gpu3: Replica (standalone) - no KV transfer
    selector = RuleBasedSelector(verbose=True)

    print("=" * 60)
    print("Rule-Based Mode Selector Demo (V2 Architecture)")
    print("=" * 60)

    # Test case 1: TTFT optimization, Turn 2
    print("\n--- Test 1: TTFT, Turn 2, Medium request ---")
    features = RequestFeatures(
        input_length=1000,
        output_length=200,
        turn_number=2,
        has_cache=True,
        cached_gpu="gpu1",
        queue_depths={"gpu1": 2, "gpu2": 0, "gpu3": 1},
        objective=OptimizationObjective.TTFT
    )
    decision = selector.select_mode(features)
    print(f"Decision: {decision.mode.value} → {decision.target_gpu}")
    print(f"Reason: {decision.reason}")

    # Test case 2: TPOT optimization, Big-Paste
    print("\n--- Test 2: TPOT, Turn 1, Big-Paste ---")
    features = RequestFeatures(
        input_length=4000,
        output_length=100,
        turn_number=1,
        has_cache=False,
        cached_gpu=None,
        queue_depths={"gpu1": 0, "gpu2": 1, "gpu3": 2},
        objective=OptimizationObjective.TPOT
    )
    decision = selector.select_mode(features)
    print(f"Decision: {decision.mode.value} → {decision.target_gpu}")
    print(f"Reason: {decision.reason}")

    # Test case 3: Throughput optimization
    print("\n--- Test 3: Throughput, Turn 1 ---")
    features = RequestFeatures(
        input_length=500,
        output_length=500,
        turn_number=1,
        has_cache=False,
        cached_gpu=None,
        queue_depths={"gpu1": 3, "gpu2": 1, "gpu3": 0},
        objective=OptimizationObjective.THROUGHPUT
    )
    decision = selector.select_mode(features)
    print(f"Decision: {decision.mode.value} → {decision.target_gpu}")
    print(f"Reason: {decision.reason}")

    # Test case 4: E2E optimization, Turn 2
    print("\n--- Test 4: E2E, Turn 2, Long output ---")
    features = RequestFeatures(
        input_length=500,
        output_length=700,
        turn_number=2,
        has_cache=True,
        cached_gpu="gpu2",  # Cache on pD (PPD mode)
        queue_depths={"gpu1": 0, "gpu2": 1, "gpu3": 0},
        objective=OptimizationObjective.E2E
    )
    decision = selector.select_mode(features)
    print(f"Decision: {decision.mode.value} → {decision.target_gpu}")
    print(f"Reason: {decision.reason}")

    # Test case 5: P99 TTFT optimization, Turn 2
    print("\n--- Test 5: P99_TTFT, Turn 2, Medium context ---")
    features = RequestFeatures(
        input_length=512,
        output_length=256,
        turn_number=2,
        has_cache=True,
        cached_gpu="gpu2",
        queue_depths={"gpu1": 1, "gpu2": 2, "gpu3": 1},
        objective=OptimizationObjective.P99_TTFT
    )
    decision = selector.select_mode(features)
    print(f"Decision: {decision.mode.value} → {decision.target_gpu}")
    print(f"Reason: {decision.reason}")

    # Test case 6: P99 TPOT optimization, Big-Paste
    print("\n--- Test 6: P99_TPOT, Big-Paste pattern (512→64) ---")
    features = RequestFeatures(
        input_length=512,
        output_length=64,
        turn_number=2,
        has_cache=False,
        cached_gpu=None,
        queue_depths={"gpu1": 0, "gpu2": 1, "gpu3": 2},
        objective=OptimizationObjective.P99_TPOT,
        current_qps=4.0
    )
    decision = selector.select_mode(features)
    print(f"Decision: {decision.mode.value} → {decision.target_gpu}")
    print(f"Reason: {decision.reason}")

    # Test case 7: TPOT optimization, NOT Big-Paste (should use Replica)
    print("\n--- Test 7: TPOT, Balanced workload (NOT Big-Paste) ---")
    features = RequestFeatures(
        input_length=256,
        output_length=256,
        turn_number=2,
        has_cache=False,
        cached_gpu=None,
        queue_depths={"gpu1": 0, "gpu2": 1, "gpu3": 0},
        objective=OptimizationObjective.TPOT
    )
    decision = selector.select_mode(features)
    print(f"Decision: {decision.mode.value} → {decision.target_gpu}")
    print(f"Reason: {decision.reason}")

    # Evaluate on training data
    print("\n" + "=" * 60)
    print("Evaluating on Training Data")
    print("=" * 60)

    try:
        results = evaluate_on_training_data(
            RuleBasedSelector(verbose=False),
            "optimizer/data/training_data.json"
        )
        print(f"\nAccuracy: {results['accuracy']:.2%}")
        print(f"Correct: {results['correct']}/{results['total']}")
        print("\nConfusion Matrix (actual, predicted): count")
        for (actual, pred), count in sorted(results['confusion_matrix'].items()):
            print(f"  ({actual}, {pred}): {count}")
    except FileNotFoundError:
        print("Training data not found. Run build_training_data.py first.")
