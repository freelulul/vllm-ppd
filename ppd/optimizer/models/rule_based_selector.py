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
    TTFT = "ttft"
    TPOT = "tpot"
    THROUGHPUT = "throughput"
    E2E = "e2e"


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

    @property
    def input_output_ratio(self) -> float:
        return self.input_length / max(self.output_length, 1)

    @property
    def is_big_paste(self) -> bool:
        """Long input, short output"""
        return self.input_length > 1000 and self.output_length < 256

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

    # GPU configuration (matches our 1P+1D+2R setup)
    GPU_CONFIG = {
        "pd": {"gpus": ["gpu0", "gpu1"], "decode_gpu": "gpu1"},
        "ppd": {"gpus": ["gpu0", "gpu1"], "decode_gpu": "gpu1"},
        "replica0": {"gpus": ["gpu2"], "gpu": "gpu2"},
        "replica1": {"gpus": ["gpu3"], "gpu": "gpu3"},
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

    def select_mode(self, features: RequestFeatures) -> RoutingDecision:
        """
        Main entry point: Select the best mode based on features and objective.

        Decision flow:
        1. Check cache affinity (if has_cache and cost is acceptable)
        2. Apply objective-specific rules
        3. Load balance among candidates
        """
        self.decision_log = []
        self._log(f"Input: {features.input_length} tokens, Output: {features.output_length} tokens")
        self._log(f"Turn: {features.turn_number}, Objective: {features.objective.value}")
        self._log(f"Has cache: {features.has_cache}, Cached GPU: {features.cached_gpu}")

        # Step 1: Check cache affinity for Turn 2+
        if features.has_cache and features.cached_gpu:
            cache_decision = self._check_cache_affinity(features)
            if cache_decision:
                return cache_decision

        # Step 2: Apply objective-specific rules
        if features.objective == OptimizationObjective.TTFT:
            return self._select_for_ttft(features)
        elif features.objective == OptimizationObjective.TPOT:
            return self._select_for_tpot(features)
        elif features.objective == OptimizationObjective.THROUGHPUT:
            return self._select_for_throughput(features)
        elif features.objective == OptimizationObjective.E2E:
            return self._select_for_e2e(features)
        else:
            # Default to PPD
            return self._default_decision(features)

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
        all_gpus = list(features.queue_depths.keys())
        if not all_gpus:
            all_gpus = ["gpu1", "gpu2", "gpu3"]

        min_gpu, min_queue = self._get_min_queue_gpu(features, all_gpus)
        min_wait = self._estimate_queue_wait(min_queue)

        # Decision: Use cache if cache_miss_cost > (cached_wait - min_wait)
        # i.e., the cost of missing cache is higher than extra queue wait
        benefit_of_switching = cached_wait - min_wait
        self._log(f"Cache miss cost: {cache_miss_cost:.0f}ms, Queue wait diff: {benefit_of_switching:.0f}ms")

        if cache_miss_cost > benefit_of_switching:
            self._log(f"→ Cache affinity wins: stay on {cached_gpu}")
            # Determine mode based on cached GPU
            if cached_gpu == "gpu1":
                mode = Mode.PPD  # GPU1 is decode, use PPD for T2+
            elif cached_gpu == "gpu2":
                mode = Mode.REPLICA
            elif cached_gpu == "gpu3":
                mode = Mode.REPLICA
            else:
                mode = Mode.PPD

            return RoutingDecision(
                mode=mode,
                target_gpu=cached_gpu,
                reason=f"Cache affinity: miss cost ({cache_miss_cost:.0f}ms) > switch benefit ({benefit_of_switching:.0f}ms)",
                cache_hit=True
            )

        self._log(f"→ Cache affinity loses: considering other options")
        return None

    def _select_for_ttft(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode optimizing for TTFT (Time To First Token).

        From REAL benchmark (2026-01-13):
        - Replica wins 6/7 workloads for T2 TTFT (no proxy overhead)
        - PD wins for XL_big_paste only
        - PPD is close but slightly slower due to proxy overhead
        """
        self._log("Objective: Minimize TTFT")

        # Rule 1: XL Big-Paste (>3000 input, <100 output) → PD
        # PD handles very long prefill better with P→D parallelization
        if features.input_length > 3000 and features.output_length < 100:
            self._log("XL Big-Paste: PD for long prefill efficiency")
            return RoutingDecision(
                mode=Mode.PD,
                target_gpu="gpu1",
                reason="XL Big-Paste + TTFT: PD prefill parallelization"
            )

        # Rule 2: For most cases, Replica has lowest TTFT (no proxy overhead)
        # This applies to both Turn 1 and Turn 2+
        gpu, queue = self._get_min_queue_gpu(features, ["gpu2", "gpu3"])
        self._log(f"TTFT default: Replica {gpu} (queue: {queue}) - no proxy overhead")
        return RoutingDecision(
            mode=Mode.REPLICA,
            target_gpu=gpu,
            reason=f"TTFT: Replica has lowest latency (no proxy overhead)"
        )

    def _select_for_tpot(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode optimizing for TPOT (Time Per Output Token).

        From REAL benchmark (2026-01-13):
        - PD/PPD consistently have better TPOT (~3.5-3.8ms) vs Replica (~3.8-7.5ms)
        - This is because PD/PPD use dedicated decode GPU
        - PPD is slightly better for Turn 2+ (no P→D transfer overhead)
        """
        self._log("Objective: Minimize TPOT")

        # Rule 1: Turn 2+ → PPD (best TPOT with cache)
        if features.turn_number >= 2:
            self._log("Turn 2+: PPD for best TPOT with cache")
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu1",
                reason="Turn 2+ + TPOT: PPD has dedicated decode GPU with cache"
            )

        # Rule 2: Turn 1 with long output → PPD (TPOT matters more)
        if features.output_length > 256:
            self._log("Long output: PPD for better TPOT")
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu1",
                reason="Long output + TPOT: PPD dedicated decode GPU"
            )

        # Rule 3: Big-Paste → PD (P→D parallelization helps)
        if features.is_big_paste:
            self._log("Big-Paste detected: PD handles long input efficiently")
            return RoutingDecision(
                mode=Mode.PD,
                target_gpu="gpu1",
                reason="Big-Paste + TPOT: PD prefill parallelization"
            )

        # Rule 4: Default to PPD for TPOT (better than Replica)
        return RoutingDecision(
            mode=Mode.PPD,
            target_gpu="gpu1",
            reason="TPOT default: PPD has dedicated decode GPU"
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
        gpu, queue = self._get_min_queue_gpu(features, ["gpu2", "gpu3"])
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

        From Strict SLO test results:
        - E2E: Replica 62.5% vs Optimizer 37.5% (when routing to PPD)
        - Key insight: E2E should ALWAYS prefer Replica (no KV transfer overhead)
        - Even Turn 2 cache benefit in PPD doesn't compensate for Turn 1 KV overhead

        Strategy:
        - E2E tasks should stay on Replica for entire conversation
        - Only exception: very long output where TPOT dominates
        """
        self._log("Objective: Minimize E2E latency")

        # Rule 1: Very long generation → PPD (TPOT dominates E2E)
        # Only for output > 600 tokens where decode time >> prefill time
        if features.output_length > 600:
            return RoutingDecision(
                mode=Mode.PPD,
                target_gpu="gpu1",
                reason="Very long output + E2E: PPD better TPOT"
            )

        # Rule 2: For E2E, Replica is almost always better
        # - No KV transfer overhead
        # - Each replica handles both prefill and decode locally
        # - Turn 2 can still benefit from prefix cache on same replica
        gpu, _ = self._get_min_queue_gpu(features, ["gpu2", "gpu3"])
        return RoutingDecision(
            mode=Mode.REPLICA,
            target_gpu=gpu,
            reason="E2E: Replica no KV transfer, local prefill+decode"
        )

    def _default_decision(self, features: RequestFeatures) -> RoutingDecision:
        """Default decision when no specific rule applies"""
        gpu, _ = self._get_min_queue_gpu(features, ["gpu2", "gpu3"])
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
    selector = RuleBasedSelector(verbose=True)

    print("=" * 60)
    print("Rule-Based Mode Selector Demo")
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
