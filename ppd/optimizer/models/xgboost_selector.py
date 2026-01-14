#!/usr/bin/env python3
"""
XGBoost-Based Mode Selector

Machine learning model that captures non-linear patterns in mode selection.
Trained on benchmark data with optimization objective as a key feature.
"""

import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")


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
    cached_gpu: Optional[str]
    queue_depths: Dict[str, int]
    objective: OptimizationObjective

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for XGBoost"""
        return np.array([
            self.input_length,
            self.output_length,
            self.input_length / max(self.output_length, 1),  # ratio
            self.input_length + self.output_length,  # total tokens
            self.turn_number,
            int(self.has_cache),
            int(self.input_length > 1000 and self.output_length < 256),  # is_big_paste
            int(self.output_length > 512),  # is_long_generation
            self.queue_depths.get('gpu1', 0),
            self.queue_depths.get('gpu2', 0),
            self.queue_depths.get('gpu3', 0),
            # One-hot encoding for objective
            int(self.objective == OptimizationObjective.TTFT),
            int(self.objective == OptimizationObjective.TPOT),
            int(self.objective == OptimizationObjective.THROUGHPUT),
            int(self.objective == OptimizationObjective.E2E),
        ], dtype=np.float32)


FEATURE_NAMES = [
    'input_length',
    'output_length',
    'input_output_ratio',
    'total_tokens',
    'turn_number',
    'has_cache',
    'is_big_paste',
    'is_long_generation',
    'queue_gpu1',
    'queue_gpu2',
    'queue_gpu3',
    'obj_ttft',
    'obj_tpot',
    'obj_throughput',
    'obj_e2e',
]


@dataclass
class RoutingDecision:
    """Result of mode selection"""
    mode: Mode
    target_gpu: str
    reason: str
    confidence: float = 0.0
    probabilities: Optional[Dict[str, float]] = None


class XGBoostSelector:
    """
    XGBoost-based mode selector.

    Captures non-linear patterns in the relationship between:
    - Request features (input/output length, turn number)
    - System state (queue depths, cache status)
    - Optimization objective
    - Best mode choice
    """

    MODE_TO_IDX = {'pd': 0, 'ppd': 1, 'replica': 2}
    IDX_TO_MODE = {0: 'pd', 1: 'ppd', 2: 'replica'}

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_importance = None

        if model_path:
            json_path = Path(model_path).with_suffix('.json')
            if json_path.exists():
                self.load_model(model_path)

    def train(self, data_path: str, save_path: Optional[str] = None) -> Dict:
        """
        Train XGBoost model on training data.

        Args:
            data_path: Path to training_data.json
            save_path: Path to save trained model

        Returns:
            Training results including accuracy and feature importance
        """
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost not installed")

        # Load training data
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Build feature matrix and labels
        X = []
        y = []

        for row in data:
            features = np.array([
                row['t1_input'],
                row['t1_output'],
                row['input_output_ratio'],
                row['t1_input'] + row['t1_output'],
                row['num_turns'],
                0,  # has_cache (not in static training data)
                row['is_big_paste'],
                row['is_long_output'],
                0, 0, 0,  # queue depths (not in static training data)
                row['objective_ttft'],
                row['objective_tpot'],
                row['objective_throughput'],
                row['objective_e2e'],
            ], dtype=np.float32)
            X.append(features)
            y.append(self.MODE_TO_IDX[row['best_mode']])

        X = np.array(X)
        y = np.array(y)

        # Split data
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        split = int(0.8 * n_samples)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'seed': 42,
        }

        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )

        # Evaluate
        y_pred = self.model.predict(dval)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_class == y_val)

        # Feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')

        results = {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'feature_importance': self.feature_importance,
            'best_iteration': self.model.best_iteration,
        }

        # Save model
        if save_path:
            self.save_model(save_path)
            results['model_path'] = save_path

        return results

    def select_mode(self, features: RequestFeatures) -> RoutingDecision:
        """
        Select mode using trained XGBoost model.

        Args:
            features: Request features

        Returns:
            Routing decision with mode, target GPU, and confidence
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Convert to feature vector
        X = features.to_feature_vector().reshape(1, -1)
        dmatrix = xgb.DMatrix(X, feature_names=FEATURE_NAMES)

        # Predict probabilities
        probs = self.model.predict(dmatrix)[0]
        pred_idx = int(np.argmax(probs))
        pred_mode = Mode(self.IDX_TO_MODE[pred_idx])
        confidence = float(probs[pred_idx])

        # Determine target GPU based on mode
        if pred_mode == Mode.PD or pred_mode == Mode.PPD:
            target_gpu = "gpu1"  # Decode GPU for PD/PPD
        else:
            # For replica, choose least loaded
            if features.queue_depths.get('gpu2', 0) <= features.queue_depths.get('gpu3', 0):
                target_gpu = "gpu2"
            else:
                target_gpu = "gpu3"

        return RoutingDecision(
            mode=pred_mode,
            target_gpu=target_gpu,
            reason=f"XGBoost prediction (confidence: {confidence:.2%})",
            confidence=confidence,
            probabilities={
                'pd': float(probs[0]),
                'ppd': float(probs[1]),
                'replica': float(probs[2])
            }
        )

    def save_model(self, path: str):
        """Save trained model to file"""
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(path.with_suffix('.json')))

        # Save feature importance
        with open(path.with_suffix('.importance.json'), 'w') as f:
            json.dump(self.feature_importance, f, indent=2)

        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model from file"""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost not installed")

        path = Path(path)
        self.model = xgb.Booster()
        self.model.load_model(str(path.with_suffix('.json')))

        # Load feature importance if available
        importance_path = path.with_suffix('.importance.json')
        if importance_path.exists():
            with open(importance_path, 'r') as f:
                self.feature_importance = json.load(f)

        print(f"Model loaded from {path}")


def train_and_evaluate():
    """Train XGBoost model and evaluate"""
    if not XGBOOST_AVAILABLE:
        print("XGBoost not installed. Run: pip install xgboost")
        return

    selector = XGBoostSelector()

    print("=" * 60)
    print("Training XGBoost Mode Selector")
    print("=" * 60)

    results = selector.train(
        data_path="optimizer/data/training_data.json",
        save_path="optimizer/models/xgboost_model"
    )

    print(f"\nTraining Results:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Train samples: {results['train_samples']}")
    print(f"  Val samples: {results['val_samples']}")
    print(f"  Best iteration: {results['best_iteration']}")

    print(f"\nFeature Importance (top 10):")
    if results['feature_importance']:
        sorted_importance = sorted(
            results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for name, score in sorted_importance:
            print(f"  {name}: {score:.2f}")

    # Demo predictions
    print("\n" + "=" * 60)
    print("Demo Predictions")
    print("=" * 60)

    test_cases = [
        RequestFeatures(
            input_length=1000, output_length=200, turn_number=2,
            has_cache=True, cached_gpu="gpu1",
            queue_depths={"gpu1": 2, "gpu2": 0, "gpu3": 1},
            objective=OptimizationObjective.TTFT
        ),
        RequestFeatures(
            input_length=4000, output_length=100, turn_number=1,
            has_cache=False, cached_gpu=None,
            queue_depths={"gpu1": 0, "gpu2": 1, "gpu3": 2},
            objective=OptimizationObjective.TPOT
        ),
        RequestFeatures(
            input_length=500, output_length=500, turn_number=1,
            has_cache=False, cached_gpu=None,
            queue_depths={"gpu1": 3, "gpu2": 1, "gpu3": 0},
            objective=OptimizationObjective.THROUGHPUT
        ),
    ]

    for i, features in enumerate(test_cases):
        decision = selector.select_mode(features)
        print(f"\nTest {i+1}: {features.objective.value}, input={features.input_length}, output={features.output_length}")
        print(f"  Prediction: {decision.mode.value} → {decision.target_gpu}")
        print(f"  Confidence: {decision.confidence:.2%}")
        print(f"  Probabilities: pd={decision.probabilities['pd']:.2%}, ppd={decision.probabilities['ppd']:.2%}, replica={decision.probabilities['replica']:.2%}")


if __name__ == "__main__":
    train_and_evaluate()
