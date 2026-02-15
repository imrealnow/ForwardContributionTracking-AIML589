"""
Early Stage Diabetes Risk Prediction Dataset Preprocessing for GP with FCP Tracking.

This module loads and preprocesses the UCI Early Stage Diabetes Risk Prediction
dataset for use as a binary classification demonstration of Forward Contribution
Propagation, with emphasis on decision contribution tracking through conditional
branching operations.

Dataset: https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
Source: Islam et al. (2020), "Likelihood Prediction of Diabetes at Early Stage
        Using Data Mining Techniques", CVMI.

Why this dataset for FCP:
    - 15 of 16 features are binary (Yes/No symptoms), making it ideal for
      showcasing FCP's decision contribution channel — when GP uses IFLTE-style
      conditionals on binary features, the attributions become directly
      interpretable as "this prediction was driven by the presence/absence
      of symptom X"
    - One continuous feature (age) provides contrast for value contributions
    - Binary classification target (Positive/Negative diabetes risk)
    - Medically interpretable features — polyuria, polydipsia, sudden weight
      loss are well-known diabetes indicators, allowing domain validation
      of FCP attributions
    - 520 instances, no missing values — compact and GP-friendly
    - Healthcare context where transparency is paramount

Features:
    1.  age             - Patient age (integer, 16-90)
    2.  gender          - Male/Female (binary)
    3.  polyuria        - Excessive urination (binary)
    4.  polydipsia      - Excessive thirst (binary)
    5.  sudden_weight_loss - Sudden weight loss (binary)
    6.  weakness        - General weakness (binary)
    7.  polyphagia      - Excessive hunger (binary)
    8.  genital_thrush  - Genital thrush infection (binary)
    9.  visual_blurring - Visual blurring (binary)
    10. itching         - Itching (binary)
    11. irritability    - Irritability (binary)
    12. delayed_healing - Delayed wound healing (binary)
    13. partial_paresis - Partial muscle weakness (binary)
    14. muscle_stiffness - Muscle stiffness (binary)
    15. alopecia        - Hair loss (binary)
    16. obesity         - Obesity (binary)

Target:
    class - Positive (diabetic) / Negative (not diabetic)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# Feature Metadata
# =============================================================================

DIABETES_FEATURES = [
    "age",
    "gender",
    "polyuria",
    "polydipsia",
    "sudden_weight_loss",
    "weakness",
    "polyphagia",
    "genital_thrush",
    "visual_blurring",
    "itching",
    "irritability",
    "delayed_healing",
    "partial_paresis",
    "muscle_stiffness",
    "alopecia",
    "obesity",
]

DIABETES_DESCRIPTIONS = {
    "age": "Patient age in years (continuous, 16-90)",
    "gender": "Gender (binary: 1=Male, 0=Female)",
    "polyuria": "Excessive urination (binary: 1=Yes, 0=No) — classic diabetes symptom",
    "polydipsia": "Excessive thirst (binary: 1=Yes, 0=No) — classic diabetes symptom",
    "sudden_weight_loss": "Sudden weight loss (binary: 1=Yes, 0=No) — common early sign",
    "weakness": "General weakness/fatigue (binary: 1=Yes, 0=No)",
    "polyphagia": "Excessive hunger (binary: 1=Yes, 0=No) — classic diabetes symptom",
    "genital_thrush": "Genital thrush/yeast infection (binary: 1=Yes, 0=No)",
    "visual_blurring": "Visual blurring (binary: 1=Yes, 0=No)",
    "itching": "Itching (binary: 1=Yes, 0=No)",
    "irritability": "Irritability (binary: 1=Yes, 0=No)",
    "delayed_healing": "Delayed wound healing (binary: 1=Yes, 0=No)",
    "partial_paresis": "Partial loss of voluntary movement (binary: 1=Yes, 0=No)",
    "muscle_stiffness": "Muscle stiffness (binary: 1=Yes, 0=No)",
    "alopecia": "Hair loss (binary: 1=Yes, 0=No)",
    "obesity": "Obesity (binary: 1=Yes, 0=No)",
}

# Features that are well-established diabetes indicators
# (useful for validating that FCP attributions make clinical sense)
STRONG_INDICATORS = ["polyuria", "polydipsia", "polyphagia", "sudden_weight_loss"]
MODERATE_INDICATORS = [
    "genital_thrush",
    "visual_blurring",
    "partial_paresis",
    "irritability",
]
WEAK_INDICATORS = [
    "itching",
    "delayed_healing",
    "muscle_stiffness",
    "alopecia",
    "obesity",
]


@dataclass
class DiabetesFeatureInfo:
    """Metadata about preprocessed diabetes features."""

    names: List[str]
    descriptions: Dict[str, str]
    n_instances: int
    n_positive: int
    n_negative: int
    binary_features: List[str]  # Names of binary-encoded features
    continuous_features: List[str]  # Names of continuous features
    age_range: Tuple[int, int] = (16, 90)
    age_scaled: bool = False
    age_scale_params: Optional[Tuple[float, float]] = None  # (mean, std)

    @property
    def class_balance(self) -> float:
        """Fraction of positive class."""
        return self.n_positive / self.n_instances

    def __repr__(self) -> str:
        lines = [
            f"Diabetes Risk Prediction Features (binary classification):",
            f"  Instances: {self.n_instances} ({self.n_positive} positive, {self.n_negative} negative)",
            f"  Class balance: {self.class_balance:.1%} positive",
            f"  Binary features: {len(self.binary_features)}",
            f"  Continuous features: {len(self.continuous_features)}",
            f"  Age range: {self.age_range[0]}-{self.age_range[1]}",
            f"  Age scaled: {self.age_scaled}",
            "",
        ]
        for name in self.names:
            desc = self.descriptions.get(name, "")
            indicator = ""
            if name in STRONG_INDICATORS:
                indicator = " [STRONG indicator]"
            elif name in MODERATE_INDICATORS:
                indicator = " [moderate indicator]"
            lines.append(f"  {name}: {desc}{indicator}")
        return "\n".join(lines)


# =============================================================================
# Data Loading
# =============================================================================


def load_diabetes_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load early stage diabetes data from CSV file.

    Handles the encoding of Yes/No and Male/Female to 1/0, and
    Positive/Negative class labels to 1/0.

    Args:
        filepath: Path to diabetes_data_upload.csv

    Returns:
        X: Feature matrix (n_samples, 16)
        y: Target vector (1=Positive, 0=Negative)
    """
    try:
        import pandas as pd

        df = pd.read_csv(filepath)
    except ImportError:
        raise ImportError(
            "pandas is required to read the diabetes CSV. "
            "Install with: pip install pandas"
        )

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Encode binary columns: Yes->1, No->0, Male->1, Female->0
    binary_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
    gender_map = {"Male": 1, "Female": 0, "male": 1, "female": 0}

    for col in df.columns:
        if col == "gender":
            df[col] = df[col].map(gender_map)
        elif col == "class":
            df[col] = df[col].map(
                {"Positive": 1, "Negative": 0, "positive": 1, "negative": 0}
            )
        elif col != "age":
            df[col] = df[col].map(binary_map)

    # Separate features and target
    target_col = "class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    # Validate
    assert X.shape[1] == 16, (
        f"Expected 16 features, got {X.shape[1]}. " f"Columns: {feature_cols}"
    )
    assert not np.any(np.isnan(X)), "Unexpected NaN values in features"
    assert not np.any(np.isnan(y)), "Unexpected NaN values in target"

    return X, y


# =============================================================================
# Preprocessing
# =============================================================================


def preprocess_diabetes(
    X: np.ndarray,
    y: np.ndarray,
    scale_age: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DiabetesFeatureInfo]:
    """
    Preprocess diabetes data for GP evolution.

    This dataset requires minimal preprocessing — binary features are already
    0/1, and age is the only continuous feature. The main decision is whether
    to scale age.

    Scaling age:
        - False (default): Age stays in raw years (16-90). GP's ephemeral
          constants can adapt. Binary features are 0/1 so IFLTE thresholds
          of ~0.5 work naturally, while age requires different thresholds.
          This scale difference may actually help GP evolve distinct handling
          for age vs symptoms.
        - True: Standardise age to zero mean, unit variance. Puts age on a
          similar scale to the binary features, which may simplify GP's job
          but loses the intuitive interpretation of age in years.

    Args:
        X: Feature matrix (n_samples, 16)
        y: Target vector (1=Positive, 0=Negative)
        scale_age: If True, standardise the age feature
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, y_train, X_test, y_test, feature_info
    """
    n_samples = len(X)

    # Identify feature types
    age_idx = DIABETES_FEATURES.index("age")
    binary_features = [f for f in DIABETES_FEATURES if f != "age"]
    continuous_features = ["age"]

    # Record age range
    age_range = (int(X[:, age_idx].min()), int(X[:, age_idx].max()))

    # Shuffle and split
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx].copy()
    y_train = y[train_idx].copy()
    X_test = X[test_idx].copy()
    y_test = y[test_idx].copy()

    # Optional age scaling
    age_scale_params = None
    if scale_age:
        age_mean = X_train[:, age_idx].mean()
        age_std = X_train[:, age_idx].std()
        if age_std == 0:
            age_std = 1.0

        X_train[:, age_idx] = (X_train[:, age_idx] - age_mean) / age_std
        X_test[:, age_idx] = (X_test[:, age_idx] - age_mean) / age_std
        age_scale_params = (float(age_mean), float(age_std))

    # Count classes
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)

    feature_info = DiabetesFeatureInfo(
        names=list(DIABETES_FEATURES),
        descriptions=dict(DIABETES_DESCRIPTIONS),
        n_instances=n_samples,
        n_positive=n_positive,
        n_negative=n_negative,
        binary_features=binary_features,
        continuous_features=continuous_features,
        age_range=age_range,
        age_scaled=scale_age,
        age_scale_params=age_scale_params,
    )

    return X_train, y_train, X_test, y_test, feature_info


def load_and_preprocess_diabetes(
    filepath: str,
    scale_age: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DiabetesFeatureInfo]:
    """
    Convenience function: load from file and preprocess in one step.

    Args:
        filepath: Path to diabetes_data_upload.csv
        scale_age: Standardise the age feature
        test_size: Fraction for test set
        random_state: Random seed
        verbose: Print summary

    Returns:
        X_train, y_train, X_test, y_test, feature_info
    """
    X, y = load_diabetes_data(filepath)

    X_train, y_train, X_test, y_test, info = preprocess_diabetes(
        X,
        y,
        scale_age=scale_age,
        test_size=test_size,
        random_state=random_state,
    )

    if verbose:
        print(f"Early Stage Diabetes Risk Prediction Dataset")
        print(f"  Total samples: {info.n_instances}")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(
            f"  Features: {len(info.names)} ({len(info.binary_features)} binary + age)"
        )
        print(
            f"  Class balance: {info.n_positive} positive ({info.class_balance:.1%}), "
            f"{info.n_negative} negative ({1-info.class_balance:.1%})"
        )
        print(f"  Age range: {info.age_range[0]}-{info.age_range[1]} years")
        print(f"  Age scaled: {info.age_scaled}")

    return X_train, y_train, X_test, y_test, info


# =============================================================================
# GP Configuration Suggestion
# =============================================================================


def get_diabetes_gp_config() -> dict:
    """
    Suggested GP configuration for diabetes classification.

    Returns a dict that can be unpacked into GPConfig.
    Since the features are mostly binary, we emphasise conditional
    operations (IFLTE) which will naturally produce decision-tree-like
    branching structures — ideal for demonstrating FCP's decision
    contribution tracking.
    """
    return {
        "feature_names": list(DIABETES_FEATURES),
        "generations": 300,
        "mu": 150,
        "lambda_": 300,
        "max_tree_depth": 6,
        "min_tree_depth": 2,
        "crossover_prob": 0.7,
        "mutation_prob": 0.2,
        "tournament_size": 5,
        "include_conditionals": True,
        "include_comparisons": True,
        "include_activations": True,
        "parsimony_coefficient": 0.001,
    }


# =============================================================================
# Evaluation Utilities
# =============================================================================


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate binary classification predictions.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted values (will be thresholded)
        threshold: Decision threshold for converting predictions to 0/1

    Returns:
        Dict with accuracy, balanced_accuracy, precision, recall, f1, specificity
    """
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)

    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python diabetes_preprocessing.py <path/to/diabetes_data_upload.csv>"
        )
        print()
        print("Downloads available from:")
        print(
            "  https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset"
        )
        sys.exit(1)

    filepath = sys.argv[1]

    print("=" * 70)
    print("Early Stage Diabetes Risk Prediction - Preprocessing Demo")
    print("=" * 70)
    print()

    X_train, y_train, X_test, y_test, info = load_and_preprocess_diabetes(
        filepath,
        scale_age=False,
        verbose=True,
    )

    print()
    print(info)

    print()
    print("Sample instances (first 3 training rows):")
    for i in range(min(3, len(X_train))):
        label = "Positive" if y_train[i] == 1 else "Negative"
        print(f"\n  Instance {i} ({label}):")
        for j, name in enumerate(info.names):
            val = X_train[i, j]
            if name in info.binary_features:
                display = "Yes" if val == 1 else "No"
                print(f"    {name}: {display}")
            else:
                print(f"    {name}: {val:.0f}")

    # Show symptom prevalence
    print()
    print("Symptom prevalence (training set):")
    for j, name in enumerate(info.names):
        if name in info.binary_features:
            prev_pos = X_train[y_train == 1, j].mean()
            prev_neg = X_train[y_train == 0, j].mean()
            print(
                f"  {name:20s}  positive: {prev_pos:.0%}  negative: {prev_neg:.0%}  "
                f"diff: {prev_pos - prev_neg:+.0%}"
            )

    print()
    print("=" * 70)
    print("GP Config Suggestion:")
    print("=" * 70)
    config = get_diabetes_gp_config()
    for k, v in config.items():
        print(f"  {k}: {v}")
