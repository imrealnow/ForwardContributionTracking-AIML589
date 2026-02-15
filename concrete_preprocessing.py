"""
Concrete Compressive Strength Dataset Preprocessing for GP with FCP Tracking.

This module loads and preprocesses the UCI Concrete Compressive Strength dataset
for use as a symbolic regression demonstration of Forward Contribution Propagation.

Dataset: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
Source: I-Cheng Yeh (1998), "Modeling of strength of high-performance concrete
        using artificial neural networks", Cement and Concrete Research, Vol. 28.

Why this dataset for FCP:
    - All 8 features are continuous (kg/m³ or days), showcasing FCP's value
      contribution tracking through arithmetic operations
    - The target (compressive strength in MPa) is a well-understood nonlinear
      function of mixture components and curing age
    - Domain experts can validate attributions: more cement generally increases
      strength, higher water-to-cement ratio decreases it, age increases it
    - Known feature interactions (e.g. water/cement ratio) provide ground truth
      for whether FCP correctly attributes joint effects
    - 1030 instances with no missing values — clean and GP-friendly

Features (all continuous):
    1. cement         - Cement content (kg/m³)
    2. blast_furnace_slag - Blast Furnace Slag content (kg/m³)
    3. fly_ash        - Fly Ash content (kg/m³)
    4. water          - Water content (kg/m³)
    5. superplasticizer - Superplasticizer content (kg/m³)
    6. coarse_aggregate - Coarse Aggregate content (kg/m³)
    7. fine_aggregate - Fine Aggregate content (kg/m³)
    8. age            - Age of concrete (days, 1-365)

Target:
    concrete_compressive_strength - Compressive strength (MPa)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# Feature Metadata
# =============================================================================

CONCRETE_FEATURES = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

CONCRETE_DESCRIPTIONS = {
    "cement": "Cement content (kg/m³) — primary binder, generally increases strength",
    "blast_furnace_slag": "Blast Furnace Slag (kg/m³) — supplementary cite material, partial cement replacement",
    "fly_ash": "Fly Ash (kg/m³) — supplementary cementitious material, slow strength gain",
    "water": "Water content (kg/m³) — higher water/cement ratio generally decreases strength",
    "superplasticizer": "Superplasticizer (kg/m³) — chemical admixture improving workability",
    "coarse_aggregate": "Coarse Aggregate (kg/m³) — gravel/crusite stone skeleton",
    "fine_aggregate": "Fine Aggregate (kg/m³) — sand, fills gaps between coarse aggregate",
    "age": "Age at testing (days, 1-365) — strength increases with curing time",
}


@dataclass
class ConcreteFeatureInfo:
    """Metadata about preprocessed concrete strength features."""

    names: List[str]
    descriptions: Dict[str, str]
    n_instances: int
    target_range: Tuple[float, float]  # (min, max) compressive strength in dataset
    feature_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    scaled: bool = False
    scale_params: Optional[Dict[str, Tuple[float, float]]] = None  # {name: (mean, std)}

    def __repr__(self) -> str:
        lines = [
            f"Concrete Compressive Strength Features (regression task):",
            f"  Instances: {self.n_instances}",
            f"  Target range: {self.target_range[0]:.1f} - {self.target_range[1]:.1f} MPa",
            f"  Scaled: {self.scaled}",
            "",
        ]
        for name in self.names:
            desc = self.descriptions.get(name, "")
            if name in self.feature_ranges:
                lo, hi = self.feature_ranges[name]
                lines.append(f"  {name} [{lo:.1f}, {hi:.1f}]: {desc}")
            else:
                lines.append(f"  {name}: {desc}")
        return "\n".join(lines)


# =============================================================================
# Data Loading
# =============================================================================


def load_concrete_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load concrete compressive strength data from file.

    Supports:
        - .csv files (comma or semicolon separated)
        - .xls/.xlsx files (requires openpyxl or xlrd)

    Args:
        filepath: Path to data file.

    Returns:
        X: Feature matrix (n_samples, 8)
        y: Target vector (compressive strength in MPa)
    """
    import os

    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".xls", ".xlsx"):
        try:
            import pandas as pd

            df = pd.read_excel(filepath)
        except ImportError:
            raise ImportError(
                "pandas and openpyxl are required to read Excel files. "
                "Install with: pip install pandas openpyxl"
            )
    elif ext == ".csv":
        try:
            import pandas as pd

            # Try comma first, then semicolon
            df = pd.read_csv(filepath)
            if df.shape[1] == 1:
                df = pd.read_csv(filepath, sep=";")
        except ImportError:
            # Fallback to numpy for simple CSVs
            data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
            if data.shape[1] == 1:
                data = np.genfromtxt(filepath, delimiter=";", skip_header=1)
            return data[:, :-1], data[:, -1]
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xls, or .xlsx")

    # The last column is the target
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)

    assert X.shape[1] == 8, (
        f"Expected 8 features, got {X.shape[1]}. " f"Columns found: {list(df.columns)}"
    )

    return X, y


# =============================================================================
# Preprocessing
# =============================================================================


def preprocess_concrete(
    X: np.ndarray,
    y: np.ndarray,
    scale_features: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ConcreteFeatureInfo]:
    """
    Preprocess concrete data for GP evolution.

    This dataset requires minimal preprocessing — all features are already
    continuous numeric values with no missing data. The main decisions are:

    1. Whether to standardise features (scale_features):
       - False (default): GP works with raw values in their natural units.
         This is often preferable for explainability since attributions
         are in physically meaningful units. GP's ephemeral constants and
         evolved coefficients can adapt to the raw scales.
       - True: Standardise to zero mean, unit variance. May help GP
         convergence if the scale differences between features (e.g.
         cement ~100-500 vs superplasticizer ~0-30) cause issues.

    2. Train/test split with shuffling.

    Args:
        X: Feature matrix (n_samples, 8)
        y: Target vector (compressive strength, MPa)
        scale_features: If True, standardise features to zero mean, unit variance
        test_size: Fraction of data for test set
        random_state: Random seed for reproducible splits

    Returns:
        X_train, y_train, X_test, y_test, feature_info
    """
    n_samples = len(X)

    # Record raw feature ranges before any transformation
    feature_ranges = {}
    for i, name in enumerate(CONCRETE_FEATURES):
        feature_ranges[name] = (float(X[:, i].min()), float(X[:, i].max()))

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

    # Optional standardisation (fit on train, apply to both)
    scale_params = None
    if scale_features:
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        stds[stds == 0] = 1.0  # Avoid division by zero

        X_train = (X_train - means) / stds
        X_test = (X_test - means) / stds

        scale_params = {
            name: (float(means[i]), float(stds[i]))
            for i, name in enumerate(CONCRETE_FEATURES)
        }

    feature_info = ConcreteFeatureInfo(
        names=list(CONCRETE_FEATURES),
        descriptions=dict(CONCRETE_DESCRIPTIONS),
        n_instances=n_samples,
        target_range=(float(y.min()), float(y.max())),
        feature_ranges=feature_ranges,
        scaled=scale_features,
        scale_params=scale_params,
    )

    return X_train, y_train, X_test, y_test, feature_info


def load_and_preprocess_concrete(
    filepath: str,
    scale_features: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ConcreteFeatureInfo]:
    """
    Convenience function: load from file and preprocess in one step.

    Args:
        filepath: Path to Concrete_Data.xls or .csv
        scale_features: Standardise features to zero mean, unit variance
        test_size: Fraction for test set
        random_state: Random seed
        verbose: Print summary

    Returns:
        X_train, y_train, X_test, y_test, feature_info
    """
    X, y = load_concrete_data(filepath)

    X_train, y_train, X_test, y_test, info = preprocess_concrete(
        X,
        y,
        scale_features=scale_features,
        test_size=test_size,
        random_state=random_state,
    )

    if verbose:
        print(f"Concrete Compressive Strength Dataset")
        print(f"  Total samples: {info.n_instances}")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Features: {len(info.names)} (all continuous)")
        print(
            f"  Target range: {info.target_range[0]:.1f} - {info.target_range[1]:.1f} MPa"
        )
        print(f"  Target mean: {y.mean():.1f} MPa, std: {y.std():.1f} MPa")
        print(f"  Scaled: {info.scaled}")
        if info.scaled:
            print(f"  (Fitted on training set, applied to both train and test)")

    return X_train, y_train, X_test, y_test, info


# =============================================================================
# GP Configuration Suggestion
# =============================================================================


def get_concrete_gp_config() -> dict:
    """
    Suggested GP configuration for concrete strength regression.

    Returns a dict that can be unpacked into GPConfig.
    The concrete dataset has continuous features and a continuous target,
    so we emphasise arithmetic primitives over conditionals.
    """
    return {
        "feature_names": list(CONCRETE_FEATURES),
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
        "parsimony_coefficient": 0.5,
    }


# =============================================================================
# Evaluation Utilities
# =============================================================================


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate regression predictions with standard metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dict with rmse, mae, r2, and mape
    """
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Mean absolute percentage error (avoiding division by zero)
    nonzero = np.abs(y_true) > 1e-8
    if nonzero.any():
        mape = float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100)
    else:
        mape = float("inf")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python concrete_preprocessing.py <path/to/Concrete_Data.xls>")
        print()
        print("Downloads available from:")
        print("  https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength")
        sys.exit(1)

    filepath = sys.argv[1]

    print("=" * 70)
    print("Concrete Compressive Strength - Preprocessing Demo")
    print("=" * 70)
    print()

    X_train, y_train, X_test, y_test, info = load_and_preprocess_concrete(
        filepath,
        scale_features=False,
        verbose=True,
    )

    print()
    print(info)

    print()
    print("Sample instances (first 3 training rows):")
    for i in range(min(3, len(X_train))):
        print(f"\n  Instance {i} (strength: {y_train[i]:.1f} MPa):")
        for j, name in enumerate(info.names):
            print(f"    {name}: {X_train[i, j]:.1f}")

    print()
    print("=" * 70)
    print("GP Config Suggestion:")
    print("=" * 70)
    config = get_concrete_gp_config()
    for k, v in config.items():
        print(f"  {k}: {v}")
