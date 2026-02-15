"""
Generic GP Evolution and FCP Explanation Framework

This module provides a reusable framework for:
1. Evolving GP classifiers/regressors on any dataset
2. Generating FCP explanations for predictions
3. Visualizing decision paths

Usage:
    from gp_evolve_explain import GPConfig, evolve_gp, explain_prediction

    # Configure
    config = GPConfig(
        feature_names=['age', 'income', 'score'],
        protected_features=['age'],
        actionable_features=['income', 'score'],
    )

    # Define custom fitness
    def my_fitness(individual, toolbox, X, y):
        # Your evaluation logic
        return (score,)

    # Evolve
    best_tree, pset = evolve_gp(X_train, y_train, config, fitness_fn=my_fitness)

    # Explain
    result = explain_prediction(best_tree, instance, config)
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# DEAP imports
from deap import base, creator, tools, gp, algorithms


# =============================================================================
# CONFIGURATION
# =============================================================================


class TaskType(Enum):
    """Type of prediction task."""

    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"  # Future support


@dataclass
class GPConfig:
    """
    Configuration for GP evolution and explanation.

    Attributes:
        feature_names: List of feature names (must match X columns)
        protected_features: Features that shouldn't be changed in counterfactuals
        actionable_features: Features that can be changed in counterfactuals
        task_type: Type of prediction task

        # Primitive set options
        include_trig: Include sin/cos primitives
        include_exp_log: Include exp/log primitives
        include_comparisons: Include comparison primitives (LT, GT, etc.)
        include_conditionals: Include if-then-else primitives
        include_activations: Include activation functions (sigmoid, relu, etc.)

        # Named constants to include
        named_constants: Dict of name -> value for fixed constants

        # Ephemeral constant ranges
        ephemeral_weights: Tuple of (min, max) for weight constants
        ephemeral_thresholds: List of possible threshold values

        # Evolution parameters
        mu: Number of parents in (μ + λ) strategy
        lambda_: Number of offspring per generation
        generations: Number of generations
        tournament_size: Tournament selection size
        crossover_prob: Probability of crossover
        mutation_prob: Probability of mutation
        max_tree_size: Maximum tree size (bloat control)
        min_tree_depth: Minimum initial tree depth
        max_tree_depth: Maximum initial tree depth
        parsimony_coefficient: Penalty for tree size

        # Random seed
        seed: Random seed for reproducibility
    """

    # Required
    feature_names: List[str]

    # Feature metadata
    protected_features: List[str] = field(default_factory=list)
    actionable_features: List[str] = field(default_factory=list)
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION

    # Primitive set options
    include_trig: bool = False
    include_exp_log: bool = False
    include_comparisons: bool = True
    include_conditionals: bool = True
    include_activations: bool = True

    # Constants
    named_constants: Dict[str, float] = field(
        default_factory=lambda: {
            "ZERO": 0.0,
            "ONE": 1.0,
            "NEGONE": -1.0,
        }
    )
    ephemeral_weights: Tuple[float, float] = (-5.0, 5.0)
    ephemeral_thresholds: List[float] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 10, 50, 100]
    )

    # Evolution parameters
    mu: int = 100
    lambda_: int = 200
    generations: int = 40
    tournament_size: int = 5
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    max_tree_size: int = 50
    min_tree_depth: int = 2
    max_tree_depth: int = 6
    parsimony_coefficient: float = 0.001

    # Random seed
    seed: Optional[int] = 42

    def validate(self):
        """Validate configuration."""
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")

        for pf in self.protected_features:
            if pf not in self.feature_names:
                raise ValueError(f"Protected feature '{pf}' not in feature_names")

        for af in self.actionable_features:
            if af not in self.feature_names:
                raise ValueError(f"Actionable feature '{af}' not in feature_names")


# =============================================================================
# PRIMITIVE SET CREATION
# =============================================================================


def create_evolution_pset(config: GPConfig) -> gp.PrimitiveSet:
    """
    Create a primitive set for evolution (plain Python operations).

    This pset uses plain Python for maximum speed during evolution.
    """
    pset = gp.PrimitiveSet("main", len(config.feature_names))

    # Rename arguments to feature names
    for i, name in enumerate(config.feature_names):
        pset.renameArguments(**{f"ARG{i}": name})

    # === ARITHMETIC OPERATIONS ===
    pset.addPrimitive(lambda x, y: x + y, 2, name="ADD")
    pset.addPrimitive(lambda x, y: x - y, 2, name="SUB")
    pset.addPrimitive(lambda x, y: x * y, 2, name="MUL")

    def safe_div(x, y):
        return x / y if abs(y) > 1e-10 else 1.0

    pset.addPrimitive(safe_div, 2, name="DIV")

    # === UNARY OPERATIONS ===
    pset.addPrimitive(lambda x: -x, 1, name="NEG")
    pset.addPrimitive(lambda x: abs(x), 1, name="ABS")

    # === COMPARISON/SELECTION ===
    pset.addPrimitive(lambda x, y: min(x, y), 2, name="MIN")
    pset.addPrimitive(lambda x, y: max(x, y), 2, name="MAX")

    if config.include_activations:
        # Sigmoid
        def sigmoid(x):
            if x >= 0:
                return 1 / (1 + np.exp(-min(x, 500)))
            else:
                exp_x = np.exp(max(x, -500))
                return exp_x / (1 + exp_x)

        pset.addPrimitive(sigmoid, 1, name="SIG")

        # Sign
        pset.addPrimitive(lambda x: 1.0 if x >= 0 else -1.0, 1, name="SIGN")

        # ReLU
        pset.addPrimitive(lambda x: max(0.0, x), 1, name="RELU")

        # Compare
        def compare(a, b):
            if a < b:
                return -1.0
            elif a > b:
                return 1.0
            else:
                return 0.0

        pset.addPrimitive(compare, 2, name="CMP")

    if config.include_conditionals:

        def if_lte(a, b, then_val, else_val):
            return then_val if a <= b else else_val

        pset.addPrimitive(if_lte, 4, name="IFLTE")

    if config.include_trig:
        pset.addPrimitive(np.sin, 1, name="SIN")
        pset.addPrimitive(np.cos, 1, name="COS")

    if config.include_exp_log:
        pset.addPrimitive(lambda x: np.exp(np.clip(x, -500, 500)), 1, name="EXP")
        pset.addPrimitive(lambda x: np.log(max(abs(x), 1e-10)), 1, name="LOG")

    # === CONSTANTS ===
    # Named constants
    for name, value in config.named_constants.items():
        pset.addTerminal(value, name=name)

    # Ephemeral constants
    w_min, w_max = config.ephemeral_weights
    pset.addEphemeralConstant("W", lambda: random.uniform(w_min, w_max))

    if config.ephemeral_thresholds:
        thresholds = config.ephemeral_thresholds
        pset.addEphemeralConstant("T", lambda: random.choice(thresholds))

    return pset


# =============================================================================
# BUILT-IN FITNESS FUNCTIONS
# =============================================================================


def fitness_accuracy(individual, toolbox, X, y) -> Tuple[float]:
    """Simple accuracy fitness (for maximization, negate for minimization)."""
    func = toolbox.compile(expr=individual)

    correct = 0
    total = 0
    for xi, yi in zip(X, y):
        try:
            score = func(*xi)
            pred = 1 if score > 0 else 0
            if pred == yi:
                correct += 1
            total += 1
        except Exception:
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return (-accuracy,)  # Negative for minimization


def fitness_balanced_accuracy(individual, toolbox, X, y) -> Tuple[float]:
    """Balanced accuracy (average of sensitivity and specificity)."""
    func = toolbox.compile(expr=individual)

    tp = fp = tn = fn = 0
    for xi, yi in zip(X, y):
        try:
            score = func(*xi)
            pred = 1 if score > 0 else 0

            if yi == 1 and pred == 1:
                tp += 1
            elif yi == 0 and pred == 1:
                fp += 1
            elif yi == 0 and pred == 0:
                tn += 1
            else:
                fn += 1
        except Exception:
            fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2

    return (-balanced_acc,)


def fitness_f1(individual, toolbox, X, y) -> Tuple[float]:
    """F1 score fitness."""
    func = toolbox.compile(expr=individual)

    tp = fp = fn = 0
    for xi, yi in zip(X, y):
        try:
            score = func(*xi)
            pred = 1 if score > 0 else 0

            if yi == 1 and pred == 1:
                tp += 1
            elif yi == 0 and pred == 1:
                fp += 1
            elif yi == 1 and pred == 0:
                fn += 1
        except Exception:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return (-f1,)


def fitness_mse(individual, toolbox, X, y) -> Tuple[float]:
    """Mean squared error fitness (for regression)."""
    func = toolbox.compile(expr=individual)

    total_error = 0
    count = 0
    for xi, yi in zip(X, y):
        try:
            pred = func(*xi)
            total_error += (pred - yi) ** 2
            count += 1
        except Exception:
            total_error += 1000  # Penalty
            count += 1

    mse = total_error / count if count > 0 else float("inf")
    return (mse,)


def fitness_cost_matrix(
    fp_cost: float = 1.0,
    fn_cost: float = 1.0,
) -> Callable:
    """
    Create a cost-matrix fitness function.

    Args:
        fp_cost: Cost of false positive (predicting 1 when true is 0)
        fn_cost: Cost of false negative (predicting 0 when true is 1)

    Returns:
        Fitness function
    """

    def fitness(individual, toolbox, X, y) -> Tuple[float]:
        func = toolbox.compile(expr=individual)

        total_cost = 0
        for xi, yi in zip(X, y):
            try:
                score = func(*xi)
                pred = 1 if score > 0 else 0

                if yi == 1 and pred == 0:
                    total_cost += fn_cost
                elif yi == 0 and pred == 1:
                    total_cost += fp_cost
            except Exception:
                total_cost += max(fp_cost, fn_cost) * 10

        return (total_cost,)

    return fitness


# Built-in fitness functions registry
BUILTIN_FITNESS = {
    "accuracy": fitness_accuracy,
    "balanced_accuracy": fitness_balanced_accuracy,
    "f1": fitness_f1,
    "mse": fitness_mse,
}


# =============================================================================
# GP EVOLUTION
# =============================================================================


def evolve_gp(
    X: np.ndarray,
    y: np.ndarray,
    config: GPConfig,
    fitness_fn: Union[str, Callable] = "balanced_accuracy",
    verbose: bool = True,
) -> Tuple[gp.PrimitiveTree, gp.PrimitiveSet]:
    """
    Evolve a GP model using (μ + λ) evolutionary strategy.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        config: GPConfig with evolution parameters
        fitness_fn: Either a string key for built-in fitness, or a callable
                   with signature (individual, toolbox, X, y) -> (score,)
        verbose: Print progress

    Returns:
        best_tree: Best individual found
        pset: The primitive set used
    """
    config.validate()

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    pset = create_evolution_pset(config)

    # Resolve fitness function
    if isinstance(fitness_fn, str):
        if fitness_fn not in BUILTIN_FITNESS:
            raise ValueError(
                f"Unknown fitness function: {fitness_fn}. "
                f"Choose from: {list(BUILTIN_FITNESS.keys())}"
            )
        base_fitness = BUILTIN_FITNESS[fitness_fn]
    else:
        base_fitness = fitness_fn

    # Create DEAP types (only if not already created)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Tree generation
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=config.min_tree_depth,
        max_=config.max_tree_depth,
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Fitness with parsimony pressure
    def evaluate_with_parsimony(individual):
        fitness = base_fitness(individual, toolbox, X, y)[0]
        size_penalty = len(individual) * config.parsimony_coefficient
        return (fitness + size_penalty,)

    toolbox.register("evaluate", evaluate_with_parsimony)

    # Selection and variation
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    toolbox.register("cx_onepoint", gp.cxOnePoint)
    toolbox.register("cx_leaf", gp.cxOnePointLeafBiased, termpb=0.3)

    toolbox.register("expr_grow", gp.genGrow, pset=pset, min_=1, max_=3)
    toolbox.register("mut_uniform", gp.mutUniform, expr=toolbox.expr_grow, pset=pset)
    toolbox.register("mut_node", gp.mutNodeReplacement, pset=pset)
    toolbox.register("mut_ephemeral", gp.mutEphemeral, mode="one")
    toolbox.register("mut_shrink", gp.mutShrink)
    toolbox.register("mut_insert", gp.mutInsert, pset=pset)

    # Bloat control
    toolbox.decorate(
        "cx_onepoint", gp.staticLimit(key=len, max_value=config.max_tree_size)
    )
    toolbox.decorate("cx_leaf", gp.staticLimit(key=len, max_value=config.max_tree_size))
    toolbox.decorate(
        "mut_uniform", gp.staticLimit(key=len, max_value=config.max_tree_size)
    )
    toolbox.decorate(
        "mut_insert", gp.staticLimit(key=len, max_value=config.max_tree_size)
    )

    # Initialize population
    population = toolbox.population(n=config.mu)
    hof = tools.HallOfFame(10)

    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_fit.register("min", np.min)
    stats_fit.register("avg", np.mean)

    stats_size = tools.Statistics(lambda ind: len(ind))
    stats_size.register("size", np.mean)

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(population)

    if verbose:
        print(f"(μ + λ) Evolution: μ={config.mu}, λ={config.lambda_}")
        print(f"Fitness: {fitness_fn if isinstance(fitness_fn, str) else 'custom'}")
        print()
        print(f"{'Gen':>4} {'Fitness':>12} {'Avg':>10} {'Size':>6}")
        print("-" * 36)
        record = stats_fit.compile(population)
        size_record = stats_size.compile(population)
        print(
            f"{0:>4} {record['min']:>12.4f} {record['avg']:>10.4f} {size_record['size']:>6.1f}"
        )

    # Evolution loop
    for gen in range(1, config.generations + 1):
        # Generate offspring
        offspring = []
        while len(offspring) < config.lambda_:
            parents = toolbox.select(population, 2)
            child1, child2 = [toolbox.clone(p) for p in parents]

            # Crossover
            if random.random() < config.crossover_prob:
                if random.random() < 0.7:
                    child1, child2 = toolbox.cx_onepoint(child1, child2)
                else:
                    child1, child2 = toolbox.cx_leaf(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            # Mutation
            for child in [child1, child2]:
                if random.random() < config.mutation_prob:
                    mutation_choice = random.random()
                    try:
                        if mutation_choice < 0.30:
                            (child,) = toolbox.mut_uniform(child)
                        elif mutation_choice < 0.50:
                            (child,) = toolbox.mut_node(child)
                        elif mutation_choice < 0.70:
                            (child,) = toolbox.mut_ephemeral(child)
                        elif mutation_choice < 0.85:
                            (child,) = toolbox.mut_shrink(child)
                        else:
                            (child,) = toolbox.mut_insert(child)
                        if hasattr(child, "fitness"):
                            del child.fitness.values
                    except Exception:
                        pass

            offspring.append(child1)
            if len(offspring) < config.lambda_:
                offspring.append(child2)

        offspring = offspring[: config.lambda_]

        # Evaluate offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # (μ + λ) selection
        combined = population + offspring
        combined_sorted = sorted(combined, key=lambda ind: ind.fitness.values[0])
        population = combined_sorted[: config.mu]

        hof.update(population)

        if verbose:
            record = stats_fit.compile(population)
            size_record = stats_size.compile(population)
            print(
                f"{gen:>4} {record['min']:>12.4f} {record['avg']:>10.4f} {size_record['size']:>6.1f}"
            )

    if verbose:
        print("-" * 36)
        best = hof[0]
        print(f"Best fitness: {best.fitness.values[0]:.4f}")
        print(f"Best tree size: {len(best)}")

    return hof[0], pset


# =============================================================================
# FCP EXPLANATION
# =============================================================================


def explain_prediction(
    tree: gp.PrimitiveTree,
    instance: Dict[str, float],
    config: GPConfig,
    threshold: float = 0.0,
    constant_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Generate FCP explanation for a single prediction.

    Args:
        tree: The evolved GP tree
        instance: Dict mapping feature names to values
        config: GPConfig used for evolution
        threshold: Decision threshold (default 0 for binary classification)
        constant_map: Optional pre-built constant map (name -> value).
                      If provided, these constants are registered in the
                      tracked pset before parsing. Useful when the tree
                      contains terminals not in config.named_constants
                      (e.g. _K0, _K1 from algebraic simplification).

    Returns:
        Dict with prediction, score, explanation, visualization
    """
    # Import FCP modules
    from fcp_tracker import (
        create_tracked_pset,
        TrackedValue,
        evaluate_with_tracking,
        synthesize_explanation,
    )
    from fcp_tree_viz import visualize_tree
    import re

    # Create tracked pset matching evolution pset
    pset = create_tracked_pset(
        "main",
        config.feature_names,
        include_trig=config.include_trig,
        include_exp_log=config.include_exp_log,
        include_comparisons=config.include_comparisons,
        include_conditionals=config.include_conditionals,
        include_activations=config.include_activations,
    )

    # Add named constants from config
    for name, value in config.named_constants.items():
        pset.addTerminal(TrackedValue.from_constant(value, name), name=name)

    # Track constants for visualization
    all_constants = dict(config.named_constants)
    constant_names = list(config.named_constants.keys())

    # Register any pre-provided constants (e.g. from simplification)
    if constant_map:
        for name, value in constant_map.items():
            if name not in all_constants:
                all_constants[name] = value
                constant_names.append(name)
                pset.addTerminal(TrackedValue.from_constant(value, name), name=name)

    # Parse tree string and replace numeric literals with terminals
    tree_str = str(tree)

    def replace_literal(match):
        lit = match.group(1)
        try:
            val = float(lit)
            safe_name = f"C{len(all_constants) - len(config.named_constants)}"
            all_constants[safe_name] = val
            constant_names.append(safe_name)
            pset.addTerminal(TrackedValue.from_constant(val, safe_name), name=safe_name)
            return safe_name
        except ValueError:
            return lit

    numeric_pattern = r"(?<![a-zA-Z_\d])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?![a-zA-Z_\d])"
    tree_str_safe = re.sub(numeric_pattern, replace_literal, tree_str)

    # Parse and evaluate
    tracked_tree = gp.PrimitiveTree.from_string(tree_str_safe, pset)
    result = evaluate_with_tracking(tracked_tree, pset, instance)

    score = result.output.value

    # Determine prediction based on task type
    if config.task_type == TaskType.BINARY_CLASSIFICATION:
        prediction = 1 if score > threshold else 0
        prediction_label = "positive" if prediction == 1 else "negative"
    else:
        prediction = score
        prediction_label = f"{score:.4f}"

    # Generate explanation
    terminal_values_for_explanation = dict(instance)
    terminal_values_for_explanation.update(all_constants)

    explanation = synthesize_explanation(
        result,
        terminal_values=terminal_values_for_explanation,
        tree=tracked_tree,
        pset=pset,
        frozen_features=config.protected_features,
        actionable_features=config.actionable_features,
        constant_names=constant_names,
    )

    # Generate visualization
    terminal_values = dict(instance)
    terminal_values.update(all_constants)

    viz = visualize_tree(
        tracked_tree,
        pset,
        terminal_values=terminal_values,
        evaluation_result=result,
        constant_names=constant_names,
    )

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "score": score,
        "explanation": explanation,
        "visualization": viz,
        "result": result,
        "constant_map": all_constants,
    }


def evaluate_model(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    X: np.ndarray,
    y: np.ndarray,
    config: GPConfig,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

    Returns dict with various metrics.
    """
    func = gp.compile(tree, pset)

    tp = fp = tn = fn = 0
    total_error = 0
    predictions = []

    for xi, yi in zip(X, y):
        try:
            score = func(*xi)
            predictions.append(score)

            if config.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = 1 if score > 0 else 0
                if yi == 1 and pred == 1:
                    tp += 1
                elif yi == 0 and pred == 1:
                    fp += 1
                elif yi == 0 and pred == 0:
                    tn += 1
                else:
                    fn += 1
            else:
                total_error += (score - yi) ** 2
        except Exception:
            fn += 1
            predictions.append(0)

    metrics = {}

    if config.task_type == TaskType.BINARY_CLASSIFICATION:
        metrics["accuracy"] = (
            (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        )
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2
        metrics["f1"] = (
            2
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"])
            if (metrics["precision"] + metrics["recall"]) > 0
            else 0
        )
        metrics["tp"] = tp
        metrics["fp"] = fp
        metrics["tn"] = tn
        metrics["fn"] = fn
    else:
        metrics["mse"] = total_error / len(y) if len(y) > 0 else float("inf")
        metrics["rmse"] = np.sqrt(metrics["mse"])

    return metrics


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: GPConfig,
    fitness_fn: Union[str, Callable] = "balanced_accuracy",
    n_explanations: int = 5,
    verbose: bool = True,
    save_svgs: bool = False,
    svg_output_dir: Optional[str] = None,
    simplify: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete experiment: evolve, evaluate, and explain.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: GPConfig
        fitness_fn: Fitness function
        n_explanations: Number of test instances to explain
        verbose: Print progress
        save_svgs: If True, save SVG files to disk
        svg_output_dir: Directory for SVG files (default: current directory)
        simplify: If True, algebraically simplify the best tree post-evolution
                  using SymPy. Folds constants, cancels redundant terms, and
                  produces a smaller equivalent tree. Requires sympy and
                  gp_simplify module.

    Returns:
        Dict with:
            - best_tree: Best evolved GP tree (simplified if simplify=True)
            - original_tree: Original unsimplified tree (only if simplify=True)
            - pset: Primitive set used (may be augmented if simplify=True)
            - train_metrics: Training set metrics
            - test_metrics: Test set metrics
            - explanations: List of explanation dicts
            - svgs: List of SVG strings for each explanation
            - config: The GPConfig used
            - simplified: Whether simplification was applied
            - sympy_expr: SymPy expression of simplified tree (only if simplify=True)
    """
    if verbose:
        print("=" * 60)
        print("GP Evolution Experiment")
        print("=" * 60)
        print(f"Training samples: {len(y_train)}")
        print(f"Test samples: {len(y_test)}")
        print(f"Features: {len(config.feature_names)}")
        print()

    # Evolve
    best_tree, pset = evolve_gp(X_train, y_train, config, fitness_fn, verbose)

    if verbose:
        print()
        print(f"Best tree: {best_tree}")
        print(f"Tree size: {len(best_tree)} nodes")
        print()

    # Post-evolution simplification
    original_tree = None
    sympy_expr = None
    simplified = False

    if simplify:
        try:
            from gp_simplify import simplify_to_primitive_tree
            import re

            # Extract constant map from the tree string using the same
            # approach as explain_prediction: named constants from config
            # plus ephemeral constants parsed from the tree string.
            constant_map = dict(config.named_constants)
            tree_str = str(best_tree)

            # Find all numeric literals in the tree string
            numeric_pattern = (
                r"(?<![a-zA-Z_\d])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?![a-zA-Z_\d])"
            )
            for match in re.finditer(numeric_pattern, tree_str):
                lit = match.group(1)
                try:
                    val = float(lit)
                    # Generate a name consistent with explain_prediction
                    safe_name = f"C{len(constant_map) - len(config.named_constants)}"
                    if safe_name not in constant_map:
                        constant_map[safe_name] = val
                except ValueError:
                    pass

            original_tree = best_tree
            original_size = len(best_tree)

            new_tree, new_pset, sympy_expr, _ = simplify_to_primitive_tree(
                best_tree,
                pset,
                constant_map=constant_map,
            )

            # Verify the simplified tree produces equivalent outputs on
            # a sample of training data before committing to it.
            func_orig = gp.compile(best_tree, pset)
            func_new = gp.compile(new_tree, new_pset)

            n_verify = min(50, len(X_train))
            verify_indices = np.random.choice(len(X_train), n_verify, replace=False)
            max_diff = 0.0
            for idx in verify_indices:
                try:
                    orig_out = float(func_orig(*X_train[idx]))
                    new_out = float(func_new(*X_train[idx]))
                    max_diff = max(max_diff, abs(orig_out - new_out))
                except Exception:
                    max_diff = float("inf")
                    break

            if max_diff < 1e-6:
                best_tree = new_tree
                pset = new_pset
                simplified = True

                if verbose:
                    new_size = len(new_tree)
                    reduction = (1 - new_size / original_size) * 100
                    print(
                        f"Simplified: {original_size} → {new_size} nodes ({reduction:.0f}% reduction)"
                    )
                    print(f"Expression: {sympy_expr}")
                    print()
            else:
                if verbose:
                    print(
                        f"Simplification skipped: verification failed (max diff: {max_diff:.2e})"
                    )
                    print()
                original_tree = None
                sympy_expr = None

        except ImportError as e:
            if verbose:
                print(f"Simplification skipped: {e}")
                print("Install sympy and ensure gp_simplify.py is available.")
                print()
        except Exception as e:
            if verbose:
                print(f"Simplification failed: {e}")
                print()

    # Evaluate
    train_metrics = evaluate_model(best_tree, pset, X_train, y_train, config)
    test_metrics = evaluate_model(best_tree, pset, X_test, y_test, config)

    if verbose:
        print("Training Metrics:")
        for k, v in train_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print()

    # Generate explanations
    explanations = []
    svgs = []

    if n_explanations > 0 and verbose:
        print(f"Generating {n_explanations} explanations...")
        print()

    # Build the constant map to pass to explain_prediction.
    # If we simplified, the pset has _K-prefixed constants that
    # explain_prediction needs to know about.
    explanation_constant_map = None
    if simplified:
        from gp_simplify import _extract_constants_from_pset

        explanation_constant_map = _extract_constants_from_pset(pset)

    for i in range(min(n_explanations, len(X_test))):
        instance = {
            name: float(val) for name, val in zip(config.feature_names, X_test[i])
        }
        true_label = y_test[i]

        try:
            result = explain_prediction(
                best_tree,
                instance,
                config,
                constant_map=explanation_constant_map,
            )
            result["instance_idx"] = i
            result["true_label"] = true_label
            explanations.append(result)

            # Generate SVG
            svg = result["visualization"].to_svg()
            svgs.append(svg)

            # Optionally save to file
            if save_svgs:
                output_dir = svg_output_dir or "."
                svg_path = f"{output_dir}/explanation_{i}.svg"
                with open(svg_path, "w") as f:
                    f.write(svg)
                if verbose:
                    print(f"Saved: {svg_path}")

            if verbose:
                print(f"--- Instance {i} (True: {true_label}) ---")
                print(
                    f"Prediction: {result['prediction_label']} (score: {result['score']:.2f})"
                )

                exp = result["explanation"]
                print("\nKey factors:")
                for f in exp.primary_factors[:3]:
                    val = instance.get(f.name, "?")
                    print(f"  • {f.name}={val} ({f.importance:.0%})")

                print("\nDecision path:")
                print(result["visualization"].to_ascii())
                print()

        except Exception as e:
            explanations.append(
                {
                    "instance_idx": i,
                    "true_label": true_label,
                    "error": str(e),
                }
            )
            svgs.append(None)

            if verbose:
                print(f"--- Instance {i} ---")
                print(f"Error: {e}")
                print()

    result_dict = {
        "best_tree": best_tree,
        "pset": pset,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "explanations": explanations,
        "svgs": svgs,
        "config": config,
        "simplified": simplified,
    }

    if simplified:
        result_dict["original_tree"] = original_tree
        result_dict["sympy_expr"] = sympy_expr

    return result_dict


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    print("GP Evolution Framework - Example")
    print("=" * 60)

    # Create synthetic binary classification data
    np.random.seed(42)
    n_samples = 500

    X = np.random.randn(n_samples, 4)
    # True function: sign(x0 + 2*x1 - x2)
    y = ((X[:, 0] + 2 * X[:, 1] - X[:, 2]) > 0).astype(int)

    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Configure
    config = GPConfig(
        feature_names=["x0", "x1", "x2", "x3"],
        protected_features=[],
        actionable_features=["x0", "x1", "x2", "x3"],
        mu=50,
        lambda_=100,
        generations=20,
    )

    # Run experiment
    results = run_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        config,
        fitness_fn="balanced_accuracy",
        n_explanations=3,
    )
