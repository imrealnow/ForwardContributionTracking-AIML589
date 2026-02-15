"""
German Credit Data Preprocessing Module

This module provides transparent, documented preprocessing of the UCI German Credit
dataset for use with explainable GP models. Unlike the pre-processed numeric version,
this gives full control over:

1. Feature encoding (ordinal vs one-hot vs dropped)
2. Feature scaling (or lack thereof - GP can handle raw scales with appropriate constants)
3. Protected attribute handling (drop, encode, or factor out)
4. Feature naming alignment with actual semantics

The original german.data file uses symbolic codes (A11, A12, etc.) which we map
to meaningful numeric values with clear documentation.

Usage:
    X, y, feature_info = load_german_credit_original('german.data')

    # feature_info contains:
    # - names: list of feature names
    # - descriptions: dict of feature descriptions
    # - protected: list of protected feature indices
    # - actionable: list of actionable feature indices
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field


@dataclass
class FeatureInfo:
    """Metadata about preprocessed features."""

    names: List[str]
    descriptions: Dict[str, str]
    protected_indices: List[int]
    actionable_indices: List[int]
    categorical_mappings: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def get_protected_names(self) -> List[str]:
        return [self.names[i] for i in self.protected_indices]

    def get_actionable_names(self) -> List[str]:
        return [self.names[i] for i in self.actionable_indices]


# =============================================================================
# ATTRIBUTE ENCODINGS
# =============================================================================

# Attribute 1: Checking account status (ordinal - higher = better)
CHECKING_STATUS = {
    "A11": 0,  # < 0 DM (overdrawn)
    "A12": 1,  # 0 <= x < 200 DM
    "A13": 2,  # >= 200 DM or salary assignment
    "A14": 3,  # No checking account (could be good or bad - treated as neutral-high)
}

# Attribute 3: Credit history (ordinal - higher = better history)
CREDIT_HISTORY = {
    "A34": 0,  # Critical account / other credits existing elsewhere
    "A33": 1,  # Delay in paying off in the past
    "A32": 2,  # Existing credits paid back duly till now
    "A31": 3,  # All credits at this bank paid back duly
    "A30": 4,  # No credits taken / all credits paid back duly
}

# Attribute 4: Purpose (nominal - could group into categories)
# Option 1: Drop (too many categories, hard to encode meaningfully)
# Option 2: Group into risk tiers based on typical default rates
PURPOSE = {
    "A40": "car_new",
    "A41": "car_used",
    "A42": "furniture",
    "A43": "radio_tv",
    "A44": "appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "vacation",
    "A48": "retraining",
    "A49": "business",
    "A410": "other",
}

# Purpose grouped by typical risk (subjective categorization)
PURPOSE_RISK = {
    "A40": 1,  # car_new - moderate (depreciating asset)
    "A41": 1,  # car_used - moderate
    "A42": 1,  # furniture - moderate
    "A43": 2,  # radio_tv - higher (non-essential)
    "A44": 1,  # appliances - moderate
    "A45": 0,  # repairs - lower (maintaining assets)
    "A46": 0,  # education - lower (investment)
    "A47": 2,  # vacation - higher (consumption)
    "A48": 0,  # retraining - lower (investment)
    "A49": 1,  # business - moderate (variable)
    "A410": 1,  # other - moderate (unknown)
}

# Attribute 6: Savings account (ordinal - higher = more savings)
SAVINGS_STATUS = {
    "A65": 0,  # Unknown / no savings
    "A61": 1,  # < 100 DM
    "A62": 2,  # 100-500 DM
    "A63": 3,  # 500-1000 DM
    "A64": 4,  # >= 1000 DM
}

# Attribute 7: Employment duration (ordinal - higher = longer employment)
EMPLOYMENT_DURATION = {
    "A71": 0,  # Unemployed
    "A72": 1,  # < 1 year
    "A73": 2,  # 1-4 years
    "A74": 3,  # 4-7 years
    "A75": 4,  # >= 7 years
}

# Attribute 9: Personal status and sex
# PROBLEMATIC: Encodes both gender and marital status together
# Options:
#   1. Drop entirely (lose marital status info)
#   2. Encode marital status only (factor out gender)
#   3. Keep as-is (perpetuates gender bias)

PERSONAL_STATUS_SEX = {
    "A91": ("male", "divorced/separated"),
    "A92": ("female", "divorced/separated/married"),
    "A93": ("male", "single"),
    "A94": ("male", "married/widowed"),
    "A95": ("female", "single"),
}

# Marital status only (factoring out gender)
MARITAL_STATUS = {
    "A91": 1,  # Divorced/separated
    "A92": 1,  # Divorced/separated/married -> treat as "has been married"
    "A93": 0,  # Single
    "A94": 2,  # Married/widowed
    "A95": 0,  # Single
}

# Gender only (for analysis, NOT for model input)
GENDER = {
    "A91": "male",
    "A92": "female",
    "A93": "male",
    "A94": "male",
    "A95": "female",
}

# Attribute 10: Other debtors/guarantors (ordinal - higher = more security)
OTHER_DEBTORS = {
    "A101": 0,  # None
    "A102": 1,  # Co-applicant
    "A103": 2,  # Guarantor
}

# Attribute 12: Property (ordinal - higher = more valuable assets)
PROPERTY = {
    "A124": 0,  # Unknown / no property
    "A123": 1,  # Car or other
    "A122": 2,  # Building society savings / life insurance
    "A121": 3,  # Real estate
}

# Attribute 14: Other installment plans (ordinal - none is best)
OTHER_INSTALLMENTS = {
    "A141": 2,  # Bank (more concerning)
    "A142": 1,  # Stores (less concerning)
    "A143": 0,  # None (best)
}

# Attribute 15: Housing (nominal, but can order by stability)
HOUSING = {
    "A151": 0,  # Rent
    "A153": 1,  # For free (somewhat stable)
    "A152": 2,  # Own (most stable)
}

# Attribute 17: Job qualification (ordinal - higher = more qualified)
JOB = {
    "A171": 0,  # Unemployed/unskilled non-resident
    "A172": 1,  # Unskilled resident
    "A173": 2,  # Skilled employee
    "A174": 3,  # Management/self-employed/highly qualified
}

# Attribute 19: Telephone (binary)
TELEPHONE = {
    "A191": 0,  # No
    "A192": 1,  # Yes
}

# Attribute 20: Foreign worker (binary)
# PROBLEMATIC: Could perpetuate discrimination
FOREIGN_WORKER = {
    "A201": 1,  # Yes (foreign)
    "A202": 0,  # No (domestic)
}


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def parse_original_row(row: str) -> Tuple[List, int]:
    """Parse a single row from german.data (original format)."""
    parts = row.strip().split()

    # Last element is the class label (1=good, 2=bad)
    label = int(parts[-1])
    features = parts[:-1]

    return features, label


def load_german_credit_original(
    filepath: str,
    drop_gender: bool = True,
    drop_foreign_worker: bool = True,
    drop_purpose: bool = False,
    scale_amount: bool = False,
    scale_duration: bool = False,
    scale_age: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, FeatureInfo]:
    """
    Load and preprocess the original German Credit data with full transparency.

    Args:
        filepath: Path to german.data file
        drop_gender: If True, encode only marital status (factor out gender)
        drop_foreign_worker: If True, exclude foreign worker feature
        drop_purpose: If True, exclude purpose feature entirely
        scale_amount: If True, scale credit amount by 1000
        scale_duration: If True, scale duration by 12 (to years)
        scale_age: If True, scale age by 10
        verbose: Print preprocessing summary

    Returns:
        X: Feature matrix
        y: Labels (1=good, 0=bad)
        feature_info: FeatureInfo with names, descriptions, protected/actionable indices
    """

    # Read raw data
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Parse all rows
    raw_data = []
    labels = []
    for line in lines:
        if line.strip():
            features, label = parse_original_row(line)
            raw_data.append(features)
            labels.append(label)

    y = np.array(labels)
    # Convert: 1=good stays 1, 2=bad becomes 0
    y = (y == 1).astype(int)

    # Build feature matrix with documented transformations
    feature_columns = []
    feature_names = []
    feature_descriptions = {}
    categorical_mappings = {}
    protected_indices = []
    actionable_indices = []

    def add_feature(
        name: str,
        values: List[float],
        description: str,
        protected: bool = False,
        actionable: bool = False,
        mapping: Optional[Dict] = None,
    ):
        """Helper to add a feature with metadata."""
        idx = len(feature_names)
        feature_columns.append(values)
        feature_names.append(name)
        feature_descriptions[name] = description
        if protected:
            protected_indices.append(idx)
        if actionable:
            actionable_indices.append(idx)
        if mapping:
            categorical_mappings[name] = mapping

    # Process each attribute
    n_samples = len(raw_data)

    # Attribute 1: Checking status (ordinal 0-3)
    values = [CHECKING_STATUS[row[0]] for row in raw_data]
    add_feature(
        "checking_status",
        values,
        "Status of checking account: 0=<0DM, 1=0-200DM, 2=>=200DM, 3=none",
        actionable=True,
        mapping=CHECKING_STATUS,
    )

    # Attribute 2: Duration in months (numeric)
    values = [int(row[1]) for row in raw_data]
    if scale_duration:
        values = [v / 12.0 for v in values]
        add_feature("duration_years", values, "Loan duration in years", actionable=True)
    else:
        add_feature(
            "duration_months", values, "Loan duration in months", actionable=True
        )

    # Attribute 3: Credit history (ordinal 0-4)
    values = [CREDIT_HISTORY[row[2]] for row in raw_data]
    add_feature(
        "credit_history",
        values,
        "Credit history: 0=critical, 1=delays, 2=current ok, 3=all paid, 4=no credits",
        mapping=CREDIT_HISTORY,
    )

    # Attribute 4: Purpose
    if not drop_purpose:
        # Use risk-based grouping (ordinal 0-2)
        values = [PURPOSE_RISK[row[3]] for row in raw_data]
        add_feature(
            "purpose_risk",
            values,
            "Purpose risk category: 0=investment, 1=moderate, 2=consumption",
            mapping=PURPOSE_RISK,
        )

    # Attribute 5: Credit amount (numeric)
    values = [int(row[4]) for row in raw_data]
    if scale_amount:
        values = [v / 1000.0 for v in values]
        add_feature(
            "credit_amount_k", values, "Credit amount in thousands DM", actionable=True
        )
    else:
        add_feature("credit_amount", values, "Credit amount in DM", actionable=True)

    # Attribute 6: Savings status (ordinal 0-4)
    values = [SAVINGS_STATUS[row[5]] for row in raw_data]
    add_feature(
        "savings_status",
        values,
        "Savings: 0=unknown, 1=<100DM, 2=100-500DM, 3=500-1000DM, 4=>=1000DM",
        actionable=True,
        mapping=SAVINGS_STATUS,
    )

    # Attribute 7: Employment duration (ordinal 0-4)
    values = [EMPLOYMENT_DURATION[row[6]] for row in raw_data]
    add_feature(
        "employment_years",
        values,
        "Employment duration: 0=unemployed, 1=<1yr, 2=1-4yr, 3=4-7yr, 4=>=7yr",
        mapping=EMPLOYMENT_DURATION,
    )

    # Attribute 8: Installment rate (numeric 1-4)
    values = [int(row[7]) for row in raw_data]
    add_feature(
        "installment_rate",
        values,
        "Installment rate as % of disposable income (1-4)",
        actionable=True,
    )

    # Attribute 9: Personal status/sex
    if drop_gender:
        # Factor out gender, keep only marital status
        values = [MARITAL_STATUS[row[8]] for row in raw_data]
        add_feature(
            "marital_status",
            values,
            "Marital status: 0=single, 1=divorced/separated, 2=married/widowed",
            mapping=MARITAL_STATUS,
        )
    else:
        # Keep combined (problematic for fairness)
        mapping = {"A91": 0, "A92": 1, "A93": 2, "A94": 3, "A95": 4}
        values = [mapping[row[8]] for row in raw_data]
        add_feature(
            "personal_status",
            values,
            "Personal status and sex (combined - potentially discriminatory)",
            protected=True,
            mapping=mapping,
        )

    # Attribute 10: Other debtors (ordinal 0-2)
    values = [OTHER_DEBTORS[row[9]] for row in raw_data]
    add_feature(
        "other_debtors",
        values,
        "Other debtors/guarantors: 0=none, 1=co-applicant, 2=guarantor",
        mapping=OTHER_DEBTORS,
    )

    # Attribute 11: Residence duration (numeric 1-4)
    values = [int(row[10]) for row in raw_data]
    add_feature("residence_years", values, "Years at current residence (1-4)")

    # Attribute 12: Property (ordinal 0-3)
    values = [PROPERTY[row[11]] for row in raw_data]
    add_feature(
        "property",
        values,
        "Property: 0=none, 1=car/other, 2=savings/insurance, 3=real estate",
        mapping=PROPERTY,
    )

    # Attribute 13: Age (numeric)
    values = [int(row[12]) for row in raw_data]
    if scale_age:
        values = [v / 10.0 for v in values]
        add_feature("age_decades", values, "Age in decades", protected=True)
    else:
        add_feature("age", values, "Age in years", protected=True)

    # Attribute 14: Other installment plans (ordinal 0-2)
    values = [OTHER_INSTALLMENTS[row[13]] for row in raw_data]
    add_feature(
        "other_installments",
        values,
        "Other installment plans: 0=none, 1=stores, 2=bank",
        mapping=OTHER_INSTALLMENTS,
    )

    # Attribute 15: Housing (ordinal 0-2)
    values = [HOUSING[row[14]] for row in raw_data]
    add_feature("housing", values, "Housing: 0=rent, 1=free, 2=own", mapping=HOUSING)

    # Attribute 16: Number of existing credits (numeric)
    values = [int(row[15]) for row in raw_data]
    add_feature(
        "num_credits",
        values,
        "Number of existing credits at this bank",
        actionable=True,
    )

    # Attribute 17: Job (ordinal 0-3)
    values = [JOB[row[16]] for row in raw_data]
    add_feature(
        "job",
        values,
        "Job: 0=unemployed unskilled, 1=unskilled resident, 2=skilled, 3=management/highly qualified",
        mapping=JOB,
    )

    # Attribute 18: Number of dependents (numeric)
    values = [int(row[17]) for row in raw_data]
    add_feature(
        "num_dependents", values, "Number of people liable to provide maintenance for"
    )

    # Attribute 19: Telephone (binary)
    values = [TELEPHONE[row[18]] for row in raw_data]
    add_feature("telephone", values, "Has telephone: 0=no, 1=yes", mapping=TELEPHONE)

    # Attribute 20: Foreign worker
    if not drop_foreign_worker:
        values = [FOREIGN_WORKER[row[19]] for row in raw_data]
        add_feature(
            "foreign_worker",
            values,
            "Foreign worker: 0=no, 1=yes (potentially discriminatory)",
            protected=True,
            mapping=FOREIGN_WORKER,
        )

    # Build feature matrix
    X = np.column_stack(feature_columns)

    # Create feature info
    feature_info = FeatureInfo(
        names=feature_names,
        descriptions=feature_descriptions,
        protected_indices=protected_indices,
        actionable_indices=actionable_indices,
        categorical_mappings=categorical_mappings,
    )

    if verbose:
        print(f"Loaded {n_samples} samples with {len(feature_names)} features")
        print(
            f"Class distribution: {sum(y)} good ({sum(y)/len(y):.1%}), {len(y)-sum(y)} bad"
        )
        print(f"\nFeatures ({len(feature_names)}):")
        for i, name in enumerate(feature_names):
            markers = []
            if i in protected_indices:
                markers.append("PROTECTED")
            if i in actionable_indices:
                markers.append("actionable")
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            print(f"  {i:2d}. {name}{marker_str}")
            print(f"      {feature_descriptions[name]}")

        if drop_gender:
            print("\n⚠ Gender factored out (marital status retained)")
        if drop_foreign_worker:
            print("⚠ Foreign worker feature dropped")
        if drop_purpose:
            print("⚠ Purpose feature dropped")

    return X, y, feature_info


def get_feature_ranges(
    X: np.ndarray, feature_info: FeatureInfo
) -> Dict[str, Tuple[float, float]]:
    """Get min/max ranges for each feature (useful for counterfactual constraints)."""
    ranges = {}
    for i, name in enumerate(feature_info.names):
        ranges[name] = (X[:, i].min(), X[:, i].max())
    return ranges
