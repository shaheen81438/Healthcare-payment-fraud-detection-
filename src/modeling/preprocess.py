"""Shared preprocessing: one fitted transformer serves LR, XGBoost, and SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    numeric_features: list[str], categorical_features: list[str], scale_numeric: bool
) -> ColumnTransformer:
    """Impute+optionally-scale numerics; one-hot encode low-cardinality categoricals."""
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform X and return a DataFrame with post-encoding feature names."""
    matrix = preprocessor.transform(X)
    names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(matrix, columns=names, index=X.index)
