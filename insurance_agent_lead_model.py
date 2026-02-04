#!/usr/bin/env python3
"""
Insurance agent lead prediction pipeline.

This script performs:
1) Exploratory data analysis (EDA) and summary report generation.
2) Feature preprocessing with numerical/categorical pipelines.
3) Model training for agent click and purchase conversion prediction.
4) Evaluation and artifact saving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


DEFAULT_CLICK_TARGET = "clicked"
DEFAULT_PURCHASE_TARGET = "purchased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insurance agent click + purchase conversion prediction"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV data containing customer and agent features.",
    )
    parser.add_argument(
        "--click-target",
        default=DEFAULT_CLICK_TARGET,
        help=f"Column name for click label (default: {DEFAULT_CLICK_TARGET}).",
    )
    parser.add_argument(
        "--purchase-target",
        default=DEFAULT_PURCHASE_TARGET,
        help=f"Column name for purchase label (default: {DEFAULT_PURCHASE_TARGET}).",
    )
    parser.add_argument(
        "--id-columns",
        nargs="*",
        default=[],
        help="Optional identifier columns to exclude from modeling.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split proportion (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest"],
        default="logistic",
        help="Model type to train (default: logistic).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write EDA reports and metrics (default: reports).",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input data is empty. Please provide a valid dataset.")
    return df


def infer_feature_columns(
    df: pd.DataFrame, target_columns: List[str], id_columns: List[str]
) -> Tuple[List[str], List[str]]:
    feature_df = df.drop(columns=target_columns + id_columns, errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in feature_df.columns if col not in numeric_cols
    ]
    return numeric_cols, categorical_cols


def generate_eda_reports(
    df: pd.DataFrame,
    target_columns: List[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "eda_summary.md"
    missing = df.isna().mean().sort_values(ascending=False)
    numeric_stats = df.select_dtypes(include=[np.number]).describe().T

    target_rates = {
        target: df[target].mean() if target in df.columns else None
        for target in target_columns
    }

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# EDA Summary\n\n")
        f.write(f"Rows: {len(df)}\n\n")
        f.write(f"Columns: {len(df.columns)}\n\n")
        f.write("## Target Rates\n")
        for target, rate in target_rates.items():
            if rate is not None:
                f.write(f"- {target}: {rate:.4f}\n")
            else:
                f.write(f"- {target}: column not found\n")
        f.write("\n## Missingness (Top 20)\n")
        f.write(missing.head(20).to_string())
        f.write("\n\n## Numeric Feature Stats\n")
        f.write(numeric_stats.to_string())

    missing.to_csv(output_dir / "missingness.csv", header=True)
    numeric_stats.to_csv(output_dir / "numeric_stats.csv")


def build_preprocessor(
    numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def build_model(model_type: str, random_state: int):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    return LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    )


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    id_columns: List[str],
    model_type: str,
    test_size: float,
    random_state: int,
    output_dir: Path,
) -> Dict[str, float]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing from the dataset.")

    feature_df = df.drop(columns=[target] + id_columns, errors="ignore")
    labels = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        feature_df,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_model(model_type, random_state)

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(x_train, y_train)

    probas = clf.predict_proba(x_test)[:, 1]
    preds = (probas >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, probas),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"metrics_{target}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    args = parse_args()
    df = load_data(args.data)

    output_dir = Path(args.output_dir)
    target_columns = [args.click_target, args.purchase_target]

    generate_eda_reports(df, target_columns, output_dir)

    numeric_cols, categorical_cols = infer_feature_columns(
        df, target_columns, args.id_columns
    )

    metrics_summary = {}
    for target in target_columns:
        metrics_summary[target] = train_and_evaluate(
            df=df,
            target=target,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            id_columns=args.id_columns,
            model_type=args.model,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=output_dir,
        )

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print("Training complete. Metrics saved to", output_dir)


if __name__ == "__main__":
    main()
