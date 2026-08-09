"""Summarize feature-engineering experiment results."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .selection import (
    CV_STD,
    delta_in_cv_std,
    fold2_confirmation,
    next_step,
    screening_decision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "feature_engineering"

SUMMARY_COLUMNS = [
    "protocol_id",
    "config_id",
    "experiment_id",
    "fold",
    "auc",
    "delta_vs_baseline",
    "delta_in_cv_std",
    "decision",
    "next_step",
    "best_iteration",
    "elapsed_seconds",
    "thread_count",
    "features",
]

FULL_CV_COLUMNS = [
    "protocol_id",
    "config_id",
    "experiment_id",
    "mean_auc",
    "mean_delta",
    "positive_folds",
    "delta_std",
    "thread_count",
]


def load_results(results_dir):
    results = []
    ignored = []
    required = {
        "protocol_id",
        "config_id",
        "experiment_id",
        "fold",
        "auc",
        "best_iteration",
        "thread_count",
    }

    for path in sorted(results_dir.glob("*_fold*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))

        if "protocol_id" not in result or "config_id" not in result:
            ignored.append(path.name)
            continue

        missing = required - result.keys()
        if missing:
            raise ValueError(f"{path.name} is missing: {', '.join(sorted(missing))}")

        result["source_file"] = path.name
        results.append(result)

    if not results:
        raise FileNotFoundError("No current runner results found. Run BASE first.")

    return pd.DataFrame(results), ignored


def add_decisions(summary):
    key_cols = ["config_id", "experiment_id", "thread_count"]

    def row_key(row):
        return tuple(row[col] for col in key_cols)

    fold1 = {
        row_key(row): row["delta_vs_baseline"]
        for _, row in summary.query("experiment_id != 'BASE' and fold == 1").iterrows()
    }
    fold2 = {
        row_key(row): row["delta_vs_baseline"]
        for _, row in summary.query("experiment_id != 'BASE' and fold == 2").iterrows()
    }

    for index, row in summary.iterrows():
        if row["experiment_id"] == "BASE":
            decision, step = "Reference", "-"
        elif row["fold"] == 1:
            decision = screening_decision(row["delta_vs_baseline"])
            step = next_step(
                decision,
                row["delta_vs_baseline"],
                fold2.get(row_key(row)),
            )
        elif row["fold"] == 2:
            fold1_delta = fold1.get(row_key(row))
            if fold1_delta is None:
                decision, step = "Missing Fold 1", "Run Fold 1"
            elif screening_decision(fold1_delta) in {"Promote", "Strong candidate"}:
                decision = fold2_confirmation(
                    fold1_delta,
                    row["delta_vs_baseline"],
                )
                step = next_step(
                    screening_decision(fold1_delta),
                    fold1_delta,
                    row["delta_vs_baseline"],
                )
            else:
                decision, step = "Outside gate", "Hold"
        else:
            decision, step = "5-Fold evidence", "Review all folds"

        summary.loc[index, ["decision", "next_step"]] = [decision, step]


def build_summary(results):
    baseline_keys = ["config_id", "fold", "thread_count"]
    baselines = (
        results.query("experiment_id == 'BASE'")[baseline_keys + ["auc"]]
        .rename(columns={"auc": "baseline_auc"})
    )

    if baselines.empty:
        raise FileNotFoundError("Run BASE before summarizing.")

    results_for_summary = results.drop(
        columns=["baseline_auc", "delta_vs_baseline"],
        errors="ignore",
    )
    summary = results_for_summary.merge(baselines, on=baseline_keys, how="left")
    ignored = summary.loc[summary["baseline_auc"].isna(), "source_file"].tolist()
    summary = summary.dropna(subset=["baseline_auc"]).copy()

    summary["delta_vs_baseline"] = summary["auc"] - summary["baseline_auc"]
    summary["delta_in_cv_std"] = summary["delta_vs_baseline"].map(delta_in_cv_std)
    add_decisions(summary)

    summary["is_feature"] = summary["experiment_id"] != "BASE"
    summary = summary.sort_values(["fold", "is_feature", "experiment_id"])
    return summary.drop(columns="is_feature"), ignored


def build_full_cv_summary(summary):
    key_cols = ["protocol_id", "config_id", "experiment_id", "thread_count"]
    feature_rows = summary.query("experiment_id != 'BASE'")
    complete = feature_rows.groupby(key_cols).filter(
        lambda group: set(group["fold"]) == set(range(1, 6))
    )

    if complete.empty:
        return pd.DataFrame(columns=FULL_CV_COLUMNS)

    return (
        complete.groupby(key_cols)
        .agg(
            mean_auc=("auc", "mean"),
            mean_delta=("delta_vs_baseline", "mean"),
            positive_folds=("delta_vs_baseline", lambda values: (values > 0).sum()),
            delta_std=("delta_vs_baseline", "std"),
        )
        .reset_index()[FULL_CV_COLUMNS]
    )


def print_summary(summary, full_cv):
    display = summary.copy()
    display["auc"] = display["auc"].map(lambda value: f"{value:.6f}")
    display["delta_vs_baseline"] = display["delta_vs_baseline"].map(
        lambda value: f"{value:+.6f}"
    )
    display["delta_in_cv_std"] = display["delta_in_cv_std"].map(
        lambda value: f"{value:.2f} sigma"
    )

    print(f"Screening scale: sigma = {CV_STD:.6f}")
    print("Reject < 0.25 sigma | Maybe < 0.75 | Promote < 1.00 | Strong >= 1.00\n")
    print(
        display[
            [
                "experiment_id",
                "fold",
                "auc",
                "delta_vs_baseline",
                "delta_in_cv_std",
                "decision",
                "next_step",
                "best_iteration",
            ]
        ].to_string(index=False)
    )

    if not full_cv.empty:
        print("\nCompleted 5-Fold evidence:")
        print(full_cv.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def save_csv(summary, full_cv, output_path, full_cv_output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = summary[SUMMARY_COLUMNS].copy()
    output["features"] = output["features"].map(lambda names: " + ".join(names))
    output.to_csv(output_path, index=False, float_format="%.6f")

    if not full_cv.empty:
        full_cv_output_path.parent.mkdir(parents=True, exist_ok=True)
        full_cv.to_csv(full_cv_output_path, index=False, float_format="%.6f")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-cv-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or args.results_dir / "summary.csv"
    full_cv_output_path = (
        args.full_cv_output or args.results_dir / "full_cv_summary.csv"
    )

    results, legacy_files = load_results(args.results_dir)
    summary, unmatched_files = build_summary(results)
    full_cv = build_full_cv_summary(summary)

    print_summary(summary, full_cv)
    save_csv(summary, full_cv, output_path, full_cv_output_path)

    ignored = sorted(set(legacy_files + unmatched_files))
    if ignored:
        print("\nIgnored non-comparable results:")
        print(*[f"- {name}" for name in ignored], sep="\n")

    print(f"\nSaved summary: {output_path}")
    if not full_cv.empty:
        print(f"Saved full CV summary: {full_cv_output_path}")


if __name__ == "__main__":
    main()
