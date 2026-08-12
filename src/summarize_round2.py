"""Summarize complete 5-Fold Round 2 experiment results."""

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUND2_RESULTS_DIR = PROJECT_ROOT / "results" / "round2"
RESULTS_DIR = ROUND2_RESULTS_DIR / "first_stage"
SHARED_PROTOCOL_PATH = RESULTS_DIR / "protocol.json"

SUMMARY_COLUMNS = [
    "protocol_id",
    "config_id",
    "experiment_id",
    "mean_auc",
    "mean_delta",
    "positive_folds",
    "delta_std",
    "mean_abs_delta",
    "max_abs_delta",
    "min_delta",
    "max_delta",
    "mean_best_iteration",
    "thread_count",
    "feature_ids",
    "features",
]


def load_protocol(protocol_path):
    if not protocol_path.exists():
        raise FileNotFoundError(f"Missing Round 2 protocol: {protocol_path}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = {"protocol_id", "config_id", "thread_count"}
    missing = required - protocol.keys()

    if missing:
        raise ValueError(
            f"protocol.json is missing: {', '.join(sorted(missing))}"
        )

    return protocol


def load_results(results_dir):
    results = []
    required = {
        "experiment_id",
        "feature_ids",
        "features",
        "fold",
        "reference_auc",
        "auc",
        "delta_vs_reference",
        "best_iteration",
    }

    for path in sorted(results_dir.glob("*_fold*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        missing = required - result.keys()

        if missing:
            raise ValueError(f"{path.name} is missing: {', '.join(sorted(missing))}")

        comparison_fields = {
            "comparison_experiment",
            "comparison_auc",
            "delta_vs_comparison",
        }
        present_comparison_fields = comparison_fields & result.keys()

        if present_comparison_fields and present_comparison_fields != comparison_fields:
            missing_comparison = comparison_fields - result.keys()
            raise ValueError(
                f"{path.name} is missing: {', '.join(sorted(missing_comparison))}"
            )

        result["source_file"] = path.name
        results.append(result)

    if not results:
        raise FileNotFoundError("No Round 2 results found.")

    return pd.DataFrame(results)


def build_summary(results, protocol):
    results = results.copy()
    results["delta_vs_reference"] = results["auc"] - results["reference_auc"]
    has_comparison = "comparison_experiment" in results.columns

    if has_comparison:
        comparison_columns = [
            "comparison_experiment",
            "comparison_auc",
            "delta_vs_comparison",
        ]
        if results[comparison_columns].isna().any().any():
            raise ValueError("Comparison fields must be present in every result.")

        results["delta_vs_comparison"] = (
            results["auc"] - results["comparison_auc"]
        )

    rows = []
    incomplete = []

    for experiment_id, group in results.groupby("experiment_id"):
        folds = sorted(group["fold"].tolist())

        if folds != list(range(1, 6)):
            incomplete.append(experiment_id)
            continue

        feature_ids = group["feature_ids"].map(tuple)
        features = group["features"].map(tuple)

        if feature_ids.nunique() != 1 or features.nunique() != 1:
            raise ValueError(
                f"{experiment_id} has inconsistent feature definitions across folds."
            )

        deltas = group["delta_vs_reference"]

        row = {
            "protocol_id": protocol["protocol_id"],
            "config_id": protocol["config_id"],
            "experiment_id": experiment_id,
            "mean_auc": group["auc"].mean(),
            "mean_delta": deltas.mean(),
            "positive_folds": int((deltas > 0).sum()),
            "delta_std": deltas.std(),
            "mean_abs_delta": deltas.abs().mean(),
            "max_abs_delta": deltas.abs().max(),
            "min_delta": deltas.min(),
            "max_delta": deltas.max(),
            "mean_best_iteration": group["best_iteration"].mean(),
            "thread_count": protocol["thread_count"],
            "feature_ids": list(feature_ids.iloc[0]),
            "features": list(features.iloc[0]),
        }

        if has_comparison:
            comparison_experiments = group["comparison_experiment"].unique()
            if len(comparison_experiments) != 1:
                raise ValueError(
                    f"{experiment_id} has inconsistent comparison experiments."
                )

            comparison_deltas = group["delta_vs_comparison"]
            row["comparison_experiment"] = comparison_experiments[0]
            row["mean_delta_vs_comparison"] = comparison_deltas.mean()
            row["better_than_comparison_folds"] = int(
                (comparison_deltas > 0).sum()
            )
            row["comparison_delta_std"] = comparison_deltas.std()

        rows.append(row)

    if not rows:
        raise FileNotFoundError("No complete 5-Fold Round 2 results found.")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("mean_delta", ascending=False)
    columns = list(SUMMARY_COLUMNS)

    if has_comparison:
        columns.extend(
            [
                "comparison_experiment",
                "mean_delta_vs_comparison",
                "better_than_comparison_folds",
                "comparison_delta_std",
            ]
        )

    return summary[columns], incomplete


def print_summary(summary):
    display = summary.copy()
    display["mean_auc"] = display["mean_auc"].map(lambda value: f"{value:.6f}")
    display["mean_delta"] = display["mean_delta"].map(
        lambda value: f"{value:+.6f}"
    )
    display["delta_std"] = display["delta_std"].map(lambda value: f"{value:.6f}")

    columns = [
        "experiment_id",
        "mean_auc",
        "mean_delta",
        "positive_folds",
        "delta_std",
        "mean_best_iteration",
    ]

    if "mean_delta_vs_comparison" in display.columns:
        display["mean_delta_vs_comparison"] = display[
            "mean_delta_vs_comparison"
        ].map(lambda value: f"{value:+.6f}")
        display["comparison_delta_std"] = display["comparison_delta_std"].map(
            lambda value: f"{value:.6f}"
        )
        columns.extend(
            [
                "comparison_experiment",
                "mean_delta_vs_comparison",
                "better_than_comparison_folds",
                "comparison_delta_std",
            ]
        )

    print(display[columns].to_string(index=False))


def save_csv(summary, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = summary.copy()
    output["feature_ids"] = output["feature_ids"].map(lambda names: " + ".join(names))
    output["features"] = output["features"].map(lambda names: " + ".join(names))
    output.to_csv(output_path, index=False, float_format="%.6f")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or args.results_dir / "summary.csv"
    protocol_path = args.protocol or args.results_dir / "protocol.json"

    if not protocol_path.exists() and args.protocol is None:
        protocol_path = SHARED_PROTOCOL_PATH

    protocol = load_protocol(protocol_path)
    results = load_results(args.results_dir)
    summary, incomplete = build_summary(results, protocol)

    print_summary(summary)
    save_csv(summary, output_path)

    if incomplete:
        print("\nIgnored incomplete experiments:")
        print(*[f"- {experiment_id}" for experiment_id in incomplete], sep="\n")

    print(f"\nSaved summary: {output_path}")


if __name__ == "__main__":
    main()
