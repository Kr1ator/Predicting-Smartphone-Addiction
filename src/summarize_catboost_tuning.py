"""Summarize the CatBoost reference check and completed Optuna trials."""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "catboost_tuning"
CHECK_PATH = RESULTS_DIR / "reference_check.json"
TRIALS_DIR = RESULTS_DIR / "trials"
VALIDATION_DIR = RESULTS_DIR / "validation"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(result, source):
    folds = {fold["fold"]: fold for fold in result["folds"]}
    params = result["params"]
    iterations = [fold["best_iteration"] for fold in result["folds"]]
    elapsed = [fold["elapsed_seconds"] for fold in result["folds"]]

    row = {
        "trial_number": result.get("trial_number", -1),
        "source": source,
        "mean_auc": result["mean_auc"],
        "mean_delta": result["mean_delta"],
        "positive_folds": result["positive_folds"],
        "mean_best_iteration": sum(iterations) / len(iterations),
        "mean_elapsed_seconds": sum(elapsed) / len(elapsed),
        **params,
    }

    for fold_number, fold in folds.items():
        row[f"fold{fold_number}_auc"] = fold["auc"]
        row[f"fold{fold_number}_delta"] = fold["delta_vs_reference"]

    if set(range(3, 6)).issubset(folds):
        validation = [folds[fold] for fold in range(3, 6)]
        row["validation_mean_delta"] = sum(
            fold["delta_vs_reference"] for fold in validation
        ) / len(validation)
        row["validation_positive_folds"] = sum(
            fold["delta_vs_reference"] > 0 for fold in validation
        )
        row["passes_validation"] = (
            row["validation_mean_delta"] > 0
            and row["validation_positive_folds"] >= 2
            and row["mean_delta"] > 0
            and row["positive_folds"] >= 4
        )

    return row


def load_results():
    if not CHECK_PATH.exists():
        raise FileNotFoundError(f"Missing reference check: {CHECK_PATH}")

    reference = load_json(CHECK_PATH)
    if not reference["passed"]:
        raise ValueError("The saved reference check did not pass.")

    trial_paths = sorted(TRIALS_DIR.glob("trial_*.json"))
    if not trial_paths:
        raise FileNotFoundError(f"No completed Optuna trials in: {TRIALS_DIR}")

    rows = [build_row(reference, "reference")]
    rows.extend(build_row(load_json(path), "optuna") for path in trial_paths)
    return pd.DataFrame(rows)


def save_best_trial(summary, file_name):
    best = summary.iloc[0]
    param_names = [
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
        "subsample",
    ]
    output = {
        "trial_number": int(best["trial_number"]),
        "source": best["source"],
        "mean_auc": float(best["mean_auc"]),
        "mean_delta": float(best["mean_delta"]),
        "params": {name: float(best[name]) for name in param_names},
    }

    if "validation_mean_delta" in best:
        output["validation_mean_delta"] = float(best["validation_mean_delta"])
        output["validation_positive_folds"] = int(
            best["validation_positive_folds"]
        )

    output["params"]["depth"] = int(output["params"]["depth"])

    path = RESULTS_DIR / file_name
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved best parameters: {path}")


def load_validation_results():
    rows = []
    incomplete = []

    for path in sorted(VALIDATION_DIR.glob("trial_*.json")):
        result = load_json(path)
        fold_numbers = sorted(fold["fold"] for fold in result["folds"])

        if fold_numbers != list(range(1, 6)):
            incomplete.append(result["trial_number"])
            continue

        rows.append(build_row(result, "validation"))

    return pd.DataFrame(rows), incomplete


def main():
    summary = load_results()
    summary = summary.sort_values("mean_delta", ascending=False)

    display_columns = [
        "trial_number",
        "source",
        "mean_auc",
        "mean_delta",
        "positive_folds",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
        "subsample",
    ]
    print(summary[display_columns].head(10).to_string(index=False))

    summary_path = RESULTS_DIR / "search_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Saved search summary: {summary_path}")
    save_best_trial(summary, "best_search_params.json")

    validation, incomplete = load_validation_results()
    if validation.empty:
        return

    validation = validation.sort_values("validation_mean_delta", ascending=False)
    validation_path = RESULTS_DIR / "full_cv_summary.csv"
    validation.to_csv(validation_path, index=False, float_format="%.6f")
    print(f"Saved full CV summary: {validation_path}")

    display_columns = [
        "trial_number",
        "validation_mean_delta",
        "validation_positive_folds",
        "mean_delta",
        "positive_folds",
        "passes_validation",
    ]
    print(validation[display_columns].to_string(index=False))

    passed = validation[validation["passes_validation"]]
    if not passed.empty:
        save_best_trial(passed, "best_validation_params.json")
    else:
        print("No candidate passed the Stage C rules.")

    if incomplete:
        print(f"Ignored incomplete validation trials: {incomplete}")


if __name__ == "__main__":
    main()
