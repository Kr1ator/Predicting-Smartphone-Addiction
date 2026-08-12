"""Run Round 2 feature experiments.

Reference: FE10, matched by fold
CV/model parameters: shared with src.experiment
Thread count: 10
Early stopping: 200 rounds
Comparison: paired AUC delta against FE10
R_GROUP LOO: also compared with the complete R_GROUP on the same fold
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .experiment import (
    CAT_COLS,
    CONFIG_ID as REFERENCE_CONFIG_ID,
    CV_PARAMS,
    MODEL_PARAMS,
    PROJECT_ROOT,
    PROTOCOL_ID as REFERENCE_PROTOCOL_ID,
    THREAD_COUNT,
)
from .round2_features import (
    ROUND2_FEATURE_SETS,
    R_GROUP_LOO_EXPERIMENTS,
    get_round2_cat_cols,
    make_round2_features,
)

DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_RESULTS_DIR = PROJECT_ROOT / "results" / "feature_engineering"
ROUND2_RESULTS_DIR = PROJECT_ROOT / "results" / "round2"
FIRST_STAGE_RESULTS_DIR = ROUND2_RESULTS_DIR / "first_stage"
R_GROUP_LOO_RESULTS_DIR = ROUND2_RESULTS_DIR / "r_group_loo"

REFERENCE_EXPERIMENT = "FE10"
EARLY_STOPPING_ROUNDS = 200


def load_reference_result(reference_results_dir, fold):
    reference_path = reference_results_dir / f"fe10_fold{fold}.json"

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing FE10 reference for Fold {fold}: {reference_path}"
        )

    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    if reference["experiment_id"] != REFERENCE_EXPERIMENT:
        raise ValueError("The reference experiment ID is not FE10.")
    if reference["protocol_id"] != REFERENCE_PROTOCOL_ID:
        raise ValueError("The FE10 reference protocol does not match.")
    if reference.get("config_id") != REFERENCE_CONFIG_ID:
        raise ValueError("The FE10 reference configuration does not match.")
    if reference["fold"] != fold:
        raise ValueError("The FE10 reference fold does not match.")
    if reference["thread_count"] != THREAD_COUNT:
        raise ValueError("The FE10 reference thread count does not match.")

    return reference


def load_r_group_result(fold, reference_auc):
    result_path = FIRST_STAGE_RESULTS_DIR / f"r_group_fold{fold}.json"

    if not result_path.exists():
        raise FileNotFoundError(
            f"Missing R_GROUP result for Fold {fold}: {result_path}"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))

    if result["experiment_id"] != "R_GROUP":
        raise ValueError("The comparison experiment ID is not R_GROUP.")
    if result["feature_ids"] != ["FE20", "FE22", "FE25"]:
        raise ValueError("The R_GROUP feature definition does not match.")
    if result["fold"] != fold:
        raise ValueError("The R_GROUP fold does not match.")
    if float(result["reference_auc"]) != reference_auc:
        raise ValueError(
            "R_GROUP and the LOO experiment do not share the same FE10 reference."
        )

    return result


def get_results_dir(experiment_id):
    if experiment_id in R_GROUP_LOO_EXPERIMENTS:
        return R_GROUP_LOO_RESULTS_DIR
    return FIRST_STAGE_RESULTS_DIR


def format_elapsed_time(seconds):
    total_seconds = round(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m {seconds}s"


def run_experiment(
    experiment_id,
    fold=1,
    data_dir=DATA_DIR,
    reference_results_dir=REFERENCE_RESULTS_DIR,
    results_dir=None,
):
    experiment_id = experiment_id.upper()

    if experiment_id not in ROUND2_FEATURE_SETS:
        raise ValueError(f"Unknown Round 2 experiment: {experiment_id}")
    if fold not in range(1, 6):
        raise ValueError("fold must be between 1 and 5")

    reference = load_reference_result(
        reference_results_dir,
        fold,
    )
    reference_auc = float(reference["auc"])

    comparison = None
    if experiment_id in R_GROUP_LOO_EXPERIMENTS:
        comparison = load_r_group_result(fold, reference_auc)

    if results_dir is None:
        results_dir = get_results_dir(experiment_id)

    # Load and prepare the training data.
    train_path = data_dir / "train.csv"
    train = pd.read_csv(train_path)

    X = train.drop(columns=["addicted_label", "id"]).copy()
    y = train["addicted_label"].copy()

    original_columns = list(X.columns)
    X = make_round2_features(X, experiment_id)
    added_columns = [col for col in X.columns if col not in original_columns]

    cat_cols = list(CAT_COLS) + get_round2_cat_cols(experiment_id)
    for col in cat_cols:
        X[col] = X[col].fillna("Missing")

    # Select the requested fold.
    splitter = StratifiedKFold(
        n_splits=CV_PARAMS["n_splits"],
        shuffle=CV_PARAMS["shuffle"],
        random_state=CV_PARAMS["random_state"],
    )

    for current_fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        if current_fold == fold:
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]
            break

    # Train the model.
    model = CatBoostClassifier(
        **MODEL_PARAMS,
        thread_count=THREAD_COUNT,
    )

    started_at = perf_counter()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    elapsed_seconds = perf_counter() - started_at

    # Evaluate the model.
    valid_pred = model.predict_proba(X_valid)[:, 1]
    valid_auc = float(roc_auc_score(y_valid, valid_pred))
    best_iteration = model.get_best_iteration() + 1

    # Save only the information that changes by experiment or fold.
    result = {
        "experiment_id": experiment_id,
        "feature_ids": list(ROUND2_FEATURE_SETS[experiment_id]),
        "features": added_columns,
        "fold": fold,
        "reference_auc": reference_auc,
        "auc": valid_auc,
        "delta_vs_reference": valid_auc - reference_auc,
        "best_iteration": best_iteration,
        "elapsed": format_elapsed_time(elapsed_seconds),
    }

    if comparison is not None:
        comparison_auc = float(comparison["auc"])
        result["comparison_experiment"] = "R_GROUP"
        result["comparison_auc"] = comparison_auc
        result["delta_vs_comparison"] = valid_auc - comparison_auc

    # Save the result as JSON.
    results_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{experiment_id.lower()}_fold{fold}.json"
    result_path = results_dir / file_name

    result_text = json.dumps(result, indent=2, ensure_ascii=False)
    result_path.write_text(result_text + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved result:", result_path)

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_ids",
        nargs="+",
        type=str.upper,
        choices=ROUND2_FEATURE_SETS,
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 6),
        help="Run one fold; omit to run all five folds.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--reference-results-dir",
        type=Path,
        default=REFERENCE_RESULTS_DIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Optional output override. By default, LOO results use their own folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folds = [args.fold] if args.fold else range(1, 6)

    for experiment_id in args.experiment_ids:
        for fold in folds:
            run_experiment(
                experiment_id,
                fold=fold,
                data_dir=args.data_dir,
                reference_results_dir=args.reference_results_dir,
                results_dir=args.results_dir,
            )


if __name__ == "__main__":
    main()
