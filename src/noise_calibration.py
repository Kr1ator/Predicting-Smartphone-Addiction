"""Calibrate CatBoost feature-addition noise against the FE10 reference."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
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
from .features import make_experiment_features

DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_RESULTS_DIR = PROJECT_ROOT / "results" / "feature_engineering"
RESULTS_DIR = PROJECT_ROOT / "results" / "noise_calibration"

PROTOCOL_ID = "catboost_fe10_noise_calibration_v1"
REFERENCE_EXPERIMENT = "FE10"
EARLY_STOPPING_ROUNDS = 200
SCREEN_COL = "daily_screen_time_hours"

NULL_EXPERIMENTS = {
    "NULL_A_AFFINE": {
        "control_type": "affine_redundant",
        "feature_name": "null_affine",
        "permutation_seed": None,
        "file_stem": "null_a_affine",
    },
    "NULL_B_PERM_1001": {
        "control_type": "permutation",
        "feature_name": "null_permutation_1001",
        "permutation_seed": 1001,
        "file_stem": "null_b_perm_1001",
    },
    "NULL_B_PERM_1002": {
        "control_type": "permutation",
        "feature_name": "null_permutation_1002",
        "permutation_seed": 1002,
        "file_stem": "null_b_perm_1002",
    },
}


def build_config_id():
    config = {
        "protocol_id": PROTOCOL_ID,
        "reference_protocol_id": REFERENCE_PROTOCOL_ID,
        "reference_config_id": REFERENCE_CONFIG_ID,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "model_params": MODEL_PARAMS,
        "cv_params": CV_PARAMS,
        "cat_cols": CAT_COLS,
        "thread_count": THREAD_COUNT,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "null_experiments": NULL_EXPERIMENTS,
    }
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:12]


CONFIG_ID = build_config_id()


def result_path(results_dir, experiment_id, fold):
    file_stem = NULL_EXPERIMENTS[experiment_id]["file_stem"]
    return results_dir / f"{file_stem}_fold{fold}.json"


def load_reference_result(reference_results_dir, fold):
    path = reference_results_dir / f"fe10_fold{fold}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing FE10 reference for Fold {fold}: {path}")

    reference = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": REFERENCE_EXPERIMENT,
        "protocol_id": REFERENCE_PROTOCOL_ID,
        "config_id": REFERENCE_CONFIG_ID,
        "fold": fold,
        "thread_count": THREAD_COUNT,
    }
    for key, expected_value in expected.items():
        if reference.get(key) != expected_value:
            raise ValueError(
                f"FE10 reference mismatch in {path}: "
                f"{key}={reference.get(key)!r}, expected {expected_value!r}"
            )

    if reference.get("features") != ["unknown_activity_time"]:
        raise ValueError(f"Unexpected FE10 feature definition in {path}")

    return reference, path


def validate_existing_result(path, experiment_id, fold, reference_auc):
    result = json.loads(path.read_text(encoding="utf-8"))
    spec = NULL_EXPERIMENTS[experiment_id]
    expected = {
        "experiment_id": experiment_id,
        "protocol_id": PROTOCOL_ID,
        "config_id": CONFIG_ID,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "reference_auc": reference_auc,
        "fold": fold,
        "thread_count": THREAD_COUNT,
        "control_type": spec["control_type"],
        "permutation_seed": spec["permutation_seed"],
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            raise ValueError(
                f"Existing result is incompatible: {path}; "
                f"{key}={result.get(key)!r}, expected {expected_value!r}"
            )
    return result


def add_control_feature(X, experiment_id):
    X = X.copy()
    spec = NULL_EXPERIMENTS[experiment_id]
    feature_name = spec["feature_name"]

    if spec["control_type"] == "affine_redundant":
        X[feature_name] = 2 * X[SCREEN_COL] - 3
    elif spec["control_type"] == "permutation":
        rng = np.random.default_rng(spec["permutation_seed"])
        values = X[SCREEN_COL].to_numpy(copy=True)
        X[feature_name] = rng.permutation(values)
    else:
        raise ValueError(f"Unknown control type: {spec['control_type']}")

    return X


def prepare_training_data(data_dir, experiment_id):
    train_path = data_dir / "train.csv"
    train = pd.read_csv(train_path)

    X = train.drop(columns=["addicted_label", "id"]).copy()
    y = train["addicted_label"].copy()

    X = make_experiment_features(X, REFERENCE_EXPERIMENT)
    X = add_control_feature(X, experiment_id)

    for col in CAT_COLS:
        X[col] = X[col].fillna("Missing")

    return X, y


def select_fold(X, y, fold):
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
            return (
                X.iloc[train_idx],
                X.iloc[valid_idx],
                y.iloc[train_idx],
                y.iloc[valid_idx],
            )

    raise ValueError(f"Fold {fold} was not produced by the CV splitter")


def run_null_experiment(
    experiment_id,
    fold,
    data_dir=DATA_DIR,
    reference_results_dir=REFERENCE_RESULTS_DIR,
    results_dir=RESULTS_DIR,
):
    experiment_id = experiment_id.upper()
    if experiment_id not in NULL_EXPERIMENTS:
        raise ValueError(f"Unknown NULL experiment: {experiment_id}")
    if fold not in range(1, CV_PARAMS["n_splits"] + 1):
        raise ValueError(f"fold must be between 1 and {CV_PARAMS['n_splits']}")

    reference, reference_path = load_reference_result(reference_results_dir, fold)
    reference_auc = float(reference["auc"])
    path = result_path(results_dir, experiment_id, fold)

    if path.exists():
        existing = validate_existing_result(
            path,
            experiment_id,
            fold,
            reference_auc,
        )
        print(f"Validated existing result, skipping: {path}")
        return existing

    X, y = prepare_training_data(data_dir, experiment_id)
    X_train, X_valid, y_train, y_valid = select_fold(X, y, fold)

    model = CatBoostClassifier(
        **MODEL_PARAMS,
        thread_count=THREAD_COUNT,
    )

    started_at = perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=CAT_COLS,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    elapsed_seconds = perf_counter() - started_at

    valid_pred = model.predict_proba(X_valid)[:, 1]
    valid_auc = float(roc_auc_score(y_valid, valid_pred))
    best_iteration = model.get_best_iteration() + 1

    spec = NULL_EXPERIMENTS[experiment_id]
    saved_model_params = MODEL_PARAMS.copy()
    saved_model_params["thread_count"] = THREAD_COUNT

    result = {
        "protocol_id": PROTOCOL_ID,
        "config_id": CONFIG_ID,
        "source": "src.noise_calibration",
        "experiment_id": experiment_id,
        "feature_group": "noise_calibration",
        "control_type": spec["control_type"],
        "permutation_seed": spec["permutation_seed"],
        "features": ["unknown_activity_time", spec["feature_name"]],
        "reference_experiment": REFERENCE_EXPERIMENT,
        "reference_protocol_id": REFERENCE_PROTOCOL_ID,
        "reference_config_id": REFERENCE_CONFIG_ID,
        "reference_result": str(reference_path.relative_to(PROJECT_ROOT)),
        "reference_auc": reference_auc,
        "fold": fold,
        "auc": valid_auc,
        "delta_vs_reference": valid_auc - reference_auc,
        "best_iteration": best_iteration,
        "elapsed_seconds": elapsed_seconds,
        "model_params": saved_model_params,
        "effective_model_params": model.get_all_params(),
        "cv_params": CV_PARAMS,
        "cat_cols": list(CAT_COLS),
        "thread_count": THREAD_COUNT,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    result_text = json.dumps(result, indent=2, ensure_ascii=False)
    path.write_text(result_text + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved result:", path)
    return result


def summarize_results(results_dir=RESULTS_DIR):
    rows = []

    for experiment_id, spec in NULL_EXPERIMENTS.items():
        fold_results = []
        for fold in range(1, CV_PARAMS["n_splits"] + 1):
            path = result_path(results_dir, experiment_id, fold)
            if not path.exists():
                continue
            fold_results.append(json.loads(path.read_text(encoding="utf-8")))

        if not fold_results:
            continue

        deltas = np.array(
            [result["delta_vs_reference"] for result in fold_results],
            dtype=float,
        )
        aucs = np.array([result["auc"] for result in fold_results], dtype=float)
        iterations = np.array(
            [result["best_iteration"] for result in fold_results],
            dtype=float,
        )

        rows.append(
            {
                "experiment_id": experiment_id,
                "control_type": spec["control_type"],
                "permutation_seed": spec["permutation_seed"],
                "folds_completed": len(fold_results),
                "complete": len(fold_results) == CV_PARAMS["n_splits"],
                "mean_auc": aucs.mean(),
                "mean_delta": deltas.mean(),
                "std_delta": deltas.std(ddof=0),
                "mean_abs_delta": np.abs(deltas).mean(),
                "max_abs_delta": np.abs(deltas).max(),
                "min_delta": deltas.min(),
                "max_delta": deltas.max(),
                "positive_folds": int((deltas > 0).sum()),
                "mean_best_iteration": iterations.mean(),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = results_dir / "summary.csv"

    if not summary.empty:
        results_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print("Saved summary:", summary_path)
    else:
        print("No NULL results found; summary was not created.")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FE10 NULL / negative-control calibration experiments."
    )
    parser.add_argument(
        "experiment_ids",
        nargs="*",
        type=str.upper,
        choices=NULL_EXPERIMENTS,
        help="Defaults to all NULL experiments.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, CV_PARAMS["n_splits"] + 1),
        help="Run one fold; omit to run all five folds.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--reference-results-dir",
        type=Path,
        default=REFERENCE_RESULTS_DIR,
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only rebuild summary.csv from existing JSON results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.summarize_only:
        experiment_ids = args.experiment_ids or list(NULL_EXPERIMENTS)
        folds = [args.fold] if args.fold else range(1, CV_PARAMS["n_splits"] + 1)

        for experiment_id in experiment_ids:
            for fold in folds:
                run_null_experiment(
                    experiment_id,
                    fold,
                    data_dir=args.data_dir,
                    reference_results_dir=args.reference_results_dir,
                    results_dir=args.results_dir,
                )

    summarize_results(args.results_dir)


if __name__ == "__main__":
    main()
