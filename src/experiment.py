"""Run one feature-engineering experiment on one CV fold."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import FEATURE_SETS, make_experiment_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "feature_engineering"

PROTOCOL_ID = "catboost_fe_single_fold_v1"

CAT_COLS = [
    "gender",
    "stress_level",
    "academic_work_impact",
]

MODEL_PARAMS = {
    "iterations": 10000,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 500,
    "allow_writing_files": False,
}

CV_PARAMS = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}


def build_config_id():
    config = {
        "model_params": MODEL_PARAMS,
        "cv_params": CV_PARAMS,
        "cat_cols": CAT_COLS,
    }
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:12]


CONFIG_ID = build_config_id()


def load_baseline_auc(results_dir, fold, thread_count):
    baseline_path = results_dir / f"base_fold{fold}.json"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Run BASE Fold {fold} first: {baseline_path}")

    baseline = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )

    if baseline["experiment_id"] != "BASE":
        raise ValueError("The baseline experiment ID is not BASE.")
    if baseline["protocol_id"] != PROTOCOL_ID:
        raise ValueError("The baseline protocol does not match.")
    if baseline.get("config_id") != CONFIG_ID:
        raise ValueError(
            f"The baseline configuration does not match. Run BASE Fold {fold} again."
        )
    if baseline["fold"] != fold:
        raise ValueError("The baseline fold does not match.")
    if baseline["thread_count"] != thread_count:
        raise ValueError("The baseline thread count does not match.")

    return float(baseline["auc"])


def run_experiment(
    experiment_id,
    fold=1,
    thread_count=8,
    data_dir=DATA_DIR,
    results_dir=RESULTS_DIR,
):
    experiment_id = experiment_id.upper()

    if experiment_id not in FEATURE_SETS:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    if fold not in range(1, 6):
        raise ValueError("fold must be between 1 and 5")
    if thread_count < 1:
        raise ValueError("thread_count must be at least 1")

    baseline_auc = None

    if experiment_id != "BASE":
        baseline_auc = load_baseline_auc(
            results_dir,
            fold,
            thread_count,
        )

    # Load and prepare the training data.
    train_path = data_dir / "train.csv"
    train = pd.read_csv(train_path)

    X = train.drop(columns=["addicted_label", "id"]).copy()
    y = train["addicted_label"].copy()

    X = make_experiment_features(X, experiment_id)

    for col in CAT_COLS:
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
        thread_count=thread_count,
    )

    started_at = perf_counter()

    model.fit(
        X_train,
        y_train,
        cat_features=CAT_COLS,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=200,
    )

    elapsed_seconds = perf_counter() - started_at

    # Evaluate the model.
    valid_pred = model.predict_proba(X_valid)[:, 1]
    valid_auc = float(roc_auc_score(y_valid, valid_pred))
    best_iteration = model.get_best_iteration() + 1

    if experiment_id == "BASE":
        baseline_auc = valid_auc

    delta_vs_baseline = valid_auc - baseline_auc

    # Build the result record.
    saved_model_params = MODEL_PARAMS.copy()
    saved_model_params["thread_count"] = thread_count

    result = {
        "protocol_id": PROTOCOL_ID,
        "config_id": CONFIG_ID,
        "source": "src.experiment",
        "model_params": saved_model_params,
        "cv_params": CV_PARAMS,
        "cat_cols": list(CAT_COLS),
        "experiment_id": experiment_id,
        "features": list(FEATURE_SETS[experiment_id]),
        "fold": fold,
        "auc": valid_auc,
        "baseline_auc": baseline_auc,
        "delta_vs_baseline": delta_vs_baseline,
        "best_iteration": best_iteration,
        "elapsed_seconds": elapsed_seconds,
        "thread_count": thread_count,
    }

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
        choices=FEATURE_SETS,
    )
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--thread-count", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    for experiment_id in args.experiment_ids:
        run_experiment(
            experiment_id,
            fold=args.fold,
            thread_count=args.thread_count,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
