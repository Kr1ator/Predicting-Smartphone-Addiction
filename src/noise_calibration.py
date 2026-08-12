"""Run the three FE10 NULL controls with a simple, fixed 5-Fold workflow."""

import json
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .experiment import (
    CAT_COLS,
    CV_PARAMS,
    MODEL_PARAMS,
    PROJECT_ROOT,
    THREAD_COUNT,
)
from .features import make_experiment_features

DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
FE10_RESULTS_DIR = PROJECT_ROOT / "results" / "feature_engineering"
RESULTS_DIR = PROJECT_ROOT / "results" / "noise_calibration"

SCREEN_COL = "daily_screen_time_hours"

# None means the affine control; an integer means a permutation seed.
NULL_EXPERIMENTS = {
    "NULL_A_AFFINE": ("null_affine", None),
    "NULL_B_PERM_1001": ("null_permutation_1001", 1001),
    "NULL_B_PERM_1002": ("null_permutation_1002", 1002),
}


def load_training_data():
    train = pd.read_csv(DATA_PATH)
    X = train.drop(columns=["addicted_label", "id"]).copy()
    y = train["addicted_label"].copy()

    X = make_experiment_features(X, "FE10")

    for col in CAT_COLS:
        X[col] = X[col].fillna("Missing")

    return X, y


def add_null_feature(X, feature_name, permutation_seed):
    X = X.copy()

    if permutation_seed is None:
        X[feature_name] = 2 * X[SCREEN_COL] - 3
    else:
        rng = np.random.default_rng(permutation_seed)
        values = X[SCREEN_COL].to_numpy(copy=True)
        X[feature_name] = rng.permutation(values)  # 把 values 里面的元素随机打乱顺序，返回新的打乱后的数组

    return X


def load_fe10_auc(fold):
    """Read the FE10 AUC for the matching fold."""
    path = FE10_RESULTS_DIR / f"fe10_fold{fold}.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    return float(reference["auc"])


def format_elapsed_time(seconds):
    total_seconds = round(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m {seconds}s"


def run_fold(X, y, train_idx, valid_idx, experiment_id, fold):
    reference_auc = load_fe10_auc(fold)

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = CatBoostClassifier(**MODEL_PARAMS, thread_count=THREAD_COUNT)

    started_at = perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=CAT_COLS,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=200,
    )
    elapsed_seconds = perf_counter() - started_at
    print(f"Fold {fold} elapsed: {format_elapsed_time(elapsed_seconds)}")

    valid_pred = model.predict_proba(X_valid)[:, 1]
    valid_auc = float(roc_auc_score(y_valid, valid_pred))

    result = {
        "experiment_id": experiment_id,
        "fold": fold,
        "reference_auc": reference_auc,
        "auc": valid_auc,
        "delta_vs_reference": valid_auc - reference_auc,
        "best_iteration": model.get_best_iteration() + 1,
        "elapsed": format_elapsed_time(elapsed_seconds),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{experiment_id.lower()}_fold{fold}.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved result: {path}")
    return result, elapsed_seconds


def save_summary(results):
    """Create one summary row for each complete 5-Fold NULL experiment."""
    rows = []

    for experiment_id in NULL_EXPERIMENTS:
        experiment_results = [
            (result, elapsed_seconds)
            for result, elapsed_seconds in results
            if result["experiment_id"] == experiment_id
        ]
        deltas = pd.Series([
            result["delta_vs_reference"]
            for result, _ in experiment_results
        ])
        aucs = pd.Series([
            result["auc"] for result, _ in experiment_results
        ])
        iterations = pd.Series([
            result["best_iteration"] for result, _ in experiment_results
        ])
        elapsed_seconds = pd.Series([
            seconds for _, seconds in experiment_results
        ])

        rows.append({
            "experiment_id": experiment_id,
            "mean_auc": aucs.mean(),
            "mean_delta": deltas.mean(),
            "std_delta": deltas.std(ddof=0),
            "mean_abs_delta": deltas.abs().mean(),
            "max_abs_delta": deltas.abs().max(),
            "min_delta": deltas.min(),
            "max_delta": deltas.max(),
            "positive_folds": int((deltas > 0).sum()),
            "mean_best_iteration": iterations.mean(),
            "mean_elapsed": format_elapsed_time(elapsed_seconds.mean()),
        })

    summary = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


def main():
    X_base, y = load_training_data()
    splitter = StratifiedKFold(
        n_splits=CV_PARAMS["n_splits"],
        shuffle=CV_PARAMS["shuffle"],
        random_state=CV_PARAMS["random_state"],
    )
    folds = list(splitter.split(X_base, y))
    results = []

    for experiment_id, (feature_name, seed) in NULL_EXPERIMENTS.items():
        X = add_null_feature(X_base, feature_name, seed)

        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            result = run_fold(
                X,
                y,
                train_idx,
                valid_idx,
                experiment_id,
                fold,
            )
            results.append(result)

    save_summary(results)


if __name__ == "__main__":
    main()
