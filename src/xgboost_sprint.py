"""One-day XGBoost sprint using the final FE10 + FE25 feature set.

The script deliberately keeps selection small: three fixed ``hist`` configurations
are compared only on folds 1--2, then the selected configuration is evaluated
once with the established five-fold split.  It writes OOF and test predictions
that can later be used for a CatBoost/XGBoost blend.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .experiment import CAT_COLS, CV_PARAMS, PROJECT_ROOT, THREAD_COUNT
from .round2_features import make_round2_features


DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "xgboost_sprint"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
EARLY_STOPPING_ROUNDS = 200
MAX_ESTIMATORS = 10_000
SEARCH_FOLDS = (1, 2)

# These are intentionally conservative, fixed CPU-hist candidates; this sprint
# is a parameter probe, not an Optuna search.
CANDIDATES = (
    {
        "name": "balanced_depth6",
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "shallower_depth5",
        "max_depth": 5,
        "learning_rate": 0.06,
        "min_child_weight": 8,
        "subsample": 0.90,
        "colsample_bytree": 0.95,
        "reg_lambda": 8.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "deeper_depth7",
        "max_depth": 7,
        "learning_rate": 0.04,
        "min_child_weight": 8,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
    },
)


def prepare_features(train, test):
    """Build FE25 and give train/test identical categorical definitions."""
    X_train = make_round2_features(
        train.drop(columns=["addicted_label", "id"]).copy(), "FE25"
    )
    X_test = make_round2_features(test.drop(columns=["id"]).copy(), "FE25")

    for column in CAT_COLS:
        categories = pd.Index(
            sorted(
                set(X_train[column].dropna().astype(str))
                | set(X_test[column].dropna().astype(str))
            )
        )
        categorical_dtype = pd.CategoricalDtype(categories=categories)
        X_train[column] = X_train[column].astype("string").astype(categorical_dtype)
        X_test[column] = X_test[column].astype("string").astype(categorical_dtype)

    return X_train, X_test


def build_model(params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=4,
        n_estimators=MAX_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        n_jobs=THREAD_COUNT,
        random_state=42,
        verbosity=0,
        **{key: value for key, value in params.items() if key != "name"},
    )


def fit_one_fold(X, y, folds, fold_number, params, X_test=None):
    train_index, valid_index = folds[fold_number - 1]
    model = build_model(params)

    started_at = perf_counter()
    model.fit(
        X.iloc[train_index],
        y.iloc[train_index],
        eval_set=[(X.iloc[valid_index], y.iloc[valid_index])],
        verbose=False,
    )
    elapsed_seconds = perf_counter() - started_at
    best_iteration = int(model.best_iteration) + 1

    valid_prediction = model.predict_proba(
        X.iloc[valid_index],
        iteration_range=(0, best_iteration),
    )[:, 1]
    result = {
        "fold": fold_number,
        "auc": float(roc_auc_score(y.iloc[valid_index], valid_prediction)),
        "best_iteration": best_iteration,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }

    test_prediction = None
    if X_test is not None:
        test_prediction = model.predict_proba(
            X_test,
            iteration_range=(0, best_iteration),
        )[:, 1]

    return valid_index, valid_prediction, test_prediction, result


def summarize_fold_results(fold_results):
    return {
        "folds": fold_results,
        "mean_auc": float(np.mean([result["auc"] for result in fold_results])),
        "mean_best_iteration": float(
            np.mean([result["best_iteration"] for result in fold_results])
        ),
        "total_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in fold_results), 1
        ),
    }


def run_search(X, y, folds):
    search_results = []

    for params in CANDIDATES:
        fold_results = []
        print(f"Testing {params['name']} on folds {SEARCH_FOLDS}", flush=True)
        for fold_number in SEARCH_FOLDS:
            _, _, _, result = fit_one_fold(X, y, folds, fold_number, params)
            fold_results.append(result)
            print(
                f"  fold={fold_number} auc={result['auc']:.6f} "
                f"best_iteration={result['best_iteration']} "
                f"seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )

        search_results.append(
            {
                "params": params,
                **summarize_fold_results(fold_results),
            }
        )

    return search_results


def run_full_cv(X, y, X_test, folds, params):
    oof_prediction = np.zeros(len(X), dtype=np.float64)
    test_prediction = np.zeros(len(X_test), dtype=np.float64)
    fold_results = []

    for fold_number in range(1, len(folds) + 1):
        valid_index, valid_prediction, fold_test_prediction, result = fit_one_fold(
            X, y, folds, fold_number, params, X_test
        )
        oof_prediction[valid_index] = valid_prediction
        test_prediction += fold_test_prediction / len(folds)
        fold_results.append(result)
        print(
            f"Full CV fold={fold_number} auc={result['auc']:.6f} "
            f"best_iteration={result['best_iteration']} "
            f"seconds={result['elapsed_seconds']:.1f}",
            flush=True,
        )

    return oof_prediction, test_prediction, summarize_fold_results(fold_results)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    started_at = perf_counter()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train["addicted_label"]
    X, X_test = prepare_features(train, test)
    splitter = StratifiedKFold(**CV_PARAMS)
    folds = list(splitter.split(X, y))

    print(f"Training rows={len(X):,}; test rows={len(X_test):,}", flush=True)
    search_results = run_search(X, y, folds)
    selected = max(search_results, key=lambda result: result["mean_auc"])
    screening = {
        "experiment": "xgboost_sprint_screening",
        "feature_set": "FE10 + FE25",
        "cv_params": CV_PARAMS,
        "search_folds": list(SEARCH_FOLDS),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "max_estimators": MAX_ESTIMATORS,
        "candidates": search_results,
        "selected_params": selected["params"],
        "selected_search_mean_auc": selected["mean_auc"],
    }
    write_json(RESULTS_DIR / "screening.json", screening)
    print(
        f"Selected {selected['params']['name']} "
        f"with fold 1-2 mean AUC={selected['mean_auc']:.6f}",
        flush=True,
    )

    oof_prediction, test_prediction, final_result = run_full_cv(
        X, y, X_test, folds, selected["params"]
    )
    # Averaging float32 fold probabilities can exceed 1 by machine epsilon.
    test_prediction = np.clip(test_prediction, 0.0, 1.0)
    total_elapsed_seconds = perf_counter() - started_at

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_frame = pd.DataFrame({"id": train["id"], "prediction": oof_prediction})
    test_frame = pd.DataFrame({"id": test["id"], "prediction": test_prediction})
    oof_frame.to_csv(RESULTS_DIR / "oof.csv.gz", index=False, compression="gzip")
    test_frame.to_csv(
        RESULTS_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
    )

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "addicted_label": test_prediction}).to_csv(
        SUBMISSION_DIR / "xgboost_sprint.csv", index=False
    )

    summary = {
        "experiment": "xgboost_sprint",
        "feature_set": "FE10 + FE25",
        "categorical_columns": CAT_COLS,
        "categorical_handling": (
            "Train and test share each categorical column's combined category "
            "set and are passed to XGBoost as pandas categorical columns."
        ),
        "cv_params": CV_PARAMS,
        "search_folds": list(SEARCH_FOLDS),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "max_estimators": MAX_ESTIMATORS,
        "candidates": search_results,
        "selected_params": selected["params"],
        "selected_search_mean_auc": selected["mean_auc"],
        "final_cv": final_result,
        "oof_auc": float(roc_auc_score(y, oof_prediction)),
        "total_elapsed_seconds": round(total_elapsed_seconds, 1),
    }
    write_json(RESULTS_DIR / "summary.json", summary)

    print(
        f"Complete: OOF AUC={summary['oof_auc']:.6f}; "
        f"total_seconds={summary['total_elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
