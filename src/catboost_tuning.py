"""Check the CatBoost reference and run a small Optuna search on Folds 1-2."""

import argparse
import json
from time import perf_counter

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .experiment import CAT_COLS, CV_PARAMS, MODEL_PARAMS, PROJECT_ROOT, THREAD_COUNT
from .round2_features import make_round2_features

DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
REFERENCE_DIR = PROJECT_ROOT / "results" / "round2" / "r_group_single"
RESULTS_DIR = PROJECT_ROOT / "results" / "catboost_tuning"
TRIALS_DIR = RESULTS_DIR / "trials"
VALIDATION_DIR = RESULTS_DIR / "validation"
CHECK_PATH = RESULTS_DIR / "reference_check.json"
STUDY_PATH = RESULTS_DIR / "optuna_study.db"

STUDY_NAME = "catboost_fe10_fe25_fold12"
SEARCH_FOLDS = (1, 2)
VALIDATION_FOLDS = (3, 4, 5)
REFERENCE_TOLERANCE = 0.000001
EARLY_STOPPING_ROUNDS = 200

# CatBoost uses MVS by default for this CPU binary-classification setup.
# These values describe the current v4 model inside the Optuna search space.
BASELINE_PARAMS = {
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "subsample": 0.8,
}


def load_data():
    train = pd.read_csv(DATA_PATH)
    X = train.drop(columns=["addicted_label", "id"]).copy()
    y = train["addicted_label"].copy()

    X = make_round2_features(X, "FE25")

    for col in CAT_COLS:
        X[col] = X[col].fillna("Missing")

    splitter = StratifiedKFold(**CV_PARAMS)
    folds = list(splitter.split(X, y))
    return X, y, folds


def load_reference_auc(fold):
    path = REFERENCE_DIR / f"fe25_fold{fold}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing FE25 reference: {path}")

    result = json.loads(path.read_text(encoding="utf-8"))
    if result["experiment_id"] != "FE25" or result["fold"] != fold:
        raise ValueError(f"Invalid FE25 reference: {path}")

    return float(result["auc"])


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_fold(X, y, folds, fold, tuned_params):
    train_idx, valid_idx = folds[fold - 1]
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model_params = MODEL_PARAMS.copy()
    model_params.update(tuned_params)
    model_params["bootstrap_type"] = "MVS"

    model = CatBoostClassifier(
        **model_params,
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

    prediction = model.predict_proba(X_valid)[:, 1]
    auc = float(roc_auc_score(y_valid, prediction))
    reference_auc = load_reference_auc(fold)

    return {
        "fold": fold,
        "auc": auc,
        "reference_auc": reference_auc,
        "delta_vs_reference": auc - reference_auc,
        "best_iteration": model.get_best_iteration() + 1,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def summarize_folds(fold_results):
    return {
        "mean_auc": sum(result["auc"] for result in fold_results) / len(fold_results),
        "mean_delta": sum(
            result["delta_vs_reference"] for result in fold_results
        ) / len(fold_results),
        "positive_folds": sum(
            result["delta_vs_reference"] > 0 for result in fold_results
        ),
    }


def check_reference(X, y, folds):
    print("Stage A: checking the current CatBoost v4 reference")
    fold_results = []

    for fold in SEARCH_FOLDS:
        result = run_fold(X, y, folds, fold, BASELINE_PARAMS)
        fold_results.append(result)
        print(
            f"Fold {fold}: AUC={result['auc']:.6f}, "
            f"delta={result['delta_vs_reference']:+.6f}"
        )

    summary = summarize_folds(fold_results)
    passed = all(
        abs(result["delta_vs_reference"]) <= REFERENCE_TOLERANCE
        for result in fold_results
    )

    output = {
        "passed": passed,
        "tolerance": REFERENCE_TOLERANCE,
        "params": BASELINE_PARAMS,
        "folds": fold_results,
        **summary,
    }
    save_json(output, CHECK_PATH)
    print(f"Saved reference check: {CHECK_PATH}")

    if not passed:
        raise RuntimeError("Reference check failed. Stop before running Optuna.")


def load_reference_check():
    if not CHECK_PATH.exists():
        raise FileNotFoundError(
            f"Run Stage A first: python -m src.catboost_tuning check"
        )

    check = json.loads(CHECK_PATH.read_text(encoding="utf-8"))
    if not check["passed"]:
        raise ValueError("The saved reference check did not pass.")
    return check


def suggest_params(trial):
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.04, 0.08, log=True
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 1.0, 30.0, log=True
        ),
        "random_strength": trial.suggest_float(
            "random_strength", 0.1, 5.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def build_objective(X, y, folds):
    def objective(trial):
        params = suggest_params(trial)
        fold_results = []

        for fold in SEARCH_FOLDS:
            result = run_fold(X, y, folds, fold, params)
            fold_results.append(result)
            print(
                f"Trial {trial.number}, Fold {fold}: "
                f"delta={result['delta_vs_reference']:+.6f}"
            )

        summary = summarize_folds(fold_results)
        output = {
            "trial_number": trial.number,
            "params": params,
            "folds": fold_results,
            **summary,
        }
        save_json(output, TRIALS_DIR / f"trial_{trial.number:03d}.json")
        return summary["mean_delta"]

    return objective


def run_search(X, y, folds, total_trials):
    try:
        import optuna
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Stage B requires Optuna. Install it before running search."
        ) from None

    load_reference_check()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_PATH}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )
    remaining_trials = max(0, total_trials - completed_trials)

    print(
        f"Stage B: {completed_trials} completed, "
        f"{remaining_trials} remaining"
    )

    if remaining_trials:
        study.optimize(
            build_objective(X, y, folds),
            n_trials=remaining_trials,
            n_jobs=1,
            gc_after_trial=True,
        )

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best mean delta: {study.best_value:+.6f}")
    print("Best parameters:")
    print(json.dumps(study.best_params, indent=2, ensure_ascii=False))


def load_search_trials():
    paths = sorted(TRIALS_DIR.glob("trial_*.json"))
    if not paths:
        raise FileNotFoundError(f"No completed Optuna trials in: {TRIALS_DIR}")

    trials = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return sorted(trials, key=lambda trial: trial["mean_delta"], reverse=True)


def update_validation_summary(result):
    result.update(summarize_folds(result["folds"]))
    validation_folds = [
        fold for fold in result["folds"] if fold["fold"] in VALIDATION_FOLDS
    ]
    validation = summarize_folds(validation_folds)
    result["validation_mean_auc"] = validation["mean_auc"]
    result["validation_mean_delta"] = validation["mean_delta"]
    result["validation_positive_folds"] = validation["positive_folds"]


def run_validation(X, y, folds, top_k):
    load_reference_check()
    candidates = load_search_trials()[:top_k]

    print("Stage C candidates:")
    for candidate in candidates:
        print(
            f"Trial {candidate['trial_number']}: "
            f"search delta={candidate['mean_delta']:+.6f}"
        )

    for candidate in candidates:
        trial_number = candidate["trial_number"]
        path = VALIDATION_DIR / f"trial_{trial_number:03d}.json"

        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["params"] != candidate["params"]:
                raise ValueError(f"Trial {trial_number} parameters do not match.")
        else:
            result = {
                "trial_number": trial_number,
                "params": candidate["params"],
                "folds": list(candidate["folds"]),
            }

        completed_folds = {fold["fold"] for fold in result["folds"]}

        for fold in VALIDATION_FOLDS:
            if fold in completed_folds:
                print(f"Trial {trial_number}, Fold {fold}: already completed")
                continue

            fold_result = run_fold(X, y, folds, fold, result["params"])
            result["folds"].append(fold_result)
            result["folds"].sort(key=lambda item: item["fold"])
            update_validation_summary(result)
            save_json(result, path)

            print(
                f"Trial {trial_number}, Fold {fold}: "
                f"delta={fold_result['delta_vs_reference']:+.6f}"
            )

        update_validation_summary(result)
        save_json(result, path)
        print(
            f"Trial {trial_number}: validation delta="
            f"{result['validation_mean_delta']:+.6f}, "
            f"full 5-Fold delta={result['mean_delta']:+.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["check", "search", "validate"])
    parser.add_argument(
        "--total-trials",
        type=int,
        default=20,
        help="Total number of completed Optuna configurations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top search trials to validate on Folds 3-5.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, folds = load_data()

    if args.stage == "check":
        check_reference(X, y, folds)
    elif args.stage == "search":
        run_search(X, y, folds, args.total_trials)
    else:
        run_validation(X, y, folds, args.top_k)


if __name__ == "__main__":
    main()
