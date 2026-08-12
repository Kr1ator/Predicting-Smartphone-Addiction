"""Feature definitions for Feature Engineering Round 2."""

from .features import (
    add_leisure_screen_ratio,
    add_work_study_ratio,
    make_experiment_features,
)

DAILY_SCREEN = "daily_screen_time_hours"
SOCIAL = "social_media_hours"
GAMING = "gaming_hours"
WORK_STUDY = "work_study_hours"
WEEKEND_SCREEN = "weekend_screen_time"
SLEEP = "sleep_hours"

GENDER = "gender"
STRESS = "stress_level"
IMPACT = "academic_work_impact"


# Numerical features used in the first stage.
def add_fe20(df):
    df["screen_minus_social"] = df[DAILY_SCREEN] - df[SOCIAL]


def add_fe22(df):
    df["screen_minus_work_study"] = df[DAILY_SCREEN] - df[WORK_STUDY]


def add_fe25(df):
    df["screen_minus_gaming_work_study"] = (
        df[DAILY_SCREEN]
        - df[GAMING]
        - df[WORK_STUDY]
    )


def add_fe26(df):
    df["leisure_minus_work_study"] = (
        df[SOCIAL]
        + df[GAMING]
        - df[WORK_STUDY]
    )


def add_fe27(df):
    df["mean_daily_weekend_screen"] = df[
        [DAILY_SCREEN, WEEKEND_SCREEN]
    ].mean(axis=1)


def add_fe28(df):
    # FE28 reuses the old FE07 definition.
    add_work_study_ratio(df)


def add_fe29(df):
    # FE29 reuses the old FE16 definition.
    add_leisure_screen_ratio(df)


def _safe_level_name(value):
    return str(value).strip().replace(" ", "_").replace("/", "_")


def add_gated_numerical(df, category_col, numerical_col, feature_prefix):
    """Create one numerical feature for each category level."""
    category = df[category_col].fillna("Missing").astype(str)

    for level in sorted(category.unique()):
        level_name = _safe_level_name(level)
        feature_name = f"{feature_prefix}__{level_name}"
        df[feature_name] = df[numerical_col].where(category == level)

    missing_flag = f"{numerical_col}_missing"
    if missing_flag not in df.columns:
        df[missing_flag] = df[numerical_col].isna().astype("int8")


# Categorical x numerical features used in CAT_NUM_GROUP.
def add_fe30(df):
    add_gated_numerical(df, GENDER, GAMING, "gaming_by_gender")


def add_fe31(df):
    add_gated_numerical(df, GENDER, SOCIAL, "social_by_gender")


def add_fe34(df):
    add_gated_numerical(df, STRESS, SLEEP, "sleep_by_stress")


def add_fe39(df):
    add_gated_numerical(df, IMPACT, WORK_STUDY, "work_by_impact")


def add_fe40(df):
    add_gated_numerical(df, IMPACT, DAILY_SCREEN, "screen_by_impact")


def add_categorical_combination(df, columns, feature_name):
    """Join categorical values into one explicit CatBoost category."""
    combined = df[columns[0]].fillna("Missing").astype(str)

    for col in columns[1:]:
        values = df[col].fillna("Missing").astype(str)
        combined = combined + "__" + values

    df[feature_name] = combined


# Categorical x categorical features used in CAT2_GROUP.
def add_fe43(df):
    add_categorical_combination(
        df,
        [GENDER, STRESS],
        "gender_stress_combo",
    )


def add_fe44(df):
    add_categorical_combination(
        df,
        [GENDER, IMPACT],
        "gender_impact_combo",
    )


def add_fe45(df):
    add_categorical_combination(
        df,
        [STRESS, IMPACT],
        "stress_impact_combo",
    )


ROUND2_FEATURE_BUILDERS = {
    "FE20": add_fe20,
    "FE22": add_fe22,
    "FE25": add_fe25,
    "FE26": add_fe26,
    "FE27": add_fe27,
    "FE28": add_fe28,
    "FE29": add_fe29,
    "FE30": add_fe30,
    "FE31": add_fe31,
    "FE34": add_fe34,
    "FE39": add_fe39,
    "FE40": add_fe40,
    "FE43": add_fe43,
    "FE44": add_fe44,
    "FE45": add_fe45,
}


# First-stage experiments and the three R_GROUP leave-one-out experiments.
ROUND2_FEATURE_SETS = {
    "FE27": ("FE27",),
    "R_GROUP": ("FE20", "FE22", "FE25"),
    "FE26": ("FE26",),
    "CAT_NUM_GROUP": ("FE30", "FE31", "FE34", "FE39", "FE40"),
    "CAT2_GROUP": ("FE43", "FE44", "FE45"),
    "O_GROUP": ("FE28", "FE29"),
    "R_MINUS_FE20": ("FE22", "FE25"),
    "R_MINUS_FE22": ("FE20", "FE25"),
    "R_MINUS_FE25": ("FE20", "FE22"),
}


R_GROUP_LOO_EXPERIMENTS = (
    "R_MINUS_FE20",
    "R_MINUS_FE22",
    "R_MINUS_FE25",
)


ROUND2_CAT_COLS = {
    "CAT2_GROUP": (
        "gender_stress_combo",
        "gender_impact_combo",
        "stress_impact_combo",
    ),
}


def make_round2_features(df, experiment_id):
    experiment_id = experiment_id.upper()

    if experiment_id not in ROUND2_FEATURE_SETS:
        raise ValueError(f"Unknown Round 2 experiment: {experiment_id}")

    df = make_experiment_features(df, "FE10")

    for feature_id in ROUND2_FEATURE_SETS[experiment_id]:
        ROUND2_FEATURE_BUILDERS[feature_id](df)

    return df


def get_round2_cat_cols(experiment_id):
    experiment_id = experiment_id.upper()

    if experiment_id not in ROUND2_FEATURE_SETS:
        raise ValueError(f"Unknown Round 2 experiment: {experiment_id}")

    return list(ROUND2_CAT_COLS.get(experiment_id, ()))
