"""Feature definitions for feature-engineering experiments."""

import numpy as np


def add_num_missing(df):
    df["num_missing"] = df.isna().sum(axis=1)


def add_missing_flags(df):
    for col in list(df.columns):
        df[f"{col}_missing"] = df[col].isna().astype("int8")


def add_weekend_screen_gap(df):
    df["weekend_screen_gap"] = (
        df["weekend_screen_time"]
        - df["daily_screen_time_hours"]
    )


def _daily_screen_time(df):
    return df["daily_screen_time_hours"].replace(0, np.nan)


def add_weekend_screen_ratio(df):
    df["weekend_screen_ratio"] = (
        df["weekend_screen_time"] / _daily_screen_time(df)
    )


def add_social_media_ratio(df):
    df["social_media_ratio"] = (
        df["social_media_hours"] / _daily_screen_time(df)
    )


def add_gaming_ratio(df):
    df["gaming_ratio"] = (
        df["gaming_hours"] / _daily_screen_time(df)
    )


def add_work_study_ratio(df):
    df["work_study_ratio"] = (
        df["work_study_hours"] / _daily_screen_time(df)
    )


def _leisure_screen_time(df):
    return df["social_media_hours"] + df["gaming_hours"]


def add_leisure_screen_time(df):
    df["leisure_screen_time"] = _leisure_screen_time(df)



def add_known_activity_time(df):
    df["known_activity_time"] = (
        df["social_media_hours"]
        + df["gaming_hours"]
        + df["work_study_hours"]
    )


def add_unknown_activity_time(df):
    df["unknown_activity_time"] = (
        _daily_screen_time(df)
        - df["social_media_hours"]
        - df["gaming_hours"]
        - df["work_study_hours"]
    )


def add_app_opens_per_screen_hour(df):
    df["app_opens_per_screen_hour"] = (
        df["app_opens_per_day"] / _daily_screen_time(df)
    )


def add_notifications_per_screen_hour(df):
    df["notifications_per_screen_hour"] = (
        df["notifications_per_day"] / _daily_screen_time(df)
    )


def add_notifications_per_app_open(df):
    df["notifications_per_app_open"] = (
        df["notifications_per_day"] / df["app_opens_per_day"]
    )


def add_screen_per_awake_hour(df):
    df["screen_per_awake_hour"] = (
        _daily_screen_time(df) / (24 - df["sleep_hours"])
    )


def add_screen_sleep_ratio(df):
    df["screen_sleep_ratio"] = (
        _daily_screen_time(df) / df["sleep_hours"]
    )


def add_leisure_screen_ratio(df):
    df["leisure_screen_ratio"] = (
        _leisure_screen_time(df) / _daily_screen_time(df)
    )


def add_work_leisure_ratio(df):
    leisure_screen_time = _leisure_screen_time(df).replace(0, np.nan)
    df["work_leisure_ratio"] = (
        df["work_study_hours"] / leisure_screen_time
    )


def add_offline_awake_hours(df):
    df["offline_awake_hours"] = (
        24
        - df["sleep_hours"]
        - df["daily_screen_time_hours"]
    )


FEATURE_BUILDERS = {
    "num_missing": add_num_missing,
    "missing_flags": add_missing_flags,
    "weekend_screen_gap": add_weekend_screen_gap,
    "weekend_screen_ratio": add_weekend_screen_ratio,
    "social_media_ratio": add_social_media_ratio,
    "gaming_ratio": add_gaming_ratio,
    "work_study_ratio": add_work_study_ratio,
    "leisure_screen_time": add_leisure_screen_time,
    "known_activity_time": add_known_activity_time,
    "unknown_activity_time": add_unknown_activity_time,
    "app_opens_per_screen_hour": add_app_opens_per_screen_hour,
    "notifications_per_screen_hour": add_notifications_per_screen_hour,
    "notifications_per_app_open": add_notifications_per_app_open,
    "screen_per_awake_hour": add_screen_per_awake_hour,
    "screen_sleep_ratio": add_screen_sleep_ratio,
    "leisure_screen_ratio": add_leisure_screen_ratio,
    "work_leisure_ratio": add_work_leisure_ratio,
    "offline_awake_hours": add_offline_awake_hours,
}

FEATURE_SETS = {
    "BASE": (),
    "FE01": ("num_missing",),
    "FE02": ("missing_flags",),
    "FE03": ("weekend_screen_gap",),
    "FE04": ("weekend_screen_ratio",),
    "FE05": ("social_media_ratio",),
    "FE06": ("gaming_ratio",),
    "FE07": ("work_study_ratio",),
    "FE08": ("leisure_screen_time",),
    "FE09": ("known_activity_time",),
    "FE10": ("unknown_activity_time",),
    "FE11": ("app_opens_per_screen_hour",),
    "FE12": ("notifications_per_screen_hour",),
    "FE13": ("notifications_per_app_open",),
    "FE14": ("screen_per_awake_hour",),
    "FE15": ("screen_sleep_ratio",),
    "FE16": ("leisure_screen_ratio",),
    "FE17": ("work_leisure_ratio",),
    "FE18": ("offline_awake_hours",),
}


def make_features(df, feature_names):
    df = df.copy()

    for name in feature_names:
        FEATURE_BUILDERS[name](df)

    return df


def make_experiment_features(df, experiment_id):
    experiment_id = experiment_id.upper()

    if experiment_id not in FEATURE_SETS:
        raise ValueError(f"Unknown experiment: {experiment_id}")

    return make_features(df, FEATURE_SETS[experiment_id])
