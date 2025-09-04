
import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/JuneJulyTrackedProject.csv.csv"
OUT_PATH = "data/processed/wellbeing_clean.csv"

YES_NO_MAP = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}

def yes_no_to_int(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(YES_NO_MAP).astype("Int64")
    return df

def time_to_minutes(t):
    """Convert time strings like '2:56 AM' to minutes since midnight."""
    if pd.isna(t):
        return np.nan
    if isinstance(t, (int, float)):
        return t
    s = str(t).strip()
    # Try multiple common formats
    fmts = ["%I:%M %p", "%H:%M", "%I %p"]
    for fmt in fmts:
        try:
            parsed = pd.to_datetime(s, format=fmt)
            return parsed.hour * 60 + parsed.minute
        except Exception:
            continue
    return np.nan

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Date handling
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)

    # Yes/No columns
    yes_no_cols = [
        "worked_today","gym_today","socialized_today",
        "cheated_diet","practiced_today"
    ]
    df = yes_no_to_int(df, yes_no_cols)

    # Numeric columns
    numeric_cols = [
        "mode_score","mood_score","productivity_score","screen_time_min",
        "chores_min","practice_time_min","sleep_min","hangout_time_min",
        "schoolwork_min","job_time_min","money_spent","calories","wake_minutes"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Times → minutes
    for col in ["bed_time","wake_time","first_meal_time","study_start_time"]:
        if col in df.columns:
            df[col + "_min"] = df[col].apply(time_to_minutes)

    # Derived features
    if "sleep_min" in df.columns:
        df["sleep_hours"] = df["sleep_min"] / 60.0
    if "wake_minutes" in df.columns:
        df["wake_hours"] = df["wake_minutes"] / 60.0

    # Simple missingness flags on key columns
    for col in ["mood_score","productivity_score","sleep_min","screen_time_min"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    return df

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    df_clean = clean_data(df)
    df_clean.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data → {OUT_PATH}")
    print(df_clean.head().to_string())

if __name__ == "__main__":
    main()
