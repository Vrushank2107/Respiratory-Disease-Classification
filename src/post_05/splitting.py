from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .common import SEED, TEST_SIZE, VALIDATION_SIZE, require_columns


def make_patient_splits(features: pd.DataFrame, test_size: float = TEST_SIZE, validation_size: float = VALIDATION_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(features, {"patient_id", "diagnosis"}, "feature table")
    patients = features[["patient_id", "diagnosis"]].drop_duplicates()
    if patients["patient_id"].duplicated().any():
        raise ValueError("Each patient must have exactly one diagnosis")
    if len(patients) < 5:
        raise ValueError("Too few patients for patient-wise train/validation/test splitting")
    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_val_idx, test_idx = next(first.split(patients, groups=patients["patient_id"]))
    train_val = patients.iloc[train_val_idx].copy()
    test = patients.iloc[test_idx].copy()
    second = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=SEED + 1)
    train_idx, val_idx = next(second.split(train_val, groups=train_val["patient_id"]))
    train = train_val.iloc[train_idx]
    val = train_val.iloc[val_idx]
    split = pd.concat([
        train.assign(split="train"),
        val.assign(split="validation"),
        test.assign(split="test"),
    ], ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    sets = {name: set(split.loc[split.split.eq(name), "patient_id"]) for name in ("train", "validation", "test")}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise AssertionError("Patient leakage detected across splits")
    class_counts = split.groupby(["split", "diagnosis"])["patient_id"].nunique().unstack(fill_value=0)
    return split, class_counts


def attach_splits(frame: pd.DataFrame, split_table: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, {"patient_id"}, "feature table")
    require_columns(split_table, {"patient_id", "split"}, "patient split table")
    merged = frame.merge(split_table[["patient_id", "split"]], on="patient_id", how="left", validate="many_to_one")
    if merged["split"].isna().any():
        raise ValueError("Some feature rows have no patient split")
    return merged
