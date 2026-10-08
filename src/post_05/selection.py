from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from .common import CORRELATION_THRESHOLD, FEATURE_MAX_COUNT, SEED, feature_columns


def rank_training_features(train: pd.DataFrame, max_features: int = FEATURE_MAX_COUNT) -> tuple[pd.DataFrame, list[str]]:
    """Rank features using training rows only; caller must pass training patients only."""
    cols = feature_columns(train)
    if not cols:
        raise ValueError("No numeric feature columns found")
    x = train[cols].replace([np.inf, -np.inf], np.nan)
    medians = x.median().fillna(0)
    x = x.fillna(medians)
    y = train["diagnosis"].astype(str)
    if y.nunique() < 2:
        raise ValueError("Feature selection requires at least two diagnoses in training data")
    encoded = LabelEncoder().fit_transform(y)
    f_values, p_values = f_classif(x, encoded)
    mi = mutual_info_classif(x, encoded, random_state=SEED)
    forest = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
    forest.fit(x, y)
    corr = x.corr().abs()
    keep, dropped = [], set()
    for column in cols:
        if column in dropped:
            continue
        keep.append(column)
        dropped.update(corr.index[(corr[column] > CORRELATION_THRESHOLD) & (corr.index != column)].tolist())
    corr_score = pd.Series({c: float(c in keep) for c in cols}, index=cols)
    ranking = pd.DataFrame({
        "feature": cols,
        "correlation_filter_keep": corr_score.reindex(cols).to_numpy(),
        "anova_f": np.nan_to_num(f_values, nan=0.0, posinf=0.0),
        "anova_p": np.nan_to_num(p_values, nan=1.0, posinf=1.0),
        "mutual_information": np.nan_to_num(mi, nan=0.0),
        "random_forest_importance": forest.feature_importances_,
    })
    for metric in ("anova_f", "mutual_information", "random_forest_importance"):
        ranking[f"{metric}_rank"] = ranking[metric].rank(method="average", ascending=False)
    ranking["combined_rank"] = ranking[["anova_f_rank", "mutual_information_rank", "random_forest_importance_rank"]].mean(axis=1)
    ranking = ranking.sort_values(["combined_rank", "feature"]).reset_index(drop=True)
    selected = ranking.loc[ranking["correlation_filter_keep"].eq(1), "feature"].head(max_features).tolist()
    if not selected:
        selected = ranking["feature"].head(min(max_features, len(cols))).tolist()
    return ranking, selected
