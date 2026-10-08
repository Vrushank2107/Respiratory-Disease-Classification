from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import SEED


def pca_and_cluster(train: pd.DataFrame, feature_names: list[str], max_clusters: int = 10) -> dict:
    x = train[feature_names].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median()).fillna(0)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    pca = PCA(random_state=SEED).fit(z)
    coords = pca.transform(z)
    rows = []
    limit = min(max_clusters, len(x) - 1)
    for k in range(2, max(2, limit) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(z)
        score = silhouette_score(z, model.labels_, sample_size=min(2500, len(z)), random_state=SEED) if len(np.unique(model.labels_)) > 1 else np.nan
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(score), "ari_after_fit": float(adjusted_rand_score(train["diagnosis"], model.labels_))})
    if not rows:
        raise ValueError("At least three training cycles are needed for clustering")
    scores = pd.DataFrame(rows)
    best_k = int(scores.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(z)
    projection = pd.DataFrame(coords[:, :min(3, coords.shape[1])], columns=[f"PC{i+1}" for i in range(min(3, coords.shape[1]))])
    projection["diagnosis"] = train["diagnosis"].astype(str).to_numpy()
    projection["cluster"] = kmeans.labels_
    projection["patient_id"] = train["patient_id"].to_numpy()
    projection["explained_variance_ratio"] = np.nan
    return {"scaler": scaler, "pca": pca, "kmeans": kmeans, "scores": scores, "projection": projection, "best_k": best_k}
