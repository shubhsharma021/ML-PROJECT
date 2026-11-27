"""
Clustering Indian States by School Infrastructure (UDISE 2022-23)
Run from repository root: python src/run_analysis.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "UDISE_2022_23_Table_2.5_0.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df[df["India/State/UT"].str.strip().str.lower() != "india"].copy().reset_index(drop=True)
total = df["Total Schools"].astype(float)

# Four dimensions. Toilet coverage is the mean of functional girls' and boys' toilet rates.
features = pd.DataFrame({"State_UT": df["India/State/UT"].values})
features["Functional_Girls_Toilet_pct"] = (
    df["Number of School having - Functional Girls' Toilet"] / total * 100
).clip(0, 100)
features["Functional_Boys_Toilet_pct"] = (
    df["Number of School having - Functional Boys' Toilet"] / total * 100
).clip(0, 100)
features["Toilet_Coverage_pct"] = features[
    ["Functional_Girls_Toilet_pct", "Functional_Boys_Toilet_pct"]
].mean(axis=1)
features["Functional_Electricity_pct"] = (
    df["Number of School having - Functional Electricity"] / total * 100
).clip(0, 100)
features["Functional_Computer_pct"] = (
    df["Number of School having - Functional Computer facility for pedagogical purposes"]
    / total * 100
).clip(0, 100)
features["Internet_pct"] = (
    df["Number of School having - Internet Facility"] / total * 100
)
features["Total_Schools"] = total.values

MODEL_COLS = [
    "Toilet_Coverage_pct",
    "Functional_Electricity_pct",
    "Functional_Computer_pct",
    "Internet_pct",
]
X = features[MODEL_COLS].fillna(features[MODEL_COLS].median())
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Choose k from 2..6 by the maximum silhouette score.
scores = []
fitted = {}
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = model.fit_predict(Xs)
    fitted[k] = (model, labels)
    scores.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(Xs, labels),
        "davies_bouldin": davies_bouldin_score(Xs, labels),
        "calinski_harabasz": calinski_harabasz_score(Xs, labels),
    })
metrics = pd.DataFrame(scores)
best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
kmeans, labels = fitted[best_k]
features["Cluster_raw"] = labels

# Order cluster IDs by their average four-feature profile (low to high).
means_raw = features.groupby("Cluster_raw")[MODEL_COLS].mean().mean(axis=1).sort_values()
order = {old: new + 1 for new, old in enumerate(means_raw.index)}
features["Cluster"] = features["Cluster_raw"].map(order)
features["Cluster_Label"] = features["Cluster"].map(lambda x: f"Cluster {x}")

features["Infrastructure_Index_pct"] = features[MODEL_COLS].mean(axis=1)
features.to_csv(RESULTS / "state_cluster_assignments.csv", index=False)
metrics.to_csv(RESULTS / "k_selection_metrics.csv", index=False)
summary = features.groupby("Cluster_Label")[MODEL_COLS].mean()
summary.to_csv(RESULTS / "cluster_profile.csv")

# Compare algorithms at the selected k.
agg_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(Xs)
db_labels = DBSCAN(eps=1.25, min_samples=3).fit_predict(Xs)
comparison = pd.DataFrame([
    {"Method": "K-Means", "Silhouette": silhouette_score(Xs, labels)},
    {"Method": "Agglomerative (Ward)", "Silhouette": silhouette_score(Xs, agg_labels)},
])
comparison.to_csv(RESULTS / "algorithm_comparison.csv", index=False)
pd.DataFrame([{
    "DBSCAN_clusters_excluding_noise": len(set(db_labels) - {-1}),
    "DBSCAN_noise_points": int((db_labels == -1).sum()),
}]).to_csv(RESULTS / "dbscan_exploration.csv", index=False)

# Figures
plt.figure(figsize=(7, 4.5))
plt.plot(metrics.k, metrics.inertia, marker="o")
plt.xlabel("Number of clusters (k)"); plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve"); plt.grid(alpha=.25); plt.tight_layout()
plt.savefig(FIGURES / "elbow_curve.png", dpi=180); plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(metrics.k, metrics.silhouette, marker="o")
plt.xlabel("Number of clusters (k)"); plt.ylabel("Silhouette score")
plt.title("Silhouette Score by Number of Clusters"); plt.grid(alpha=.25); plt.tight_layout()
plt.savefig(FIGURES / "silhouette_scores.png", dpi=180); plt.close()

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(Xs)
features["PC1"], features["PC2"] = coords[:, 0], coords[:, 1]
plt.figure(figsize=(8, 6))
for cluster in sorted(features.Cluster.unique()):
    group = features[features.Cluster == cluster]
    plt.scatter(group.PC1, group.PC2, s=55, label=f"Cluster {cluster}")
    for _, row in group.iterrows():
        plt.annotate(row.State_UT, (row.PC1, row.PC2), fontsize=6, alpha=.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("State/UT Clusters (PCA Projection)")
plt.legend(); plt.grid(alpha=.2); plt.tight_layout()
plt.savefig(FIGURES / "pca_cluster_plot.png", dpi=200); plt.close()

plt.figure(figsize=(8, 4.8))
plt.imshow(summary.values, aspect="auto", vmin=0, vmax=100)
plt.colorbar(label="Mean facility coverage (%)")
plt.xticks(range(4), ["Toilets", "Electricity", "Computers", "Internet"], rotation=20)
plt.yticks(range(len(summary.index)), summary.index)
plt.title("Average Infrastructure Profile by Cluster")
for i in range(summary.shape[0]):
    for j in range(summary.shape[1]):
        plt.text(j, i, f"{summary.iloc[i,j]:.1f}", ha="center", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIGURES / "cluster_profile_heatmap.png", dpi=200); plt.close()

ranked = features.sort_values("Infrastructure_Index_pct", ascending=True)
plt.figure(figsize=(9, 9))
plt.barh(ranked.State_UT, ranked.Infrastructure_Index_pct)
plt.xlabel("Mean coverage across four dimensions (%)")
plt.title("School Infrastructure Coverage Index by State/UT")
plt.tight_layout(); plt.savefig(FIGURES / "state_infrastructure_index.png", dpi=180); plt.close()

print(f"Analysed {len(features)} States/UTs; selected k={best_k}.")
print(metrics.round(3).to_string(index=False))
print("\\nCluster profile (%):\\n", summary.round(2).to_string())
print("\\nCluster sizes:\\n", features.Cluster_Label.value_counts().sort_index().to_string())
print("\\nResults saved under results/ and figures/.")
