"""
Clustering Workflow:
- Load data
- Run Elbow Method to find optimal k
- Fit K-Means Clustering (TensorFlow)
- Visualize clusters and export results to CSV
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np

import data_loader
import visualize
from kmeans_tf import TFKMeans
from knn_tools import find_cluster_representatives

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def title(text):
    print("\n" + "--" * 30)
    print(text)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1 : load data for clustering")
    data = data_loader.load_data()
    df = data["df"]
    X_scaled = data["X_scaled"]
    features = data["features"]

    print(f"data of all : {data['n_rows']} rows")
    print(f"clustering features : {features}")

    title("STEP 2 : Elbow method for finding optimal k")
    k_values = list(range(2, 9))
    inertias = []

    for k in k_values:
        model = TFKMeans(n_clusters=k)
        model.fit(X_scaled)
        inertias.append(model.inertia_)
        print(f"   k = {k}  ->  Inertia (WCSS) = {model.inertia_:.4f}")

    visualize.plot_elbow(k_values, inertias, OUT_DIR / "01_elbow.png")

    title("STEP 3 : Fit final model with optimal k")
    # ค่า k=3 หรือ k=4 เหมาะสมที่สุดตามโครงสร้างประชากร (อายุ vs เงินเดือน)
    optimal_k = 3
    print(f"Fitting K-Means with optimal k = {optimal_k}")

    final_model = TFKMeans(n_clusters=optimal_k)
    final_model.fit(X_scaled)
    labels = final_model.predict(X_scaled)

    title("STEP 4 : Find cluster representatives")
    reps = find_cluster_representatives(X_scaled, labels, optimal_k, n_samples=2)
    for c_id, sample_indices in reps.items():
        print(f"Cluster {c_id} Representative Row IDs: {sample_indices.tolist()}")

    title("STEP 5 : Export plots and CSV files")
    visualize.plot_clusters(df, labels, features, OUT_DIR / "02_clusters.png")
    
    visualize.save_cluster_summary(
        df,
        labels,
        features,
        OUT_DIR / "cluster_summary.csv",
        OUT_DIR / "clustered_animals.csv",
    )

    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")


if __name__ == "__main__":
    main()