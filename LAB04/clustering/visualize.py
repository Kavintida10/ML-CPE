"""
Visualization functions for Clustering outputs
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_elbow(k_values, inertias, out_path):
    """บันทึกกราฟ Elbow Method"""
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, inertias, "o-", color="purple")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_clusters(df, labels, features, out_path):
    """บันทึกกราฟกระจายตัวของแต่ละ Cluster"""
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        df[features[0]], 
        df[features[1]], 
        c=labels, 
        cmap="tab10", 
        alpha=0.75, 
        edgecolors="k", 
        linewidths=0.5
    )
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_cluster_summary(df, labels, features, out_summary_path, out_detail_path):
    """บันทึกผลการจัดกลุ่มลงไฟล์ CSV"""
    df_result = df.copy()
    df_result["Cluster"] = labels
    df_result.to_csv(out_detail_path, index=False, encoding="utf-8-sig")

    # สรุปสถิติเฉลี่ยของแต่ละกลุ่ม
    summary = df_result.groupby("Cluster")[features].agg(["mean", "count"]).reset_index()
    summary.to_csv(out_summary_path, index=False, encoding="utf-8-sig")
    return summary