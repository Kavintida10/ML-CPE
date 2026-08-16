"""
Read CSV and prepare scaled features for Clustering
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "data-Social Network" / "Social_Network_Ads.csv"

if not CSV_PATH.exists():
    CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data-Social Network" / "Social_Network_Ads.csv"

# เลือก Feature หลักสำหรับจัดกลุ่มพฤติกรรมลูกค้า (อายุ และ เงินเดือนโดยประมาณ)
FEATURES = [
    "Age",
    "EstimatedSalary",
]


def load_data():
    # step 1 : read CSV
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    # step 2 : extract numeric features
    X = df[FEATURES].copy().to_numpy(dtype="float32")

    # step 3 : Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")

    return {
        "df": df,
        "X_scaled": X_scaled,
        "features": FEATURES,
        "n_rows": len(df),
    }


if __name__ == "__main__":
    data = load_data()
    print("clustering rows :", data["n_rows"])
    print("X_scaled shape  :", data["X_scaled"].shape)
    print("features        :", data["features"])