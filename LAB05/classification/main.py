import json
import os

import joblib
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

# PetImages sits one level up from this file (see project structure in README)
DATA_PATH = "../PetImages"
OUTPUT_DIR = "outputs"
IMG_SIZE = 100
TEST_SIZE = 0.2
# Mini-dataset only has ~500 images per class, so None just loads all of them.
# Bump this down (e.g. 300) first if you swap in the full ~12k/class dataset.
MAX_PER_CLASS = None
KERNELS = ["linear", "poly", "rbf"]


def main():

    print("--" * 30)
    print("SVM Image Recognition: Cat vs Dog")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] preprocess images...")

    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Step 4-6: Train + predict + evaluate for EACH kernel, so we can
    # compare Linear vs Poly vs RBF like the assignment asks for.
    results = {}

    for kernel in KERNELS:
        print("\n" + "=" * 60)
        print(f"[Kernel: {kernel}] Training SVM...")
        model, scaler = train_svm(X_train, y_train, kernel=kernel)

        print(f"[Kernel: {kernel}] Testing model...")
        predictions = predict_svm(model, scaler, X_test)

        print(f"[Kernel: {kernel}] Evaluating model...")
        accuracy = evaluate_model(
            y_test, predictions, classes,
            save_path=f"{OUTPUT_DIR}/confusion_matrix_{kernel}.png"
        )

        results[kernel] = {
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
        }

    # Step 7: Compare kernels and keep the best one as the "official" model
    print("\n" + "=" * 60)
    print("Kernel comparison:")
    for kernel in KERNELS:
        print(f"  {kernel:<8}: {results[kernel]['accuracy'] * 100:.2f}%")

    best_kernel = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest kernel: {best_kernel} "
          f"({results[best_kernel]['accuracy'] * 100:.2f}%)")

    joblib.dump(results[best_kernel]["model"], f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(results[best_kernel]["scaler"], f"{OUTPUT_DIR}/scaler.pkl")

    with open(f"{OUTPUT_DIR}/kernel_comparison.json", "w") as f:
        json.dump(
            {k: results[k]["accuracy"] for k in KERNELS} | {"best": best_kernel},
            f, indent=2
        )

    print(f"\nSaved best model ({best_kernel}) as {OUTPUT_DIR}/svm_model.pkl")


if __name__ == "__main__":
    main()