import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from data_loader import load_data
from preprocessing import preprocess_features
from split_data import split_and_save_data
from nn_model import build_nn_model
from evaluate import evaluate_and_plot
from test_nn import test_random_samples

def main():
    output_dir = "outputs"
    
    # 1. โหลดข้อมูล
    print("1. Loading dataset...")
    X_raw, y, categories = load_data(data_dir="../PetImages", img_size=(64, 64))
    
    # 2. Preprocessing & Reshape
    print("2. Preprocessing...")
    X = preprocess_features(X_raw)
    X = X.reshape(X.shape[0], -1)
    
    # 3. แบ่งชุดข้อมูลและเซฟไฟล์ .npy ทั้งหมดลง outputs/
    print("3. Splitting and saving data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save_data(X, y, categories, output_dir)
    
    # 4. โมเดลหลัก (Hidden Units: 128, 64 / Epochs: 30)
    print("4. Training primary model...")
    main_model = build_nn_model(hidden_units=(128, 64), max_iter=30)
    main_model.fit(X_train, y_train)
    
    # เซฟ Model ลง outputs/
    joblib.dump(main_model, os.path.join(output_dir, "nn_model.pkl"))
    
    # 5. ประเมินผลและเซฟกราฟ/รายงานทั้งหมดลง outputs/
    print("5. Evaluating primary model...")
    evaluate_and_plot(main_model, None, X_test, y_test, categories, output_dir)
    test_random_samples(main_model, X_test, y_test, categories, output_dir)
    
    # -------------------------------------------------------------
    # 6. เปรียบเทียบผลเพื่อนำตัวเลขไปเขียนสรุปในเล่มรายงาน
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("Comparing Epochs (10, 30, 50)")
    print("="*50)
    epoch_results = []
    for ep in [10, 30, 50]:
        m = build_nn_model(hidden_units=(128, 64), max_iter=ep)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        epoch_results.append({"Epochs": ep, "Test Accuracy": round(acc, 4)})
    print(pd.DataFrame(epoch_results).to_string(index=False))
    
    print("\n" + "="*50)
    print("Comparing Configurations")
    print("="*50)
    config_results = []
    configs = [
        {"Name": "Config 1 (1 Layer: 64)", "layers": (64,)},
        {"Name": "Config 2 (2 Layers: 128, 64)", "layers": (128, 64)},
        {"Name": "Config 3 (3 Layers: 256, 128, 64)", "layers": (256, 128, 64)}
    ]
    for cfg in configs:
        m = build_nn_model(hidden_units=cfg["layers"], max_iter=30)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        config_results.append({"Configuration": cfg["Name"], "Test Accuracy": round(acc, 4)})
    print(pd.DataFrame(config_results).to_string(index=False))
    
    print(f"\nDone! All required files created in: {output_dir}/")

if __name__ == "__main__":
    main()
