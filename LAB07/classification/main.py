import os
import pandas as pd
from sklearn.metrics import accuracy_score
from data_loader import load_data
from preprocessing import preprocess_features
from split_data import split_and_save_data
from cnn_model import build_cnn_model
from evaluate import evaluate_and_plot
from test_cnn import test_random_samples

def main():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. โหลดข้อมูล
    print("1. Loading dataset...")
    X_raw, y, categories = load_data(data_dir="../PetImages", img_size=(64, 64))
    print(f"Total loaded: {len(X_raw)} images.")
    
    # 2. Preprocessing (รักษา shape 4D สำหรับ CNN: [N, 64, 64, 3])
    print("2. Preprocessing...")
    X = preprocess_features(X_raw)
    
    # 3. แบ่งชุดข้อมูลและบันทึกไฟล์ .npy ลง outputs/
    print("3. Splitting and saving data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save_data(X, y, categories, output_dir)
    
    # 4. สร้างและเทรน Primary CNN Model
    print("4. Training primary CNN model...")
    main_model = build_cnn_model(input_shape=(64, 64, 3), num_conv_blocks=2, dense_units=64)
    history = main_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # บันทึก Model เป็น .keras ตามข้อกำหนดของอาจารย์
    main_model.save(os.path.join(output_dir, "cnn_model.keras"))
    
    # 5. ประเมินผลและสร้างกราฟรายงาน
    print("5. Evaluating primary model...")
    evaluate_and_plot(main_model, history, X_test, y_test, categories, output_dir)
    test_random_samples(main_model, X_test, y_test, categories, output_dir)
    
    # -------------------------------------------------------------
    # 6. เปรียบเทียบผลลัพธ์ (Epochs & Configurations)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("Comparing Epochs (5, 10, 20)")
    print("="*50)
    epoch_results = []
    for ep in [5, 10, 20]:
        m = build_cnn_model(input_shape=(64, 64, 3), num_conv_blocks=2, dense_units=64)
        m.fit(X_train, y_train, epochs=ep, batch_size=32, verbose=0)
        y_pred = (m.predict(X_test) > 0.5).astype(int).reshape(-1)
        acc = accuracy_score(y_test, y_pred)
        epoch_results.append({"Epochs": ep, "Test Accuracy": round(acc, 4)})
    print(pd.DataFrame(epoch_results).to_string(index=False))
    
    print("\n" + "="*50)
    print("Comparing Configurations")
    print("="*50)
    config_results = []
    configs = [
        {"Name": "Config 1 (1 Conv Block, 32 Dense)", "blocks": 1, "dense": 32},
        {"Name": "Config 2 (2 Conv Blocks, 64 Dense)", "blocks": 2, "dense": 64},
        {"Name": "Config 3 (3 Conv Blocks, 128 Dense)", "blocks": 3, "dense": 128}
    ]
    for cfg in configs:
        m = build_cnn_model(input_shape=(64, 64, 3), num_conv_blocks=cfg["blocks"], dense_units=cfg["dense"])
        m.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        y_pred = (m.predict(X_test) > 0.5).astype(int).reshape(-1)
        acc = accuracy_score(y_test, y_pred)
        config_results.append({"Configuration": cfg["Name"], "Test Accuracy": round(acc, 4)})
    print(pd.DataFrame(config_results).to_string(index=False))
    
    print(f"\nDone! All required files created in: {output_dir}/")

if __name__ == "__main__":
    main()
    