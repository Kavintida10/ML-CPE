# LAB 07: Convolutional Neural Network (CNN) และการประยุกต์ใช้งาน

**รายวิชา:** Machine Learning (Sec 2, 1/2569)  
**ผู้จัดทำ:** นางสาวกวินธิดา สุขโฉม (รหัสนักศึกษา: 116710400602-4, Sec 2)  
**GitHub:** [@Kavintida10](https://github.com/Kavintida10)

---

## 📌 บทนำ (Introduction)
โครงงานนี้เป็นส่วนหนึ่งของปฏิบัติการวิชา Machine Learning เพื่อศึกษาการทำงานของโครงข่ายประสาทแบบคอนโวลูชัน (Convolutional Neural Network: CNN) ในการจำแนกประเภทรูปภาพ (Image Classification) ระหว่างสุนัข (Dog) และแมว (Cat) โดยมุ่งเน้นการดึงลักษณะเด่นของภาพ (Feature Extraction) ผ่าน Convolutional Layers และ Pooling Layers ควบคู่กับการทดลองเปรียบเทียบโครงสร้างโมเดลในรูปแบบต่าง ๆ (Configurations) ตลอดจนการสรุปผลการประเมินและการแสดงผลตามมาตรฐานงานวิจัยวิชาการ (IEEE Style)

---

## 📁 แหล่งที่มาของข้อมูล (Dataset)
* **Dataset:** Cats-And-Dogs-Mini-Dataset
* **Link:** [Kaggle - Cats-And-Dogs-Mini-Dataset](https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset)
* **จำนวนข้อมูล:** ทั้งหมด 1,000 ภาพ (Cat: 500 ภาพ, Dog: 500 ภาพ)
* **ขนาดข้อมูล:** ภาพสี RGB ปรับขนาดเป็น 64 × 64 พิกเซล ($64 \times 64 \times 3 = 12,288$ features)
* **การแบ่งชุดข้อมูล (Data Split):**
  * Training Set: 70% (700 ภาพ)
  * Validation Set: 15% (150 ภาพ)
  * Test Set: 15% (150 ภาพ)
---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB07-CNN/
├── PetImages/
│   ├── Cat/                            # โฟลเดอร์เก็บรูปภาพแมว (500 ไฟล์)
│   └── Dog/                            # โฟลเดอร์เก็บรูปภาพสุนัข (500 ไฟล์)
├── classification/
│   ├── data_loader.py                  # โหลดภาพจาก PetImages, ปรับขนาดเป็น 64x64 และจัดการไฟล์ที่เสียหาย
│   ├── preprocessing.py                # ปรับสเกลข้อมูล (Normalize) ค่าพิกเซลให้อยู่ในช่วง 0-1
│   ├── split_data.py                   # แบ่งชุดข้อมูลเป็น Train, Val, Test พร้อมบันทึกไฟล์ .npy
│   ├── cnn_model.py                    # ออกแบบโครงสร้างโมเดล CNN (Conv2D, Pooling, Dense Layers)
│   ├── evaluate.py                     # ประเมินผลความแม่นยำ สร้าง Confusion Matrix และกราฟ IEEE Style
│   ├── test_cnn.py                     # สุ่มเลือกภาพจาก Test Set (แมว 2, หมา 2) ทดสอบและพล็อตผล
│   ├── main.py                         # สคริปต์หลักสำหรับรัน Training Pipeline และเปรียบเทียบ Configs
│   ├── __pycache__/                    # โฟลเดอร์แคชไพธอน
│   └── outputs/
│       ├── confusion_matrix.png        # แผนภาพ Confusion Matrix แสดงความถูกต้องรายคลาส
│       ├── prediction_sample.png       # ภาพตัวอย่างผลการทำนาย 4 รูป (แมว 2, หมา 2) พร้อมค่าความมั่นใจ
│       ├── training_history.png        # กราฟแสดงแนวโน้ม Accuracy & Loss สไตล์เปเปอร์วิชาการ IEEE
└── requirements.txt                    # รายการไลบรารีและแพ็กเกจที่ต้องใช้งาน (TensorFlow, OpenCV ฯลฯ)
```

