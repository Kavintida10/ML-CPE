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
## 📊 สรุปผลการทดลอง (Experimental Results)

### 1. การเปรียบเทียบโครงสร้างสถาปัตยกรรมโมเดล (Model Configurations)
ผลการทดสอบประสิทธิภาพของโมเดล CNN แต่ละโครงสร้างบนชุดข้อมูลทดสอบ (Test Set) หลังเสร็จสิ้นการฝึกสอน 15 Epochs:

| Configuration | สถาปัตยกรรมเครือข่าย | Conv Blocks | Dense Units | Test Accuracy | ผลการประเมิน / ข้อสังเกต |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Config 1** | Simple CNN | 1 Block | 32 Units | 0.5000[cite: 1] | โครงสร้างตื้นเกินไป ไม่สามารถดึง Feature สำคัญได้เพียงพอ มีประสิทธิภาพเทียบเท่าการเดาสุ่ม (Underfitting) |
| **Config 2 (Best)** | Moderate CNN | 2 Blocks | 64 Units | **0.6933**[cite: 1] | **ประสิทธิภาพดีที่สุด** มีระดับความซับซ้อนที่สมดุลกับปริมาณข้อมูล สามารถเรียนรู้รูปแบบลักษณะภาพได้ดีที่สุด |
| **Config 3** | Deep CNN | 3 Blocks | 128 Units | 0.6333[cite: 1] | มีจำนวนพารามิเตอร์มากเกินไปสำหรับชุดข้อมูลขนาดเล็ก ส่งผลให้เกิดการจำข้อมูล (Overfitting) และความแม่นยำลดลง |

---

### 2. ผลการเรียนรู้และแนวโน้มการลู่เข้า (Training & Validation Performance)
จากผลการฝึกสอนของโมเดลที่ดีที่สุด (Config 2) ตลอด 15 Epochs พบพฤติกรรมการเรียนรู้ดังนี้:

* **Model Accuracy:** ค่าความแม่นยำบนชุดฝึกสอน (Training Accuracy) เพิ่มขึ้นอย่างต่อเนื่องจนถึง **97.00%**[cite: 1] ขณะที่ความแม่นยำบนชุดตรวจสอบ (Validation Accuracy) สูงสุดอยู่ที่ **72.67%**[cite: 1]
* **Model Loss:** ค่าฟังก์ชันต้นทุนบนชุดฝึกสอน (Training Loss) ลดลงอย่างต่อเนื่องจนเหลือ **0.1153**[cite: 1] ในขณะที่ Validation Loss มีแนวโน้มดีดตัวขึ้นอยู่ที่ **0.8112** หลังผ่านรอบการฝึกสอนช่วงกลาง[cite: 1]
* **ปัญหา Overfitting:** กราฟแสดงช่องว่างความแม่นยำที่ห่างกันอย่างเห็นได้ชัดระหว่างชุด Train และ Validation รวมทั้งค่า Validation Loss ที่เริ่มปรับตัวสูงขึ้น ชี้ให้เห็นว่าโมเดลเริ่มจดจำชุดข้อมูลฝึกสอนมากกว่าการสรุปความรู้ทั่วไป (Generalization)[cite: 1]

---

### 3. ตัวอย่างการทดสอบเชิงคุณภาพ (Prediction Samples)
* การทดสอบพยากรณ์ภาพตัวอย่างแบบสมดุล (แมว 2 ภาพ และ สุนัข 2 ภาพ) โมเดลสามารถจำแนกได้อย่างถูกต้องครบถ้วน (**4/4 correct**)[cite: 5]
* ค่าความมั่นใจในการจำแนกประเภท (Confidence Score) อยู่ในระดับสูงมาก:
  * Cat ตัวที่ 1: ความมั่นใจ **94%**[cite: 5]
  * Cat ตัวที่ 2: ความมั่นใจ **97%**[cite: 5]
  * Dog ตัวที่ 1: ความมั่นใจ **100%**[cite: 5]
  * Dog ตัวที่ 2: ความมั่นใจ **100%**[cite: 5]
* การแสดงผลผ่านการเกลี่ยพิกเซลแบบ Bicubic Interpolation ช่วยให้มองเห็นรายละเอียดของภาพขนาด $64 \times 64$ พิกเซล ได้ชัดเจนและราบเรียบขึ้น[cite: 5]
