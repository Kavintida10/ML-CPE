# LAB 06: Neural Network และการประยุกต์ใช้งาน 

**รายวิชา:** Machine Learning (Sec 2, 1/2569)  
**ผู้จัดทำ:** นางสาวกวินธิดา สุขโฉม (รหัสนักศึกษา: 116710400602-4, Sec 2)  
**GitHub:** [@Kavintida10](https://github.com/Kavintida10)

---

## 📌 บทนำ (Introduction)
โครงงานนี้เป็นส่วนหนึ่งของปฏิบัติการวิชา Machine Learning เพื่อศึกษาการทำงานของโครงข่ายประสาทเทียม (Neural Network: Fully Connected / Multi-Layer Perceptron) ในการจำแนกประเภทรูปภาพ (Image Classification) ระหว่างสุนัข (Dog) และแมว (Cat) พร้อมทั้งศึกษาเปรียบเทียบผลกระทบของการปรับแต่งโครงสร้างโมเดล (Configurations) และจำนวนรอบการเรียนรู้ (Epochs/Iterations) ต่อประสิทธิภาพของโมเดล

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
LAB06-NN/
├── PetImages/
│   ├── Cat/                           # โฟลเดอร์เก็บรูปภาพแมว (500 ไฟล์)
│   └── Dog/                           # โฟลเดอร์เก็บรูปภาพสุนัข (500 ไฟล์)
├── classification/
│   ├── data_loader.py                 # โหลดภาพจาก PetImages และข้ามไฟล์ที่เสียหาย
│   ├── preprocessing.py               # ปรับสเกลข้อมูล (Normalize) ค่าพิกเซล ให้อยู่ในช่วง 0-1
│   ├── split_data.py                  # แบ่งชุดข้อมูลเป็น Train, Val, Test พร้อมบันทึกไฟล์ .npy
│   ├── nn_model.py                    # สร้างและกำหนดพารามิเตอร์ของโมเดล Neural Network
│   ├── evaluate.py                    # ประเมินผลความแม่นยำและสร้างกราฟ/ภาพสรุปผล
│   ├── test_nn.py                     # สุ่มเลือกภาพจาก Test Set มาทดสอบและพล็อตผล
│   ├── main.py                        # สคริปต์หลักสำหรับรัน Training Pipeline
│   └── outputs/
│       ├── classes.json               # บันทึกรายชื่อคลาส (Cat, Dog)
│       ├── confusion_matrix.png       # แผนภาพ Confusion Matrix
│       ├── features.npy               # ข้อมูลภาพทั้งหมดหลังผ่าน Preprocessing
│       ├── history.json               # ประวัติค่า Loss ในแต่ละ Iteration
│       ├── labels.npy                 # ป้ายกำกับ (Labels) ทั้งหมด
│       ├── nn_model.pkl               # ไฟล์โมเดล Neural Network ที่ฝึกสอนแล้ว
│       ├── prediction_sample.png      # ภาพตัวอย่างผลการทำนาย 4 รูป
│       ├── training_history.png       # กราฟแสดงแนวโน้ม Training Loss
│       ├── X_train.npy                # ข้อมูล Feature สำหรับ Training Set
│       ├── X_val.npy                  # ข้อมูล Feature สำหรับ Validation Set
│       ├── X_test.npy                 # ข้อมูล Feature สำหรับ Test Set
│       ├── y_train.npy                # ข้อมูล Label สำหรับ Training Set
│       ├── y_val.npy                  # ข้อมูล Label สำหรับ Validation Set
│       └── y_test.npy                 # ข้อมูล Label สำหรับ Test Set
└── requirements.txt                   # รายการไลบรารีและแพ็กเกจที่ต้องใช้งาน
