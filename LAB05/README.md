# LAB 05: Support Vector Machine และการประยุกต์ใช้งาน SVM

**รายวิชา:** Machine Learning (Sec 2, 1/2569)  
**ผู้จัดทำ:** นางสาวกวินธิดา สุขโฉม (รหัสนักศึกษา: 116710400602-4, Sec 2)  
**GitHub:** [@Kavintida10](https://github.com/Kavintida10)

---

## 📌 บทนำ (Introduction)
โปรเจกต์นี้เป็นการจำแนกรูปภาพสุนัขและแมวโดยใช้แบบจำลอง **Support Vector Machine (SVM)** พร้อมทั้งศึกษาและเปรียบเทียบประสิทธิภาพการทำงานของ Kernel รูปแบบต่าง ๆ ได้แก่ **Linear**, **Polynomial** และ **Radial Basis Function (RBF)** เพื่อประเมินความแม่นยำในการสร้างเส้นแบ่งขอบเขตการตัดสินใจ (Decision Boundary) ผ่านการลดมิติข้อมูลด้วย PCA

---

## 📂 แหล่งที่มาของข้อมูล (Dataset)
* **Dataset:** Cats-And-Dogs-Mini-Dataset  
* **Link:** [Kaggle - Cats-And-Dogs-Mini-Dataset](https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB05-SVM/
├── PetImages/
│   ├── cats_set/                     # Dataset รูปภาพแมว
│   └── dogs_set/                     # Dataset รูปภาพสุนัข
│
├── classification/
│   ├── outputs/                      # โฟลเดอร์เก็บผลลัพธ์จากการรัน
│   │   ├── confusion_matrix_linear.png
│   │   ├── confusion_matrix_poly.png
│   │   ├── confusion_matrix_rbf.png
│   │   ├── prediction_sample.png     # รูปตัวอย่างผลการทำนาย (Pred vs True)
│   │   ├── kernel_comparison.json    # ค่า Accuracy เปรียบเทียบทั้ง 3 kernels
│   │   ├── scaler.pkl                # Pipeline scaler + PCA ที่ fit แล้ว
│   │   ├── svm_model.pkl             # Best SVM model ที่ผ่านการบันทึก
│   │   ├── images.npy                # ข้อมูลรูปภาพในรูปแบบ NumPy array
│   │   ├── labels.npy                # ป้ายกำกับคลาส (Labels)
│   │   ├── X_train.npy               # Feature set สำหรับ Train
│   │   ├── X_test.npy                # Feature set สำหรับ Test
│   │   ├── y_train.npy               # Label สำหรับ Train
│   │   └── y_test.npy                # Label สำหรับ Test
│   │
│   ├── data_loader.py                # ฟังก์ชันโหลดและแปลงรูปภาพ
│   ├── preprocessing.py              # ฟังก์ชันปรับขนาดภาพและแปลงเป็นฟีเจอร์
│   ├── split_data.py                 # แบ่งข้อมูลเป็น Train / Test set
│   ├── svm_model.py                  # นิยามและเทรน SVM kernels (Linear, Poly, RBF)
│   ├── evaluate.py                   # คำนวณ Accuracy, Report และพล็อต Confusion Matrix
│   ├── main.py                       # สคริปต์หลักสำหรับรันกระบวนการทั้งหมด
│   └── test_svm.py                   # สุ่มทดสอบภาพและแสดงผลการพยากรณ์
│
├── requirements.txt                  # รายการ Dependencies
├── link-data.txt                     # ลิงก์ดาวน์โหลด Dataset
└── README.md                         # คำอธิบายโปรเจกต์
