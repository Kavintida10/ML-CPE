# LAB 03: Regression & Classification 

**รายวิชา:** Machine Learning (04-624-201)  
**ผู้จัดทำ:** นางสาวกวินธิดา สุขโฉม (รหัสนักศึกษา: 116710400602-4, Sec 2)  
**GitHub:** [@Kavintida10](https://github.com/Kavintida10)

---

## 📌 บทนำ (Introduction)
ใบงานนี้ศึกษาและประยุกต์ใช้แบบจำลอง Machine Learning ใน 2 โจทย์หลัก คือ **Regression (การทำนายอายุ)** และ **Classification (การจำแนกเพศ)** โดยใช้ชุดข้อมูลรูปภาพใบหน้า **UTKFace (Aligned & Cropped)** ผ่านกระบวนการเตรียมข้อมูล สกัดฟีเจอร์ด้วยความสว่าง (Brightness) และการลดมิติข้อมูลพิกเซลด้วย **Principal Component Analysis (PCA)** พร้อมทั้งวิเคราะห์ปัญหา Overfitting/Underfitting และประเมินประสิทธิภาพของโมเดลอย่างละเอียด

---

## 📂 แหล่งที่มาของข้อมูล (Dataset)
* **Dataset:** UTKFace (Aligned & Cropped Face Dataset)
* **Link:** [Kaggle - UTKFace Cropped](https://www.kaggle.com/datasets/moritzm00/utkface-cropped)
* **จำนวนรูปภาพทั้งหมด:** 23,708 ภาพ (สุ่มตัวอย่าง 3,000 ภาพสำหรับการทดลอง)
* **รูปแบบการตั้งชื่อไฟล์:** `[age]_[gender]_[race]_[date].jpg.chip.jpg`
  * `age`: อายุของบุคคลในภาพ (Target สำหรับ Regression)
  * `gender`: เพศของบุคคลในภาพ (0 = ชาย, 1 = หญิง สำหรับ Classification)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB03/
├── LAB03.ipynb                 # Jupyter Notebook บันทึกขั้นตอนการทดลองและโค้ดทั้งหมด
└── README.md                       # รายงานสรุปผลการทดลอง
