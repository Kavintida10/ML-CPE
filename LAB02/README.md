# LAB 02: Data Preprocessing & Feature Engineering

**รายวิชา:** Machine Learning (04-624-201)  
**ผู้จัดทำ:** นางสาวกวินธิดา สุขโฉม (รหัสนักศึกษา: 116710400602-4, Sec 2)  
**GitHub:** [@Kavintida10](https://github.com/Kavintida10)

---

## 📌 บทนำ (Introduction)
ใบงานนี้ศึกษาและปฏิบัติตามกระบวนการ **Data Preprocessing** และ **Feature Engineering** แบบครบวงจร ซึ่งเป็นขั้นตอนรากฐานสำคัญก่อนนำข้อมูลไปพัฒนาแบบจำลอง Machine Learning โดยใช้ชุดข้อมูล **Doraemon Complete Data** เพื่อวิเคราะห์โครงสร้าง ทำความเข้าใจการกระจายตัวของตัวละคร จัดการกับค่าสูญหายและความผิดปกติ พร้อมทั้งแปลงข้อมูลหมวดหมู่ (Categorical Data) ให้อยู่ในรูปตัวเลขที่พร้อมสำหรับการเทรนโมเดล

---

## 📂 แหล่งที่มาของข้อมูล (Dataset)
* **Dataset:** Doraemon Complete Data (`character_list.csv`)
* **Source:** [Kaggle - Doraemon Complete Data](https://www.kaggle.com/datasets/samasiayushman/doraemon-complete-data)
* **ขนาดข้อมูลเริ่มต้น:** 100 แถว, 20 คอลัมน์

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
ML-LAB02/
├── character_list.csv              # ไฟล์ชุดข้อมูลตัวละครโดราเอมอนจาก Kaggle
├── ML-LAB-02.ipynb                 # Jupyter Notebook บันทึกขั้นตอนการทดลองและโค้ดทั้งหมด
├── link-data.txt                   # ลิงก์แหล่งที่มาของชุดข้อมูล
└── README.md                       # รายงานสรุปผลการทดลอง
