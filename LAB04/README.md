# LAB 04: k-Nearest Neighbor (KNN) และการประยุกต์ใช้งาน KNN 

**รายวิชา:** Machine Learning (Sec 2, 1/2569)  
**ผู้จัดทำ:** Kavintida Sukchom (GitHub: [@Kavintida10](https://github.com/Kavintida10)) 

---
โปรเจกต์นี้เป็นการศึกษาและประยุกต์ใช้อัลกอริทึม k-Nearest Neighbor (KNN) ในการจำแนกประเภทข้อมูล (Classification) รวมถึงการทดลองปรับค่าพารามิเตอร์ $k$ (Number of Neighbors) และการวัดระยะทาง (Distance Metrics) เพื่อเปรียบเทียบและประเมินประสิทธิภาพความแม่นยำของแบบจำลอง

---
## Dataset
https://www.kaggle.com/datasets/rakeshrau/social-network-ads?resource=download

---
```text
LAB04-KNN/
├── classification/
│   ├── outputs/
│   │   ├── 01_k_curve.png          # กราฟเปรียบเทียบค่าความแม่นยำ (Accuracy) กับค่า K ต่างๆ
│   │   ├── 02_confusion_matrix.png # เมทริกซ์ความสับสน (Confusion Matrix) ของโมเดลที่ดีที่สุด
│   │   └── predictions.csv         # ไฟล์บันทึกผลการทำนายเทียบกับค่าจริง (Test Set)
│   ├── data_loader.py              # โหลดข้อมูล แปลงค่า Text เป็นตัวเลข และทำ StandardScaler
│   ├── evaluate.py                 # ฟังก์ชันวาดกราฟ K-Curve, Confusion Matrix และ Export รายงาน
│   ├── knn_tf.py                   # ตัวแบบ K-Nearest Neighbors Classifier
│   └── main.py                     # สคริปต์หลักสำหรับเทรนและประเมินผล Classification
│
├── clustering/
│   ├── outputs/
│   │   ├── 01_elbow.png            # กราฟ Elbow Method เพื่อหาจำนวนกลุ่ม (K) ที่เหมาะสม
│   │   ├── 02_clusters.png         # ภาพ 2D Scatter Plot แสดงการจัดกลุ่มลูกค้า
│   │   ├── cluster_summary.csv     # สรุปค่าเฉลี่ยสถิติของแต่ละกลุ่ม (Cluster)
│   │   └── clustered_animals.csv   # ชุดข้อมูลเดิมพร้อมระบุหมายเลข Cluster
│   ├── data_loader.py              # โหลดและเตรียมฟีเจอร์สำหรับงาน Clustering
│   ├── kmeans_tf.py                # ตัวแบบ K-Means Clustering
│   ├── knn_tools.py                # เครื่องมือหาตัวแทนจุดศูนย์กลางของแต่ละกลุ่ม
│   ├── main.py                     # สคริปต์หลักสำหรับรันกระบวนการ Clustering ทั้งหมด
│   └── visualize.py                # ฟังก์ชันสร้างกราฟ Elbow และ Cluster Visualization
│
├── data-Social Network/
│   └── Social_Network_Ads.csv      # ไฟล์ชุดข้อมูล Social Network Ads จาก Kaggle
│
├── requirements.txt                # รายการไลบรารีที่จำเป็นสำหรับโปรเจกต์
└── link-data.txt                   # ลิงก์แหล่งที่มาของชุดข้อมูล

---
## จัดทำโดย
นางสาวกวินธิดา สุขโฉม sec2 116710400602-4
