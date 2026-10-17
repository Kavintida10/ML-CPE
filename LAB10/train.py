import csv, time
from pathlib import Path
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RUNS = HERE / "runs"
QUICK_TEST = False    

GPU = torch.cuda.is_available()
DEVICE = 0 if GPU else "cpu"

# เปลี่ยนทีละตัวแปรเทียบกับค่าตั้งต้น e50_s640_b16
# (ชื่อการทดลอง, epochs, imgsz, batch)
FULL = [
    ("e10_s640_b16", 10, 640, 16),    # เทียบจำนวน epoch
    ("e30_s640_b16", 30, 640, 16),
    ("e50_s640_b16", 50, 640, 16),    # ค่าตั้งต้น
    ("e100_s640_b16", 100, 640, 16),
    ("e50_s320_b16", 50, 320, 16),    # เทียบขนาดภาพ
    ("e50_s640_b8", 50, 640, 8),      # เทียบ batch size
    ("e50_s640_b32", 50, 640, 32),
]
LIGHT = [                          
    ("e5_s320_b16", 5, 320, 16),      # เทียบจำนวน epoch
    ("e10_s320_b16", 10, 320, 16),    # ค่าตั้งต้น
    ("e20_s320_b16", 20, 320, 16),
    ("e10_s416_b16", 10, 416, 16),    # เทียบขนาดภาพ
    ("e10_s320_b8", 10, 320, 8),      # เทียบ batch size

]


def make_yaml():
    p = HERE / "data_abs.yaml"        # ใช้ path เต็ม กันหาโฟลเดอร์ผิดที่
    p.write_text(
        f"path: {DATASET.as_posix()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        "nc: 3\nnames: ['with_mask', 'without_mask', 'mask_weared_incorrect']\n",
        encoding="utf-8")
    return p


def main():
    data = make_yaml()
    exps = FULL if GPU else LIGHT
    if QUICK_TEST:
        exps = [("quick_test", 1, exps[0][2], exps[0][3])]
    RUNS.mkdir(exist_ok=True)
    print("อุปกรณ์:", "GPU" if GPU else "CPU", "| จำนวนการทดลอง:", len(exps))
    for name, epochs, imgsz, batch in exps:
        done = RUNS / name / "done.txt"
        if done.exists() and not QUICK_TEST:
            print("ข้าม", name, "(ฝึกเสร็จแล้ว)")
            continue
        model = YOLO("yolo26n.pt")
        t0 = time.time()
        model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
                    device=DEVICE, workers=8 if GPU else 2, seed=42, cache=True,
                    project=str(RUNS), name=name, exist_ok=True)
        sec = time.time() - t0
        done.write_text(f"{sec:.0f}")
        log = RUNS / "train_times.csv"
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["name", "epochs", "imgsz", "batch", "device", "seconds"])
            w.writerow([name, epochs, imgsz, batch, "GPU" if GPU else "CPU", round(sec)])
        print(f"== {name}: ใช้เวลา {sec / 60:.1f} นาที ==")


if __name__ == "__main__":
    main()