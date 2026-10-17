import random, sys, time
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "outputs"
TEST = HERE / "dataset" / "test" / "images"
CONF = 0.25
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def best_model_path():
    table = pd.read_csv(RUNS / "comparison.csv", encoding="utf-8-sig")
    run = table.sort_values("mAP50", ascending=False).iloc[0]
    print("ใช้โมเดล", run["run"], "(mAP50 สูงสุดบนชุด test =", run["mAP50"], ")")
    return RUNS / run["run"] / "weights" / "best.pt", int(run["imgsz"])


def run_video(model, src, imgsz):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = OUT / f"pred_{src.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=CONF, device="cpu", verbose=False)[0]
        writer.write(r.plot())
        n += 1
        if n % 30 == 0:
            print(f"ประมวลผลแล้ว {n} เฟรม")
    cap.release(); writer.release()
    print(f"เสร็จ {n} เฟรม ใช้เวลา {time.time() - t0:.1f} วินาที | บันทึกที่ {out_path}")


def main():
    OUT.mkdir(exist_ok=True)
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else random.choice(list(TEST.glob("*")))
    weights, imgsz = best_model_path()
    model = YOLO(str(weights))
    if src.suffix.lower() in VID_EXT:
        run_video(model, src, imgsz)
        return
    t0 = time.time()
    r = model.predict(str(src), imgsz=imgsz, conf=CONF, device="cpu", verbose=False)[0]
    print(f"ตรวจจับใช้เวลา {time.time() - t0:.2f} วินาที")
    for b in r.boxes:
        print(r.names[int(b.cls)], round(float(b.conf), 2), [int(v) for v in b.xyxy[0]])
    out = OUT / f"pred_{src.stem}.png"
    cv2.imwrite(str(out), r.plot())
    print("พบ", len(r.boxes), "วัตถุ | บันทึกที่", out)


if __name__ == "__main__":
    main()