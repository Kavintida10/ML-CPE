import sys, time
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CONF = 0.4
WIN = "YOLO26n realtime"

table = pd.read_csv(RUNS / "comparison.csv", encoding="utf-8-sig")
run = table.sort_values("mAP50", ascending=False).iloc[0]
model = YOLO(str(RUNS / run["run"] / "weights" / "best.pt"))
imgsz = int(run["imgsz"])
print("ใช้โมเดล", run["run"], "| กด Q หรือ ESC หรือปุ่ม X เพื่อออก")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("เปิดกล้องไม่ได้")
fps, t_prev = 0.0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.predict(frame, imgsz=imgsz, conf=CONF, device="cpu", verbose=False)[0]
    show = r.plot()
    now = time.time()
    inst = 1 / max(now - t_prev, 1e-6)
    fps = 0.9 * fps + 0.1 * inst if fps else inst
    t_prev = now
    cv2.putText(show, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(WIN, show)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()