import sys, time
from pathlib import Path
import cv2, joblib
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import image_test as it

# ตั้งค่าในไฟล์นี้เลย ไม่ต้องไปแก้ image_test.py
it.N_PROPOSALS = 200
it.MIN_SCORE = 1.0
it.NMS_IOU = 0.3
WIN = "R-CNN realtime"
BGR = [None, (0, 160, 0), (0, 0, 255), (0, 165, 255)]   # เขียว, แดง, ส้ม

m = joblib.load(HERE / "outputs" / "rcnn_head.joblib")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("เปิดกล้องไม่ได้")
print("กด SPACE = ตรวจจับเฟรมนี้ | กด Q หรือ ESC = ออก | หรือกดปุ่ม X ของหน้าต่าง")

def draw(frame, dets, text):
    show = frame.copy()
    for box, c, s in dets:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(show, (x1, y1), (x2, y2), BGR[c], 2)
        cv2.putText(show, f"{it.NAMES[c]} {s:.1f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR[c], 1)
    cv2.putText(show, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return show

dets, info = [], "SPACE = detect, Q = quit"
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))
    cv2.imshow(WIN, draw(frame, dets, info))
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord(" "):
        cv2.imshow(WIN, draw(frame, [], "Detecting... please wait"))
        cv2.waitKey(1)
        t0 = time.time()
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets = it.detect(pil, m["clf"], m["reg"])
        info = f"{len(dets)} objects, {time.time() - t0:.1f}s/frame"
        print(info)

cap.release()
cv2.destroyAllWindows()