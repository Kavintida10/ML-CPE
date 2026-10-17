import sys, time
from pathlib import Path
import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import image_test as it

it.MIN_SCORE, it.NMS_IOU = 0.7, 0.3
WIN = "Fast R-CNN realtime"
BGR = [None, (0, 160, 0), (0, 0, 255), (0, 165, 255)]   # เขียว, แดง, ส้ม

model = it.load_model()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("เปิดกล้องไม่ได้")
print("SPACE = ตรวจจับเฟรมนี้ | A = เปิด/ปิดโหมดอัตโนมัติ | Q หรือ ESC = ออก")

def draw(frame, dets, text):
    show = frame.copy()
    for box, c, s in dets:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(show, (x1, y1), (x2, y2), BGR[c], 2)
        cv2.putText(show, f"{it.tr.NAMES[c]} {s:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR[c], 1)
    cv2.putText(show, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return show

def run(frame):
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = it.detect(pil, model)
    return d, f"{len(d)} objects, {time.time() - t0:.1f}s/frame"

dets, info, auto = [], "SPACE = detect, A = auto, Q = quit", False
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))
    cv2.imshow(WIN, draw(frame, dets, ("AUTO | " if auto else "") + info))
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord("a"):
        auto = not auto
    if key == ord(" ") or auto:
        dets, info = run(frame)
        print(info)

cap.release()
cv2.destroyAllWindows()