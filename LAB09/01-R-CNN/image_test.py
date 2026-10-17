import random, sys
from pathlib import Path
import cv2, joblib, numpy as np, torch, torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
TEST = HERE.parent / "dataset" / "test" / "images"
NAMES = ["background", "with_mask", "without_mask", "mask_weared_incorrect"]
COLORS = [None, "green", "red", "orange"]
N_PROPOSALS, MIN_SCORE, NMS_IOU = 500, -5.0, 0.3
cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
cnn.fc = torch.nn.Identity()
cnn.eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def features(crops):
    return np.vstack([cnn(torch.stack([tf(c) for c in crops[i:i + 32]])).numpy()
                      for i in range(0, len(crops), 32)])

def proposals(bgr):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(bgr); ss.switchToSelectiveSearchFast()
    r = ss.process()[:N_PROPOSALS]
    return np.array([[x, y, x + w, y + h] for x, y, w, h in r], dtype=float)

def crop(pil, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    return pil.crop((x1, y1, max(x2, x1 + 2), max(y2, y1 + 2)))

def detect(pil, clf, reg):
    W, H = pil.size
    props = proposals(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    F = features([crop(pil, b) for b in props])
    S = clf.decision_function(F)
    cls, score = clf.classes_[S.argmax(1)], S.max(1)
    # ปรับกรอบด้วย regression
    t = reg.predict(F)
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    cx = (props[:, 0] + props[:, 2]) / 2 + t[:, 0] * pw
    cy = (props[:, 1] + props[:, 3]) / 2 + t[:, 1] * ph
    w = pw * np.exp(np.clip(t[:, 2], -2, 2)); h = ph * np.exp(np.clip(t[:, 3], -2, 2))
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)
    keep = (cls != 0) & (score > MIN_SCORE)
    if not keep.any():
        return []
    b, s, c = boxes[keep], score[keep], cls[keep]
    k = torchvision.ops.batched_nms(torch.tensor(b).float(), torch.tensor(s).float(),
                                    torch.tensor(c), NMS_IOU).numpy()
    return [(b[i], int(c[i]), float(s[i])) for i in k]

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else random.choice(list(TEST.glob("*")))
    m = joblib.load(HERE / "outputs" / "rcnn_head.joblib")
    pil = Image.open(path).convert("RGB")
    dets = detect(pil, m["clf"], m["reg"])
    d = ImageDraw.Draw(pil)
    for box, c, s in dets:
        d.rectangle(list(box), outline=COLORS[c], width=3)
        d.text((box[0], max(0, box[1] - 12)), f"{NAMES[c]} {s:.2f}", fill=COLORS[c])
        print(NAMES[c], round(s, 2), [int(v) for v in box])
    out = HERE / "outputs" / f"pred_{path.stem}.png"
    pil.save(out)
    print("พบ", len(dets), "วัตถุ | บันทึกที่", out)