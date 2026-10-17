import random
from pathlib import Path
import cv2, joblib, numpy as np, torch, torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent / "dataset" / "train"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
N_IMAGES, N_PROPOSALS = 200, 200   # ทดสอบก่อนด้วย N_IMAGES = 20
random.seed(42); np.random.seed(42)

cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
cnn.fc = torch.nn.Identity()
cnn.eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def features(crops):
    return cnn(torch.stack([tf(c) for c in crops])).numpy()

def proposals(bgr):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(bgr); ss.switchToSelectiveSearchFast()
    r = ss.process()[:N_PROPOSALS]
    return np.array([[x, y, x + w, y + h] for x, y, w, h in r], dtype=float)

def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)

def read_gt(path, W, H):
    boxes, cls = [], []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        c, x, y, w, h = line.split()
        x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2]); cls.append(int(c) + 1)
    return np.array(boxes), np.array(cls)   # คลาส 0 = พื้นหลัง

def crop(pil, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    return pil.crop((x1, y1, max(x2, x1 + 2), max(y2, y1 + 2)))

imgs = sorted((ROOT / "images").glob("*"))
random.shuffle(imgs)
X, y, Xr, Tr = [], [], [], []
for i, p in enumerate(imgs[:N_IMAGES]):
    pil = Image.open(p).convert("RGB"); W, H = pil.size
    gt, gcls = read_gt(ROOT / "labels" / f"{p.stem}.txt", W, H)
    if len(gt) == 0: continue
    props = np.vstack([proposals(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)), gt])
    ious = iou(props, gt); best, mx = ious.argmax(1), ious.max(1)
    pos = np.where(mx >= 0.5)[0]; neg = np.where(mx < 0.3)[0]
    pos = np.random.choice(pos, min(len(pos), 8), replace=False)
    neg = np.random.choice(neg, min(len(neg), 24), replace=False)
    sel = np.concatenate([pos, neg]).astype(int)
    f = features([crop(pil, props[j]) for j in sel])
    X.append(f); y += [int(gcls[best[j]]) if mx[j] >= 0.5 else 0 for j in sel]
    for k, j in enumerate(sel[:len(pos)]):          # ตัวแรก ๆ คือ positive
        px1, py1, px2, py2 = props[j]; g = gt[best[j]]
        pw, ph = px2 - px1, py2 - py1
        Xr.append(f[k])
        Tr.append([((g[0] + g[2]) / 2 - (px1 + px2) / 2) / pw, ((g[1] + g[3]) / 2 - (py1 + py2) / 2) / ph,
                   np.log((g[2] - g[0]) / pw), np.log((g[3] - g[1]) / ph)])
    print(f"{i + 1}/{N_IMAGES} {p.name}")

X = np.vstack(X)
clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000).fit(X, y)
reg = Ridge(alpha=1.0).fit(np.array(Xr), np.array(Tr))
joblib.dump({"clf": clf, "reg": reg}, OUT / "rcnn_head.joblib")
print("เสร็จแล้ว บันทึกที่", OUT / "rcnn_head.joblib", "| ตัวอย่างฝึก:", len(y))