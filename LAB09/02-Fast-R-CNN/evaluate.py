import sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import image_test as it
tr = it.tr

TEST = HERE.parent / "dataset" / "test"
N_TEST = None     
IOU_T = 0.5       # เกณฑ์ IoU ว่าตรวจถูก
SCORE_T = 0.5     # ความน่าจะเป็นขั้นต่ำสำหรับคิด Precision/Recall
it.MIN_SCORE = 0.01   # เก็บกรอบทุกคะแนนไว้ เพื่อคำนวณ mAP
NAMES = tr.NAMES

# แคชกรอบเสนอของชุด test เพื่อให้รันซ้ำเร็วขึ้น
CACHE_T = tr.OUT / "proposals_cache" / "test"
_orig, _cur = tr.proposals, {"stem": ""}
tr.proposals = lambda pil, cache_file=None: _orig(pil, CACHE_T / f"{_cur['stem']}.npy")

def read_gt(path, W, H):
    boxes, cls = [], []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        c, x, y, w, h = line.split()
        x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2]); cls.append(int(c) + 1)
    return np.array(boxes), np.array(cls)

def iou1(b, gts):
    x1 = np.maximum(b[0], gts[:, 0]); y1 = np.maximum(b[1], gts[:, 1])
    x2 = np.minimum(b[2], gts[:, 2]); y2 = np.minimum(b[3], gts[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1])
    g = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    return inter / (a + g - inter + 1e-9)

def ap_score(rec, prec):
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

model = it.load_model()
stats = {c: {"s": [], "tp": []} for c in (1, 2, 3)}
n_gt = {c: 0 for c in (1, 2, 3)}
imgs = sorted((TEST / "images").glob("*"))
if N_TEST: imgs = imgs[:N_TEST]
t_start = time.time()

for i, p in enumerate(imgs):
    pil = Image.open(p).convert("RGB"); W, H = pil.size
    gt, gc = read_gt(TEST / "labels" / f"{p.stem}.txt", W, H)
    for c in gc: n_gt[int(c)] += 1
    _cur["stem"] = p.stem
    dets = sorted(it.detect(pil, model), key=lambda d: -d[2])
    used = set()
    for box, c, s in dets:
        idx = np.where(gc == c)[0]
        tp = 0
        if len(idx):
            ious = iou1(box, gt[idx]); j = ious.argmax()
            if ious[j] >= IOU_T and idx[j] not in used:
                used.add(idx[j]); tp = 1
        stats[c]["s"].append(s); stats[c]["tp"].append(tp)
    print(f"{i + 1}/{len(imgs)} {p.name}")

lines, aps, TP, FP, GT = [], [], 0, 0, 0
lines.append(f"Fast R-CNN | IoU={IOU_T} | คะแนนขั้นต่ำสำหรับ P/R={SCORE_T} | จำนวนรูป={len(imgs)}")
for c in (1, 2, 3):
    s, tp = np.array(stats[c]["s"]), np.array(stats[c]["tp"])
    if n_gt[c] == 0:
        lines.append(f"{NAMES[c]}: ไม่มีวัตถุจริงในรูปที่ทดสอบ"); continue
    if len(s):
        o = np.argsort(-s); tpc = np.cumsum(tp[o]); fpc = np.cumsum(1 - tp[o])
        ap = ap_score(tpc / n_gt[c], tpc / (tpc + fpc))
    else:
        ap = 0.0
    k = s >= SCORE_T
    t, f = int(tp[k].sum()), int((1 - tp[k]).sum())
    TP += t; FP += f; GT += n_gt[c]; aps.append(ap)
    prec = t / (t + f) if t + f else 0.0
    lines.append(f"{NAMES[c]}: AP={ap:.3f} | Precision={prec:.3f} | Recall={t / n_gt[c]:.3f} | GT={n_gt[c]}")
lines.append(f"รวม: Precision={TP / (TP + FP) if TP + FP else 0:.3f} | Recall={TP / GT if GT else 0:.3f}")
lines.append(f"mAP@{IOU_T} = {np.mean(aps) if aps else 0:.3f}")
lines.append(f"เวลารวม {time.time() - t_start:.0f} วินาที (รันครั้งแรกรวมเวลา Selective Search ด้วย)")
print("\n".join(lines))
(tr.OUT / "eval_results.txt").write_text("\n".join(lines), encoding="utf-8")