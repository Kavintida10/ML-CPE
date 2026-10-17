import random, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.ops import roi_pool, box_iou
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "dataset"
OUT = HERE / "outputs"
CACHE = OUT / "proposals_cache"
NAMES = ["background", "with_mask", "without_mask", "mask_weared_incorrect"]
N_IMAGES, EPOCHS = 200, 10      
N_PROPOSALS, STRIDE = 200, 16
TSTD = torch.tensor([0.1, 0.1, 0.2, 0.2])   # ตัวปรับสเกลของค่าชดเชยกรอบ
prep = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class FastRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        for p in self.backbone.parameters():
            p.requires_grad = False                 # ล็อก CNN ไว้ เพราะรันบน CPU
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5))
        self.cls = nn.Linear(512, 4)                # จำแนกคลาส (รวมพื้นหลัง)
        self.reg = nn.Linear(512, 4)                # ปรับกรอบ

    def features(self, x):                          # รูปทั้งใบเข้า CNN ครั้งเดียว
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone(x)

    def head(self, fmap, boxes):                    # RoI Pooling แล้วเข้า FC
        x = roi_pool(fmap, [boxes], output_size=(7, 7), spatial_scale=1.0 / STRIDE)
        h = self.fc(x)
        return self.cls(h), self.reg(h)


def to_tensor(pil):
    return prep(pil).unsqueeze(0)


def proposals(pil, cache_file=None):
    if cache_file is not None and cache_file.exists():
        return np.load(cache_file)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    ss.switchToSelectiveSearchFast()
    r = ss.process()[:N_PROPOSALS]
    b = np.array([[x, y, x + w, y + h] for x, y, w, h in r], dtype=np.float32)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, b)
    return b


def read_gt(path, W, H):
    boxes, cls = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = line.split()
        x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2]); cls.append(int(c) + 1)
    return np.array(boxes, dtype=np.float32), np.array(cls, dtype=np.int64)


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    OUT.mkdir(exist_ok=True)
    split = ROOT / "train"
    imgs = sorted((split / "images").glob("*"))
    random.shuffle(imgs)
    imgs = imgs[:N_IMAGES]
    model = FastRCNN()

    # 1) เตรียมข้อมูล: feature map ของทั้งภาพ + กรอบเสนอ + ป้ายกำกับ
    data, t0 = [], time.time()
    for i, p in enumerate(imgs):
        pil = Image.open(p).convert("RGB"); W, H = pil.size
        gt, gc = read_gt(split / "labels" / f"{p.stem}.txt", W, H)
        if len(gt) == 0:
            continue
        props = np.vstack([proposals(pil, CACHE / "train" / f"{p.stem}.npy"), gt])
        P, G = torch.from_numpy(props).float(), torch.from_numpy(gt).float()
        mx, best = box_iou(P, G).max(1)
        labels = torch.where(mx >= 0.5, torch.from_numpy(gc)[best], torch.zeros_like(best))
        g = G[best]
        pw, ph = P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]
        t = torch.stack([((g[:, 0] + g[:, 2]) / 2 - (P[:, 0] + P[:, 2]) / 2) / pw,
                         ((g[:, 1] + g[:, 3]) / 2 - (P[:, 1] + P[:, 3]) / 2) / ph,
                         torch.log((g[:, 2] - g[:, 0]) / pw),
                         torch.log((g[:, 3] - g[:, 1]) / ph)], 1) / TSTD
        data.append((model.features(to_tensor(pil)), P, labels, t, mx))
        print(f"เตรียมข้อมูล {i + 1}/{len(imgs)} {p.name}")
    print(f"เตรียมข้อมูลเสร็จ ใช้เวลา {time.time() - t0:.0f} วินาที")

    # 2) ฝึกส่วนหัว: cross-entropy + smooth L1 รวมในโมเดลเดียว
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-4, weight_decay=1e-4)
    t1 = time.time()
    for ep in range(EPOCHS):
        model.train()
        random.shuffle(data)
        total = 0.0
        for fmap, P, labels, t, mx in data:
            pos = torch.where(mx >= 0.5)[0]
            neg = torch.where(mx < 0.3)[0]
            pos = pos[torch.randperm(len(pos))[:16]]               # positive ไม่เกิน 16
            neg = neg[torch.randperm(len(neg))[:64 - len(pos)]]    # รวมไม่เกิน 64 กรอบต่อรูป
            sel = torch.cat([pos, neg])
            logits, deltas = model.head(fmap, P[sel])
            loss = F.cross_entropy(logits, labels[sel])
            if len(pos):
                loss = loss + F.smooth_l1_loss(deltas[:len(pos)], t[pos])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"epoch {ep + 1}/{EPOCHS} | loss เฉลี่ย {total / len(data):.4f}")
    print(f"ฝึกเสร็จ ใช้เวลา {time.time() - t1:.0f} วินาที")

    head = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone")}
    torch.save(head, OUT / "fast_rcnn_head.pt")
    print("บันทึกที่", OUT / "fast_rcnn_head.pt")


if __name__ == "__main__":
    main()