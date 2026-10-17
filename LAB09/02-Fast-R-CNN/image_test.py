import random, sys, time
from pathlib import Path
import torch, torch.nn.functional as F, torchvision
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as tr          # ใช้โมเดลและฟังก์ชันจาก train.py (ไม่เริ่มฝึกใหม่)

TEST = HERE.parent / "dataset" / "test" / "images"
COLORS = [None, "green", "red", "orange"]
MIN_SCORE, NMS_IOU = 0.5, 0.3     # คะแนนที่นี่เป็นความน่าจะเป็น 0-1


def load_model():
    model = tr.FastRCNN()
    model.load_state_dict(torch.load(tr.OUT / "fast_rcnn_head.pt"), strict=False)
    model.eval()
    return model


@torch.no_grad()
def detect(pil, model):
    W, H = pil.size
    props = torch.from_numpy(tr.proposals(pil)).float()      # Selective Search
    fmap = model.features(tr.to_tensor(pil))                  # CNN รูปทั้งใบ ครั้งเดียว
    logits, deltas = model.head(fmap, props)                  # RoI Pooling + FC
    score, cls = F.softmax(logits, 1).max(1)
    d = deltas * tr.TSTD
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    cx = (props[:, 0] + props[:, 2]) / 2 + d[:, 0] * pw
    cy = (props[:, 1] + props[:, 3]) / 2 + d[:, 1] * ph
    w = pw * torch.exp(d[:, 2].clamp(-2, 2))
    h = ph * torch.exp(d[:, 3].clamp(-2, 2))
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H)
    keep = (cls != 0) & (score > MIN_SCORE)
    if not keep.any():
        return []
    b, s, c = boxes[keep], score[keep], cls[keep]
    k = torchvision.ops.batched_nms(b, s, c, NMS_IOU)
    return [(b[i].numpy(), int(c[i]), float(s[i])) for i in k]


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else random.choice(list(TEST.glob("*")))
    model = load_model()
    pil = Image.open(path).convert("RGB")
    t0 = time.time()
    dets = detect(pil, model)
    print(f"ตรวจจับใช้เวลา {time.time() - t0:.2f} วินาที")
    d = ImageDraw.Draw(pil)
    for box, c, s in dets:
        d.rectangle(list(box), outline=COLORS[c], width=3)
        d.text((box[0], max(0, box[1] - 12)), f"{tr.NAMES[c]} {s:.2f}", fill=COLORS[c])
        print(tr.NAMES[c], round(s, 2), [int(v) for v in box])
    out = tr.OUT / f"pred_{path.stem}.png"
    pil.save(out)
    print("พบ", len(dets), "วัตถุ | บันทึกที่", out)