import random
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent / "dataset" / "train"
NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]
COLORS = ["green", "red", "orange"]

imgs = list((ROOT / "images").glob("*"))
for p in random.sample(imgs, 3):
    im = Image.open(p).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    label = ROOT / "labels" / f"{p.stem}.txt"
    for line in label.read_text().splitlines():
        c, x, y, w, h = line.split()
        c = int(c)
        x, y, w, h = map(float, (x, y, w, h))
        x1, y1 = (x - w / 2) * W, (y - h / 2) * H
        x2, y2 = (x + w / 2) * W, (y + h / 2) * H
        d.rectangle([x1, y1, x2, y2], outline=COLORS[c], width=3)
        d.text((x1, max(0, y1 - 12)), NAMES[c], fill=COLORS[c])
    im.save(f"check_{p.stem}.png")
    print("saved", f"check_{p.stem}.png")