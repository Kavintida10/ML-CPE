import random, shutil
import xml.etree.ElementTree as ET
from pathlib import Path

RAW, OUT = Path("raw"), Path("dataset")
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
random.seed(42)

xmls = sorted((RAW / "annotations").glob("*.xml"))
random.shuffle(xmls)
n = len(xmls)
a, b = int(n * 0.7), int(n * 0.85)
splits = {"train": xmls[:a], "val": xmls[a:b], "test": xmls[b:]}

for split, files in splits.items():
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    for xml in files:
        root = ET.parse(xml).getroot()
        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)
        lines = []
        for obj in root.findall("object"):
            c = CLASSES.index(obj.find("name").text)
            bb = obj.find("bndbox")
            x1, y1, x2, y2 = [float(bb.find(t).text)
                              for t in ("xmin", "ymin", "xmax", "ymax")]
            lines.append(f"{c} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        (OUT / split / "labels" / f"{xml.stem}.txt").write_text("\n".join(lines))
        img = list((RAW / "images").glob(xml.stem + ".*"))[0]
        shutil.copy(img, OUT / split / "images")

(OUT / "data.yaml").write_text(
    "path: dataset\ntrain: train/images\nval: val/images\ntest: test/images\n"
    "nc: 3\nnames: ['with_mask', 'without_mask', 'mask_weared_incorrect']\n"
)
print({k: len(v) for k, v in splits.items()})