from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATA = HERE / "data_abs.yaml"          # สร้างโดย train.py
SPLIT = "test"                         # วัดผลบนชุด test
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def finished_runs():
    return [d for d in sorted(RUNS.iterdir())
            if d.is_dir() and d.name != "quick_test"
            and (d / "weights" / "best.pt").exists() and (d / "done.txt").exists()]


def main():
    rows, cls_rows, curves = [], [], {}
    for d in finished_runs():
        args = yaml.safe_load((d / "args.yaml").read_text(encoding="utf-8"))
        imgsz, epochs, batch = args["imgsz"], args["epochs"], args["batch"]
        model = YOLO(str(d / "weights" / "best.pt"))
        m = model.val(data=str(DATA), split=SPLIT, imgsz=imgsz, batch=16, device=DEVICE,
                      workers=2, plots=False, verbose=False,
                      project=str(RUNS / "eval"), name=d.name, exist_ok=True)

        df = pd.read_csv(d / "results.csv")
        df.columns = df.columns.str.strip()
        tr = df[[c for c in df.columns if c.startswith("train/") and c.endswith("loss")]].sum(axis=1)
        va = df[[c for c in df.columns if c.startswith("val/") and c.endswith("loss")]].sum(axis=1)
        curves[d.name] = (tr, va)

        minutes = float((d / "done.txt").read_text()) / 60
        rows.append([d.name, epochs, imgsz, batch, round(m.box.mp, 3), round(m.box.mr, 3),
                     round(m.box.map50, 3), round(m.box.map, 3),
                     round(float(tr.iloc[-1]), 3), round(float(va.iloc[-1]), 3), round(minutes, 1)])
        for i, c in enumerate(m.box.ap_class_index):
            cls_rows.append([d.name, m.names[int(c)], round(float(m.box.p[i]), 3),
                             round(float(m.box.r[i]), 3), round(float(m.box.ap50[i]), 3),
                             round(float(m.box.ap[i]), 3)])

    out = pd.DataFrame(rows, columns=["run", "epochs", "imgsz", "batch", "precision", "recall",
                                      "mAP50", "mAP50-95", "train_loss", "val_loss", "train_min"])
    out = out.sort_values(["epochs", "imgsz", "batch"]).reset_index(drop=True)
    per_class = pd.DataFrame(cls_rows, columns=["run", "class", "precision", "recall", "AP50", "AP50-95"])
    print(f"\nผลบนชุด {SPLIT}")
    print(out.to_string(index=False))
    print("\nแยกรายคลาส")
    print(per_class.to_string(index=False))
    out.to_csv(RUNS / "comparison.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(RUNS / "per_class.csv", index=False, encoding="utf-8-sig")

    # กราฟ 1: mAP50 ของแต่ละการทดลอง
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(out["run"], out["mAP50"])
    ax.set_ylabel("mAP50 (test)"); ax.set_title("YOLO26n: mAP50 per configuration")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    fig.savefig(RUNS / "comparison_map.png", dpi=150); plt.close(fig)

    # กราฟ 2: train / validation loss ตาม epoch
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for name, (tr, va) in curves.items():
        axs[0].plot(range(1, len(tr) + 1), tr, label=name)
        axs[1].plot(range(1, len(va) + 1), va, label=name)
    axs[0].set_title("Train loss"); axs[1].set_title("Validation loss")
    for a in axs:
        a.set_xlabel("epoch"); a.legend(fontsize=7)
    plt.tight_layout(); fig.savefig(RUNS / "comparison_loss.png", dpi=150); plt.close(fig)
    print("\nบันทึกใน", RUNS)


if __name__ == "__main__":
    main()