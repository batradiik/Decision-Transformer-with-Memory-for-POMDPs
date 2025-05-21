from __future__ import annotations
import argparse, textwrap, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(path: Path):
    data = np.load(path)
    return np.asarray(data["train_losses"]), np.asarray(data["val_returns"])


def first_epoch_above(arr, thr):
    ok = np.where(arr >= thr)[0]
    return int(ok[0]) if ok.size else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,)
    p.add_argument("--out", default="comparison_plot.png",)
    p.add_argument("--thr", type=float, default=300.0,)
    p.add_argument("--table", default="comparison.csv",)
    args = p.parse_args()

    rows = []                     
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for f in args.runs:
        tl, _ = load_metrics(Path(f))
        if not np.isfinite(tl).any():
            continue
        label = Path(f).stem.replace("memory_dt_velocity_cartpole_", "")
        plt.plot(tl, label=label)
    plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(alpha=.3); plt.legend()
    plt.subplot(1, 2, 2)
    for f in args.runs:
        tl, vr = load_metrics(Path(f))
        label = Path(f).stem.replace("memory_dt_velocity_cartpole_", "")
        plt.plot(vr, label=label)
        conv = first_epoch_above(vr, args.thr)
        rows.append(dict(model=label,
                         epochs_to_300=conv,
                         final_return=vr[-1],
                         final_loss=tl[-1] if np.isfinite(tl[-1]) else np.nan))
    plt.title("Validation Mean Return"); plt.xlabel("Epoch"); plt.ylabel("Return")
    plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(args.out)
    print(f"Plots saved  to  {args.out}")

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(args.table, float_format="%.2f")


if __name__ == "__main__":
    main()