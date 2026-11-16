"""畫 ViT 微調的 Test confusion matrix（列＝Haiku 標籤，欄＝ViT 預測；顏色＝該列的比例，格內是天數）。

讀 experiments/vision/vit_pattern.py 寫出的結果，圖也存在同一個資料夾（data/outputs/experiments/vit_pattern/）。
用法（repo 根目錄）：PYTHONPATH=. python experiments/vision/vit_pattern_confusion.py [AAPL | NVDA | AAPL+NVDA]
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from shared import paths

HERE = paths.OUTPUTS / "experiments" / "vit_pattern"
TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
PRE = "" if TICKER == "AAPL" else f"{TICKER}_"
ZH = ["持續上漲", "持續下跌", "盤整", "突破", "跌破", "頂部反轉", "底部反轉"]
plt.rcParams["font.family"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS"]
cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

def load(name):
    if not (HERE / name).exists():
        return None
    return np.array(json.load(open(HERE / name))["confusion_test"]["matrix"])

panels = [
    ("K 線圖，3 個 seed 加總\n（每個 seed 都評同樣 203 天，共 609 筆）",
     (None if load(f"{PRE}finetune_candlestick_seed42.json") is None else
      sum(load(f"{PRE}finetune_candlestick_seed{s}.json") for s in (42, 43, 44)))),
    ("K 線＋成交量雙圖，3 個 seed 加總\n（共 609 筆）",
     sum(load(f"{PRE}finetune_both_seed{s}.json") for s in (42, 43, 44))
     if (HERE / f"{PRE}finetune_both_seed42.json").exists() else None),
]
if "+" in TICKER:   # 合併訓練：同一個模型，Test 分股票各畫一張
    runs = [json.load(open(HERE / f"{PRE}finetune_candlestick_seed{s}.json")) for s in (42, 43, 44)]
    panels = [(f"{t} Test（{TICKER} 合併訓練）\nK 線圖，3 個 seed 加總",
               sum(np.array(r["test_by_ticker"][t]["confusion"]) for r in runs)) for t in TICKER.split("+")]
panels = [p for p in panels if p[1] is not None]
fig, axes = plt.subplots(1, len(panels), squeeze=False, figsize=(7.5 * len(panels), 7), facecolor="#fcfcfb")
for ax, (title, m) in zip(axes[0], panels):
    frac = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            if m[i, j] == 0:
                continue
            ax.text(j, i, f"{m[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if frac[i, j] > 0.5 else "#2b2b29")
    ax.set_xticks(range(7), ZH, rotation=35, ha="right")
    ax.set_yticks(range(7), [f"{z}（{n}）" for z, n in zip(ZH, m.sum(1))])
    ax.set_xlabel("ViT 預測", color="#55554f")
    ax.set_ylabel("Haiku 標籤（括號內是筆數）", color="#55554f")
    acc = np.trace(m) / m.sum()
    ax.set_title(f"{title}\naccuracy {acc:.3f}", fontsize=11, color="#2b2b29")
    ax.set_xticks(np.arange(-.5, 7), minor=True); ax.set_yticks(np.arange(-.5, 7), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle(f"ViT 微調看 K 線型態：Test confusion matrix（{TICKER}）\n顏色＝該列（真實類別）裡被預測成各類的比例",
             fontsize=12, color="#2b2b29")
fig.tight_layout()
fig.savefig(HERE / f"{PRE}confusion_test.png", dpi=150)
print("saved")
