"""Frozen ViT：正式 mplfinance 圖只換配色的對照。

畫法與縮圖流程和 chart_generator 相同，只改漲跌顏色；K 線與 Volume 圖用同一組配色重畫：

- mpl    ：正式圖，綠漲紅跌（基準）。
- swap   ：紅漲綠跌。資訊與基準完全相同，只對調顏色，可當作雜訊大小的對照。
- mono   ：黑白。漲為白底黑框空心棒，跌為實心黑棒。
- blueor ：藍漲橘跌。

樣本、標籤、凍結 ViT、分類頭、seeds、指標與報告格式全部沿用
run_mpl_wick_comparison.py：本檔只替換該模組的圖別、設定與繪圖函式後呼叫其 main()。
新圖畫到本實驗輸出目錄，不寫入 data/raw/charts。Test 不評估。

用法（需先跑過 run_pixel_render_comparison.py）：
    python experiments/vision/run_mpl_color_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import paths
from experiments.vision import run_mpl_wick_comparison as shared

PALETTES = {
    "swap": {"up": "red", "down": "green", "edge": "inherit"},
    "mono": {"up": "white", "down": "black", "edge": "black"},
    "blueor": {"up": "#1f77b4", "down": "#ff7f0e", "edge": "inherit"},
}
RUN_NAME = "mpl_color_w20"


def render(window: pd.DataFrame, kind: str, target: Path) -> None:
    """kind = '{palette}_k' 或 '{palette}_vol'。"""
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    from module_a_data.preprocess.chart_generator import _save_figure

    name, chart = kind.rsplit("_", 1)
    palette = PALETTES[name]
    if chart == "k":
        style = mpf.make_mpf_style(
            marketcolors=mpf.make_marketcolors(
                up=palette["up"], down=palette["down"], edge=palette["edge"],
                wick=palette["edge"]),
            gridstyle="",
        )
        fig, _ = mpf.plot(
            window, type="candle", style=style, volume=False, axisoff=True,
            returnfig=True, figsize=(3, 3), scale_padding=0,
        )
    else:
        up = window["Close"].to_numpy() >= window["Open"].to_numpy()
        colors = np.where(up, palette["up"], palette["down"])
        edge = "black" if palette["edge"] == "black" else None
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.bar(np.arange(len(window)), window["Volume"].to_numpy(dtype=float), color=colors,
               width=0.75, edgecolor=edge, linewidth=0.8 if edge else 0)
        ax.set_xlim(-0.5, len(window) - 0.5)
        ax.set_ylim(bottom=0)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save_figure(fig, target, (224, 224))


def save_samples(data: dict, output: Path, n: int = 3) -> list[str]:
    """每檔挑 Train 等距 n 天：上列 K 線、下列 Volume；左到右 mpl｜swap｜mono｜blueor。"""
    out_dir = output / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ticker, payload in data.items():
        dates = payload["rows"]["train"]["dates"]
        for i in np.linspace(0, len(dates) - 1, n, dtype=int):
            date = dates[i]
            top = [paths.RAW_CHARTS / ticker / f"{date}.png"] + [
                shared.chart_path(output, ticker, f"{name}_k", date) for name in PALETTES]
            bottom = [paths.RAW_CHARTS / ticker / "volume" / f"{date}.png"] + [
                shared.chart_path(output, ticker, f"{name}_vol", date) for name in PALETTES]
            sheet = Image.new("RGB", (228 * 4 + 4, 228 * 2 + 4), (128, 128, 128))
            for row, tiles in enumerate((top, bottom)):
                for column, tile_path in enumerate(tiles):
                    with Image.open(tile_path) as tile:
                        sheet.paste(tile.convert("RGB"), (4 + column * 228, 4 + row * 228))
            target = out_dir / f"{ticker}_{date}.png"
            sheet.save(target)
            saved.append(str(target.relative_to(ROOT)))
    return saved


def main() -> None:
    shared.PROTOCOL = "vit_mpl_color_comparison_v1"
    shared.OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "vit_mpl_color"
    shared.NEW_KINDS = tuple(f"{name}_{chart}" for name in PALETTES for chart in ("k", "vol"))
    shared.CONFIGS = {
        "mpl_single": ["mpl_candlestick"],
        **{f"{name}_single": [f"{name}_k"] for name in PALETTES},
        "mpl_dual": ["mpl_candlestick", "mpl_volume"],
        **{f"{name}_dual": [f"{name}_k", f"{name}_vol"] for name in PALETTES},
    }
    shared.COMPARISONS = [(f"{name}_{mode}", f"mpl_{mode}")
                          for mode in ("single", "dual") for name in PALETTES]
    shared.render = render
    shared.save_samples = save_samples
    if not any(arg.startswith("--run-name") for arg in sys.argv[1:]):
        sys.argv += ["--run-name", RUN_NAME]
    shared.main()

    for report in shared.OUTPUT_ROOT.glob("*/report.md"):
        text = report.read_text(encoding="utf-8").replace(
            "# Frozen ViT：mplfinance K 線影線加粗對照", "# Frozen ViT：mplfinance 圖配色對照")
        report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
