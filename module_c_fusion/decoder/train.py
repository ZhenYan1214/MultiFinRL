"""訓練 decoder：用 Z_fused 當 soft-prompt 前綴，微調 LLaMA-2 產生 y_belief（計畫書 3.5 節）。

只實作 L_belief 一項 loss（計畫書原文「λ3 is prioritized to ensure the richness of the
belief state」，且 L_align 需要聯合訓練 encoder——目前 encoder 是凍結的，架構上還沒開放；
L_ground 需要另一份目前也不存在的 oracle relevance scores 標記——這兩項先不做，
見 module_c_fusion/decoder/model.py 檔頭與 docs/decisions.md）。

backbone LLaMA-2 全程凍結，只訓練：
    1. ZFusedProjector（Z_fused -> soft-prompt 前綴 embedding）
    2. LoRA adapters（q/k/v/o_proj + gate/up/down_proj，見 model.py 檔頭說明為何全掛）

訓練設定依 2026 年 QLoRA 微調的常見共識調整過（不是隨便選的預設值）：
    - 學習率預設 2e-4（LoRA/QLoRA 常見穩定區間 1e-4~3e-4，2e-4 是最常見的起手式）。
    - Cosine 學習率排程 + warmup（前 5% 的 step 讓學習率從 0 慢慢升到設定值，之後照
      cosine 曲線遞減到接近 0），比全程固定學習率更穩定，是目前的標準做法。
    - Gradient clipping（max_norm=1.0），避免某一步梯度爆掉，訓練穩定性的基本防護。
    - checkpoint 只在 val loss 創新低時才覆寫存檔，不是無條件存最後一個 epoch的
      結果——如果過擬合在後面幾個 epoch 才出現，最後一個 epoch 反而不是最好的版本。

train/val/test 依時間切分（70/15/15，跟 classifier.py 同一套慣例，不可隨機打散，避免
時間洩漏）：train 拿去訓練，val 每個 epoch 結束後算一次 loss（不參與訓練、不更新參數），
用來跟 train loss 對照——如果 val loss 跟 train loss 差不多，代表學到的是可以類化的規律；
如果 val loss 明顯比 train loss差很多，代表在死記硬背訓練集，不是真的學會。test 這次先
保留不用，是給之後要做「生成文字品質」人工檢查用的（模型完全沒看過的天，見 decisions.md）。

前置（照順序）：
    1. Z_fused 已存在：python -m module_c_fusion.fusion.train --ticker AAPL
       python -m module_c_fusion.fusion.consolidate --ticker AAPL
    2. y_belief 已存在：python -m module_c_fusion.decoder.generate_y_belief --ticker AAPL
    3. 本機已 `hf auth login`，且帳號已通過 meta-llama/Llama-2-7b-hf 的存取申請
    4. 有 GPU（本檔預設抓 meta-llama/Llama-2-7b-hf，4-bit 量化約需 5-6GB VRAM，
       RTX 5060 Ti 16GB 跑得動）

用法：
    python -m module_c_fusion.decoder.train --ticker AAPL --epochs 3

本檔案需要 torch/transformers/peft，無法在沒有 GPU 的環境執行，只驗證過語法
（py_compile），實際訓練需要在你自己機器上跑。重點看兩件事：(1) train loss 跟 val loss
有沒有一起往下降，(2) val loss 是在哪個 epoch 最低——如果不是最後一個 epoch，代表
之後的訓練其實在讓模型過擬合，checkpoint 邏輯已經處理好這件事，只會存 val loss 最低
那次的結果，不用你自己判斷。
"""
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from shared import paths
from shared.utils import load_config, read_json, write_json
from module_c_fusion.fusion.consolidate import load_index
from module_c_fusion.decoder.model import ZFusedDecoder, load_backbone


def load_paired_samples(ticker: str) -> list[tuple[str, "np.ndarray", str]]:
    """(date, Z_fused, y_belief 文字) 三元組，依日期排序，只取兩邊都有的日期——Z_fused
    跟 y_belief 是分開產出的兩支腳本，範圍不一定完全一致，交集才是能訓練的樣本。
    """
    index = load_index(ticker)
    if index is None:
        raise SystemExit(
            f"找不到 Z_fused 索引，先跑 python -m module_c_fusion.fusion.consolidate --ticker {ticker}")
    y_belief_path = paths.y_belief_path(ticker)
    if not y_belief_path.exists():
        raise SystemExit(
            f"找不到 y_belief，先跑 python -m module_c_fusion.decoder.generate_y_belief --ticker {ticker}")
    y_belief = read_json(y_belief_path)

    dates = index["dates"]
    z = index["z"]
    samples = [(str(date), z[i], y_belief[str(date)])
              for i, date in enumerate(dates) if str(date) in y_belief]
    if not samples:
        raise SystemExit("Z_fused 跟 y_belief 沒有任何日期交集，檢查兩邊的 ticker/日期範圍是否一致")
    return samples  # index 的 dates 本來就已依時間排序，這裡不再重新排序


def time_split(rows: list, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """時間序切分：前 70% train、中 15% val、後 15% test（跟 classifier.py 同一套慣例）。"""
    n = len(rows)
    i, j = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    return rows[:i], rows[i:j], rows[j:]


class YBeliefDataset(Dataset):
    """單純包裝一份 (date, Z_fused, y_belief 文字) 清單成 PyTorch Dataset，tokenize 交給
    __getitem__ 做。切分邏輯在 main() 裡處理，這裡只負責把切好的其中一段包成 Dataset。
    """

    def __init__(self, samples: list, tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, z, text = self.samples[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {
            "z_fused": torch.tensor(z, dtype=torch.float32),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def eval_loss(decoder, loader, device) -> float:
    """算一次平均 loss，不更新參數（用在 val set 上，torch.no_grad() 不計算梯度）。"""
    decoder.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss = decoder(batch["z_fused"].to(device),
                           batch["input_ids"].to(device),
                           batch["attention_mask"].to(device))
            total += loss.item()
            n += 1
    decoder.train()
    return total / max(n, 1)


def save_checkpoint(decoder, llm, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.projector.state_dict(), out_dir / "projector.pt")
    llm.save_pretrained(out_dir / "lora_adapter")  # peft 只存 adapter 權重，不含凍結的 backbone


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA/QLoRA 常見穩定區間 1e-4~3e-4，2e-4 是最常見的起手式")
    ap.add_argument("--warmup_ratio", type=float, default=0.05,
                    help="學習率從 0 升到設定值要花的 step 比例，之後 cosine 遞減")
    ap.add_argument("--n_prefix_tokens", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512,
                    help="y_belief 文字 tokenize 後的長度上限。LLaMA-2 的 tokenizer 對中文"
                         "不友善（一個中文字常常要拆成 2-3 個 token），預設 512 是為了避免"
                         "把敘述文字截斷太多，不是隨便設的數字")
    args = ap.parse_args()

    samples = load_paired_samples(args.ticker)
    train_rows, val_rows, test_rows = time_split(samples)
    print(f"[decoder.train] {args.ticker}: 共 {len(samples)} 天可訓練樣本（Z_fused ∩ y_belief），"
         f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}（依時間序切分，不打散）")

    print(f"[decoder.train] 載入 backbone: {args.base_model}（4-bit 量化 + LoRA 全掛 attention+MLP，backbone 凍結）")
    tokenizer, llm = load_backbone(args.base_model)
    decoder = ZFusedDecoder(llm, tokenizer, z_dim=cfg["fusion"]["z_dim"],
                            n_prefix_tokens=args.n_prefix_tokens)

    train_ds = YBeliefDataset(train_rows, tokenizer, max_length=args.max_length)
    val_ds = YBeliefDataset(val_rows, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    trainable = [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    out_dir = paths.OUTPUTS / "checkpoints" / "decoder"
    best_val = float("inf")
    best_epoch = None
    epoch_history = []

    for epoch in range(args.epochs):
        running_train_loss, n_train_steps = 0.0, 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = decoder(batch["z_fused"].to(llm.device),
                           batch["input_ids"].to(llm.device),
                           batch["attention_mask"].to(llm.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_train_loss += loss.item()
            n_train_steps += 1
            print(f"  epoch {epoch + 1}/{args.epochs} step {step} "
                 f"L_belief={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        train_l = running_train_loss / max(n_train_steps, 1)
        val_l = eval_loss(decoder, val_loader, llm.device)
        print(f"[decoder.train] epoch {epoch + 1}/{args.epochs} 結束 -> "
             f"train_L_belief(avg)={train_l:.4f} val_L_belief={val_l:.4f}")
        epoch_history.append({"epoch": epoch + 1, "train_loss_avg": train_l, "val_loss": val_l})

        if val_l < best_val:
            best_val = val_l
            best_epoch = epoch + 1
            save_checkpoint(decoder, llm, out_dir)
            print(f"[decoder.train] val loss 創新低（{val_l:.4f}），checkpoint 已更新 -> {out_dir}")

    print(f"[decoder.train] 訓練結束，最終存檔的是 epoch {best_epoch}（val_L_belief={best_val:.4f} 最低的那次）")

    report = {
        "ticker": args.ticker,
        "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
        "test_period": [test_rows[0][0], test_rows[-1][0]] if test_rows else [],
        "best_epoch": best_epoch, "best_val_loss": best_val,
        "epoch_history": epoch_history,
    }
    report_path = paths.OUTPUTS / "metrics" / "decoder_train_report.json"
    write_json(report, report_path)
    print(f"[decoder.train] 訓練報告 -> {report_path}")


if __name__ == "__main__":
    main()
