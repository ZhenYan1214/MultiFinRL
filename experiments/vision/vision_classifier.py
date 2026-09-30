"""ViT-only 雙圖分類模型，供視覺特徵調整實驗使用。

正式 pipeline 的 :mod:`vision_encoder` 只負責產生 token-level ``H_v``；這個模組
另外接上一個很小的分類頭，讓我們可以先隔離文字與 RAG，直接量測 ViT 圖像表示能否
預測 BEARISH / NEUTRAL / BULLISH。
"""
from collections.abc import Iterable

import torch
import torch.nn as nn


def build_classification_head(hidden: int, n_classes: int = 3,
                              dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(hidden * 2),
        nn.Dropout(dropout),
        nn.Linear(hidden * 2, n_classes),
    )


class DualImageViTClassifier(nn.Module):
    """共享權重 ViT + 兩張圖 CLS 串接分類頭。

    Input: ``pixel_values [B, 2, 3, 224, 224]``
    Output: ``logits [B, n_classes]``
    """

    def __init__(self, model_id: str, n_classes: int = 3, dropout: float = 0.1,
                 head_names: list[str] | None = None):
        super().__init__()
        from transformers import AutoModel

        self.model_id = model_id
        try:
            # 已下載過時直接使用快取，避免離線環境仍先送出 Hub HEAD request。
            self.backbone = AutoModel.from_pretrained(model_id, local_files_only=True)
        except OSError:
            self.backbone = AutoModel.from_pretrained(model_id)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.feature_dim = self.hidden_size * 2
        self.head_names = tuple(head_names or [])
        if self.head_names:
            self.classifier = None
            self.stock_heads = nn.ModuleDict({
                name: build_classification_head(self.hidden_size, n_classes, dropout)
                for name in self.head_names
            })
        else:
            self.classifier = build_classification_head(
                self.hidden_size, n_classes, dropout
            )
            self.stock_heads = nn.ModuleDict()
        self.adaptation = "frozen"

    def encode_cls(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5 or pixel_values.size(1) != 2:
            raise ValueError(
                "pixel_values 應為 [B,2,3,H,W]，"
                f"實際 shape={tuple(pixel_values.shape)}"
            )
        batch, n_images, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch * n_images, channels, height, width)
        output = self.backbone(pixel_values=flat).last_hidden_state[:, 0]
        return output.reshape(batch, n_images * output.size(-1))

    def classify(self, features: torch.Tensor,
                 tickers: list[str] | tuple[str, ...] | None = None) -> torch.Tensor:
        if not self.head_names:
            return self.classifier(features)
        if tickers is None or len(tickers) != len(features):
            raise ValueError("多股票分類需要為 batch 內每筆樣本提供 ticker")
        unknown = set(tickers) - set(self.head_names)
        if unknown:
            raise ValueError(f"找不到股票分類頭：{sorted(unknown)}")

        logits = features.new_zeros((len(features), 3))
        for name, head in self.stock_heads.items():
            indices = [i for i, ticker in enumerate(tickers) if ticker == name]
            if not indices:
                continue
            index = torch.tensor(indices, device=features.device)
            values = head(features.index_select(0, index))
            logits = logits.index_copy(0, index, values)
        return logits

    def forward(self, pixel_values: torch.Tensor,
                tickers: list[str] | tuple[str, ...] | None = None) -> torch.Tensor:
        return self.classify(self.encode_cls(pixel_values), tickers)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        module = self.stock_heads if self.head_names else self.classifier
        return module.parameters()

    def configure_adaptation(self, strategy: str) -> dict[str, int]:
        """凍結 backbone，再依策略解凍 blocks 或掛 LoRA（分類頭永遠可訓練）。"""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.head_parameters():
            parameter.requires_grad = True

        if strategy == "frozen":
            n_unfrozen = 0
            lora_rank = None
        elif strategy == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            n_unfrozen = len(self.backbone.encoder.layer)
            lora_rank = None
        elif strategy.startswith("lora_r") and strategy[6:].isdigit():
            from peft import LoraConfig, get_peft_model

            lora_rank = int(strategy[6:])
            if lora_rank < 1:
                raise ValueError("LoRA rank 必須 >= 1")
            blocks = self.backbone.encoder.layer
            target_blocks = list(range(max(0, len(blocks) - 4), len(blocks)))
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                layers_to_transform=target_blocks,
                layers_pattern="layer",
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            n_unfrozen = 0
        elif strategy.startswith("last") and strategy[4:].isdigit():
            n_unfrozen = int(strategy[4:])
            lora_rank = None
            blocks = self.backbone.encoder.layer
            if not 1 <= n_unfrozen <= len(blocks):
                raise ValueError(f"{strategy} 超出 ViT blocks 數量 {len(blocks)}")
            for block in blocks[-n_unfrozen:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            # 最後的 LayerNorm 直接影響 CLS 表示，跟最後幾層一起調整。
            if hasattr(self.backbone, "layernorm"):
                for parameter in self.backbone.layernorm.parameters():
                    parameter.requires_grad = True
        else:
            raise ValueError(f"不支援的 adaptation strategy: {strategy}")

        self.adaptation = strategy
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "unfrozen_blocks": n_unfrozen,
            "lora_rank": lora_rank,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen baseline 不應讓 backbone dropout 在每個 epoch 改變固定特徵。
        if self.adaptation == "frozen":
            self.backbone.eval()
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """只保存被調整的參數，避免每個實驗重複存整份 86M backbone。"""
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable
        }
