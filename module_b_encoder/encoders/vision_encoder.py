"""共享權重的雙圖 ViT 視覺編碼器：兩張市場圖 PNG -> H_v。

- 從 HuggingFace 載入預訓練 ViT（config.yaml 的 encoders.vision）。
- 兩張 224x224 RGB PNG 以同一個 batch 通過同一組 ViT 權重。
- 輸出 token-level 特徵 [2, 197, 768]（每張圖各有 CLS + 196 patches）。
- 第 0 張目前是 K 線、第 1 張目前是成交量；圖別辨識由 module_c 的 slot embedding 負責。
- 換不同 ViT 版本只要改 model_id（比較實驗、domain gap 見 docs/decisions.md #10）。
- 輸出 shape 一旦定案不可再變（docs/data_format.md 第 2 節）。
"""
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class VisionEncoder:
    def __init__(self, model_id: str = "google/vit-base-patch16-224", device: str | None = None,
                 n_images: int = 2):
        # torch/transformers 延遲載入：fake 模式不需要安裝它們
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.model_id = model_id
        self.n_images = n_images
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def encode(self, png_paths: Sequence[str | Path]) -> np.ndarray:
        """依給定順序編碼兩張圖，回傳 H_v [2, 197, hidden]、float32。"""
        paths = list(png_paths)
        if len(paths) != self.n_images:
            raise ValueError(f"VisionEncoder 預期 {self.n_images} 張圖，實際收到 {len(paths)} 張")
        return self.encode_batch(paths)

    def encode_batch(self, png_paths: Sequence[str | Path]) -> np.ndarray:
        """批次編碼任意張圖片，供大量產生候選輔助圖向量；不改變雙圖輸出契約。"""
        from PIL import Image

        images = []
        for path in png_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if not images:
            raise ValueError("encode_batch 至少需要一張圖片")
        with self.torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
        return out.last_hidden_state.cpu().numpy().astype(np.float32)


def fake_h_v(shape=(2, 197, 768), seed: int | None = None) -> np.ndarray:
    """第一階段假資料：隨機 H_v（shape 同真實輸出）。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)
