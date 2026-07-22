"""ViT 視覺編碼器：K 線圖 PNG -> H_v。

- 從 HuggingFace 載入預訓練 ViT（config.yaml 的 encoders.vision）。
- 輸入 224x224 RGB PNG，輸出 token-level 特徵 [197, 768]（CLS + 196 patches）。
- 換不同 ViT 版本只要改 model_id（比較實驗、domain gap 見 docs/decisions.md #10）。
- 輸出 shape 一旦定案不可再變（docs/data_format.md 第 2 節）。
"""
import numpy as np


class VisionEncoder:
    def __init__(self, model_id: str = "google/vit-base-patch16-224", device: str | None = None):
        # torch/transformers 延遲載入：fake 模式不需要安裝它們
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def encode(self, png_path: str) -> np.ndarray:
        """單張 K 線圖 -> H_v，shape [197, 768]，float32。"""
        from PIL import Image
        img = Image.open(png_path).convert("RGB")
        with self.torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
        return out.last_hidden_state[0].cpu().numpy().astype(np.float32)


def fake_h_v(shape=(197, 768), seed: int | None = None) -> np.ndarray:
    """第一階段假資料：隨機 H_v（shape 同真實輸出）。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)
