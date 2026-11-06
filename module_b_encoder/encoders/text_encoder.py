"""
文字編碼器（凍結的 FinBERT）：K 線文字敘述 -> H_t、RAG-1 型態定義 -> H_r。

- 預設 FinBERT（config.yaml 的 encoders.text），全程凍結，只做前向運算。
- encode()：單段文字 -> token-level [512, 768]，padding 位置為 0。下游用「非零列」當 mask，
  融合層只取實際的 token（fusion.max_caption_tokens）。
- encode_pooled()：單段文字 -> masked mean [768]（α* 的相似度、RAG-1 檢索 key 用）。
- 舊架構把新聞編碼成 H_t（decisions.md #99），現在新聞改走 RAG-2 原文，不再經過 FinBERT（#110）。
"""
import numpy as np

MAX_TOKENS = 512


class TextEncoder:
    def __init__(self, model_id: str = "ProsusAI/finbert", device: str | None = None):
        # torch/transformers 延遲載入：fake 模式不需要安裝它們
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def n_tokens(self, text: str) -> int:
        """含 [CLS]/[SEP] 的 token 數（檢查是否超過 512 用）。"""
        return len(self.tokenizer(text)["input_ids"])

    def encode(self, text: str) -> np.ndarray:
        """單段文字 -> [512, 768]，float32；padding 部分為 0。"""
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=MAX_TOKENS, padding="max_length",
            ).to(self.device)
            out = self.model(**inputs)
            h = out.last_hidden_state[0]                      # [512, 768]
            h = h * inputs["attention_mask"][0].unsqueeze(-1)  # padding 歸零
        return h.cpu().numpy().astype(np.float32)

    def encode_pooled(self, text: str) -> np.ndarray:
        """單段文字 -> 池化向量 [768]（mean pooling，不含 padding）。"""
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS,
            ).to(self.device)
            out = self.model(**inputs)
            mask = inputs["attention_mask"][0].unsqueeze(-1)
            h = (out.last_hidden_state[0] * mask).sum(0) / mask.sum()
        return h.cpu().numpy().astype(np.float32)


def token_mask(h: np.ndarray) -> np.ndarray:
    """[..., L, 768] -> [..., L] bool：非零列就是實際 token（encode() 把 padding 歸零）。"""
    return np.abs(h).sum(axis=-1) > 0


def masked_mean(h: np.ndarray) -> np.ndarray:
    """[..., L, 768] -> [..., 768]：只平均實際 token。"""
    m = token_mask(h)[..., None].astype(np.float32)
    return (h * m).sum(axis=-2) / np.maximum(m.sum(axis=-2), 1.0)


def fake_h_t(shape=(512, 768), seed: int | None = None, n_tokens: int = 120) -> np.ndarray:
    """假資料：前 n_tokens 列隨機、其餘為 0（模擬 padding）。"""
    rng = np.random.default_rng(seed)
    h = np.zeros(shape, dtype=np.float32)
    h[:n_tokens] = rng.standard_normal((n_tokens, shape[-1])).astype(np.float32)
    return h
