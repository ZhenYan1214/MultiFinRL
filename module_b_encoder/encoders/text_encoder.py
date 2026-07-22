"""文字編碼器：新聞 / 文件文字 -> H_t。

- 預設 FinBERT（config.yaml 的 encoders.text）；LLaMA 系列比較後擇優。
- 輸出 token-level 特徵 [512, 768]（不足 512 token 以 0 padding，超過截斷），float32。
- days_ago 的併入方式（負責人建議一併丟給 encoder）：目前做法是把
  "[N days ago] headline. content" 接成一段文字，讓模型從字面判斷相關性強度；
  B 可自行改進（例如加 embedding）。
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

    def encode_news(self, news: list[dict]) -> np.ndarray:
        """每日新聞列表 -> H_t [512, 768]。無新聞時回傳全 0。"""
        if not news:
            return np.zeros((MAX_TOKENS, 768), dtype=np.float32)
        parts = [f"[{n['days_ago']} days ago] {n['headline']}. {n['content']}" for n in news]
        return self.encode(" ".join(parts))

    def encode_pooled(self, text: str) -> np.ndarray:
        """單段文字 -> 池化向量 [768]（RAG 向量庫 key 用；mean pooling）。"""
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS,
            ).to(self.device)
            out = self.model(**inputs)
            mask = inputs["attention_mask"][0].unsqueeze(-1)
            h = (out.last_hidden_state[0] * mask).sum(0) / mask.sum()
        return h.cpu().numpy().astype(np.float32)


def fake_h_t(shape=(512, 768), seed: int | None = None) -> np.ndarray:
    """第一階段假資料：隨機 H_t。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)
