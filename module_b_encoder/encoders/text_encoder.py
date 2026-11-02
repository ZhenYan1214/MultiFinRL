"""文字編碼器：新聞 / 文件文字 -> H_t。

- 預設 FinBERT（config.yaml 的 encoders.text）。
- H_t 有兩種產生方式，由 config.yaml 的 encoders.text_method 切換（docs/decisions.md #99）：
    chunk_relevance_max（預設）：encode_news_chunk_max()
        每則新聞 "[N days ago] headline. content" 依序切成 ≤510 token 的段落，各自編碼取
        masked mean；只保留內文提到公司名稱或 ticker ≥ min_mentions 次的段落（當天一段都沒有時
        用全部段落），逐維取 max 得到一個 768 維向量，放在 H_t 第 0 列，其餘列為 0。
        → 每一則新聞的全文都會被讀到，不會被截斷。
    concat_truncate（舊版）：encode_news()
        全部新聞接成一段、截到 512 token，輸出 token-level [512, 768]。實測平均只讀到 7.7% 的文字。
- 兩種輸出 shape 都是 [512, 768] float32（docs/data_format.md 第 2 節），下游不用改 shape。
"""
import re

import numpy as np

MAX_TOKENS = 512
CHUNK_TOKENS = MAX_TOKENS - 2   # 扣掉 [CLS]、[SEP]
CHUNK_BATCH = 32

# 段落相關性判斷用的公司名稱；沒列出的 ticker 只比對 ticker 本身。
COMPANY_NAMES: dict[str, list[str]] = {
    "AAPL": ["apple"],
    "NVDA": ["nvidia"],
    "MSFT": ["microsoft"],
}


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

    def encode_news_chunk_max(self, news: list[dict], ticker: str, min_mentions: int = 3) -> np.ndarray:
        """每日新聞列表 -> H_t [512, 768]：段落級編碼 + 公司相關性過濾 + 逐維 max，結果放第 0 列。

        無新聞時回傳全 0（跟 encode_news 相同）。實驗依據見 docs/decisions.md #99。
        """
        h_t = np.zeros((MAX_TOKENS, 768), dtype=np.float32)
        if not news:
            return h_t
        tok = self.tokenizer
        chunks = []
        for n in news:
            ids = tok(f"[{n['days_ago']} days ago] {n['headline']}. {n['content']}",
                      add_special_tokens=False)["input_ids"]
            chunks += [ids[s:s + CHUNK_TOKENS] for s in range(0, len(ids), CHUNK_TOKENS)]
        chunks = [c for c in chunks if c]
        if not chunks:
            return h_t

        names = [ticker.lower()] + COMPANY_NAMES.get(ticker.upper(), [])
        mention_re = re.compile(r"\b(" + "|".join(re.escape(x) for x in names) + r")\b")
        keep = [len(mention_re.findall(tok.decode(c).lower())) >= min_mentions for c in chunks]
        if any(keep):
            chunks = [c for c, k in zip(chunks, keep) if k]

        h_t[0] = self._chunk_masked_means(chunks).max(axis=0)
        return h_t

    def _chunk_masked_means(self, chunks: list[list[int]]) -> np.ndarray:
        """token id 段落（不含特殊 token）-> 每段 masked mean [n, 768]。

        依長度排序後批次編碼，padding 到 64 的倍數：長時間批次跑時，每批長度都不同會讓
        PyTorch 快取記憶體碎片累積、顯示記憶體被塞滿而大幅變慢。padding 有 mask，結果不變。
        """
        torch, tok = self.torch, self.tokenizer
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
        out = np.zeros((len(chunks), 768), dtype=np.float32)
        for b in range(0, len(order), CHUNK_BATCH):
            idx = order[b:b + CHUNK_BATCH]
            seqs = [[tok.cls_token_id] + chunks[i] + [tok.sep_token_id] for i in idx]
            length = min(MAX_TOKENS, -(-max(len(s) for s in seqs) // 64) * 64)
            input_ids = torch.full((len(seqs), length), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), length), dtype=torch.long)
            for j, s in enumerate(seqs):
                input_ids[j, :len(s)] = torch.tensor(s)
                mask[j, :len(s)] = 1
            with torch.no_grad():
                h = self.model(input_ids=input_ids.to(self.device),
                               attention_mask=mask.to(self.device),
                               token_type_ids=torch.zeros_like(input_ids).to(self.device)).last_hidden_state
                m = mask.to(self.device).unsqueeze(-1).float()
                out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        return out

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
