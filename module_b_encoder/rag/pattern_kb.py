"""
RAG-1 型態定義知識庫：7 段英文定義 -> FinBERT 編碼（decisions.md #112）。

輸入：data/knowledge/patterns/{NN}_{pattern}.md（順序 = shared/patterns.PATTERN_IDS）
輸出：data/vectors/_patterns/patterns.npz
        H_r_all  [7, 512, 768] float32   每段定義的 token-level 編碼（padding 為 0）
        pooled   [7, 768] float32        masked mean，α* 與檢索 key 用
        ids      [7]                     型態代號
      data/vectors/_patterns/meta.json   每段的 token 數與檔案 SHA256（改了定義要重建）

知識庫所有日子共用；檢索（每天取哪 K 段）在融合層裡做，見 module_c_fusion/fusion/model.py。

用法：
    python -m module_b_encoder.rag.pattern_kb
"""
import hashlib
import re

import numpy as np

from shared import paths, patterns
from shared.utils import load_config, read_json, write_json

KB_FILE = paths.PATTERN_KB / "patterns.npz"
META_FILE = paths.PATTERN_KB / "meta.json"


def read_definitions() -> list[tuple[str, str, str]]:
    """回傳 [(pattern_id, 檔名, 本文)]，依 PATTERN_IDS 順序；frontmatter 會被拿掉。"""
    files = sorted(paths.KNOWLEDGE_PATTERNS.glob("[0-9][0-9]_*.md"))
    out = []
    for i, pid in enumerate(patterns.PATTERN_IDS):
        match = [f for f in files if f.name == f"{i:02d}_{pid.lower()}.md"]
        if not match:
            raise SystemExit(f"找不到 {pid} 的定義檔 {paths.KNOWLEDGE_PATTERNS}/{i:02d}_{pid.lower()}.md")
        text = match[0].read_text(encoding="utf-8")
        text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
        text = re.sub(r"^#+\s*", "", text, flags=re.M)          # 標題符號拿掉，留文字
        out.append((pid, match[0].name, re.sub(r"\s+", " ", text).strip()))
    return out


def load_kb() -> dict:
    """讀 patterns.npz；不存在時提示先建。"""
    if not KB_FILE.exists():
        raise SystemExit(f"找不到 {KB_FILE}，先跑 python -m module_b_encoder.rag.pattern_kb")
    data = np.load(KB_FILE)
    ids = [str(x) for x in data["ids"]]
    if ids != patterns.PATTERN_IDS:
        raise SystemExit(f"{KB_FILE} 的型態 {ids} 跟 shared/patterns.PATTERN_IDS 不一致，"
                         "先重建：python -m module_b_encoder.rag.pattern_kb")
    return {"H_r_all": data["H_r_all"], "pooled": data["pooled"], "ids": [str(x) for x in data["ids"]],
            "meta": read_json(META_FILE) if META_FILE.exists() else {}}


def main():
    cfg = load_config()
    from module_b_encoder.encoders.text_encoder import TextEncoder
    enc = TextEncoder(cfg["encoders"]["text"])
    defs = read_definitions()
    h_all, pooled, meta = [], [], {"encoder": enc.model_id, "entries": []}
    for pid, name, text in defs:
        n = enc.n_tokens(text)
        if n > 512:
            raise SystemExit(f"{name} 有 {n} 個 token，超過 FinBERT 上限 512，請縮短")
        h_all.append(enc.encode(text))
        pooled.append(enc.encode_pooled(text))
        meta["entries"].append({"pattern": pid, "file": name, "n_tokens": n,
                                "sha256": hashlib.sha256(text.encode()).hexdigest()})
        print(f"[pattern_kb] {pid}: {n} tokens")
    paths.PATTERN_KB.mkdir(parents=True, exist_ok=True)
    np.savez(KB_FILE, H_r_all=np.stack(h_all), pooled=np.stack(pooled),
             ids=np.array([d[0] for d in defs]))
    write_json(meta, META_FILE)
    print(f"[pattern_kb] {len(defs)} 段定義 -> {KB_FILE}")


if __name__ == "__main__":
    main()
