"""K 線型態與成交量標籤的唯一清單（docs/decisions.md #109、#112）。

K 線文字敘述（module_a_data/chart_caption.py）、RAG-1 型態定義知識庫
（data/knowledge/patterns/*.md、module_b_encoder/rag/pattern_kb.py）、
型態一致率評估（module_c_fusion/decoder/evaluate.py）都讀這裡，順序固定不可改：
RAG-1 知識庫的第 i 段定義就是 PATTERNS[i]。
"""

# 7 類 K 線型態：代號 -> (中文, 英文名稱)
# decisions.md #128：頭肩頂、頭肩底、W 底、M 頭合併成反轉；#129：反轉再拆成頂部、底部
PATTERNS: dict[str, tuple[str, str]] = {
    "UPTREND": ("持續上漲", "Uptrend"),
    "DOWNTREND": ("持續下跌", "Downtrend"),
    "CONSOLIDATION": ("盤整", "Consolidation"),
    "BREAKOUT": ("突破", "Breakout"),
    "BREAKDOWN": ("跌破", "Breakdown"),
    "REVERSAL_TOP": ("頂部反轉", "Top Reversal"),
    "REVERSAL_BOTTOM": ("底部反轉", "Bottom Reversal"),
}
PATTERN_IDS: list[str] = list(PATTERNS)

# 成交量標籤：四選一，依序判斷（decisions.md #127）。爆量只看最後 3 天，
# 視窗前段的高量不算爆量（前段高量會拉高前 15 日均量，本來就會反映成縮量）。
VOLUME_LABELS: dict[str, tuple[str, str]] = {
    "SPIKE": ("爆量", "Volume spike"),
    "EXPANDING": ("放量", "Expanding volume"),
    "CONTRACTING": ("縮量", "Contracting volume"),
    "FLAT": ("量平", "Flat volume"),
}
VOLUME_LABEL_IDS: list[str] = list(VOLUME_LABELS)

# 成交量公式門檻（架構頁 a_caption，門檻待調；Haiku 的 prompt 用同一組定義）
VOLUME_SPIKE_DAYS = 3           # 爆量：最後 3 天內
VOLUME_SPIKE_RATIO = 2.0        #       有一天 > 20 日均量 2 倍
VOLUME_RECENT_DAYS = 5          # 放量／縮量：近 5 日均量 ÷ 前 15 日均量
VOLUME_EXPAND_RATIO = 1.3       # > 1.3 -> 放量
VOLUME_CONTRACT_RATIO = 0.77    # < 0.77 -> 縮量

# Decoder 敘述裡辨識型態用的關鍵字（中英文都收；y_belief 與生成文字是中文）
PATTERN_KEYWORDS: dict[str, list[str]] = {
    "UPTREND": ["持續上漲", "上升趨勢", "多頭排列", "uptrend"],
    "DOWNTREND": ["持續下跌", "下降趨勢", "空頭排列", "downtrend"],
    "CONSOLIDATION": ["盤整", "整理", "區間震盪", "consolidation", "sideways"],
    "BREAKOUT": ["突破", "breakout"],
    "BREAKDOWN": ["跌破", "breakdown"],
    "REVERSAL_TOP": ["頂部反轉", "頭肩頂", "M 頭", "M頭", "雙頂", "倒 V", "top reversal",
                     "head and shoulders top", "double top", "inverted v"],
    "REVERSAL_BOTTOM": ["底部反轉", "頭肩底", "W 底", "W底", "雙底", "V 型反轉", "bottom reversal",
                        "head and shoulders bottom", "inverse head and shoulders", "double bottom"],
}
