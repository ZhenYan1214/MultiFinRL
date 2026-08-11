"""路徑常數：所有模組一律從這裡取路徑，不硬編字串。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "config.yaml"

DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_OHLCV = RAW / "ohlcv"
RAW_CHARTS = RAW / "charts"
RAW_NEWS = RAW / "news"
RAW_FILINGS = RAW / "filings"
RAW_TRANSCRIPTS = RAW / "transcripts"

DATASET = DATA / "processed" / "dataset"   # A 的交付物
VECTORS = DATA / "vectors"                  # B 的交付物
OUTPUTS = DATA / "outputs"                  # C 的產出
LABELS = DATA / "labels"                    # 人工／LLM 標記的答案卷（進版本控制，見 .gitignore）


def daily_json(ticker: str, date: str) -> Path:
    """A 的每日 JSON 路徑。date 格式 YYYY-MM-DD。"""
    return DATASET / ticker / f"{date}.json"


def vector_dir(ticker: str, date: str) -> Path:
    """B 的每日向量目錄路徑。"""
    return VECTORS / ticker / date


def event_ground_truth_path(ticker: str) -> Path:
    """事件抽取 ground truth 路徑：{date: [event_type, ...]} 的字典。"""
    return LABELS / "event_ground_truth" / f"{ticker}.json"


def event_extraction_llm_cache_path(ticker: str) -> Path:
    """LLM-based 事件抽取的中斷續跑快取：{date: [event_type, ...]}，跟 ground truth 格式
    相同但語意不同——這是機器抽取的結果，不是答案卷，不進版本控制（見 .gitignore data/outputs/）。
    """
    return OUTPUTS / "metrics" / "event_extraction_llm_cache" / f"{ticker}.json"
