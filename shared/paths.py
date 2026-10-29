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
RAW_MACRO = RAW / "macro"          # ETF/指數用的總經資料（decisions.md #38）
RAW_MACRO_INDICATORS = RAW_MACRO / "indicators"
RAW_MACRO_NEWS = RAW_MACRO / "news"
RAW_MACRO_FOMC = RAW_MACRO / "fomc"

DATASET = DATA / "processed" / "dataset"   # A 的交付物
TEMPORAL_SPLITS = DATA / "processed" / "temporal_splits"  # 共用時間切分 manifest
VECTORS = DATA / "vectors"                  # B 的交付物
OUTPUTS = DATA / "outputs"                  # C 的產出
PIPELINE_REPORTS = OUTPUTS / "reports"      # 每次正式 pipeline 的 Markdown／圖表 bundle
LABELS = DATA / "labels"                    # 人工／LLM 標記的答案卷（進版本控制，見 .gitignore）


def daily_json(ticker: str, date: str) -> Path:
    """A 的每日 JSON 路徑。date 格式 YYYY-MM-DD。"""
    return DATASET / ticker / f"{date}.json"


def vectors_root(tag: str | None = None) -> Path:
    """向量根目錄；tag（例如 recency）是另一套檢索設定的實驗 H_r，存在 data/vectors_{tag}/，
    該目錄只放 H_r.npy 與 index.json，H_v／H_t 仍讀正式的 data/vectors/。"""
    return DATA / f"vectors_{tag}" if tag else VECTORS


def vector_dir(ticker: str, date: str, tag: str | None = None) -> Path:
    """B 的每日向量目錄路徑。"""
    return vectors_root(tag) / ticker / date


def temporal_split_path(ticker: str) -> Path:
    """A/B/C 共用的時間切分 manifest；所有會學習參數的模組都必須沿用。"""
    return TEMPORAL_SPLITS / f"{ticker}.json"


def event_ground_truth_path(ticker: str) -> Path:
    """事件抽取 ground truth 路徑：{date: [event_type, ...]} 的字典。"""
    return LABELS / "event_ground_truth" / f"{ticker}.json"


def event_extraction_llm_cache_path(ticker: str) -> Path:
    """LLM-based 事件抽取的中斷續跑快取：{date: [event_type, ...]}，跟 ground truth 格式
    相同但語意不同——這是機器抽取的結果，不是答案卷，不進版本控制（見 .gitignore data/outputs/）。
    """
    return OUTPUTS / "metrics" / "event_extraction_llm_cache" / f"{ticker}.json"


def y_belief_path(ticker: str) -> Path:
    """decoder 訓練目標 y_belief 路徑：{date: y_belief 文字}。是答案卷，進版本控制，
    見 module_c_fusion/decoder/generate_y_belief.py。
    """
    return LABELS / "y_belief" / f"{ticker}.json"


def grounding_labels_path(ticker: str, vectors_tag: str | None = None) -> Path:
    """L_ground 的 LLM 相關性分數：{date: {chunk_ids, scores, reasons, ...}}。是答案卷，
    進版本控制，見 module_c_fusion/grounding/generate_alpha_star.py。
    vectors_tag（例如 recency）對應另一套檢索的 H_r，答案卷另存 {ticker}_{tag}.json。
    """
    return LABELS / "grounding" / (f"{ticker}_{vectors_tag}.json" if vectors_tag else f"{ticker}.json")


def fusion_checkpoint_path(tickers: str | list[str], tag: str | None = None) -> Path:
    """Fusion checkpoint；單股與多股都把 ticker 寫進檔名，避免互相覆蓋。
    tag（例如 grounding 實驗組名）會接在檔名最後，實驗組不會蓋掉正式 checkpoint。
    """
    names = [tickers] if isinstance(tickers, str) else list(tickers)
    if tag:
        names.append(tag)
    return OUTPUTS / "checkpoints" / f"fusion_{'_'.join(names)}.pt"


def z_fused_dir(tag: str | None = None) -> Path:
    """Z_fused 輸出根目錄；有 tag 時寫到 z_fused_{tag}/，不覆蓋正式 Z_fused。"""
    return OUTPUTS / (f"z_fused_{tag}" if tag else "z_fused")


def ppo_checkpoint_path(ticker: str, *, fake: bool = False) -> Path:
    """PPO agent checkpoint；fake 使用獨立 namespace。"""
    suffix = "fake" if fake else ticker
    return OUTPUTS / "checkpoints" / f"ppo_agent_{suffix}.zip"


def classification_report_path(ticker: str, tag: str | None = None) -> Path:
    suffix = f"{ticker}_{tag}" if tag else ticker
    return OUTPUTS / "metrics" / f"classification_report_{suffix}.json"


def ppo_train_report_path(ticker: str, *, fake: bool = False) -> Path:
    suffix = "fake" if fake else ticker
    return OUTPUTS / "metrics" / f"ppo_train_report_{suffix}.json"


def backtest_report_path(ticker: str, strategy: str, run_id: str) -> Path:
    """回測報告同時含 ticker、strategy 與秒級 run_id，不同策略不會在同秒互蓋。"""
    return OUTPUTS / "backtest" / f"report_{ticker}_{strategy}_{run_id}.json"


def pipeline_report_dir(run_id: str) -> Path:
    """Pipeline 報告只以秒級 run_id 分目錄；ticker 寫在報告內容，不放入路徑。"""
    return PIPELINE_REPORTS / run_id
