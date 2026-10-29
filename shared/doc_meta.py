"""chunk_id 解析與文件時效（B 的檢索與 C 的 L_ground 共用）。

chunk_id 命名規則（module_a_data/preprocess/chunker.py）：{TICKER}_{DOC_TYPE}_{YYYY-MM-DD}_{序號}，
DOC_TYPE 可能含底線（earnings_call）。macro 模式另有兩種（generate_vectors.py 的 _macro_chunks）：
    MACRO_NEWS_{YYYYMMDD}_{digest}
    MACRO_{series_id}_{observation YYYYMMDD}_{available YYYYMMDD}   # 以 available 日期為文件日期
retriever 檢索不足 K 份時以 "PAD" 補齊。

被取代次數（superseded）：在分析日當下回頭看，這份文件發布之後、分析日（含）之前，又發布了
幾份同類新文件。10-K 與 10-Q 都報告最新財務數字，合併計算；8-K 是兩份定期財報之間的重大事件，
以「之後又出了幾份 10-K/10-Q」計算（build_dataset 也在新定期財報出來後淘汰舊 8-K）；法說會以
之後又開了幾場計算。只用分析日以前已發布的文件，不會洩漏未來資訊。macro chunk 一律記 0。
"""
import bisect
import datetime as dt
import functools
import re

from shared import paths
from shared.utils import read_json

PAD_CHUNK_ID = "PAD"
COMPANY_FORMS = ("10-K", "10-Q", "8-K", "earnings_call")

_COMPANY_RE = re.compile(r"^[^_]+_(?P<form>.+)_(?P<date>\d{4}-\d{2}-\d{2})_\d+$")


def _yyyymmdd(value: str) -> dt.date:
    return dt.date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def parse_chunk_id(chunk_id: str) -> tuple[str, dt.date | None]:
    """回傳 (來源名稱, 文件日期)；PAD 回傳 ("PAD", None)，無法解析回傳 ("unknown", None)。"""
    if chunk_id == PAD_CHUNK_ID:
        return "PAD", None
    if chunk_id.startswith("MACRO_NEWS_"):
        parts = chunk_id.split("_")
        try:
            return "macro_news", _yyyymmdd(parts[2])
        except (IndexError, ValueError):
            return "macro_news", None
    if chunk_id.startswith("MACRO_"):
        try:
            return "macro_indicator", _yyyymmdd(chunk_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return "macro_indicator", None
    m = _COMPANY_RE.match(chunk_id)
    if not m:
        return "unknown", None
    form = m.group("form")
    source = form if form in COMPANY_FORMS else "unknown"
    return source, dt.date.fromisoformat(m.group("date"))


@functools.lru_cache(maxsize=None)
def document_dates(ticker: str) -> dict[str, tuple[str, ...]]:
    """{"periodic": 10-K/10-Q 的發布日, "earnings_call": 法說會日期}，皆已排序；找不到索引檔時為空。"""
    out = {"periodic": (), "earnings_call": ()}
    filings = paths.RAW_FILINGS / ticker / "index.json"
    if filings.exists():
        out["periodic"] = tuple(sorted(f["filing_date"] for f in read_json(filings)["filings"]
                                       if f["form"] in ("10-K", "10-Q")))
    transcripts = paths.RAW_TRANSCRIPTS / ticker / "index.json"
    if transcripts.exists():
        out["earnings_call"] = tuple(sorted(t["event_date"] for t in read_json(transcripts)["transcripts"]))
    return out


def superseded_count(ticker: str | None, source: str, doc_date: dt.date | None, date: str) -> int:
    """這份文件之後、分析日（含）之前發布的同類新文件數（見檔頭說明）。"""
    if ticker is None or doc_date is None:
        return 0
    pool = {"10-K": "periodic", "10-Q": "periodic", "8-K": "periodic",
            "earnings_call": "earnings_call"}.get(source)
    if pool is None:
        return 0
    dates = document_dates(ticker)[pool]
    return max(bisect.bisect_right(dates, date) - bisect.bisect_right(dates, doc_date.isoformat()), 0)
