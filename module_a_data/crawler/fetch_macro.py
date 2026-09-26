"""下載 ETF／指數使用的美國總體經濟資料，存到 ``data/raw/macro/``。

資料來源全部免費，且優先使用官方來源：

* 指標：Federal Reserve Bank of St. Louis 的 FRED／ALFRED API。
* 總經新聞：Federal Reserve Board 的歷年 FOMC 新聞稿。
* 利率點陣圖：Federal Reserve Board 的 SEP accessible HTML；解析表格後重畫成
  224x224 PNG，讓既有 ViT 可以直接使用，同時保留結構化 JSON 供稽核。

FRED 資料使用 ``output_type=4`` 取得 initial release，並保存 ``available_date``
（FRED 回傳的 realtime_start）。下游必須依 available_date 做 as-of join，只能向後沿用，
不可把現在看到的修訂值回填到歷史日期，否則會造成 look-ahead bias。

環境變數：
    FRED_API_KEY=免費申請的 FRED API key

用法：
    python -m module_a_data.crawler.fetch_macro --kind indicators
    python -m module_a_data.crawler.fetch_macro --kind news
    python -m module_a_data.crawler.fetch_macro --kind dot_plot --meeting_date 2025-12-10
    python -m module_a_data.crawler.fetch_macro --kind all

``--kind all`` 會從 FOMC 新聞中找出 SEP 公布日並下載該日點陣圖。
"""
import argparse
import datetime as dt
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from module_a_data.preprocess.text_cleaner import clean
from shared import paths
from shared.utils import load_config, read_json, write_json

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FED_BASE_URL = "https://www.federalreserve.gov"
FED_FOMC_ARCHIVE_URL = FED_BASE_URL + "/newsevents/pressreleases/{year}-press-fomc.htm"
FED_PROJECTIONS_URLS = (
    FED_BASE_URL + "/monetarypolicy/fomcprojtabl{date}.htm",
    # 少數歷史頁面（例如 2022-03-16）使用 table，而非其他日期常見的 tabl。
    FED_BASE_URL + "/monetarypolicy/fomcprojtable{date}.htm",
)
ET = ZoneInfo("America/New_York")

REQUEST_TIMEOUT_SEC = 30
REQUEST_DELAY_SEC = 0.15
HEADERS = {
    "User-Agent": "MultiFinRL academic research project",
    "Accept-Language": "en-US,en;q=0.9",
}

# 第一版涵蓋通膨、就業、政策利率、殖利率曲線與市場風險；鍵值皆為 FRED series ID。
FRED_SERIES = {
    "CPIAUCSL": {"name": "consumer_price_index", "frequency": "monthly", "units": "index",
                 "initial_release": True},
    "PCEPILFE": {"name": "core_pce_price_index", "frequency": "monthly", "units": "index",
                  "initial_release": True},
    "PAYEMS": {"name": "nonfarm_payrolls", "frequency": "monthly", "units": "thousands",
               "initial_release": True},
    "UNRATE": {"name": "unemployment_rate", "frequency": "monthly", "units": "percent",
               "initial_release": True},
    # 下列日頻序列不使用 output_type=4；FRED 對這類序列的 initial-release 請求會回 400。
    # 它們不是會反覆修訂的總經統計值，直接以 observation_date 當 available_date。
    "DFEDTARU": {"name": "fed_target_upper", "frequency": "daily", "units": "percent",
                 "initial_release": False},
    "DGS10": {"name": "treasury_10y", "frequency": "daily", "units": "percent",
              "initial_release": False},
    "DGS2": {"name": "treasury_2y", "frequency": "daily", "units": "percent",
             "initial_release": False},
    "T10Y2Y": {"name": "treasury_10y_2y_spread", "frequency": "daily", "units": "percent",
               "initial_release": False},
    "VIXCLS": {"name": "vix_close", "frequency": "daily", "units": "index",
               "initial_release": False},
}


class FetchHTTPError(RuntimeError):
    """不暴露 query string 的 HTTP 錯誤，並保留 status code 供 fallback 判斷。"""

    def __init__(self, url: str, status_code: int, message: str):
        self.url = url
        self.status_code = status_code
        self.message = message
        super().__init__(f"GET {url} 回傳 HTTP {status_code}: {message}")


def get_fred_api_key(cli_key: str | None = None) -> str:
    """取得 FRED API key（CLI 優先，其次讀取 ``FRED_API_KEY``）。"""
    key = cli_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "缺少 FRED_API_KEY！請在專案根目錄的 .env 設定 "
            "FRED_API_KEY=your_key，或透過 --api_key 傳入。"
        )
    return key


def _get(url: str, *, params: dict | None = None, max_retries: int = 3) -> requests.Response:
    """帶 timeout 與簡單退避重試的 GET；429、5xx 與連線錯誤才重試。

    4xx 代表請求內容有誤，立即停止並顯示 API 回傳訊息。錯誤文字刻意不使用
    ``response.url``，避免 query string 中的 API key 被印到終端。
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SEC)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < max_retries:
                    wait = 2 ** attempt
                    print(f"[fetch_macro] HTTP {response.status_code}，{wait} 秒後重試")
                    time.sleep(wait)
                    continue
            if response.status_code >= 400:
                try:
                    body = response.json()
                    message = body.get("error_message") or body.get("message") or response.reason
                except (ValueError, AttributeError):
                    message = response.reason or "unknown error"
                raise FetchHTTPError(url, response.status_code, message)
            return response
        except (requests.ConnectionError, requests.Timeout) as error:
            if attempt + 1 >= max_retries:
                raise RuntimeError(f"GET {url} 連線失敗: {error}") from error
            wait = 2 ** attempt
            print(f"[fetch_macro] 連線失敗，{wait} 秒後重試")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _validate_date_range(start: str, end: str) -> None:
    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end)
    if start_date > end_date:
        raise ValueError(f"start 不可晚於 end: {start} > {end}")


def _to_float(raw: str | None) -> float | None:
    if raw in (None, "", "."):
        return None
    return float(raw)


def fetch_macro_indicators(
    start: str,
    end: str,
    api_key: str | None = None,
    series_ids: Iterable[str] | None = None,
) -> list[dict]:
    """從 FRED／ALFRED 抓初次發布值，回傳統一格式的觀測值列表。

    ``observation_date`` 是統計所屬期間，``available_date`` 才是當時市場真正可取得資料的
    日期。下游模型必須使用後者對齊交易日。
    """
    _validate_date_range(start, end)
    key = get_fred_api_key(api_key)
    requested = list(series_ids or FRED_SERIES)
    unknown = [series_id for series_id in requested if series_id not in FRED_SERIES]
    if unknown:
        raise ValueError(f"未知的 FRED series ID: {unknown}")

    records: list[dict] = []
    for series_id in requested:
        meta = FRED_SERIES[series_id]
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "asc",
        }
        if meta["initial_release"]:
            params.update({
                "realtime_start": "1776-07-04",
                "realtime_end": "9999-12-31",
                "output_type": 4,
            })
        else:
            params["output_type"] = 1

        data = _get(FRED_OBSERVATIONS_URL, params=params).json()
        observations = data.get("observations", [])
        for observation in observations:
            available_date = (
                observation.get("realtime_start")
                if meta["initial_release"]
                else observation["date"]
            )
            # observation_date 在區間內，不代表資料當時已經公布；例如月底數字可能下個月才發布。
            if available_date and available_date > end:
                continue
            records.append({
                "series_id": series_id,
                "name": meta["name"],
                "frequency": meta["frequency"],
                "units": meta["units"],
                "observation_date": observation["date"],
                "available_date": available_date,
                "value": _to_float(observation.get("value")),
                "source": (
                    "fred_alfred_initial_release"
                    if meta["initial_release"]
                    else "fred_daily_observation"
                ),
            })
        print(f"[fetch_macro] FRED {series_id}: {len(observations)} 筆")
        time.sleep(REQUEST_DELAY_SEC)

    return sorted(records, key=lambda item: (item["available_date"] or "", item["series_id"]))


def save_macro_indicators(start: str, end: str, records: list[dict]) -> Path:
    """保存總經指標；單一 index.json 方便下游一次讀取後做 as-of join。"""
    out = paths.RAW_MACRO_INDICATORS / "index.json"
    write_json({
        "source": "FRED/ALFRED",
        "start": start,
        "end": end,
        "point_in_time": True,
        "series": FRED_SERIES,
        "observations": records,
    }, out)
    print(f"[fetch_macro] indicators: {len(records)} 筆 -> {out}")
    return out


def _find_dot_plot_table(soup: BeautifulSoup):
    marker = soup.find(string=re.compile(r"Figure\s*2\.", re.IGNORECASE))
    if marker:
        table = marker.parent.find_next("table")
        if table:
            return table
    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        if "Longer run" in text and "Midpoint" in text:
            return table
    raise ValueError("SEP HTML 中找不到 Figure 2 利率點陣圖表格")


def _parse_dot_plot_html(html: str, meeting_date: str) -> dict:
    """解析 SEP accessible HTML 的 Figure 2，轉為可稽核的結構化點數。"""
    soup = BeautifulSoup(html, "lxml")
    table = _find_dot_plot_table(soup)
    rows = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)

    horizons: list[str] = []
    header_index = -1
    for index, cells in enumerate(rows):
        candidates = [cell for cell in cells if re.fullmatch(r"20\d{2}", cell) or "Longer run" in cell]
        if "Longer run" in candidates and len(candidates) >= 2:
            horizons = candidates
            header_index = index
            break
    if not horizons:
        raise ValueError("SEP Figure 2 找不到年份欄位")

    points = []
    for cells in rows[header_index + 1:]:
        if not cells:
            continue
        try:
            rate = float(cells[0])
        except ValueError:
            continue
        values = cells[1:1 + len(horizons)]
        if len(values) < len(horizons):
            values += [""] * (len(horizons) - len(values))
        counts = {}
        for horizon, raw_count in zip(horizons, values):
            raw_count = raw_count.strip()
            counts[horizon] = int(raw_count) if raw_count.isdigit() else 0
        points.append({"rate": rate, "counts": counts})

    if not points or not any(sum(point["counts"].values()) for point in points):
        raise ValueError("SEP Figure 2 沒有可用的點陣圖數值")
    return {
        "meeting_date": meeting_date,
        "available_date": meeting_date,
        "source": "federal_reserve_sep_accessible_html",
        "horizons": horizons,
        "points": points,
    }


def _render_dot_plot(data: dict, out: Path) -> None:
    """把結構化點數重畫成固定 224x224 RGB PNG，供既有 ViT 使用。"""
    # 延後匯入，讓只抓 FRED／新聞或查看 --help 時不必初始化 Matplotlib。
    import matplotlib.pyplot as plt
    from PIL import Image

    horizons = data["horizons"]
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    for x, horizon in enumerate(horizons):
        for point in data["points"]:
            count = point["counts"].get(horizon, 0)
            if count <= 0:
                continue
            offsets = [0.0] if count == 1 else [(-0.12 + 0.24 * i / (count - 1)) for i in range(count)]
            ax.scatter([x + offset for offset in offsets], [point["rate"]] * count,
                       s=9, color="#1f4e79", edgecolors="none")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([h.replace("Longer run", "Long run") for h in horizons], fontsize=5)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_ylabel("Fed funds rate (%)", fontsize=6)
    ax.set_title(f"FOMC dot plot {data['meeting_date']}", fontsize=7)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    fig.tight_layout(pad=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    # Matplotlib 預設會保存 RGBA；專案影像契約要求 3-channel RGB。
    with Image.open(out) as image:
        image.convert("RGB").save(out)


def fetch_fed_dot_plot(meeting_date: str) -> str:
    """下載並解析某次 FOMC SEP 點陣圖，回傳 224x224 PNG 的路徑。"""
    dt.date.fromisoformat(meeting_date)
    compact_date = meeting_date.replace("-", "")
    response = None
    url = ""
    not_found_urls = []
    for template in FED_PROJECTIONS_URLS:
        candidate_url = template.format(date=compact_date)
        try:
            response = _get(candidate_url)
            url = candidate_url
            break
        except FetchHTTPError as error:
            if error.status_code != 404:
                raise
            not_found_urls.append(candidate_url)
    if response is None:
        tried = ", ".join(not_found_urls)
        raise RuntimeError(f"找不到 {meeting_date} 的 FOMC SEP accessible HTML；已嘗試: {tried}")

    data = _parse_dot_plot_html(response.text, meeting_date)
    data["url"] = url

    out_dir = paths.RAW_MACRO_FOMC / meeting_date
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source.html").write_text(response.text, encoding="utf-8")
    write_json(data, out_dir / "dot_plot.json")
    image_path = out_dir / "dot_plot.png"
    _render_dot_plot(data, image_path)
    print(f"[fetch_macro] FOMC dot plot {meeting_date} -> {image_path}")
    return str(image_path)


def _fomc_category(headline: str) -> str:
    lower = headline.lower()
    if "economic projections" in lower or "projection materials" in lower:
        return "economic_projections"
    if "minutes" in lower:
        return "fomc_minutes"
    if "statement" in lower:
        return "fomc_statement"
    return "monetary_policy"


def _release_date_from_url(url: str) -> str | None:
    match = re.search(r"monetary(20\d{6})", url)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()


def _parse_fomc_archive(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, dict] = {}
    for link in soup.find_all("a", href=True):
        url = urljoin(FED_BASE_URL, link["href"])
        release_date = _release_date_from_url(url)
        if not release_date:
            continue
        headline = clean(link.get_text(" ", strip=True))
        if not headline:
            continue
        found[url] = {"url": url, "date": release_date, "headline": headline}
    return sorted(found.values(), key=lambda item: (item["date"], item["url"]))


def _parse_fomc_article(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    headline_node = (
        soup.select_one("h3.title")
        or soup.select_one("#content h3")
        or soup.select_one("main h3")
        or soup.find("h1", string=re.compile(r"\S"))
    )
    headline = clean(headline_node.get_text(" ", strip=True)) if headline_node else ""
    main = soup.select_one("#content") or soup.find("main") or soup.body
    if main is None:
        return headline, ""
    for unwanted in main.find_all(["script", "style", "nav", "footer", "form"]):
        unwanted.decompose()
    paragraphs = [clean(p.get_text(" ", strip=True)) for p in main.find_all("p")]
    content = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
    return headline, content


def _published_at(release_date: str) -> str:
    # FOMC statement、minutes 與 projections 通常在美東時間 14:00 發布。
    day = dt.date.fromisoformat(release_date)
    return dt.datetime.combine(day, dt.time(14, 0), tzinfo=ET).isoformat()


def fetch_macro_news(start: str, end: str) -> list[dict]:
    """抓取指定期間的 Fed FOMC 新聞，格式比照 ``fetch_news_alpaca.py``。"""
    _validate_date_range(start, end)
    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end)
    links: dict[str, dict] = {}
    for year in range(start_date.year, end_date.year + 1):
        url = FED_FOMC_ARCHIVE_URL.format(year=year)
        for item in _parse_fomc_archive(_get(url).text):
            if start <= item["date"] <= end:
                links[item["url"]] = item
        print(f"[fetch_macro] FOMC archive {year}: 累積 {len(links)} 個連結")
        time.sleep(REQUEST_DELAY_SEC)

    items = []
    for index, item in enumerate(sorted(links.values(), key=lambda value: value["date"]), start=1):
        response = _get(item["url"])
        parsed_headline, content = _parse_fomc_article(response.text)
        # Archive 頁的連結文字最穩定；文章頁標題只在 archive 缺字時補用。
        headline = item["headline"] or parsed_headline
        items.append({
            "headline": headline,
            "content": content,
            "source": "federal_reserve",
            "published_at": _published_at(item["date"]),
            "url": item["url"],
            "category": _fomc_category(headline),
        })
        print(f"[fetch_macro] FOMC news {index}/{len(links)}: {item['date']} {headline}")
        time.sleep(REQUEST_DELAY_SEC)
    return items


def save_macro_news(items: list[dict]) -> Path:
    """依發布日保存，既有檔案按 URL 合併去重，行為與個股新聞 crawler 一致。"""
    by_day: dict[str, list[dict]] = {}
    for item in items:
        day = dt.datetime.fromisoformat(item["published_at"]).date().isoformat()
        by_day.setdefault(day, []).append(item)

    for day, day_items in by_day.items():
        out = paths.RAW_MACRO_NEWS / f"{day}.json"
        existing = read_json(out).get("news", []) if out.exists() else []
        seen_urls = {item.get("url") for item in existing if item.get("url")}
        merged = existing + [item for item in day_items if item.get("url") not in seen_urls]
        write_json({"date": day, "news": merged}, out)
    print(f"[fetch_macro] news: {len(items)} 則 -> {len(by_day)} 天 -> {paths.RAW_MACRO_NEWS}")
    return paths.RAW_MACRO_NEWS


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="下載 ETF／指數使用的免費官方總經資料")
    ap.add_argument("--kind", choices=("all", "indicators", "news", "dot_plot"), default="all")
    ap.add_argument("--start", default=cfg["date_range"]["start"])
    ap.add_argument("--end", default=cfg["date_range"]["end"])
    ap.add_argument("--api_key", default=None, help="FRED API key（未提供則讀 FRED_API_KEY）")
    ap.add_argument("--meeting_date", action="append", default=[],
                    help="SEP 公布日 YYYY-MM-DD；可重複指定。未指定時 all 會從新聞自動找")
    args = ap.parse_args()

    if args.kind in ("all", "indicators"):
        records = fetch_macro_indicators(args.start, args.end, args.api_key)
        save_macro_indicators(args.start, args.end, records)

    news = []
    if args.kind in ("all", "news"):
        news = fetch_macro_news(args.start, args.end)
        save_macro_news(news)

    meeting_dates = list(dict.fromkeys(args.meeting_date))
    if args.kind == "all" and not meeting_dates:
        meeting_dates = sorted({
            item["published_at"][:10]
            for item in news
            if item.get("category") == "economic_projections"
        })
    if args.kind == "dot_plot" and not meeting_dates:
        ap.error("--kind dot_plot 必須至少提供一個 --meeting_date")
    if args.kind in ("all", "dot_plot"):
        for meeting_date in meeting_dates:
            fetch_fed_dot_plot(meeting_date)
            time.sleep(REQUEST_DELAY_SEC)


if __name__ == "__main__":
    main()
