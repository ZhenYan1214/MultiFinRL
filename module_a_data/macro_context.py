"""把總經 raw data 依交易日整理成 point-in-time context。

`fetch_macro.py` 負責下載；本模組只讀本機檔案。所有選取都以 `available_date <= date`
為條件，禁止用 observation_date 回填尚未公布的數字，也不會使用未來的 FOMC 點陣圖。
"""
import datetime as dt
from pathlib import Path

from shared import paths
from shared.utils import read_json


class MacroDataStore:
    """一次載入總經索引，供 `build_dataset` 對多個交易日做 as-of join。"""

    def __init__(
        self,
        indicators_path: Path | None = None,
        news_dir: Path | None = None,
        fomc_dir: Path | None = None,
        root: Path | None = None,
        news_backfill_days: int = 7,
    ):
        self.indicators_path = indicators_path or paths.RAW_MACRO_INDICATORS / "index.json"
        self.news_dir = news_dir or paths.RAW_MACRO_NEWS
        self.fomc_dir = fomc_dir or paths.RAW_MACRO_FOMC
        self.root = root or paths.ROOT
        self.news_backfill_days = news_backfill_days

        if not self.indicators_path.exists():
            raise FileNotFoundError(
                f"找不到總經指標 {self.indicators_path}；先跑 "
                "python -m module_a_data.crawler.fetch_macro --kind all"
            )
        observations = read_json(self.indicators_path).get("observations", [])
        self._observations: dict[str, list[dict]] = {}
        for item in observations:
            available_date = item.get("available_date")
            if not available_date:
                continue
            self._observations.setdefault(item["series_id"], []).append(item)
        for values in self._observations.values():
            values.sort(key=lambda item: (item["observation_date"], item["available_date"]))

        self._news_files = {
            file.stem: file for file in self.news_dir.glob("*.json") if _is_iso_date(file.stem)
        } if self.news_dir.exists() else {}

        self._dot_plots = []
        if self.fomc_dir.exists():
            for json_path in sorted(self.fomc_dir.glob("*/dot_plot.json")):
                data = read_json(json_path)
                available_date = data.get("available_date") or data.get("meeting_date")
                image_path = json_path.with_name("dot_plot.png")
                if not available_date or not image_path.exists():
                    continue
                self._dot_plots.append({
                    "available_date": available_date,
                    "meeting_date": data.get("meeting_date", available_date),
                    "path": _relative_path(image_path, self.root),
                    "source_json": _relative_path(json_path, self.root),
                    "source": data.get("source", "federal_reserve_sep_accessible_html"),
                })
        self._dot_plots.sort(key=lambda item: item["available_date"])
        if not self._dot_plots:
            raise FileNotFoundError(
                f"找不到 FOMC 點陣圖 {self.fomc_dir}/*/dot_plot.png；先跑 "
                "python -m module_a_data.crawler.fetch_macro --kind all"
            )

    def indicators_as_of(self, date: str) -> list[dict]:
        """每個 series 取交易日收盤前已公布的最新一期觀測值。"""
        dt.date.fromisoformat(date)
        selected = []
        for series_id, values in self._observations.items():
            eligible = [item for item in values if item["available_date"] <= date]
            if not eligible:
                continue
            # 先選最新統計所屬期；相同所屬期若有多筆，以當時已知的最新 vintage 為準。
            selected.append(max(
                eligible,
                key=lambda item: (item["observation_date"], item["available_date"]),
            ))
        return sorted(selected, key=lambda item: item["series_id"])

    def news_as_of(self, date: str) -> list[dict]:
        """取當日及前 N 個日曆日發布的 FOMC 新聞，不取未來檔案。"""
        current = dt.date.fromisoformat(date)
        result = []
        for days_ago in range(self.news_backfill_days + 1):
            day = (current - dt.timedelta(days=days_ago)).isoformat()
            file = self._news_files.get(day)
            if not file:
                continue
            for item in read_json(file).get("news", []):
                published_at = item.get("published_at", "")
                if published_at[:10] > date:
                    continue
                result.append({
                    "headline": item["headline"],
                    "content": item["content"],
                    "source": item["source"],
                    "published_at": published_at,
                    "days_ago": days_ago,
                    "category": item.get("category", "monetary_policy"),
                    "url": item.get("url"),
                })
        return sorted(result, key=lambda item: item["published_at"], reverse=True)

    def dot_plot_as_of(self, date: str) -> dict | None:
        """取交易日以前最新一份 SEP 點陣圖；發布前不回填。"""
        dt.date.fromisoformat(date)
        eligible = [item for item in self._dot_plots if item["available_date"] <= date]
        return dict(eligible[-1]) if eligible else None

    def build_daily_context(self, date: str) -> dict | None:
        dot_plot = self.dot_plot_as_of(date)
        if dot_plot is None:
            return None
        return {
            "as_of_date": date,
            "indicators": self.indicators_as_of(date),
            "news": self.news_as_of(date),
            "dot_plot": dot_plot,
        }


def _relative_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"總經檔案必須位於專案根目錄 {root} 內: {path}") from error
    return str(relative).replace("\\", "/")


def _is_iso_date(value: str) -> bool:
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True
