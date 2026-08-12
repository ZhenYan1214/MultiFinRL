"""總體經濟資料抓取，給 ETF/指數用，取代個股才有的財報、法說會（decisions.md #38）。

背景：ETF、指數不是單一公司，沒有自己的財報、法說會可以抓。教授確認（#38）這部分
改用總經數據替代——CPI、PCE 這類總經指標，甚至聯準會的利率點陣圖。點陣圖本身是
圖表格式，教授特別提到可以直接餵給現有的 ViT 視覺編碼器，跟 K 線圖用同一套架構，
不需要另外設計新的模態。

現況：這支腳本只先把架構寫出來，三個函式都還沒有實作、也還沒有跑過。目前計畫仍
以個股（AAPL）為主（configs/config.yaml 的 tickers），還沒有真的要擴充到 ETF，
等真的要做這塊的時候，再回來確認資料來源、把 NotImplementedError 換成真正的實作。

規劃中要抓的三類資料：
    1. 總經指標時間序列：CPI、PCE、非農就業、Fed 利率決策等（表格/數字格式）。
       候選來源：FRED（Federal Reserve Economic Data，聖路易聯準銀行維護，免費 API），
       這類總經數據最常見的免費來源，待確認是否符合需求。
    2. 總經新聞：跟總經事件相關的新聞。抓取邏輯可能可以直接沿用
       `fetch_news_alpaca.py`，只是查詢範圍從個股新聞換成總經新聞，待確認 Alpaca
       News API 有沒有涵蓋這塊。
    3. 聯準會利率點陣圖：FOMC 每季公布一次，圖片/圖表格式，教授建議可以直接用
       現有的 ViT 視覺編碼器處理（跟 K 線圖同一套架構）。來源待確認（聯準會官網
       應該有公開發布）。

輸出格式待定，暫定沿用其他 raw 資料的模式：存到 `data/raw/macro/{ticker_or_index}/`
底下，之後 build_dataset.py 要整合進每日 JSON 記錄時再一併定義格式（比照
`docs/data_format.md` 現有的做法）。
"""
import argparse

from shared import paths


def fetch_macro_indicators(start: str, end: str) -> list[dict]:
    """抓總經指標時間序列（CPI、PCE、非農就業、Fed 利率等）。

    未實作，資料來源待確認（候選：FRED API）。
    """
    raise NotImplementedError(
        "總經指標抓取還沒實作，見 docs/decisions.md #38，先確認資料來源（例如 FRED API）"
    )


def fetch_fed_dot_plot(meeting_date: str) -> str:
    """抓某次 FOMC 會議公布的利率點陣圖，回傳存檔路徑（圖片格式）。

    未實作，資料來源待確認；教授建議這張圖可以直接進現有的 ViT 視覺編碼器，
    跟 K 線圖用同一套架構，不需要另外設計新的視覺輸入格式。
    """
    raise NotImplementedError(
        "點陣圖抓取還沒實作，見 docs/decisions.md #38，先確認聯準會官網的公開資料格式"
    )


def fetch_macro_news(start: str, end: str) -> list[dict]:
    """抓總經相關新聞，格式比照 fetch_news_alpaca.py 的輸出。

    未實作；有可能可以直接重用 fetch_news_alpaca.py 的抓取邏輯，只是換查詢範圍，
    待確認 Alpaca News API 能不能涵蓋總經類新聞。
    """
    raise NotImplementedError(
        "總經新聞抓取還沒實作，見 docs/decisions.md #38，先確認能不能沿用 fetch_news_alpaca.py"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-08-09")
    ap.parse_args()
    raise NotImplementedError(
        "這支腳本目前只有架構，還沒有實作。要開始做 ETF/指數這塊時，"
        "先確認總經資料的來源（例如 FRED API）跟點陣圖的抓取方式，再回來補上三個函式。"
    )


if __name__ == "__main__":
    main()
