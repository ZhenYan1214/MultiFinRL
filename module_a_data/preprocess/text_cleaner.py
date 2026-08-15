"""清洗新聞與文件文字：去 HTML 標籤、廣告雜訊、多餘空白。

新聞、財報共用同一套 clean()，兩種來源各自的雜訊類型不同：
    - 新聞頁面：訂閱／電子報／相關文章這類版面雜訊（_NOISE_PATTERNS，逐行比對）。
    - 財報 HTML（SEC EDGAR）：現代 10-K/10-Q 幾乎都是 inline XBRL（iXBRL）格式，
      內嵌一個 <ix:header> 區塊放機器可讀的財務標記資料（給軟體解析用，不是給人看的
      正文），以及部分區塊用 inline style="display:none"／"visibility:hidden"
      刻意隱藏（例如封面頁重複揭露、輔助工具用的替代文字），這兩者不去除的話，
      get_text() 會把這些視覺上看不到、跟正文無關的標記字串一起撈進清洗後的文字。
      這兩條規則對新聞頁面不會誤傷——新聞 HTML 一般不會有這些財報特有的結構。
"""
import re

from bs4 import BeautifulSoup

# 常見雜訊行（廣告、訂閱提示等，新聞頁面適用），可持續補充
_NOISE_PATTERNS = [
    re.compile(r"(?i)subscribe to .*"),
    re.compile(r"(?i)sign up for .*newsletter.*"),
    re.compile(r"(?i)read more:.*"),
    re.compile(r"(?i)related articles?:.*"),
]

# 財報 HTML 常見的隱藏內容（display:none / visibility:hidden），視覺上人類讀者看不到
_HIDDEN_STYLE_RE = re.compile(r"(?i)display\s*:\s*none|visibility\s*:\s*hidden")


def clean_html(raw_html: str) -> str:
    """去除 HTML 標籤、script/style 等區塊、iXBRL 隱藏標記，回傳純文字。"""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "aside", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE_RE):
        tag.decompose()
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    """清洗純文字：去雜訊行、壓縮空白。"""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(p.search(line) for p in _NOISE_PATTERNS):
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def clean(raw: str) -> str:
    """一站式：HTML -> 乾淨純文字。輸入若已是純文字也安全。"""
    return clean_text(clean_html(raw))
