"""清洗新聞與文件文字：去 HTML 標籤、廣告雜訊、多餘空白。"""
import re

from bs4 import BeautifulSoup

# 常見雜訊行（廣告、訂閱提示等），可持續補充
_NOISE_PATTERNS = [
    re.compile(r"(?i)subscribe to .*"),
    re.compile(r"(?i)sign up for .*newsletter.*"),
    re.compile(r"(?i)read more:.*"),
    re.compile(r"(?i)related articles?:.*"),
]


def clean_html(raw_html: str) -> str:
    """去除 HTML 標籤與 script/style 等區塊，回傳純文字。"""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
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
