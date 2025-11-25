from pathlib import Path
from bs4 import BeautifulSoup
import json


def parse_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_markdown_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_json_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, indent=2)


def parse_html_file(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")
