"""Verify local links, assets and fragment targets in the rendered portfolio."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        for key in ("href", "src"):
            if key in attributes:
                self.links.append(attributes[key])


def check_site(site):
    site = Path(site).resolve()
    pages = {}
    for path in site.rglob("*.html"):
        parser = Page()
        parser.feed(path.read_text())
        pages[path] = parser
    if not pages:
        raise ValueError("No rendered HTML found; run quarto render first.")
    errors = []
    count = 0
    for path, page in pages.items():
        for reference in page.links:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc:
                continue
            relative = unquote(parsed.path)
            if relative.startswith("/"):
                target = site / relative.lstrip("/")
            else:
                target = path.parent / relative if relative else path
            target = target.resolve()
            if target.is_dir():
                target = target / "index.html"
            count += 1
            if not target.exists():
                errors.append(f"{path.relative_to(site)}: missing {reference}")
            elif parsed.fragment and target in pages and unquote(parsed.fragment) not in pages[target].ids:
                errors.append(f"{path.relative_to(site)}: missing anchor {reference}")
    if errors:
        raise ValueError("\n".join(errors))
    return len(pages), count


if __name__ == "__main__":
    pages, links = check_site(ROOT / "_site")
    print(f"Checked {pages} HTML pages and {links} local links/assets; no missing targets.")
