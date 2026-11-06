"""Check the buildless Pages artifact; research source stays outside docs/."""

from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1] / "docs"
PUBLIC_PDFS = {"assets/QuadFDB.pdf"}
ALLOWED = {
    ".html", ".css", ".js", ".json", ".svg", ".png", ".jpg", ".jpeg",
    ".webp", ".woff", ".woff2", ".txt", ".md", ".ico", ".mp4", ".webm",
}
errors = []


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.ids = set()
        self.links = []
        self.has_title = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "title":
            self.has_title = True
        if values.get("id"):
            if values["id"] in self.ids:
                errors.append(f"Duplicate id in {self.path.name}: {values['id']}")
            self.ids.add(values["id"])
        for key in ("src", "href", "poster"):
            if values.get(key):
                self.links.append(values[key])
        if tag == "img" and "alt" not in values:
            errors.append(f"Image lacks alt text in {self.path.name}")


if not (ROOT / "index.html").is_file():
    raise SystemExit("Missing docs/index.html")

pages = {}
files = list(ROOT.rglob("*"))
for path in files:
    if path.is_symlink():
        errors.append(f"Public artifact contains symlink: {path.relative_to(ROOT)}")
    if not path.is_file():
        continue
    is_public_pdf = path.relative_to(ROOT).as_posix() in PUBLIC_PDFS
    if path.name != ".nojekyll" and path.suffix.lower() not in ALLOWED and not is_public_pdf:
        errors.append(f"Unexpected public file type: {path.relative_to(ROOT)}")
    if any(part.startswith(".") and part != ".nojekyll" for part in path.relative_to(ROOT).parts):
        errors.append(f"Hidden file in public artifact: {path.relative_to(ROOT)}")
    if path.suffix.lower() in {".html", ".css", ".js", ".json", ".md", ".svg", ".txt"}:
        body = path.read_text(encoding="utf-8")
        if re.search(r"/Users/|file://|https?://(?:127\.0\.0\.1|localhost)|github_pat_|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY", body):
            errors.append(f"Local or credential reference in {path.relative_to(ROOT)}")
        if path.suffix.lower() == ".css":
            for ref in re.findall(r"url\(\s*['\"]?([^'\")\s]+)", body):
                url = urlsplit(ref)
                if not url.scheme and not ref.startswith("#") and not (path.parent / unquote(url.path)).is_file():
                    errors.append(f"Missing CSS asset: {ref}")
    if path.suffix.lower() == ".html":
        page = Page(path)
        page.feed(path.read_text(encoding="utf-8"))
        if not page.has_title:
            errors.append(f"Missing title: {path.name}")
        pages[path.resolve()] = page

for path, page in pages.items():
    for ref in page.links:
        url = urlsplit(ref)
        if url.scheme or url.netloc:
            continue
        if url.path.startswith("/"):
            errors.append(f"Root-relative link breaks repository Pages: {ref}")
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path
        if not target.is_relative_to(ROOT.resolve()):
            errors.append(f"Link leaves public artifact: {ref}")
            continue
        if not target.exists():
            errors.append(f"Missing linked file: {ref}")
        if url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
            errors.append(f"Missing fragment: {ref}")

if errors:
    raise SystemExit("\n".join(errors))

public_files = [path for path in files if path.is_file()]
size = sum(path.stat().st_size for path in public_files)
print(f"Public artifact verified: {len(public_files)} files, {size / 1048576:.1f} MiB; only docs/ is deployed.")
