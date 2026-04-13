"""
scrape_docs.py
Scrapes Supabase documentation pages and saves them as clean text files.

Strategy:
  1. Fetch the sitemap. If it's a sitemap INDEX (points to other sitemaps),
     follow each child sitemap until we get real page URLs.
  2. Filter to just /docs/ URLs.
  3. For each URL, fetch the page, extract main content, save as .txt.
  4. Save metadata (filename -> url + title) so we can cite sources later.

Usage:
    python src/scrape_docs.py
"""
import json
import re
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# silence the harmless "you're parsing XML with lxml" warning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---- Config ---------------------------------------------------------------
SITEMAP_URL = "https://supabase.com/sitemap.xml"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "pages"
METADATA_FILE = Path(__file__).resolve().parent.parent / "data" / "metadata.json"
URL_FILTER = "/docs/"          # only keep URLs containing this
MAX_PAGES = 80                  # start small; bump up later
REQUEST_DELAY = 0.5             # half-second between requests = polite
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DocsyBot/0.1; portfolio project)"
}
# ---------------------------------------------------------------------------


def fetch_xml(url: str) -> BeautifulSoup | None:
    """Fetch a URL and parse as XML."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.content, "xml")
    except Exception as e:
        print(f"   ⚠️  Failed to fetch {url}: {e}")
        return None


def extract_urls_from_sitemap(sitemap_url: str, depth: int = 0) -> list[str]:
    """
    Recursively walk a sitemap. If it's a <sitemapindex>, follow each child.
    If it's a <urlset>, return the URLs.
    """
    indent = "  " * depth
    print(f"{indent}📥 Sitemap: {sitemap_url}")

    soup = fetch_xml(sitemap_url)
    if soup is None:
        return []

    # Case 1: this is a sitemap index — recurse into each child sitemap
    if soup.find("sitemapindex"):
        child_urls = [loc.text.strip() for loc in soup.find_all("loc")]
        print(f"{indent}   → sitemap index with {len(child_urls)} child sitemaps")
        all_urls: list[str] = []
        for child in child_urls:
            all_urls.extend(extract_urls_from_sitemap(child, depth + 1))
        return all_urls

    # Case 2: this is a regular sitemap — return its URLs
    page_urls = [loc.text.strip() for loc in soup.find_all("loc")]
    print(f"{indent}   → {len(page_urls)} page URLs")
    return page_urls


def get_doc_urls() -> list[str]:
    all_urls = extract_urls_from_sitemap(SITEMAP_URL)
    docs_urls = [u for u in all_urls if URL_FILTER in u and not u.endswith(".xml")]
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for u in docs_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    print(f"\n✅ Found {len(unique)} unique doc URLs (using first {MAX_PAGES}).\n")
    return unique[:MAX_PAGES]


def url_to_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_")
    return f"{safe}.txt" if safe else "index.txt"


def extract_content(html: str) -> tuple[str, str]:
    """Pull main text + page title out of an HTML page."""
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else "Untitled"

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return title, ""

    for tag in main.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def scrape():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    urls = get_doc_urls()
    if not urls:
        print("❌ No URLs found. Aborting.")
        return

    metadata = {}
    saved = 0
    skipped = 0

    for i, url in enumerate(urls, start=1):
        try:
            print(f"[{i}/{len(urls)}] {url}")
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                print(f"   ⚠️  HTTP {resp.status_code}, skipping.")
                skipped += 1
                continue

            title, text = extract_content(resp.text)

            if len(text) < 200:
                print(f"   ⚠️  Too short ({len(text)} chars), skipping.")
                skipped += 1
                continue

            filename = url_to_filename(url)
            (OUTPUT_DIR / filename).write_text(text, encoding="utf-8")
            metadata[filename] = {"url": url, "title": title, "chars": len(text)}
            saved += 1

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            skipped += 1

    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"✅ Done. Saved {saved} pages, skipped {skipped}.")
    print(f"   Pages:    {OUTPUT_DIR}")
    print(f"   Metadata: {METADATA_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    scrape()