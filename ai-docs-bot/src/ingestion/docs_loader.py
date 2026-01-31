"""Docs website ingestion via sitemap parsing."""
import requests
import time
from xml.etree import ElementTree
from typing import List
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Document


def parse_sitemap(sitemap_url: str) -> List[str]:
    """Extract all URLs from a sitemap.xml file.

    Handles both:
    - <urlset> (regular sitemap with page URLs)
    - <sitemapindex> (sitemap index pointing to other sitemaps)
    """
    response = requests.get(sitemap_url, timeout=30)
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    urls = []

    # Check if this is a sitemap index (contains <sitemap> elements)
    sitemap_locs = root.findall('.//sm:sitemap/sm:loc', namespaces)
    if sitemap_locs:
        # Recursively parse each child sitemap
        for sitemap_loc in sitemap_locs:
            child_url = sitemap_loc.text
            if child_url:
                try:
                    child_urls = parse_sitemap(child_url)
                    urls.extend(child_urls)
                except Exception as e:
                    print(f"Failed to parse child sitemap {child_url}: {e}")
                    continue
    else:
        # Regular urlset - extract page URLs
        for url_elem in root.findall('.//sm:url/sm:loc', namespaces):
            if url_elem.text:
                urls.append(url_elem.text)

    # Filter to only /docs/ paths if needed
    return [u for u in urls if '/docs/' in u]


def load_docs_from_sitemap(sitemap_url: str, delay: float = 0.5) -> List[Document]:
    """Load all documents from a sitemap.

    Args:
        sitemap_url: URL to the sitemap.xml file
        delay: Delay between requests in seconds (rate limiting)

    Returns:
        List of LlamaIndex Document objects
    """
    urls = parse_sitemap(sitemap_url)

    documents = []
    reader = SimpleWebPageReader(html_to_text=True)

    for url in urls:
        try:
            docs = reader.load_data([url])
            for doc in docs:
                doc.metadata['source'] = url
                doc.metadata['source_type'] = 'docs'
            documents.extend(docs)
            time.sleep(delay)  # Rate limiting
        except Exception as e:
            print(f"Failed to load {url}: {e}")
            continue

    return documents
