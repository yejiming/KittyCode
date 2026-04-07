"""Public web search tool."""

from __future__ import annotations

from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .base import Tool

_DEFAULT_TIMEOUT = 20
_MAX_RESULTS = 8
_SEARCH_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = "KittyCode/0.1 (+https://github.com/yejiming/KittyCode)"


class WebSearchTool(Tool):
    name = "web_search"
    description = """
    Search the public web for current information and return result links with
    snippets. Use this when you need fresh external sources.
    """
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to run",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional domains to include exclusively",
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional domains to exclude from the results",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 20)",
            },
        },
        "required": ["query", "timeout"],
    }

    def execute(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> str:
        normalized_query = query.strip()
        if len(normalized_query) < 2:
            return "Error: query must be at least 2 characters long"

        allowed = _normalize_domains(allowed_domains)
        blocked = _normalize_domains(blocked_domains)
        if allowed and blocked:
            return "Error: allowed_domains and blocked_domains cannot be used together"

        try:
            results = _search(normalized_query, timeout=max(1, timeout))
        except requests.RequestException as exc:
            return f"Error searching the web: {exc}"

        filtered = [
            result
            for result in results
            if _domain_allowed(result["url"], allowed, blocked)
        ]

        if not filtered:
            return f'No search results found for "{normalized_query}".'

        visible = filtered[:_MAX_RESULTS]
        lines = [f'Web search results for "{normalized_query}":', ""]
        for index, result in enumerate(visible, 1):
            lines.append(f"{index}. {result['title']}")
            lines.append(f"   URL: {result['url']}")
            if result["snippet"]:
                lines.append(f"   Snippet: {result['snippet']}")

        lines.extend(["", "Sources:"])
        lines.extend(f"- {result['title']}: {result['url']}" for result in visible)
        lines.extend(["", "Use the URLs above as sources in your final response when relevant."])
        return "\n".join(lines)


def _search(query: str, timeout: int) -> list[dict]:
    response = requests.get(
        _SEARCH_URL,
        params={"q": query},
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict] = []

    for result in soup.select(".result"):
        link = result.select_one("a.result__a")
        if link is None:
            continue
        raw_url = link.get("href", "")
        resolved_url = _unwrap_duckduckgo_url(raw_url)
        if not resolved_url:
            continue
        title = " ".join(link.get_text(" ", strip=True).split())
        snippet_tag = result.select_one(".result__snippet")
        snippet = ""
        if snippet_tag is not None:
            snippet = " ".join(snippet_tag.get_text(" ", strip=True).split())
        results.append(
            {
                "title": title or resolved_url,
                "url": resolved_url,
                "snippet": snippet,
            }
        )
        if len(results) >= _MAX_RESULTS * 2:
            break

    return results


def _unwrap_duckduckgo_url(raw_url: str) -> str:
    if not raw_url:
        return ""

    candidate = urljoin("https://duckduckgo.com", raw_url)
    parsed = urlparse(candidate)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        redirect_target = parse_qs(parsed.query).get("uddg")
        if redirect_target:
            return unquote(redirect_target[0])
    return candidate


def _normalize_domains(domains: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for domain in domains or []:
        value = domain.strip().lower()
        if value:
            normalized.append(value)
    return normalized


def _domain_allowed(url: str, allowed: list[str], blocked: list[str]) -> bool:
    host = urlparse(url).hostname or ""
    host = host.lower()
    if allowed and not any(_matches_domain(host, domain) for domain in allowed):
        return False
    if blocked and any(_matches_domain(host, domain) for domain in blocked):
        return False
    return True


def _matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)