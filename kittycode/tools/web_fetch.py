"""Fetch and summarize web content."""

from __future__ import annotations

import json
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from ..interrupts import CancellationRequested
from .base import Tool

_DEFAULT_TIMEOUT = 20
_MAX_REDIRECTS = 5
_MAX_CONTENT_CHARS = 24_000
_USER_AGENT = "KittyCode/0.1 (+https://github.com/yejiming/KittyCode)"


class WebFetchTool(Tool):
    name = "web_fetch"
    description = """
    Fetch content from a public URL, extract readable text, and summarize it
    against a prompt. Use this to inspect current web pages or text endpoints.
    """
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
            "prompt": {
                "type": "string",
                "description": "What information to extract from the fetched content",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 20)",
            },
        },
        "required": ["url", "prompt"],
    }

    _parent_agent = None

    def execute(self, url: str, prompt: str, timeout: int = _DEFAULT_TIMEOUT, cancel_event=None) -> str:
        try:
            normalized_url = _normalize_url(url)
        except ValueError as exc:
            return f"Error: {exc}"

        _raise_if_cancelled(cancel_event)

        try:
            fetched = _fetch_url(normalized_url, timeout=max(1, timeout))
        except requests.RequestException as exc:
            return f"Error fetching {normalized_url}: {exc}"

        _raise_if_cancelled(cancel_event)

        redirect = fetched.get("redirect")
        if redirect is not None:
            return (
                "Redirect detected to a different host. "
                f"Call web_fetch again with {redirect['redirect_url']} "
                f"(original URL: {redirect['original_url']}, status: {redirect['status_code']})."
            )

        extracted = fetched["content"]
        if len(extracted) > _MAX_CONTENT_CHARS:
            extracted = extracted[:_MAX_CONTENT_CHARS] + "\n... (content truncated)"

        summary = _summarize_content(
            self._parent_agent,
            normalized_url,
            prompt,
            extracted,
            cancel_event=cancel_event,
        )

        lines = [
            f"Fetched: {fetched['final_url']}",
            f"Status: {fetched['status_code']} {fetched['reason']}",
            f"Content-Type: {fetched['content_type'] or 'unknown'}",
            f"Bytes: {fetched['bytes']}",
            "",
            summary,
        ]
        return "\n".join(lines).strip()


def _normalize_url(url: str) -> str:
    if not url.strip():
        raise ValueError("url is required")

    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse("https://" + url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme}")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not supported")
    if not parsed.netloc:
        raise ValueError(f"invalid URL: {url}")
    if parsed.scheme == "http":
        parsed = parsed._replace(scheme="https")
    return urlunparse(parsed)


def _fetch_url(url: str, timeout: int) -> dict:
    current_url = url
    headers = {
        "Accept": "text/html, text/plain, application/json, application/xml;q=0.9, */*;q=0.8",
        "User-Agent": _USER_AGENT,
    }

    for _ in range(_MAX_REDIRECTS + 1):
        response = requests.get(current_url, headers=headers, timeout=timeout, allow_redirects=False)
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            if not location:
                raise requests.RequestException("redirect missing Location header")
            redirect_url = urljoin(current_url, location)
            if _is_permitted_redirect(current_url, redirect_url):
                current_url = redirect_url
                continue
            return {
                "redirect": {
                    "original_url": current_url,
                    "redirect_url": redirect_url,
                    "status_code": response.status_code,
                }
            }

        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        return {
            "redirect": None,
            "final_url": response.url,
            "status_code": response.status_code,
            "reason": response.reason or "OK",
            "content_type": content_type.split(";", 1)[0].strip().lower(),
            "bytes": len(response.content),
            "content": _extract_response_text(response),
        }

    raise requests.RequestException(f"too many redirects while fetching {url}")


def _extract_response_text(response: requests.Response) -> str:
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if "json" in content_type:
        try:
            return json.dumps(response.json(), indent=2, ensure_ascii=False)
        except ValueError:
            return response.text

    if "html" in content_type:
        return _html_to_text(response.text, response.url)

    if (
        not content_type
        or content_type.startswith("text/")
        or "xml" in content_type
        or "javascript" in content_type
    ):
        return response.text

    return f"Binary content omitted ({content_type or 'unknown content type'})."


def _html_to_text(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for link in soup.select("a[href]"):
        href = urljoin(base_url, link.get("href", ""))
        text = " ".join(link.get_text(" ", strip=True).split())
        replacement = href if not text else f"{text} ({href})"
        link.replace_with(replacement)

    text = soup.get_text("\n", strip=True)
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _summarize_content(parent_agent, url: str, prompt: str, content: str, cancel_event=None) -> str:
    if not prompt.strip() or parent_agent is None or getattr(parent_agent, "llm", None) is None:
        return content

    llm = parent_agent.llm.clone() if hasattr(parent_agent.llm, "clone") else parent_agent.llm
    request = (
        "You are processing fetched web content for KittyCode. "
        "Use only the provided content when answering. "
        "If the content does not contain the answer, say so.\n\n"
        f"URL: {url}\n"
        f"User request: {prompt}\n\n"
        f"Fetched content:\n{content}"
    )
    try:
        response = llm.chat(
            messages=[{"role": "user", "content": request}],
            tools=None,
            cancel_event=cancel_event,
        )
    except Exception:
        return content

    return response.content.strip() or content


def _is_permitted_redirect(original_url: str, redirect_url: str) -> bool:
    original = urlparse(original_url)
    redirect = urlparse(redirect_url)
    if redirect.scheme != original.scheme:
        return False
    if redirect.port != original.port:
        return False
    if redirect.username or redirect.password:
        return False

    def _strip_www(hostname: str) -> str:
        return hostname.removeprefix("www.")

    return _strip_www(original.hostname or "") == _strip_www(redirect.hostname or "")


def _raise_if_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancellationRequested()