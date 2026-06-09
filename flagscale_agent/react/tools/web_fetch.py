"""Web fetch tool — retrieve and extract content from URLs."""

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from flagscale_agent.react.tools.base import Tool, EFFECT_NETWORK

# ── SSRF protection: blocked host patterns ──
_BLOCKED_HOSTS = (
    "localhost", "127.", "0.0.0.0", "::1",
    "169.254.",  # AWS metadata
    "metadata.google.internal",  # GCP metadata
    "100.100.100.200",  # Alibaba Cloud metadata
)
_BLOCKED_SCHEMES = frozenset({"file", "ftp", "gopher", "dict", "ldap", "tftp"})

# ── Response size limit ──
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB


def _validate_url(url: str):
    """Raise ValueError if url targets internal/protected resources."""
    parsed = urlparse(url)
    if parsed.scheme in _BLOCKED_SCHEMES:
        raise ValueError(f"Blocked URL scheme: {parsed.scheme}")
    hostname = (parsed.hostname or "").lower()
    for blocked in _BLOCKED_HOSTS:
        if blocked in hostname:
            raise ValueError(f"Blocked host: {hostname} (matches {blocked})")
    if hostname.startswith("10.") or hostname.startswith("172.16.") or hostname.startswith("192.168."):
        raise ValueError(f"Blocked private network address: {hostname}")


class WebFetchTool(Tool):
    name = "web_fetch"
    effects = EFFECT_NETWORK
    description = "Fetch a URL and extract its main text content. Useful for reading documentation, GitHub pages, error references, etc."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
        },
        "required": ["url"],
    }
    max_result_size = 80000

    def __init__(self, timeout: int = 30, proxies: dict = None):
        self._timeout = timeout
        self._proxies = proxies

    def execute(self, **kwargs) -> str:
        url = kwargs["url"]

        # SSRF protection
        try:
            _validate_url(url)
        except ValueError as e:
            return f"[WEB_FETCH_BLOCKED] {e}"

        try:
            resp = requests.get(
                url,
                timeout=self._timeout,
                headers={"User-Agent": "FlagScale-Agent/1.0"},
                allow_redirects=True,
                proxies=self._proxies,
                stream=True,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            msg = (
                f"[WEB_FETCH_NETWORK_ERROR] Could not retrieve {url}: {e}\n"
                f"This is a network-level issue, not a tool bug. The remote server may be unreachable "
                f"from this machine, or the URL may require authentication.\n"
                f"Do NOT interpret this as a tool execution error — the tool worked correctly, "
                f"the network is simply unavailable for this URL."
            )
            if not self._proxies and _is_network_error(str(e)):
                msg += _PROXY_HINT
            return msg

        # Read response with size limit
        try:
            raw = bytearray()
            for chunk in resp.iter_content(chunk_size=8192):
                raw.extend(chunk)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    return (
                        f"[WEB_FETCH_SIZE_EXCEEDED] Response from {url} "
                        f"exceeded {_MAX_RESPONSE_BYTES // (1024 * 1024)} MB limit. "
                        f"Content truncated. First {_MAX_RESPONSE_BYTES // (1024 * 1024)} MB:\n"
                        f"{raw[:_MAX_RESPONSE_BYTES].decode('utf-8', errors='replace')[:self.max_result_size]}"
                    )
            text = raw.decode(resp.encoding or "utf-8", errors="replace")
        except Exception:
            # Fallback: use resp.text directly
            text = resp.text

        content_type = resp.headers.get("Content-Type", "")

        if "text/plain" in content_type or url.endswith((".txt", ".log", ".yaml", ".yml", ".json", ".md", ".rst", ".cfg", ".ini", ".toml")):
            return text[:self.max_result_size]

        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return f"[WEB_FETCH_UNSUPPORTED] Cannot extract content from {url}: unsupported Content-Type '{content_type}'. The tool only handles text/plain, text/html, and recognized text file extensions."

        return _extract_text(text, url)[:self.max_result_size]


def _extract_text(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.find("div", class_=re.compile(r"(content|article|post|entry|readme)", re.I))
    )

    target = main or soup.body or soup
    text = target.get_text(separator="\n", strip=True)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    result = "\n".join(lines)

    if len(result) < 50:
        return f"[WEB_FETCH_LOW_CONTENT] {url} returned only {len(result)} chars of extractable content. The page may be mostly scripts/styles or behind a JavaScript wall.\n{result}"

    return result


_PROXY_HINT = (
    "\n\n💡 Network error detected and no proxy configured. "
    "Set proxy in ~/.flagscale/agent.yaml:\n"
    "  shell_env:\n"
    '    HTTP_PROXY: "http://host:port"\n'
    '    HTTPS_PROXY: "http://host:port"\n'
    "Then use /reload to apply."
)


def _is_network_error(msg: str) -> bool:
    patterns = (
        "ConnectionError", "ConnectTimeout", "ProxyError",
        "SSLError", "NewConnectionError", "MaxRetryError",
        "Connection refused", "Name or service not known",
        "Temporary failure in name resolution",
        "Network is unreachable", "No route to host",
    )
    return any(p.lower() in msg.lower() for p in patterns)
