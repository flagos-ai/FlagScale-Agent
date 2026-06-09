"""Tests for WebFetchTool."""

from unittest.mock import MagicMock, patch

import pytest

from flagscale_agent.react.tools.web_fetch import WebFetchTool, _extract_text, _is_network_error


class TestWebFetchTool:
    def test_html_fetch(self):
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = "<html><body><main><p>Hello world</p></main></body></html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("flagscale_agent.react.tools.web_fetch.requests.get", return_value=mock_resp):
            result = tool.execute(url="https://example.com")
        assert "Hello world" in result

    def test_plain_text_url(self):
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_resp.text = "raw content here"
        mock_resp.raise_for_status = MagicMock()

        with patch("flagscale_agent.react.tools.web_fetch.requests.get", return_value=mock_resp):
            result = tool.execute(url="https://example.com/file.txt")
        assert result == "raw content here"

    def test_yaml_url_returns_raw(self):
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/octet-stream"}
        mock_resp.text = "key: value\n"
        mock_resp.raise_for_status = MagicMock()

        with patch("flagscale_agent.react.tools.web_fetch.requests.get", return_value=mock_resp):
            result = tool.execute(url="https://example.com/config.yaml")
        assert result == "key: value\n"

    def test_unsupported_content_type(self):
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.text = "binary"
        mock_resp.raise_for_status = MagicMock()

        with patch("flagscale_agent.react.tools.web_fetch.requests.get", return_value=mock_resp):
            result = tool.execute(url="https://example.com/file.pdf")
        assert "WEB_FETCH_UNSUPPORTED" in result
        assert "application/pdf" in result

    def test_network_error_with_proxy_hint(self):
        import requests
        tool = WebFetchTool(proxies=None)
        with patch("flagscale_agent.react.tools.web_fetch.requests.get",
                    side_effect=requests.ConnectionError("ConnectionError: Name or service not known")):
            result = tool.execute(url="https://example.com")
        assert "WEB_FETCH_NETWORK_ERROR" in result
        assert "proxy" in result.lower()

    def test_network_error_no_hint_with_proxy(self):
        import requests
        tool = WebFetchTool(proxies={"http": "http://proxy:8080"})
        with patch("flagscale_agent.react.tools.web_fetch.requests.get",
                    side_effect=requests.ConnectionError("ConnectionError")):
            result = tool.execute(url="https://example.com")
        assert "WEB_FETCH_NETWORK_ERROR" in result
        assert "proxy" not in result.lower() or "configured" not in result.lower()

    def test_error_message_does_not_start_with_error(self):
        """web_fetch errors should use [WEB_FETCH_*] tags, not raw ERROR: prefix."""
        import requests
        tool = WebFetchTool()
        with patch("flagscale_agent.react.tools.web_fetch.requests.get",
                    side_effect=requests.ConnectionError("fail")):
            result = tool.execute(url="https://example.com")
        assert not result.startswith("ERROR:")
        assert result.startswith("[WEB_FETCH_NETWORK_ERROR]")

    def test_error_message_explains_not_a_tool_bug(self):
        """web_fetch errors should tell LLM not to interpret as tool error."""
        import requests
        tool = WebFetchTool()
        with patch("flagscale_agent.react.tools.web_fetch.requests.get",
                    side_effect=requests.ConnectionError("fail")):
            result = tool.execute(url="https://example.com")
        assert "tool execution error" in result.lower()
        assert "tool worked correctly" in result.lower()


class TestExtractText:
    def test_main_tag(self):
        html = "<html><body><nav>Nav</nav><main><p>Content</p></main></body></html>"
        result = _extract_text(html, "https://example.com")
        assert "Content" in result
        assert "Nav" not in result

    def test_article_tag(self):
        html = "<html><body><article><p>Article text</p></article><footer>Foot</footer></body></html>"
        result = _extract_text(html, "https://example.com")
        assert "Article text" in result

    def test_fallback_to_body(self):
        html = "<html><body><p>Body text</p></body></html>"
        result = _extract_text(html, "https://example.com")
        assert "Body text" in result

    def test_strips_script_style(self):
        html = "<html><body><script>alert(1)</script><style>.x{}</style><p>Clean</p></body></html>"
        result = _extract_text(html, "https://example.com")
        assert "Clean" in result
        assert "alert" not in result
        assert ".x{}" not in result

    def test_short_content_warning(self):
        html = "<html><body><p>Hi</p></body></html>"
        result = _extract_text(html, "https://example.com")
        assert "WEB_FETCH_LOW_CONTENT" in result
        assert "chars" in result


class TestIsNetworkError:
    def test_connection_error(self):
        assert _is_network_error("ConnectionError: failed")

    def test_timeout(self):
        assert _is_network_error("ConnectTimeout: timed out")

    def test_non_network_error(self):
        assert not _is_network_error("404 Not Found")
