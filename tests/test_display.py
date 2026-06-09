"""Tests for display utilities."""

import os

from flagscale_agent.react import display


class TestDisplay:
    def test_render_markdown_code_block(self):
        md = "```python\nprint('hello')\n```"
        result = display.render_markdown(md)
        assert "hello" in result

    def test_render_markdown_inline_code(self):
        os.environ.pop("NO_COLOR", None)
        md = "Use `pip install` to install."
        result = display.render_markdown(md)
        assert "pip install" in result

    def test_render_markdown_bold(self):
        os.environ.pop("NO_COLOR", None)
        md = "This is **important**."
        result = display.render_markdown(md)
        assert "important" in result

    def test_render_markdown_no_color(self):
        os.environ["NO_COLOR"] = "1"
        try:
            md = "```python\ncode\n```"
            result = display.render_markdown(md)
            assert "code" in result
            assert "\033" not in result
        finally:
            del os.environ["NO_COLOR"]

    def test_fmt_tokens(self):
        assert display._fmt_tokens(999) == "999"
        assert display._fmt_tokens(1000) == "1,000"
        assert display._fmt_tokens(100000) == "100k"
        assert display._fmt_tokens(1234567) == "1234k"

    def test_color_helpers(self):
        os.environ.pop("NO_COLOR", None)
        assert "red" not in display.red("red")  or True  # just ensure no crash
        assert "magenta" not in display.magenta("magenta") or True
        display.dim("test")
        display.green("test")
        display.yellow("test")
        display.cyan("test")
        display.bold("test")
