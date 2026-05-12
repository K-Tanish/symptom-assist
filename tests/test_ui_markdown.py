"""Verify that static frontend contains the required markdown rendering dependencies."""
import pathlib
import re

STATIC_DIR = pathlib.Path(__file__).parent.parent / "static"

def _html() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

def _js() -> str:
    js_file = STATIC_DIR / "app.js"
    return js_file.read_text(encoding="utf-8") if js_file.exists() else _html()

def _css() -> str:
    css_file = STATIC_DIR / "style.css"
    return css_file.read_text(encoding="utf-8") if css_file.exists() else _html()


def test_marked_cdn_script_present():
    """marked.js v9 CDN script tag must be included in the HTML head."""
    assert "cdn.jsdelivr.net/npm/marked@9" in _html(), (
        "marked.js CDN script not found in static/index.html"
    )


def test_dompurify_cdn_script_present():
    """DOMPurify v3 CDN script tag must be included in the HTML head."""
    assert "cdn.jsdelivr.net/npm/dompurify@3" in _html(), (
        "DOMPurify CDN script not found in static/index.html"
    )


def test_add_message_uses_dompurify_for_bot():
    """addMessage must call DOMPurify.sanitize for bot messages."""
    assert "DOMPurify.sanitize(marked.parse(text))" in _js(), (
        "addMessage() does not sanitize bot message HTML via DOMPurify"
    )


def test_user_messages_use_text_content():
    """User messages must remain as textContent (no innerHTML injection)."""
    # The else-branch should assign textContent, not innerHTML, for user messages
    assert re.search(
        r"else\s*\{\s*bubble\.textContent\s*=\s*text",
        _js(),
    ), "User messages should use textContent, not innerHTML"


def test_bot_bubble_markdown_css_present():
    """CSS rules for .bot .msg-bubble markdown elements must exist."""
    assert ".bot .msg-bubble p" in _css(), (
        "CSS for .bot .msg-bubble paragraph elements is missing"
    )
    assert ".bot .msg-bubble ul" in _css() or ".bot .msg-bubble ol" in _css(), (
        "CSS for .bot .msg-bubble list elements is missing"
    )
