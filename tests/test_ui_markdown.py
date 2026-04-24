"""Verify that static/index.html contains the required markdown rendering dependencies."""
import pathlib
import re

HTML_FILE = pathlib.Path(__file__).parent.parent / "static" / "index.html"


def _html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


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
    html = _html()
    assert "DOMPurify.sanitize(marked.parse(text))" in html, (
        "addMessage() does not sanitize bot message HTML via DOMPurify"
    )


def test_user_messages_use_text_content():
    """User messages must remain as textContent (no innerHTML injection)."""
    html = _html()
    # The else-branch should assign textContent, not innerHTML, for user messages
    assert re.search(
        r"else\s*\{\s*bubble\.textContent\s*=\s*text",
        html,
    ), "User messages should use textContent, not innerHTML"


def test_bot_bubble_markdown_css_present():
    """CSS rules for .bot .msg-bubble markdown elements must exist."""
    html = _html()
    assert ".bot .msg-bubble p" in html, (
        "CSS for .bot .msg-bubble paragraph elements is missing"
    )
    assert ".bot .msg-bubble ul" in html or ".bot .msg-bubble ol" in html, (
        "CSS for .bot .msg-bubble list elements is missing"
    )
