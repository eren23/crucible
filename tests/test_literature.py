"""Tests for crucible.researcher.literature."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crucible.researcher.literature import (
    format_literature_context,
    search_papers,
    suggest_queries,
)


# ---------------------------------------------------------------------------
# format_literature_context
# ---------------------------------------------------------------------------

class TestFormatLiteratureContext:
    def test_empty_papers(self):
        assert format_literature_context([]) == ""

    def test_formats_papers(self):
        papers = [
            {
                "id": "2401.12345",
                "title": "World Models for Code",
                "summary": "We propose a world model approach to code generation.",
                "ai_summary": "Novel world model for code.",
                "upvotes": 42,
                "keywords": ["world-model", "code"],
            },
        ]
        result = format_literature_context(papers)
        assert "## Related Literature" in result
        assert "World Models for Code" in result
        assert "2401.12345" in result
        assert "42 upvotes" in result

    def test_respects_max_papers(self):
        papers = [
            {"id": f"id{i}", "title": f"Paper {i}", "summary": "", "ai_summary": "", "upvotes": 0, "keywords": []}
            for i in range(10)
        ]
        result = format_literature_context(papers, max_papers=3)
        assert result.count("**Paper") == 3

    def test_truncates_long_summary(self):
        papers = [
            {
                "id": "1",
                "title": "T",
                "summary": "x" * 300,
                "ai_summary": "",
                "upvotes": 0,
                "keywords": [],
            },
        ]
        result = format_literature_context(papers)
        assert "..." in result


# ---------------------------------------------------------------------------
# suggest_queries
# ---------------------------------------------------------------------------

class TestSuggestQueries:
    def test_extracts_model_family(self):
        queries = suggest_queries(
            'MODEL_FAMILY="looped"',
            [],
            [],
        )
        assert any("looped" in q for q in queries)

    def test_extracts_from_beliefs(self):
        queries = suggest_queries(
            "",
            ["broader models converge faster with RMSprop"],
            [],
        )
        assert len(queries) >= 1

    def test_fallback_when_empty(self):
        queries = suggest_queries("", [], [])
        assert len(queries) >= 1

    def test_limits_to_four(self):
        queries = suggest_queries(
            'MODEL_FAMILY="test"',
            ["belief " * 20] * 10,
            [{"finding": f"finding {i} something important"} for i in range(20)],
        )
        assert len(queries) <= 4


# ---------------------------------------------------------------------------
# search_papers (mocked)
# ---------------------------------------------------------------------------

class TestSearchPapers:
    def test_empty_query_returns_empty(self):
        assert search_papers("") == []
        assert search_papers("   ") == []

    @patch("crucible.researcher.literature._search_via_hub")
    def test_uses_hub_when_available(self, mock_hub):
        mock_hub.return_value = [
            {"id": "1", "title": "Test", "summary": "", "ai_summary": "", "upvotes": 0,
             "published_at": "", "github_repo": "", "keywords": []},
        ]
        # Clear cache
        from crucible.researcher.literature import _cache
        _cache.clear()

        result = search_papers("test query", limit=5)
        assert len(result) == 1
        mock_hub.assert_called_once()

    @patch("crucible.researcher.literature._search_via_hub", return_value=None)
    @patch("crucible.researcher.literature._search_via_api")
    def test_falls_back_to_api(self, mock_api, mock_hub):
        mock_api.return_value = [
            {"id": "2", "title": "Fallback", "summary": "", "ai_summary": "", "upvotes": 0,
             "published_at": "", "github_repo": "", "keywords": []},
        ]
        from crucible.researcher.literature import _cache
        _cache.clear()

        result = search_papers("test fallback", limit=5)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    @patch("crucible.researcher.literature._search_via_hub", return_value=None)
    @patch("crucible.researcher.literature._search_via_api", return_value=[])
    def test_returns_empty_on_total_failure(self, mock_api, mock_hub):
        from crucible.researcher.literature import _cache
        _cache.clear()

        result = search_papers("will fail", limit=5)
        assert result == []
