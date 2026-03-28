"""Tests for the EndNote XML parser."""

import pytest

from src.data.endnote_parser import (
    _extract_category,
    _extract_style_text,
    _normalise_category,
    parse_endnote_xml,
)
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_RECORD = """\
<xml><records>
  <record>
    <titles><title><style>Test Paper Title</style></title></titles>
    <contributors><authors>
      <author><style>Smith, John</style></author>
      <author><style>Doe, Jane</style></author>
    </authors></contributors>
    <dates><year><style>2021</style></year></dates>
    <periodical><full-title><style>Nature Neuroscience</style></full-title></periodical>
    <electronic-resource-num><style>10.1234/test.2021</style></electronic-resource-num>
    <label><style>My Group</style></label>
  </record>
</records></xml>
"""

MULTI_RECORD = """\
<xml><records>
  <record>
    <titles><title><style>Paper A</style></title></titles>
    <electronic-resource-num><style>10.1234/a</style></electronic-resource-num>
    <label><style>Group Alpha</style></label>
  </record>
  <record>
    <titles><title><style>Paper B</style></title></titles>
    <electronic-resource-num><style>10.1234/b</style></electronic-resource-num>
    <custom1><style>Group Beta</style></custom1>
  </record>
  <record>
    <titles><title><style>Paper C — no DOI</style></title></titles>
  </record>
</records></xml>
"""


# ---------------------------------------------------------------------------
# _extract_style_text
# ---------------------------------------------------------------------------

class TestExtractStyleText:
    def test_none_element(self):
        assert _extract_style_text(None) == ""

    def test_with_style_wrapper(self):
        el = ET.fromstring("<title><style>Hello World</style></title>")
        assert _extract_style_text(el) == "Hello World"

    def test_without_style_wrapper(self):
        el = ET.fromstring("<title>Direct text</title>")
        assert _extract_style_text(el) == "Direct text"

    def test_empty_element(self):
        el = ET.fromstring("<title></title>")
        assert _extract_style_text(el) == ""


# ---------------------------------------------------------------------------
# _normalise_category
# ---------------------------------------------------------------------------

class TestNormaliseCategory:
    def test_simple(self):
        assert _normalise_category("My Group") == "my_group"

    def test_special_chars(self):
        assert _normalise_category("CA3 / Pattern-Completion") == "ca3_pattern_completion"

    def test_empty(self):
        assert _normalise_category("") == "uncategorized"

    def test_whitespace(self):
        assert _normalise_category("  lots   of   spaces  ") == "lots_of_spaces"


# ---------------------------------------------------------------------------
# _extract_category
# ---------------------------------------------------------------------------

class TestExtractCategory:
    def test_label_priority(self):
        xml = "<record><label><style>From Label</style></label><custom1><style>From Custom</style></custom1></record>"
        rec = ET.fromstring(xml)
        assert _extract_category(rec) == "from_label"

    def test_custom_fallback(self):
        xml = "<record><custom1><style>From Custom1</style></custom1></record>"
        rec = ET.fromstring(xml)
        assert _extract_category(rec) == "from_custom1"

    def test_research_notes_fallback(self):
        xml = "<record><research-notes><style>From Notes</style></research-notes></record>"
        rec = ET.fromstring(xml)
        assert _extract_category(rec) == "from_notes"

    def test_keyword_fallback(self):
        xml = "<record><keywords><keyword><style>hippocampus</style></keyword></keywords></record>"
        rec = ET.fromstring(xml)
        assert _extract_category(rec) == "hippocampus"

    def test_uncategorized(self):
        xml = "<record></record>"
        rec = ET.fromstring(xml)
        assert _extract_category(rec) == "uncategorized"


# ---------------------------------------------------------------------------
# parse_endnote_xml
# ---------------------------------------------------------------------------

class TestParseEndnoteXml:
    def test_minimal_record(self):
        records = parse_endnote_xml(MINIMAL_RECORD.encode())
        assert len(records) == 1
        r = records[0]
        assert r["doi"] == "10.1234/test.2021"
        assert r["title"] == "Test Paper Title"
        assert r["authors"] == ["Smith, John", "Doe, Jane"]
        assert r["year"] == 2021
        assert r["journal"] == "Nature Neuroscience"
        assert r["seed_category"] == "my_group"
        assert r["is_seed"] is True

    def test_multiple_records(self):
        records = parse_endnote_xml(MULTI_RECORD.encode())
        assert len(records) == 3

        # Paper A
        assert records[0]["doi"] == "10.1234/a"
        assert records[0]["seed_category"] == "group_alpha"

        # Paper B (custom1 category)
        assert records[1]["doi"] == "10.1234/b"
        assert records[1]["seed_category"] == "group_beta"

        # Paper C (no DOI)
        assert records[2]["doi"] is None
        assert records[2]["title"] == "Paper C — no DOI"
        assert records[2]["seed_category"] == "uncategorized"

    def test_doi_url_stripping(self):
        xml = b"""\
<xml><records><record>
  <titles><title><style>Test</style></title></titles>
  <electronic-resource-num><style>https://doi.org/10.1234/test</style></electronic-resource-num>
</record></records></xml>
"""
        records = parse_endnote_xml(xml)
        assert records[0]["doi"] == "10.1234/test"

    def test_deduplication(self):
        xml = b"""\
<xml><records>
  <record>
    <titles><title><style>Same Paper</style></title></titles>
    <electronic-resource-num><style>10.1234/dup</style></electronic-resource-num>
    <label><style>Group A</style></label>
  </record>
  <record>
    <titles><title><style>Same Paper Again</style></title></titles>
    <electronic-resource-num><style>10.1234/dup</style></electronic-resource-num>
    <label><style>Group B</style></label>
  </record>
</records></xml>
"""
        records = parse_endnote_xml(xml)
        assert len(records) == 1
        assert "group_a" in records[0]["seed_categories"]
        assert "group_b" in records[0]["seed_categories"]

    def test_empty_xml(self):
        xml = b"<xml><records></records></xml>"
        records = parse_endnote_xml(xml)
        assert records == []

    def test_no_title_no_doi_skipped(self):
        xml = b"""\
<xml><records>
  <record>
    <dates><year><style>2020</style></year></dates>
  </record>
</records></xml>
"""
        records = parse_endnote_xml(xml)
        assert records == []

    def test_invalid_xml_raises(self):
        with pytest.raises(ET.ParseError):
            parse_endnote_xml(b"not valid xml <<<<")
