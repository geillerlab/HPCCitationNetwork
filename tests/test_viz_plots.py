"""Tests for src/viz/plots.py — color mapping, node styling, and dynamic colors."""

from src.viz.plots import CATEGORY_COLORS, ensure_category_colors, get_node_color


def test_category_colors_has_defaults():
    """Default categories should have colors."""
    # At minimum, the base categories that ship with the module
    assert "uncategorized" in CATEGORY_COLORS
    assert "" in CATEGORY_COLORS


def test_category_colors_empty_default():
    """Empty string key provides a default color."""
    assert "" in CATEGORY_COLORS
    assert CATEGORY_COLORS[""] == "#cccccc"


def test_get_node_color_seed():
    """Seed papers get their category color."""
    # Register a test category first
    ensure_category_colors(["test_cat"])
    data = {"is_seed": True, "seed_category": "test_cat"}
    assert get_node_color(data) == CATEGORY_COLORS["test_cat"]


def test_get_node_color_seed_unknown_category():
    """Seeds with unknown category get the default color."""
    data = {"is_seed": True, "seed_category": "nonexistent"}
    assert get_node_color(data) == CATEGORY_COLORS[""]


def test_get_node_color_non_seed():
    """Non-seed papers are always light gray."""
    data = {"is_seed": False, "seed_category": "some_cat"}
    assert get_node_color(data) == "#dddddd"


def test_get_node_color_missing_seed_flag():
    """Papers without is_seed flag are treated as non-seed."""
    data = {"seed_category": "some_cat"}
    assert get_node_color(data) == "#dddddd"


def test_ensure_category_colors_adds_new():
    """ensure_category_colors should add colors for unknown categories."""
    new_cats = ["brand_new_cat_a", "brand_new_cat_b"]
    ensure_category_colors(new_cats)
    for cat in new_cats:
        assert cat in CATEGORY_COLORS
        assert CATEGORY_COLORS[cat].startswith("#")


def test_ensure_category_colors_idempotent():
    """Calling ensure_category_colors twice should not change existing colors."""
    ensure_category_colors(["idempotent_cat"])
    first_color = CATEGORY_COLORS["idempotent_cat"]
    ensure_category_colors(["idempotent_cat"])
    assert CATEGORY_COLORS["idempotent_cat"] == first_color
