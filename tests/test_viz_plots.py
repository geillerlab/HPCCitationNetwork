"""Tests for src/viz/plots.py — color mapping and node styling."""

from src.viz.plots import CATEGORY_COLORS, get_node_color


def test_category_colors_has_all_categories():
    """All expected categories have a color."""
    expected = [
        "point_attractor", "continuous_attractor", "sequence",
        "successor_representation", "btsp", "bespoke",
        "autonomous_dynamics", "general_attractor",
    ]
    for cat in expected:
        assert cat in CATEGORY_COLORS, f"Missing color for {cat}"


def test_category_colors_empty_default():
    """Empty string key provides a default color."""
    assert "" in CATEGORY_COLORS
    assert CATEGORY_COLORS[""] == "#cccccc"


def test_get_node_color_seed():
    """Seed papers get their category color."""
    data = {"is_seed": True, "seed_category": "point_attractor"}
    assert get_node_color(data) == CATEGORY_COLORS["point_attractor"]


def test_get_node_color_seed_unknown_category():
    """Seeds with unknown category get the default color."""
    data = {"is_seed": True, "seed_category": "nonexistent"}
    assert get_node_color(data) == CATEGORY_COLORS[""]


def test_get_node_color_non_seed():
    """Non-seed papers are always light gray."""
    data = {"is_seed": False, "seed_category": "point_attractor"}
    assert get_node_color(data) == "#dddddd"


def test_get_node_color_missing_seed_flag():
    """Papers without is_seed flag are treated as non-seed."""
    data = {"seed_category": "sequence"}
    assert get_node_color(data) == "#dddddd"
