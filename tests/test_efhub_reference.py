"""Tests for eFHUB card classification and database reference module."""

import pytest

from squadpredict.efhub_reference import (
    EFHUB_CARD_TYPES,
    get_efhub_search_url,
    match_efhub_player,
    resolve_efhub_card_type,
)


def test_efhub_card_types_whitelist():
    expected = {"Big Time", "Epic (Booster)", "Show Time", "Highlight", "Trending (POTW)", "Featured", "Standard"}
    assert set(EFHUB_CARD_TYPES) == expected


def test_match_efhub_player_exact():
    result = match_efhub_player("L. Messi")
    assert result is not None
    assert result["card_type"] == "Big Time"


def test_match_efhub_player_alias():
    result = match_efhub_player("Van Basten")
    assert result is not None
    assert result["card_type"] == "Epic (Booster)"


def test_match_efhub_player_showtime():
    result = match_efhub_player("Mbappe")
    assert result is not None
    assert result["card_type"] == "Show Time"


def test_get_efhub_search_url():
    url = get_efhub_search_url("L. Messi")
    assert "https://efootballhub.net/search/players" in url
    assert "Messi" in url


def test_resolve_efhub_card_type_legend():
    card_type, condition, url = resolve_efhub_card_type("Van Basten (105)", rating=105)
    assert card_type == "Epic (Booster)"
    assert condition == "Permanent B"
    assert "efootballhub.net" in url


def test_resolve_efhub_card_type_big_time():
    card_type, condition, url = resolve_efhub_card_type("Messi", rating=109)
    assert card_type == "Big Time"
    assert condition == "Permanent B"


def test_resolve_efhub_card_type_show_time():
    card_type, condition, url = resolve_efhub_card_type("Mbappe", rating=106, visual_type="Show Time")
    assert card_type == "Show Time"


def test_resolve_efhub_card_type_standard():
    card_type, condition, url = resolve_efhub_card_type("Unknown Player", rating=82)
    assert card_type == "Standard"
    assert condition == "A-E Live Rating"
