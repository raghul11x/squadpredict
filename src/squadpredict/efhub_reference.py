"""eFHUB Reference Module -- Integrates eFHUB (efootballhub.net) card taxonomy and database.

Provides official eFootball card classifications as defined by eFHUB:
- Big Time: Special defining-match cards, 108+ ceiling or iconic match commemorative editions (Permanent B condition).
- Epic (Booster): Iconic retired legends with dual/single stat boosters, permanent B condition.
- Show Time: Premium cards with exclusive game-changing skills (Blitz Curler, Fortress, Phenomenal Finishing, etc.).
- Highlight: Themed club campaigns and tournament selections.
- Trending (POTW): Weekly real-world performance cards with green frames.
- Standard: Regular base GP cards.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from rapidfuzz import fuzz

# eFHUB Official Card Categories
EFHUB_CARD_TYPES = [
    "Big Time",
    "Epic (Booster)",
    "Show Time",
    "Highlight",
    "Trending (POTW)",
    "Featured",
    "Standard",
]

# Curated reference catalog of eFHUB cards and defining editions
EFHUB_PLAYER_CATALOG: list[dict[str, Any]] = [
    # --- Big Time (Defining Moment / Historic Performances / Big Time Epics) ---
    {"name": "L. Messi", "aliases": ["Messi", "Lionel Messi"], "card_type": "Big Time", "rating": 109, "condition": "Permanent B", "note": "2022 World Cup Final Edition"},
    {"name": "P. Aubameyang", "aliases": ["Aubameyang", "Pierre-Emerick Aubameyang"], "card_type": "Big Time", "rating": 106, "condition": "Permanent B", "note": "BVB Dortmund Big Time Epic Edition"},
    {"name": "D. Rice", "aliases": ["Rice", "Declan Rice"], "card_type": "Big Time", "rating": 106, "condition": "Permanent B", "note": "Arsenal Defining Moment Big Time Edition"},
    {"name": "M. Neuer", "aliases": ["Neuer", "Manuel Neuer"], "card_type": "Big Time", "rating": 107, "condition": "Permanent B", "note": "2014 World Cup Sweeper Keeper Big Time Edition"},
    {"name": "D. Maradona", "aliases": ["Maradona", "Diego Maradona"], "card_type": "Big Time", "rating": 108, "condition": "Permanent B", "note": "1986 World Cup Wonder Goal Edition"},
    {"name": "K. Rummenigge", "aliases": ["Rummenigge"], "card_type": "Big Time", "rating": 108, "condition": "Permanent B", "note": "1981 Golden Boot Big Time Edition"},

    # --- Epic Booster (Legends with Permanent B Condition & Stat Boosters) ---
    {"name": "M. van Basten", "aliases": ["Van Basten", "Marco van Basten"], "card_type": "Epic (Booster)", "rating": 105, "condition": "Permanent B", "note": "AC Milan / Netherlands Legend"},
    {"name": "W. Sneijder", "aliases": ["Sneijder", "Wesley Sneijder"], "card_type": "Epic (Booster)", "rating": 104, "condition": "Permanent B", "note": "Inter Treble Winner"},
    {"name": "R. Gullit", "aliases": ["Gullit", "Ruud Gullit"], "card_type": "Epic (Booster)", "rating": 106, "condition": "Permanent B", "note": "Dutch Trio Milan Legend"},
    {"name": "Xavi", "aliases": ["Xavi Hernandez"], "card_type": "Epic (Booster)", "rating": 106, "condition": "Permanent B", "note": "FC Barcelona Golden Era"},
    {"name": "J. Guardiola", "aliases": ["Guardiola", "Pep Guardiola"], "card_type": "Epic (Booster)", "rating": 105, "condition": "Permanent B", "note": "Dream Team Pivot"},
    {"name": "P. Schmeichel", "aliases": ["Schmeichel", "Peter Schmeichel"], "card_type": "Epic (Booster)", "rating": 107, "condition": "Permanent B", "note": "Treble Legend Goalkeeper"},
    {"name": "F. Cannavaro", "aliases": ["Cannavaro", "Fabio Cannavaro"], "card_type": "Epic (Booster)", "rating": 106, "condition": "Permanent B", "note": "2006 Ballon d'Or Winner"},
    {"name": "J. Stam", "aliases": ["Stam", "Jaap Stam"], "card_type": "Epic (Booster)", "rating": 106, "condition": "Permanent B", "note": "Manchester United Legend"},
    {"name": "A. Nesta", "aliases": ["Nesta", "Alessandro Nesta"], "card_type": "Epic (Booster)", "rating": 105, "condition": "Permanent B", "note": "AC Milan Wall"},
    {"name": "Pepe", "aliases": ["K. Pepe"], "card_type": "Epic (Booster)", "rating": 106, "condition": "Permanent B", "note": "Euro 2016 Champion Edition"},
    {"name": "A. Costacurta", "aliases": ["Costacurta", "Alessandro Costacurta"], "card_type": "Epic (Booster)", "rating": 104, "condition": "Permanent B", "note": "Rossoneri Icon"},
    {"name": "C. Puyol", "aliases": ["Puyol", "Carles Puyol"], "card_type": "Epic (Booster)", "rating": 104, "condition": "Permanent B", "note": "Captain Courageous Barca"},
    {"name": "Marcelo", "aliases": ["Marcelo Vieira"], "card_type": "Epic (Booster)", "rating": 104, "condition": "Permanent B", "note": "Real Madrid Legend Edition"},
    {"name": "A. Cole", "aliases": ["Cole", "Ashley Cole"], "card_type": "Epic (Booster)", "rating": 102, "condition": "Permanent B", "note": "Invincibles Left Back"},
    {"name": "Gilberto Silva", "aliases": ["Gilberto", "G. Silva"], "card_type": "Epic (Booster)", "rating": 102, "condition": "Permanent B", "note": "Invisible Wall Pivot"},
    {"name": "S. Eto'o", "aliases": ["Eto'o", "Samuel Etoo"], "card_type": "Epic (Booster)", "rating": 103, "condition": "Permanent B", "note": "Inter / Barca Iconic Striker"},
    {"name": "P. Vieira", "aliases": ["Vieira", "Patrick Vieira"], "card_type": "Epic (Booster)", "rating": 105, "condition": "Permanent B", "note": "Dominant Box-to-Box Legend"},
    {"name": "P. Maldini", "aliases": ["Maldini", "Paolo Maldini"], "card_type": "Epic (Booster)", "rating": 106, "condition": "Permanent B", "note": "All-Time Greatest Defender"},
    {"name": "F. Beckenbauer", "aliases": ["Beckenbauer", "Der Kaiser"], "card_type": "Epic (Booster)", "rating": 106, "condition": "Permanent B", "note": "Der Kaiser Sweeper"},
    {"name": "Ronaldinho", "aliases": ["Ronaldinho Gaucho"], "card_type": "Epic (Booster)", "rating": 106, "condition": "Permanent B", "note": "Samba Magician Barca"},

    # --- Show Time (Active Stars with Game-Changing Skills) ---
    {"name": "K. Mbappé", "aliases": ["Mbappe", "Kylian Mbappe"], "card_type": "Show Time", "rating": 106, "condition": "A-E Live Rating", "note": "Phenomenal Finishing Skill"},
    {"name": "M. ter Stegen", "aliases": ["Ter Stegen", "Marc ter Stegen"], "card_type": "Show Time", "rating": 105, "condition": "A-E Live Rating", "note": "Fortress Goalkeeper Skill"},
    {"name": "K. De Bruyne", "aliases": ["De Bruyne", "Kevin De Bruyne"], "card_type": "Show Time", "rating": 105, "condition": "A-E Live Rating", "note": "Visionary Pass Skill"},
    {"name": "E. Haaland", "aliases": ["Haaland", "Erling Haaland"], "card_type": "Show Time", "rating": 106, "condition": "A-E Live Rating", "note": "Phenomenal Finishing Skill"},
    {"name": "J. Bellingham", "aliases": ["Bellingham", "Jude Bellingham"], "card_type": "Show Time", "rating": 105, "condition": "A-E Live Rating", "note": "Momentum Dribbling Skill"},
    {"name": "Vinícius Jr.", "aliases": ["Vinicius", "Vini Jr."], "card_type": "Show Time", "rating": 106, "condition": "A-E Live Rating", "note": "Momentum Dribbling Skill"},

    # --- Highlight / Club Selection / Featured ---
    {"name": "B. Saka", "aliases": ["Saka", "Bukayo Saka"], "card_type": "Highlight", "rating": 104, "condition": "A-E Live Rating", "note": "North London Derby Selection"},
    {"name": "V. van Dijk", "aliases": ["Van Dijk", "Virgil van Dijk"], "card_type": "Highlight", "rating": 105, "condition": "A-E Live Rating", "note": "Matchday Selection"},
]


def get_efhub_search_url(player_name: str) -> str:
    """Generate a direct search link to the eFHUB database for a given player."""
    clean_name = re.sub(r"^[A-Z]\.\s*", "", player_name).strip()
    encoded = urllib.parse.quote_plus(clean_name)
    return f"https://efootballhub.net/search/players?name={encoded}&max=true"


def match_efhub_player(query_name: str) -> dict[str, Any] | None:
    """Fuzzy-match a name against the eFHUB reference catalog."""
    if not query_name or not query_name.strip():
        return None

    clean_query = query_name.strip().lower()
    # Also strip rating suffixes like " (106)" or " (107)"
    clean_query = re.sub(r"\s*\(\d+\)\s*", "", clean_query).strip()

    best_item = None
    best_score = 0.0

    for item in EFHUB_PLAYER_CATALOG:
        candidates = [item["name"]] + item.get("aliases", [])
        for cand in candidates:
            score = max(
                fuzz.token_sort_ratio(clean_query, cand.lower()),
                fuzz.ratio(clean_query, cand.lower()),
                fuzz.token_set_ratio(clean_query, cand.lower()),
            )
            if score > best_score:
                best_score = score
                best_item = item

    if best_score >= 70 and best_item is not None:
        return best_item
    return None


def resolve_efhub_card_type(
    player_name: str,
    rating: int = 90,
    visual_type: str | None = None,
) -> tuple[str, str, str]:
    """Resolve player card type using eFHUB standards.

    Returns:
        tuple of (card_type, condition_rating, efhub_url)
    """
    matched = match_efhub_player(player_name)
    efhub_url = get_efhub_search_url(player_name)

    if matched:
        official_type = matched["card_type"]
        condition = matched.get("condition", "A-E Live Rating")

        # Honor Big Time and Epic Booster catalog mappings directly
        if official_type in ["Big Time", "Epic (Booster)"]:
            return official_type, condition, efhub_url

        if rating >= 108:
            official_type = "Big Time"
            condition = "Permanent B"
        elif visual_type == "Show Time" and "Epic" not in official_type:
            official_type = "Show Time"

        return official_type, condition, efhub_url

    # Player not in static catalog: deduce from rating & visual analysis
    if rating >= 108:
        card_type = "Big Time"
        condition = "Permanent B"
    elif visual_type == "Show Time":
        card_type = "Show Time"
        condition = "A-E Live Rating"
    elif visual_type in ["Epic (Booster)", "Epic"]:
        card_type = visual_type
        condition = "Permanent B"
    elif rating >= 104 and visual_type != "Standard":
        # Ultra-high ratings in modern eFootball are typically Big Time / Epic
        card_type = "Big Time"
        condition = "Permanent B"
    elif rating >= 98:
        card_type = "Highlight"
        condition = "A-E Live Rating"
    else:
        card_type = "Standard"
        condition = "A-E Live Rating"

    return card_type, condition, efhub_url
