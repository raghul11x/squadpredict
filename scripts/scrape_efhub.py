"""Scrape eFHUB (efootballhub.net) player data from search pages and store offline.

This script parses the server-rendered HTML from eFHUB to extract player data:
- Player name (from <img alt="..."> tags)
- Position (from position label spans, e.g., SS, CF, AMF, CMF, DMF, LWF, RWF)
- Max Overall Rating (from rating spans)
- eFHUB ID (from href="/players/{id}" links)

Since eFHUB is a Next.js app with some client-side rendering, we extract data
from the HTML structure of player card components that are server-rendered.
"""

import csv
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# eFootball position labels
VALID_POSITIONS = {
    "GK", "CB", "LB", "RB", "LWB", "RWB",
    "DMF", "CMF", "AMF", "LMF", "RMF",
    "LWF", "RWF", "SS", "CF", "ST",
    "LW", "RW", "LM", "RM", "AM", "CM", "DM"
}

# Card type categories based on eFHUB taxonomy
CARD_TYPE_MAP = {
    # Big Time cards - rating >= 108 or specific defining-moment cards
    "big_time": {"min_rating": 108, "label": "Big Time"},
    # Epic Booster - legendary cards typically 102-107
    "epic": {"min_rating": 102, "label": "Epic (Booster)"},
    # Show Time - premium cards typically 104-106
    "showtime": {"min_rating": 104, "label": "Show Time"},
    # Highlight - featured/club selection typically 98-103
    "highlight": {"min_rating": 98, "label": "Highlight"},
}


class EFHUBCardParser(HTMLParser):
    """Parse eFHUB HTML to extract player card data from the grid."""

    def __init__(self):
        super().__init__()
        self.players = []
        self._current_player = {}
        self._in_player_link = False
        self._capture_text = False
        self._text_buffer = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        # Detect player card links: <a href="/players/{id}">
        if tag == "a":
            href = attrs_dict.get("href", "")
            match = re.match(r"^/players/(\d+)$", href)
            if match:
                self._in_player_link = True
                self._current_player = {
                    "efhub_id": match.group(1),
                    "efhub_url": f"https://efootballhub.net{href}",
                }

        # Detect player name from <img alt="Player Name">
        if tag == "img" and self._in_player_link:
            alt = attrs_dict.get("alt", "")
            if alt and "logo" not in alt.lower() and "efhub" not in alt.lower():
                self._current_player["player_name"] = alt

        # Capture text content from spans (for rating and position)
        if tag == "span" and self._in_player_link:
            self._capture_text = True
            self._text_buffer = ""

    def handle_data(self, data):
        if self._capture_text:
            self._text_buffer += data.strip()

    def handle_endtag(self, tag):
        if tag == "span" and self._capture_text:
            self._capture_text = False
            text = self._text_buffer.strip()

            if self._in_player_link and text:
                # Check if it's a rating (numeric)
                if text.isdigit() and 40 <= int(text) <= 120:
                    self._current_player["overall_rating"] = int(text)
                # Check if it's a position
                elif text.upper() in VALID_POSITIONS:
                    self._current_player["position"] = text.upper()

        # When the player link closes, save the player
        if tag == "a" and self._in_player_link:
            self._in_player_link = False
            if "player_name" in self._current_player and "overall_rating" in self._current_player:
                self.players.append(self._current_player.copy())
            self._current_player = {}


def fetch_page(url: str) -> str:
    """Fetch a page with a browser-like User-Agent."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return ""


def classify_card_type(rating: int) -> str:
    """Classify card type based on rating threshold."""
    if rating >= 108:
        return "Big Time"
    elif rating >= 104:
        return "Epic (Booster)"
    elif rating >= 98:
        return "Highlight"
    elif rating >= 90:
        return "Standard"
    else:
        return "Standard"


def scrape_efhub_players(search_names: list[str]) -> list[dict]:
    """Scrape player data from eFHUB search pages."""
    all_players = {}  # Deduplicate by efhub_id

    for name in search_names:
        url = f"https://efootballhub.net/search/players?name={urllib.request.quote(name)}&max=true"
        print(f"Fetching: {name}... ", end="", flush=True)
        html = fetch_page(url)
        if not html:
            continue

        parser = EFHUBCardParser()
        parser.feed(html)
        new_count = 0
        for p in parser.players:
            pid = p.get("efhub_id", "")
            if pid and pid not in all_players:
                # Add card type classification
                p["card_type"] = classify_card_type(p.get("overall_rating", 0))
                all_players[pid] = p
                new_count += 1
        print(f"found {len(parser.players)} cards, {new_count} new")
        time.sleep(0.5)  # Be polite

    return list(all_players.values())


def main():
    output_path = Path(__file__).parent.parent / "data" / "efhub_players.csv"

    # Search for a wide range of popular players by position and name
    search_queries = [
        # Top players by name
        "messi", "ronaldo", "mbappe", "haaland", "neymar",
        "salah", "de bruyne", "bellingham", "vinicius", "saka",
        "kane", "lewandowski", "modric", "kroos", "pedri",
        "van dijk", "dias", "marquinhos", "rudiger", "saliba",
        "neuer", "courtois", "donnarumma", "alisson", "ter stegen",
        "hakimi", "davies", "robertson", "alexander-arnold", "cancelo",
        "rice", "casemiro", "de jong", "silva", "gundogan",
        "osimhen", "martinez", "alvarez", "leao", "fati",
        # Legends
        "maradona", "pele", "ronaldinho", "beckenbauer", "maldini",
        "van basten", "gullit", "xavi", "iniesta", "guardiola",
        "vieira", "cannavaro", "nesta", "stam", "puyol",
        "schmeichel", "cole", "marcelo", "costacurta", "sneijder",
        "etoo", "aubameyang", "beckham", "zidane", "henry",
        "del piero", "totti", "batistuta", "bergkamp", "pirlo",
        "rummenigge", "matthaus", "kahn", "lahm", "cafu",
        "carlos", "drogba", "lampard", "gerrard", "scholes",
        "giggs", "rooney", "torres", "forlan", "owen",
        "rivaldo", "seedorf", "nedved", "davids", "kluivert",
        # Current stars
        "palmer", "yamal", "pedri", "gavi", "foden",
        "odegaard", "havertz", "isak", "watkins", "rashford",
        "nunez", "diaz", "jota", "szoboszlai", "soule",
        "cunha", "semenyo", "kelleher", "torres", "mitchell",
        "woltemade", "ndiaye",
    ]

    print(f"Starting eFHUB scrape with {len(search_queries)} search queries...")
    print(f"Output: {output_path}")
    print()

    players = scrape_efhub_players(search_queries)

    # Sort by rating descending
    players.sort(key=lambda p: p.get("overall_rating", 0), reverse=True)

    # Write CSV
    fieldnames = ["player_name", "position", "overall_rating", "card_type", "efhub_id", "efhub_url"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(players)

    print(f"\nDone! Saved {len(players)} players to {output_path}")


if __name__ == "__main__":
    main()
