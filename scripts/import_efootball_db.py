"""Import comprehensive eFootball / eFHUB player database.

Scrapes multi-page player listings from eFootball player databases (PESDB/eFHUB),
extracting player names, in-game positions, overall ratings, and card categories.
Merges with curated Big Time & Epic Booster definitions into offline CSV databases.
"""

import csv
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EFHUB_CSV = PROJECT_ROOT / "data" / "efhub_players.csv"
REFERENCE_CSV = PROJECT_ROOT / "data" / "player_reference.csv"

# Valid eFootball positions
VALID_POSITIONS = {
    "GK", "CB", "LB", "RB", "LWB", "RWB",
    "DMF", "CMF", "AMF", "LMF", "RMF",
    "LWF", "RWF", "SS", "CF", "ST",
}

# Position normalization
POS_NORM = {
    "LW": "LWF", "RW": "RWF", "CAM": "AMF", "CM": "CMF", "CDM": "DMF",
    "LM": "LMF", "RM": "RMF",
}


def classify_tier(name: str, rating: int, card_type_text: str, pack_name: str) -> str:
    """Classify card tier using eFootball and eFHUB taxonomy."""
    c_lower = (card_type_text + " " + pack_name).lower()
    n_lower = name.lower()

    # Big Time defining cards
    if "big time" in c_lower or (rating >= 107 and "neuer" in n_lower) or (rating == 106 and "rice" in n_lower) or (rating == 106 and "aubameyang" in n_lower) or rating >= 108:
        return "Big Time"

    # Show Time cards
    if "show time" in c_lower or "showtime" in c_lower or (rating == 106 and "mbapp" in n_lower):
        return "Show Time"

    # Epic / Epic Booster
    if "epic" in c_lower or "legend" in c_lower or rating >= 102:
        return "Epic (Booster)"

    # Highlight / Featured / POTW / Strike Arena / POTD
    if any(k in c_lower for k in ["highlight", "featured", "potw", "potd", "pots", "strike arena", "trending", "selection"]):
        return "Highlight"

    if rating >= 98:
        return "Highlight"

    return "Standard"


class PesdbParser(HTMLParser):
    """Parse player cards from PESDB eFootball HTML listing."""

    def __init__(self):
        super().__init__()
        self.players = []
        self._in_article = False
        self._current_player = {}
        self._in_name = False
        self._in_card_type = False
        self._in_pack = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        tag_class = attrs_dict.get("class", "")

        if tag == "article" and "efootball-result-card" in tag_class:
            self._in_article = True
            self._current_player = {}

        if not self._in_article:
            return

        if tag == "img":
            alt = attrs_dict.get("alt", "")
            # Pattern: "Player Name — POS, OVR XX, MAX YY"
            m = re.match(r"^(.+?)\s*[—–-]\s*([A-Za-z]+),\s*OVR\s*(\d+)", alt)
            if m:
                p_name = m.group(1).strip()
                pos = m.group(2).upper()
                ovr = int(m.group(3))
                pos = POS_NORM.get(pos, pos)
                if pos in VALID_POSITIONS:
                    self._current_player["player_name"] = p_name
                    self._current_player["position"] = pos
                    self._current_player["overall_rating"] = ovr

        if tag == "a" and "href" in attrs_dict and "/efootball/players/" in attrs_dict["href"]:
            href = attrs_dict["href"]
            id_m = re.search(r"-(\d+)$", href)
            if id_m:
                self._current_player["efhub_id"] = id_m.group(1)
                self._current_player["efhub_url"] = f"https://efootballhub.net/players/{id_m.group(1)}"

        if "card-type" in tag_class:
            self._in_card_type = True

        if "efootball-result-pack" in tag_class:
            self._in_pack = True

    def handle_data(self, data):
        if self._in_card_type:
            self._current_player["raw_card_type"] = data.strip()
            self._in_card_type = False
        elif self._in_pack:
            self._current_player["pack_name"] = data.strip()
            self._in_pack = False

    def handle_endtag(self, tag):
        if tag == "article" and self._in_article:
            self._in_article = False
            if "player_name" in self._current_player and "overall_rating" in self._current_player:
                name = self._current_player["player_name"]
                ovr = self._current_player["overall_rating"]
                c_type = self._current_player.get("raw_card_type", "")
                pack = self._current_player.get("pack_name", "")
                self._current_player["card_type"] = classify_tier(name, ovr, c_type, pack)
                self.players.append(self._current_player.copy())
            self._current_player = {}


def fetch_pesdb_page(page: int) -> list[dict]:
    """Fetch and parse one page from PESDB eFootball database."""
    url = f"https://pesdb.net/efootball/players/?sort=overall_rating&page={page}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            parser = PesdbParser()
            parser.feed(html)
            return parser.players
    except Exception as e:
        print(f"  Error fetching page {page}: {e}")
        return []


def run_database_import(max_pages: int = 12) -> int:
    """Import top players across multiple pages and merge with existing databases."""
    print(f"=== Starting eFootball / eFHUB Database Import ({max_pages} pages) ===")
    all_players: dict[str, dict] = {}

    for p in range(1, max_pages + 1):
        print(f"Fetching page {p}/{max_pages}... ", end="", flush=True)
        players = fetch_pesdb_page(p)
        print(f"found {len(players)} players")
        for pl in players:
            key = f"{pl['player_name']}_{pl['position']}_{pl['overall_rating']}"
            all_players[key] = pl
        time.sleep(0.6)

    print(f"\nTotal scraped cards: {len(all_players)}")

    # Load existing reference catalog to ensure our ground-truth players are preserved
    existing_ref = []
    if REFERENCE_CSV.exists():
        with open(REFERENCE_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ref.append(row)

    # Save to data/efhub_players.csv
    EFHUB_CSV.parent.mkdir(parents=True, exist_ok=True)
    sorted_players = sorted(all_players.values(), key=lambda x: x["overall_rating"], reverse=True)
    fieldnames = ["player_name", "position", "overall_rating", "card_type", "efhub_id", "efhub_url"]
    with open(EFHUB_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_players)

    print(f"Saved {len(sorted_players)} records to {EFHUB_CSV}")

    # Merge into player_reference.csv with estimated physical & tactical attributes
    merged_catalog: dict[str, dict] = {}
    for r in existing_ref:
        merged_catalog[r["player_name"]] = r

    for pl in sorted_players:
        name = pl["player_name"]
        if name in merged_catalog:
            continue
        pos = pl["position"]
        ovr = pl["overall_rating"]
        ct = pl["card_type"]

        # Synthesize realistic eFootball attributes
        if pos == "GK":
            atk, defn, pace, drib = 45, ovr, 55, 60
        elif pos in ["CB", "LB", "RB", "LWB", "RWB"]:
            atk, defn, pace, drib = 58, ovr, 82 if pos in ["LB", "RB"] else 76, 72
        elif pos in ["DMF", "CMF", "AMF", "LMF", "RMF"]:
            atk, defn, pace, drib = 84 if pos == "AMF" else 74, 86 if pos == "DMF" else 68, 80, 85
        else:
            atk, defn, pace, drib = ovr, 45, 92, 88

        merged_catalog[name] = {
            "player_name": name,
            "position": pos,
            "overall_rating": ovr,
            "pace": pace,
            "dribbling": drib,
            "attack_rating": atk,
            "defense_rating": defn,
            "chemistry_style": "Basic",
            "card_type": ct,
        }

    ref_fieldnames = [
        "player_name", "position", "overall_rating", "pace", "dribbling",
        "attack_rating", "defense_rating", "chemistry_style", "card_type"
    ]
    with open(REFERENCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ref_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_catalog.values())

    print(f"Expanded {REFERENCE_CSV} to {len(merged_catalog)} total players!")
    return len(sorted_players)


if __name__ == "__main__":
    run_database_import(max_pages=15)
