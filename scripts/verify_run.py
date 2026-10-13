import sys
from pathlib import Path
from squadpredict.ocr_ingest import ingest_screenshot

img1 = Path("sample_screenshots/WhatsApp Image 2026-09-10 at 5.36.26 PM.jpeg")
img2 = Path("sample_screenshots/WhatsApp Image 2026-09-10 at 5.36.26 PM (1).jpeg")

print("=== SQUAD 1 INGESTION ===")
res1 = ingest_screenshot(img1)
print(f"Formation: {res1.get('formation')}")
print(f"Playstyle: {res1.get('playstyle')}")
print(f"Strength: {res1.get('collective_strength')}")
print(f"Players count: {len(res1.get('players', []))}")
print("Lineup:")
for p in res1.get('players', []):
    print(f"  [{p.get('position')}] {p.get('player_name')} | Rating: {p.get('overall_rating')} | Card: {p.get('card_type')}")

print("\n=== SQUAD 2 INGESTION ===")
res2 = ingest_screenshot(img2)
print(f"Formation: {res2.get('formation')}")
print(f"Playstyle: {res2.get('playstyle')}")
print(f"Strength: {res2.get('collective_strength')}")
print(f"Players count: {len(res2.get('players', []))}")
print("Lineup:")
for p in res2.get('players', []):
    print(f"  [{p.get('position')}] {p.get('player_name')} | Rating: {p.get('overall_rating')} | Card: {p.get('card_type')}")
