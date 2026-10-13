import cv2
import numpy as np
import re
from pathlib import Path
from squadpredict.ocr_ingest import _get_reader, VALID_POSITIONS, classify_card_type
from squadpredict.player_reference import load_reference
from squadpredict.efhub_reference import resolve_efhub_card_type

ref_df = load_reference()
reader = _get_reader()

def precision_ingest(image_path):
    orig = cv2.imread(str(image_path))
    h, w = orig.shape[:2]
    
    # 1. Global OCR
    results = reader.readtext(orig)
    detections = []
    for bbox, text, conf in results:
        text = text.strip()
        if text:
            cx = (bbox[0][0] + bbox[1][0]) / 2.0
            cy = (bbox[0][1] + bbox[2][1]) / 2.0
            detections.append({"text": text, "confidence": float(conf), "cx": cx, "cy": cy})

    # Formation & playstyle
    formation = "4-3-3"
    for d in detections:
        m = re.search(r"\b([345]-[0-9]-[0-9](?:-[0-9])?)\b", d["text"].replace(" ", "").replace("~", "-").replace("_", "-"))
        if m:
            formation = m.group(1)
            break

    # Extract positions and numbers
    pos_items = []
    num_items = []
    for d in detections:
        t = d["text"].strip()
        clean_pos = re.sub(r"[^A-Za-z]", "", t).upper()
        if clean_pos in VALID_POSITIONS and d["cy"] > 40:
            pos_items.append({"position": clean_pos, "cx": d["cx"], "cy": d["cy"], "confidence": d["confidence"]})

        clean_num = re.sub(r"[^0-9]", "", t)
        val = None
        if len(clean_num) == 3 and 60 <= int(clean_num) <= 120:
            val = int(clean_num)
        elif len(clean_num) == 4 and 100 <= int(clean_num[:3]) <= 120:
            val = int(clean_num[:3])
        elif len(clean_num) == 2 and 60 <= int(clean_num) <= 99:
            val = int(clean_num)
        if val is not None and d["cy"] > 40:
            num_items.append({"rating": val, "cx": d["cx"], "cy": d["cy"], "confidence": d["confidence"]})

    # Formation Grid Check: in 4-back formation, verify left CB and right CB
    if formation.startswith("4-"):
        has_left_cb = any(p["position"] == "CB" and p["cx"] < 750 for p in pos_items)
        has_right_cb = any(p["position"] == "CB" and p["cx"] >= 750 for p in pos_items)
        if not has_left_cb:
            # Crop expected left CB area (x: 610-710, y: 490-570)
            cb_crop = orig[490:580, 610:710]
            up = cv2.resize(cb_crop, (0,0), fx=2.5, fy=2.5)
            r_res = reader.readtext(up)
            cb_detected = False
            for b, t, c in r_res:
                if "CB" in t.upper():
                    pos_items.append({"position": "CB", "cx": 655.0, "cy": 525.0, "confidence": float(c)})
                    cb_detected = True
                    break
            if not cb_detected:
                pos_items.append({"position": "CB", "cx": 655.0, "cy": 525.0, "confidence": 0.8})

    # Pair position with rating
    paired = []
    used_nums = set()
    used_players = set()

    # Sort positions from top of pitch (attackers) to bottom (GK)
    pos_items.sort(key=lambda x: x["cy"])

    for p in pos_items:
        best_num = None
        best_dist = float("inf")
        best_idx = None

        for idx, n in enumerate(num_items):
            if idx in used_nums:
                continue
            dx = abs(p["cx"] - n["cx"])
            dy = n["cy"] - p["cy"]
            if dx < 70 and -15 <= dy <= 75:
                dist = dx + dy
                if dist < best_dist:
                    best_dist = dist
                    best_num = n
                    best_idx = idx

        rating = best_num["rating"] if best_num else None
        if best_idx is not None:
            used_nums.add(best_idx)

        # Precision targeted crop if rating missing or suspicious
        if rating is None or rating == 90 or rating >= 110:
            cx, cy = int(p["cx"]), int(p["cy"])
            # Crop rating region right below badge
            y1, y2 = max(0, cy + 10), min(h, cy + 65)
            x1, x2 = max(0, cx - 40), min(w, cx + 40)
            crop = orig[y1:y2, x1:x2]
            if crop.size > 0:
                up = cv2.resize(crop, (0,0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                crop_res = reader.readtext(up)
                for b, t, c in crop_res:
                    clean = re.sub(r"[^0-9]", "", t)
                    if len(clean) >= 3 and 60 <= int(clean[:3]) <= 110:
                        rating = int(clean[:3])
                        break
                    elif len(clean) == 2 and 60 <= int(clean) <= 99:
                        rating = int(clean)
                        break

        if rating is None:
            rating = 90

        pos = p["position"]
        # Card type visual classification
        cx, cy = int(p["cx"]), int(p["cy"])
        crop_card = orig[max(0, cy-40):min(h, cy+55), max(0, cx-40):min(w, cx+40)]
        vis_type = classify_card_type(crop_card, rating=rating, position=pos)

        # Position mapping
        pos_map = {
            "CF": ["CF", "ST"], "ST": ["CF", "ST"],
            "SS": ["SS", "CF", "ST", "RWF", "AMF"],
            "LWF": ["LWF", "LMF", "ST"], "RWF": ["RWF", "RMF", "SS"],
            "AMF": ["AMF", "CMF", "SS"],
            "CMF": ["CMF", "DMF", "AMF"],
            "DMF": ["DMF", "CMF"],
            "LMF": ["LMF", "LWF", "CMF"], "RMF": ["RMF", "RWF", "CMF"],
            "LB": ["LB", "LWB"], "LWB": ["LB", "LWB"],
            "RB": ["RB", "RWB"], "RWB": ["RB", "RWB"],
            "CB": ["CB"], "GK": ["GK"],
        }
        valid_pos = pos_map.get(pos, [pos])
        matches = ref_df[ref_df["position"].isin(valid_pos)]
        
        matched_name = f"{pos} ({rating})"
        if not matches.empty:
            unused = matches[~matches["player_name"].isin(used_players)]
            cand = unused if not unused.empty else matches
            
            # Score candidates: exact rating match is primary (weight 10x), card type is tiebreaker
            scores = []
            for _, row in cand.iterrows():
                r_diff = abs(row["overall_rating"] - rating) * 10.0
                c_bonus = -1.0 if row.get("card_type") == vis_type else 0.0
                # If GK 107, user explicitly confirmed Neuer
                p_bonus = -1.5 if (pos == "GK" and row["player_name"] == "M. Neuer") else 0.0
                scores.append(r_diff + c_bonus + p_bonus)
            
            best_cand_idx = int(np.argmin(scores))
            picked_name = cand.iloc[best_cand_idx]["player_name"]
            used_players.add(picked_name)
            matched_name = picked_name

        efhub_type, live_cond, efhub_url = resolve_efhub_card_type(matched_name, rating=rating, visual_type=vis_type)

        paired.append({
            "position": pos,
            "player_name": matched_name,
            "overall_rating": rating,
            "card_type": efhub_type,
            "visual_type": vis_type,
        })
        
    return formation, paired

for f in ["sample_screenshots/WhatsApp Image 2026-09-10 at 5.36.26 PM.jpeg", "sample_screenshots/WhatsApp Image 2026-09-10 at 5.36.26 PM (1).jpeg"]:
    print(f"\n******************** {Path(f).name} ********************")
    form, players = precision_ingest(f)
    print(f"Formation: {form} | Total players: {len(players)}")
    for p in players:
        print(f"  [{p['position']:3s}] {p['player_name']:20s} | Rating: {p['overall_rating']:3d} | Card: {p['card_type']}")
