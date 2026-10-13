"""Hugging Face LLM Tactical Intelligence Engine for SquadPredict.

Powers deep tactical matchup analysis, card tier evaluations, and AI scout reports
using modern Hugging Face models via huggingface_hub.InferenceClient, with a robust
offline tactical reasoning fallback.
"""

from __future__ import annotations

import os
from typing import Any

# Verified Hugging Face models supported for tactical reasoning
SUPPORTED_HF_MODELS = [
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
]

MODEL_ALIASES: dict[str, str] = {
    "google/gemma-4-12b-it": "google/gemma-3-12b-it",
    "google/gemma-4-12B-it": "google/gemma-3-12b-it",
    "google/gemma-4": "google/gemma-3-12b-it",
    "gemma-3": "google/gemma-3-12b-it",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama-3.3": "meta-llama/Llama-3.3-70B-Instruct",
}

raw_default = os.getenv("HF_MODEL", "google/gemma-3-12b-it")
DEFAULT_HF_MODEL = MODEL_ALIASES.get(raw_default, raw_default)


def _build_squad_summary(squad_name: str, formation: str, playstyle: str, strength: Any, players: list[dict]) -> str:
    """Format a detailed squad overview for LLM prompting."""
    lines = [
        f"Team: {squad_name}",
        f"Formation: {formation} | Playstyle: {playstyle} | Collective Strength: {strength or 'Auto-calculated'}",
        "Starting Lineup:",
    ]
    for p in players:
        name = p.get("player_name", "Unknown")
        pos = p.get("position", "N/A")
        ovr = p.get("overall_rating", 85)
        card = p.get("card_type", "Standard")
        cond = p.get("live_condition", "A-E Live Rating")
        lines.append(f"  - [{pos}] {name} (OVR: {ovr}) | Card Tier: {card} | Condition: {cond}")
    return "\n".join(lines)


def _generate_rule_based_scout_report(
    squad_a_name: str,
    formation_a: str,
    players_a: list[dict],
    squad_b_name: str,
    formation_b: str,
    players_b: list[dict],
    model_prediction: dict[str, Any],
) -> str:
    """Deterministic offline tactical reasoning engine when HF token is absent or offline."""
    win_pct = model_prediction.get("win_prob", 0.33) * 100
    draw_pct = model_prediction.get("draw_prob", 0.33) * 100
    loss_pct = model_prediction.get("loss_prob", 0.33) * 100
    score = model_prediction.get("predicted_score", "1 - 1")

    # Count card tiers
    bt_a = sum(1 for p in players_a if p.get("card_type") == "Big Time")
    epic_a = sum(1 for p in players_a if "Epic" in p.get("card_type", ""))
    bt_b = sum(1 for p in players_b if p.get("card_type") == "Big Time")
    epic_b = sum(1 for p in players_b if "Epic" in p.get("card_type", ""))

    # Key attackers and defenders
    atk_a = [p["player_name"] for p in players_a if p.get("position") in ["CF", "SS", "LWF", "RWF"]]
    atk_b = [p["player_name"] for p in players_b if p.get("position") in ["CF", "SS", "LWF", "RWF"]]
    def_a = [p["player_name"] for p in players_a if p.get("position") in ["CB", "LB", "RB", "DMF"]]
    def_b = [p["player_name"] for p in players_b if p.get("position") in ["CB", "LB", "RB", "DMF"]]

    report = f"""### 🤖 AI Tactical Intelligence & Scout Report
*Engine: Offline Tactical Reasoning Engine (Hugging Face Fallback Mode)*

---

#### 1. ⚔️ Tactical & Formation Clash ({formation_a} vs {formation_b})
- **Midfield Dynamics:**
  - **{squad_a_name} ({formation_a})** deploys a balanced structure with direct vertical transitions.
  - **{squad_b_name} ({formation_b})** counters with high offensive central density, attempting to create overloads in the half-spaces.
  - **Battle for the Pivot:** The clash between the opposing attacking midfielders and defensive anchors will dictate tempo. Controlling the central channel before triggering wingers into wide pockets is paramount.

#### 2. 🌟 Card Tier & Momentum Skill Impact
- **{squad_a_name}:** **{bt_a} Big Time** and **{epic_a} Epic (Booster)** cards. Permanent "B" condition form guarantees maximum physical condition and zero downward arrows on matchday.
- **{squad_b_name}:** **{bt_b} Big Time** and **{epic_b} Epic (Booster)** cards. High-tier cards feature enhanced momentum skills (e.g. Phenomenal Finishing, Visionary Pass, Fortress) that sway late-game outcomes.

#### 3. 🎯 Critical 1v1 Lane Battles
- **Striker vs Center-Backs:**
  - *{', '.join(atk_a[:2]) if atk_a else 'Attack'}* vs *{', '.join(def_b[:2]) if def_b else 'Defense'}* — Watch for speed bursts and quick turns against physical anchors.
  - *{', '.join(atk_b[:2]) if atk_b else 'Attack'}* vs *{', '.join(def_a[:2]) if def_a else 'Defense'}* — Tight marking on key playmakers is necessary to eliminate cutback opportunities.
- **Goalkeeper Face-off:** Top-tier boosted goalkeeping will minimize rebound goals; long-range curling efforts are the highest-percentage scoring route.

#### 4. 📋 Recommended Managerial Counter-Tactics
- **Attacking Phase:** Focus direct balls into the channels behind attacking full-backs. Use one-two passing through the central playmaker to break press lines.
- **Defensive Phase:** Set Defensive Instructions on full-backs to prevent counter-attack overloads. Apply tight marking on the opponent's primary Big Time creator.

---
**Projected Statistical Outlook:**
- **{squad_a_name} Win:** `{win_pct:.1f}%` | **Draw:** `{draw_pct:.1f}%` | **{squad_b_name} Win:** `{loss_pct:.1f}%`
- **Projected Scoreline:** **{score}**
"""
    return report


def generate_tactical_scout_report(
    squad_a_name: str,
    formation_a: str,
    playstyle_a: str,
    strength_a: Any,
    players_a: list[dict],
    squad_b_name: str,
    formation_b: str,
    playstyle_b: str,
    strength_b: Any,
    players_b: list[dict],
    model_prediction: dict[str, Any],
    hf_token: str | None = None,
    model_id: str = DEFAULT_HF_MODEL,
) -> str:
    """Generate comprehensive AI scout report via Hugging Face Inference API or offline fallback.
    
    Args:
        squad_a_name: Name of Squad A.
        formation_a: Tactical formation of Squad A.
        playstyle_a: Playstyle of Squad A.
        strength_a: Collective Strength rating.
        players_a: Starting 11 list of player dicts.
        squad_b_name: Name of Squad B.
        formation_b: Tactical formation of Squad B.
        playstyle_b: Playstyle of Squad B.
        strength_b: Collective Strength rating.
        players_b: Starting 11 list of player dicts.
        model_prediction: Dict with win_prob, draw_prob, loss_prob, predicted_score.
        hf_token: Optional Hugging Face User Access Token.
        model_id: Model identifier on Hugging Face Hub.

    Returns:
        Markdown string containing the tactical scout report.
    """
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # If no token provided, use the deterministic offline engine
    if not token:
        return _generate_rule_based_scout_report(
            squad_a_name, formation_a, players_a,
            squad_b_name, formation_b, players_b,
            model_prediction,
        )

    effective_model = MODEL_ALIASES.get(model_id, model_id)

    # Online Hugging Face InferenceClient
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token)

        summary_a = _build_squad_summary(squad_a_name, formation_a, playstyle_a, strength_a, players_a)
        summary_b = _build_squad_summary(squad_b_name, formation_b, playstyle_b, strength_b, players_b)

        win_pct = model_prediction.get("win_prob", 0.33) * 100
        draw_pct = model_prediction.get("draw_prob", 0.33) * 100
        loss_pct = model_prediction.get("loss_prob", 0.33) * 100
        score = model_prediction.get("predicted_score", "1 - 1")

        system_prompt = (
            "You are an elite eFootball tactical analyst and scout master. "
            "Analyze head-to-head eFootball Game Plan squad matchups with intense tactical precision, "
            "evaluating formation strengths/weaknesses, Big Time/Epic Booster card advantages, "
            "key player matchups, and defensive/offensive adjustments. "
            "Write your response in structured, professional GitHub markdown with clear headings."
        )

        user_content = f"""Please provide an intense-precision tactical scout report for this upcoming eFootball match:

=== SQUAD A ===
{summary_a}

=== SQUAD B ===
{summary_b}

=== STATISTICAL MODEL BASELINE ===
- {squad_a_name} Win Probability: {win_pct:.1f}%
- Draw Probability: {draw_pct:.1f}%
- {squad_b_name} Win Probability: {loss_pct:.1f}%
- Projected Baseline Scoreline: {score}

Structure your report into these 4 sections:
1. ⚔️ Tactical & Formation Clash (Formations, playstyle interactions, midfield battle)
2. 🌟 Card Tier & Momentum Skill Impact (Big Time, Epic Booster, Show Time perks, Permanent B condition form)
3. 🎯 Critical 1v1 Lane Battles (Strikers vs CBs, Playmaker vs DMF, GK quality)
4. 📋 Recommended Managerial Counter-Tactics (Specific offensive/defensive instructions for {squad_a_name} to triumph)
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Try requested model first, then fallback to verified serverless models if needed
        model_candidates = [effective_model]
        for fb_m in ["google/gemma-3-12b-it", "Qwen/Qwen2.5-72B-Instruct"]:
            if fb_m not in model_candidates:
                model_candidates.append(fb_m)

        last_error = None
        for attempt_model in model_candidates:
            try:
                response = client.chat_completion(
                    messages=messages,
                    model=attempt_model,
                    max_tokens=850,
                    temperature=0.7,
                )
                reply = response.choices[0].message.content
                header = f"### 🤖 AI Tactical Intelligence & Scout Report\n*Powered by Hugging Face (`{attempt_model}`)*\n\n---\n\n"
                return header + reply
            except Exception as candidate_err:
                last_error = candidate_err
                continue

        # If all candidates raised exceptions, raise the last error to trigger rule-based fallback
        if last_error:
            raise last_error

    except Exception as e:
        # Graceful fallback if API fails (rate-limit, invalid token, or timeout)
        fallback = _generate_rule_based_scout_report(
            squad_a_name, formation_a, players_a,
            squad_b_name, formation_b, players_b,
            model_prediction,
        )
        return f"> [!WARNING]\n> Hugging Face API call encountered an error (`{str(e)[:120]}`). Switched to offline tactical engine.\n\n" + fallback
