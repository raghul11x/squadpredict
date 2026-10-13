"""SquadPredict -- Streamlit application."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from squadpredict.config import (
    CLASSIFIER_PATH,
    DB_PATH,
    KNOWN_FORMATIONS,
    PLAYER_REFERENCE_CSV,
)
from squadpredict.db import (
    get_all_real_results,
    get_all_squads,
    get_performance_stats,
    get_recent_predictions,
    init_db,
    log_real_result,
    save_prediction,
    save_squad,
)
from squadpredict.features import build_match_feature_vector, build_team_features
from squadpredict.hf_llm import DEFAULT_HF_MODEL, SUPPORTED_HF_MODELS, generate_tactical_scout_report
from squadpredict.model import load_model, predict
from squadpredict.ocr_ingest import ingest_screenshot
from squadpredict.player_reference import load_reference, lookup_player
from squadpredict.retrain import blend_and_retrain


@st.cache_resource
def _get_cached_reader():
    """Cache EasyOCR reader in memory for fast extractions."""
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


@st.dialog("Game Plan Screenshot Preview", width="large")
def _show_screenshot_dialog(img_bytes: bytes, title: str) -> None:
    """Display a full modal dialog preview of the uploaded screenshot."""
    st.image(img_bytes, caption=title, use_container_width=True)
    st.caption("Press Escape or click outside to close.")


def _init_session_state() -> None:
    """Initialize session state variables."""
    if "db_initialized" not in st.session_state:
        init_db()
        st.session_state.db_initialized = True
    if "reference_df" not in st.session_state:
        st.session_state.reference_df = load_reference()
    if "model" not in st.session_state:
        if CLASSIFIER_PATH.exists():
            st.session_state.model = load_model()
        else:
            st.session_state.model = None
    if "ocr_cache" not in st.session_state:
        st.session_state.ocr_cache = {}


def _render_squad_input(label: str, key_prefix: str) -> tuple[str, str, list[dict], dict] | None:
    """Render squad input UI (screenshot upload or manual entry).
    
    Returns (squad_name, formation, players_list, extra_metadata) or None if not ready.
    """
    st.subheader(f"{label}")
    
    input_method = st.radio(
        f"Input method for {label}",
        ["Screenshot Upload", "Manual Entry"],
        key=f"{key_prefix}_input_method",
        horizontal=True,
    )
    
    if input_method == "Screenshot Upload":
        return _render_screenshot_input(label, key_prefix)
    else:
        return _render_manual_input(label, key_prefix)


def _render_screenshot_input(label: str, key_prefix: str) -> tuple[str, str, list[dict], dict] | None:
    """Render screenshot upload and automated OCR extraction UI."""
    uploaded_file = st.file_uploader(
        f"Upload {label} Game Plan screenshot",
        type=["png", "jpg", "jpeg"],
        key=f"{key_prefix}_upload",
        help="Upload a full Game Plan screenshot showing the pitch formation and player ratings.",
    )
    
    if uploaded_file is None:
        return None

    # Read bytes and hash for cache
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if file_hash in st.session_state.ocr_cache:
        result = st.session_state.ocr_cache[file_hash]
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            with st.spinner(f"Analyzing {label} Game Plan screenshot..."):
                result = ingest_screenshot(tmp_path, st.session_state.reference_df)
                st.session_state.ocr_cache[file_hash] = result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Uploaded Screenshot Preview & Fullscreen Modal
    prev_col1, prev_col2 = st.columns([1, 1])
    with prev_col1:
        if st.button(f"🔍 Enlarge / Zoom {label} Screenshot", key=f"{key_prefix}_preview_modal_btn", use_container_width=True):
            _show_screenshot_dialog(file_bytes, f"{label} Game Plan Screenshot")
    with prev_col2:
        with st.popover("🖼️ Quick Popover View", use_container_width=True):
            st.image(file_bytes, caption=f"{label} Screenshot Preview", use_container_width=True)

    with st.expander(f"🖼️ {label} Screenshot Preview (Click to view / collapse)", expanded=True):
        st.image(
            file_bytes,
            caption=f"{label} Game Plan — Click top-right icon for full-screen zoom",
            use_container_width=True,
        )

    # Display detected squad highlights
    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        st.metric("Formation", result.get("formation", "4-3-3"))
    with hcol2:
        st.metric("Playstyle", result.get("playstyle", "Balanced"))
    with hcol3:
        strength_str = str(result["collective_strength"]) if result.get("collective_strength") else "Auto-calc"
        st.metric("Collective Strength", strength_str)

    st.success(f"Extracted {result['player_count']}/11 starting players automatically.")

    # Display Card Tier Breakdown badges
    breakdown = result.get("card_type_breakdown", {})
    if breakdown:
        badges = []
        if breakdown.get("Big Time"):
            badges.append(f"🌟 **Big Time:** {breakdown['Big Time']}")
        if breakdown.get("Show Time"):
            badges.append(f"⚡ **Show Time:** {breakdown['Show Time']}")
        if breakdown.get("Epic (Booster)"):
            badges.append(f"🏆 **Epic (Booster):** {breakdown['Epic (Booster)']}")
        if breakdown.get("Epic"):
            badges.append(f"🎖️ **Epic:** {breakdown['Epic']}")
        if breakdown.get("Highlight"):
            badges.append(f"🟢 **Highlight:** {breakdown['Highlight']}")
        if breakdown.get("Standard"):
            badges.append(f"⚪ **Standard:** {breakdown['Standard']}")
        if badges:
            st.info(" | ".join(badges))

    players = result["players"]
    squad_name = st.text_input(
        f"{label} Name", value=label, key=f"{key_prefix}_name_ocr"
    )
    
    current_formation = result.get("formation", "4-3-3")
    form_idx = KNOWN_FORMATIONS.index(current_formation) if current_formation in KNOWN_FORMATIONS else 0
    formation = st.selectbox(
        f"Formation for {label}",
        KNOWN_FORMATIONS,
        index=form_idx,
        key=f"{key_prefix}_formation_ocr",
    )

    if players:
        player_df = pd.DataFrame(players)
        display_cols = [
            "player_name", "position", "overall_rating", "card_type", "live_condition",
            "attack_rating", "defense_rating", "pace", "dribbling"
        ]
        available_cols = [c for c in display_cols if c in player_df.columns]
        
        with st.expander("Review & Adjust Lineup Ratings (eFHUB Reference)", expanded=False):
            edited_df = st.data_editor(
                player_df[available_cols],
                num_rows="dynamic",
                key=f"{key_prefix}_players_ocr",
                use_container_width=True,
            )
            final_players = edited_df.to_dict("records")
            st.caption("Card types and condition ratings cross-referenced against [eFHUB Database](https://efootballhub.net).")
        
        meta = {
            "playstyle": result.get("playstyle", "Balanced"),
            "collective_strength": result.get("collective_strength"),
        }
        return squad_name, formation, final_players, meta
    else:
        st.warning("No players detected. Please use manual entry or re-upload a clearer screenshot.")
        return None


def _render_manual_input(label: str, key_prefix: str) -> tuple[str, str, list[dict], dict] | None:
    """Render manual squad entry form."""
    squad_name = st.text_input(
        f"{label} name", value=f"{label}", key=f"{key_prefix}_name"
    )
    formation = st.selectbox(
        f"Formation for {label}",
        KNOWN_FORMATIONS,
        key=f"{key_prefix}_formation",
    )
    
    st.caption("Enter player names (one per line). Names are fuzzy-matched against the reference database.")
    
    player_names_text = st.text_area(
        f"Player names for {label}",
        height=200,
        key=f"{key_prefix}_player_names",
        placeholder="Courtois\nVan Dijk\nR. Dias\nA. Davies\nHakimi\nDe Bruyne\nBellingham\nPedri\nVinicius Jr.\nSalah\nHaaland",
    )
    
    if not player_names_text.strip():
        return None
    
    names = [n.strip() for n in player_names_text.strip().split("\n") if n.strip()]
    ref_df = st.session_state.reference_df
    
    players = []
    unmatched = []
    for name in names:
        player = lookup_player(name, ref_df)
        if player:
            players.append(player)
        else:
            unmatched.append(name)
    
    if unmatched:
        st.warning(f"Could not match: {', '.join(unmatched)}. Add them to data/player_reference.csv.")
    
    if players:
        player_df = pd.DataFrame(players)
        display_cols = ["player_name", "position", "overall_rating", "pace",
                       "dribbling", "attack_rating", "defense_rating"]
        available_cols = [c for c in display_cols if c in player_df.columns]
        edited_df = st.data_editor(
            player_df[available_cols],
            num_rows="dynamic",
            key=f"{key_prefix}_players_manual",
            use_container_width=True,
        )
        return squad_name, formation, edited_df.to_dict("records"), {}
    
    return None


def _tab_predict() -> None:
    """Tab 1: Predict match outcome."""
    st.header("Match Prediction & Squad Intelligence")
    
    if st.session_state.model is None:
        st.error(
            "No trained model found. Run 'python scripts/train_initial_model.py' first."
        )
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        result_a = _render_squad_input("Squad A (Your Squad)", "squad_a")
    with col2:
        result_b = _render_squad_input("Squad B (Opponent)", "squad_b")
    
    st.divider()
    
    # Both squads provided: Run full match forecast
    if result_a and result_b:
        name_a, formation_a, players_a, meta_a = result_a
        name_b, formation_b, players_b, meta_b = result_b
        
        features_a = build_team_features(players_a, formation_a)
        features_b = build_team_features(players_b, formation_b)
        
        prediction = predict(features_a, features_b, st.session_state.model)
        
        st.subheader("Match Outcome Forecast")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric(f"{name_a} Win Probability", f"{prediction['win_prob']:.1%}")
        with col_r2:
            st.metric("Draw Probability", f"{prediction['draw_prob']:.1%}")
        with col_r3:
            st.metric(f"{name_b} Win Probability", f"{prediction['loss_prob']:.1%}")
        
        st.markdown(f"### Projected Scoreline: **{prediction['predicted_scoreline']}**")
        
        chart_data = pd.DataFrame({
            "Outcome": [f"{name_a} Win", "Draw", f"{name_b} Win"],
            "Probability": [
                prediction["win_prob"],
                prediction["draw_prob"],
                prediction["loss_prob"],
            ],
        })
        st.bar_chart(chart_data.set_index("Outcome"), horizontal=True)
        
        # Tactical comparison metrics
        with st.expander("Tactical Comparison & Team Attribute Differentials", expanded=True):
            comp_df = pd.DataFrame({
                "Attribute": [
                    "Attack Rating", "Midfield Rating", "Defense Rating", "Avg Pace",
                    "🌟 Big Time Cards", "🏆 Epic Cards", "⚡ Show Time Cards", "Star Players (>=90)"
                ],
                name_a: [
                    features_a["avg_attack_rating"],
                    features_a["avg_midfield_rating"],
                    features_a["avg_defense_rating"],
                    features_a["avg_pace"],
                    features_a.get("bigtime_count", 0),
                    features_a.get("epic_count", 0),
                    features_a.get("showtime_count", 0),
                    features_a["star_player_count"],
                ],
                name_b: [
                    features_b["avg_attack_rating"],
                    features_b["avg_midfield_rating"],
                    features_b["avg_defense_rating"],
                    features_b["avg_pace"],
                    features_b.get("bigtime_count", 0),
                    features_b.get("epic_count", 0),
                    features_b.get("showtime_count", 0),
                    features_b["star_player_count"],
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # AI Tactical Intelligence & Scout Report (Powered by Hugging Face)
        st.markdown("---")
        with st.spinner("Analyzing tactical matchups with Hugging Face AI..."):
            playstyle_a = meta_a.get("playstyle", "Balanced") if meta_a else "Balanced"
            playstyle_b = meta_b.get("playstyle", "Balanced") if meta_b else "Balanced"
            strength_a = meta_a.get("collective_strength") if meta_a else None
            strength_b = meta_b.get("collective_strength") if meta_b else None

            scout_report = generate_tactical_scout_report(
                name_a, formation_a, playstyle_a, strength_a, players_a,
                name_b, formation_b, playstyle_b, strength_b, players_b,
                prediction,
                hf_token=os.getenv("HF_TOKEN"),
                model_id=st.session_state.get("hf_model_select", DEFAULT_HF_MODEL),
            )
            st.markdown(scout_report)
        st.markdown("---")

        # Save to database
        if st.button("Save Prediction to History", type="secondary"):
            players_a_json = json.dumps(players_a, default=str)
            players_b_json = json.dumps(players_b, default=str)
            
            squad_a_id = save_squad(name_a, formation_a, players_a_json, "ocr")
            squad_b_id = save_squad(name_b, formation_b, players_b_json, "ocr")
            
            model_version = getattr(st.session_state, "model_version", "v1_initial")
            
            pred_id = save_prediction(
                squad_a_id, squad_b_id,
                prediction["win_prob"],
                prediction["draw_prob"],
                prediction["loss_prob"],
                prediction["predicted_scoreline"],
                model_version,
            )
            
            st.session_state.last_prediction = {
                "prediction_id": pred_id,
                "squad_a_id": squad_a_id,
                "squad_b_id": squad_b_id,
                "squad_a_name": name_a,
                "squad_b_name": name_b,
            }
            st.success(f"Saved fixture (Prediction ID: #{pred_id}). Log result in Tab 2 after the match.")

    # Only Squad A provided: Show single squad analysis
    elif result_a:
        name_a, formation_a, players_a, meta_a = result_a
        features_a = build_team_features(players_a, formation_a)
        
        st.subheader(f"Squad Intelligence: {name_a}")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Attack Rating", f"{features_a['avg_attack_rating']:.1f}")
        with mcol2:
            st.metric("Midfield Rating", f"{features_a['avg_midfield_rating']:.1f}")
        with mcol3:
            st.metric("Defense Rating", f"{features_a['avg_defense_rating']:.1f}")
        with mcol4:
            st.metric("Overall Average", f"{features_a['avg_overall']:.1f}")
        
        st.info("Upload Squad B above to see head-to-head match win/draw/loss probabilities and scoreline prediction.")
    else:
        st.info("Upload your eFootball Game Plan screenshot above. SquadPredict will automatically extract the formation, ratings, and roster.")


def _tab_log_result() -> None:
    """Tab 2: Log real match result."""
    st.header("Log Real Match Result & Continuous Retraining")
    
    st.caption("Every match result logged shifts the model from synthetic simulation toward real-world performance.")
    
    recent_preds = get_recent_predictions(limit=20)
    
    if "last_prediction" in st.session_state:
        lp = st.session_state.last_prediction
        st.info(
            f"Latest fixture: **{lp['squad_a_name']}** vs **{lp['squad_b_name']}** "
            f"(Prediction ID: #{lp['prediction_id']})"
        )
        use_last = st.checkbox("Use latest fixture", value=True)
    else:
        use_last = False
    
    if not use_last and recent_preds:
        pred_options = {
            f"#{p['id']}: {p.get('squad_a_name', 'Team A')} vs {p.get('squad_b_name', 'Team B')} ({p['created_at']})": p
            for p in recent_preds
        }
        selected_label = st.selectbox("Select fixture from history", list(pred_options.keys()))
        selected_pred = pred_options[selected_label] if selected_label else None
    elif not use_last:
        st.warning("No predictions found in history. Run a prediction in Tab 1 first.")
        return
    else:
        selected_pred = None
    
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        score_a = st.number_input("Squad A Goals", min_value=0, max_value=20, value=0, key="score_a")
    with col2:
        st.markdown("<h2 style='text-align: center; padding-top: 25px;'>-</h2>", unsafe_allow_html=True)
    with col3:
        score_b = st.number_input("Squad B Goals", min_value=0, max_value=20, value=0, key="score_b")
    
    if score_a > score_b:
        outcome = "win"
        outcome_label = "Squad A Win"
    elif score_a < score_b:
        outcome = "loss"
        outcome_label = "Squad B Win"
    else:
        outcome = "draw"
        outcome_label = "Draw"
    
    st.info(f"Actual Outcome: **{outcome_label}**")
    
    if st.button("Log Result & Retrain Model (Weighted 3.0x)", type="primary", use_container_width=True):
        if use_last and "last_prediction" in st.session_state:
            lp = st.session_state.last_prediction
            pred_id, sq_a, sq_b = lp["prediction_id"], lp["squad_a_id"], lp["squad_b_id"]
        elif selected_pred:
            pred_id, sq_a, sq_b = selected_pred["id"], selected_pred["squad_a_id"], selected_pred["squad_b_id"]
        else:
            st.error("No prediction selected.")
            return
        
        scoreline = f"{score_a}-{score_b}"
        log_real_result(pred_id, sq_a, sq_b, scoreline, outcome)
        st.success(f"Result recorded: {scoreline} ({outcome_label})")
        
        with st.spinner("Retraining model with blended real telemetry..."):
            try:
                new_ver = blend_and_retrain()
                st.session_state.model = load_model()
                st.session_state.model_version = new_ver
                st.success(f"Model retrained and updated to {new_ver}!")
            except Exception as e:
                st.error(f"Retrain encountered an issue: {e}")


def _tab_performance() -> None:
    """Tab 3: Model performance dashboard."""
    st.header("Model Performance & Continuous Learning Telemetry")
    
    stats = get_performance_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches Logged", stats["total_matches"])
    with col2:
        st.metric("Correct Predictions", stats["correct_predictions"])
    with col3:
        acc_display = f"{stats['accuracy']:.1%}" if stats["total_matches"] > 0 else "N/A"
        st.metric("Prediction Accuracy", acc_display)
    
    if stats["total_matches"] == 0:
        st.info("Log match results in Tab 2 to track model accuracy over time.")
        return
    
    st.divider()
    
    st.subheader("Accuracy Breakdown by Outcome")
    bd = stats["outcome_breakdown"]
    bd_df = pd.DataFrame({
        "Outcome": ["Win", "Draw", "Loss"],
        "Logged Matches": [bd.get("win", {}).get("total", 0), bd.get("draw", {}).get("total", 0), bd.get("loss", {}).get("total", 0)],
        "Correct Predictions": [bd.get("win", {}).get("correct", 0), bd.get("draw", {}).get("correct", 0), bd.get("loss", {}).get("correct", 0)],
    })
    bd_df["Accuracy"] = bd_df.apply(
        lambda r: f"{r['Correct Predictions']/r['Logged Matches']:.1%}" if r["Logged Matches"] > 0 else "N/A", axis=1
    )
    st.dataframe(bd_df, use_container_width=True, hide_index=True)
    
    results = get_all_real_results()
    if results:
        st.divider()
        st.subheader("Recent Match Log")
        rows = []
        for r in results:
            probs = {"win": r.get("win_prob", 0), "draw": r.get("draw_prob", 0), "loss": r.get("loss_prob", 0)}
            pred_outcome = max(probs, key=probs.get) if any(probs.values()) else "N/A"
            rows.append({
                "Fixture": f"{r.get('squad_a_name', 'A')} vs {r.get('squad_b_name', 'B')}",
                "Predicted": pred_outcome.title(),
                "Actual": r["actual_outcome"].title(),
                "Score": r["actual_scoreline"],
                "Accurate": "Yes" if pred_outcome == r["actual_outcome"] else "No",
                "Timestamp": r.get("logged_at", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="SquadPredict — eFootball Match Outcome Predictor",
        page_icon="⚽",
        layout="wide",
    )
    
    st.title("⚽ SquadPredict")
    st.caption("eFootball match outcome predictor — Computer Vision OCR + Monte Carlo Simulator + Machine Learning")
    
    _init_session_state()

    with st.sidebar:
        st.header("🤖 AI Settings")
        with st.expander("Hugging Face LLM", expanded=True):
            token_val = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if token_val and token_val.strip():
                st.success("🟢 HF Token: Loaded from `.env`")
            else:
                st.info("ℹ️ Mode: Offline Tactical Engine\n*(Add `HF_TOKEN` in `.env` to enable live LLM)*")

            default_idx = SUPPORTED_HF_MODELS.index(DEFAULT_HF_MODEL) if DEFAULT_HF_MODEL in SUPPORTED_HF_MODELS else 0
            st.selectbox(
                "Model",
                SUPPORTED_HF_MODELS,
                index=default_idx,
                key="hf_model_select",
                help="Select model for deep tactical scout reporting (or set HF_MODEL in .env).",
            )
            st.caption("Tactical scout reports are powered by the `.env` configuration.")

        st.divider()
        st.subheader("Database Stats")
        ref_count = len(st.session_state.reference_df) if "reference_df" in st.session_state else 211
        st.caption(f"📚 Offline Reference Database: **{ref_count} players**")

    tab1, tab2, tab3 = st.tabs(["Predict & Squad Analysis", "Log Real Result", "Model Performance"])
    
    with tab1:
        _tab_predict()
    with tab2:
        _tab_log_result()
    with tab3:
        _tab_performance()


if __name__ == "__main__":
    main()
