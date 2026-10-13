import os
from dotenv import load_dotenv
import pytest

load_dotenv()
from squadpredict.hf_llm import (
    DEFAULT_HF_MODEL,
    SUPPORTED_HF_MODELS,
    _build_squad_summary,
    _generate_rule_based_scout_report,
    generate_tactical_scout_report,
)


@pytest.fixture
def sample_squad():
    return [
        {"player_name": "P. Aubameyang", "position": "CF", "overall_rating": 106, "card_type": "Big Time", "live_condition": "Permanent B"},
        {"player_name": "L. Messi", "position": "SS", "overall_rating": 109, "card_type": "Big Time", "live_condition": "Permanent B"},
        {"player_name": "K. Mbappé", "position": "LWF", "overall_rating": 106, "card_type": "Show Time", "live_condition": "A-E Live Rating"},
        {"player_name": "R. Gullit", "position": "AMF", "overall_rating": 106, "card_type": "Epic (Booster)", "live_condition": "Permanent B"},
        {"player_name": "D. Rice", "position": "DMF", "overall_rating": 106, "card_type": "Big Time", "live_condition": "Permanent B"},
        {"player_name": "Marcelo", "position": "LB", "overall_rating": 104, "card_type": "Epic (Booster)", "live_condition": "Permanent B"},
        {"player_name": "A. Costacurta", "position": "RB", "overall_rating": 104, "card_type": "Epic (Booster)", "live_condition": "Permanent B"},
        {"player_name": "F. Cannavaro", "position": "CB", "overall_rating": 106, "card_type": "Epic (Booster)", "live_condition": "Permanent B"},
        {"player_name": "J. Stam", "position": "CB", "overall_rating": 106, "card_type": "Epic (Booster)", "live_condition": "Permanent B"},
        {"player_name": "M. Neuer", "position": "GK", "overall_rating": 107, "card_type": "Big Time", "live_condition": "Permanent B"},
    ]


@pytest.fixture
def sample_prediction():
    return {
        "win_prob": 0.55,
        "draw_prob": 0.20,
        "loss_prob": 0.25,
        "predicted_score": "2 - 1",
    }


def test_supported_models_list():
    assert len(SUPPORTED_HF_MODELS) >= 3
    assert DEFAULT_HF_MODEL in SUPPORTED_HF_MODELS


def test_build_squad_summary(sample_squad):
    summary = _build_squad_summary("Squad Alpha", "4-1-2-3", "Quick Counter", 3295, sample_squad)
    assert "Squad Alpha" in summary
    assert "4-1-2-3" in summary
    assert "Aubameyang" in summary
    assert "Big Time" in summary


def test_offline_scout_report_generation(sample_squad, sample_prediction):
    report = _generate_rule_based_scout_report(
        "Squad A", "4-1-2-3", sample_squad,
        "Squad B", "4-2-1-3", sample_squad,
        sample_prediction,
    )
    assert "AI Tactical Intelligence & Scout Report" in report
    assert "Tactical & Formation Clash" in report
    assert "Critical 1v1 Lane Battles" in report
    assert "Projected Scoreline" in report
    assert "55.0%" in report


def test_generate_tactical_scout_report_fallback_without_token(monkeypatch, sample_squad, sample_prediction):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    report = generate_tactical_scout_report(
        "Squad A", "4-1-2-3", "Quick Counter", 3295, sample_squad,
        "Squad B", "4-2-1-3", "Quick Counter", 3221, sample_squad,
        sample_prediction,
        hf_token="",
    )
    assert "AI Tactical Intelligence & Scout Report" in report
    assert "Recommended Managerial Counter-Tactics" in report


def test_generate_tactical_scout_report_live_token(sample_squad, sample_prediction):
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("No HF_TOKEN set in environment")
    report = generate_tactical_scout_report(
        "Squad A", "4-1-2-3", "Quick Counter", 3295, sample_squad,
        "Squad B", "4-2-1-3", "Quick Counter", 3221, sample_squad,
        sample_prediction,
        model_id="google/gemma-3-12b-it",
    )
    assert "AI Tactical Intelligence & Scout Report" in report
    assert "Powered by Hugging Face" in report
