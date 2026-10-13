"""End-to-end integration test for the full SquadPredict pipeline."""

import json
from pathlib import Path

from squadpredict.db import (
    get_all_real_results,
    get_performance_stats,
    init_db,
    log_real_result,
    save_prediction,
    save_squad,
)
from squadpredict.features import build_team_features
from squadpredict.model import load_model, predict
from squadpredict.player_reference import load_reference, lookup_player
from squadpredict.retrain import blend_and_retrain


def test_full_pipeline_end_to_end(tmp_path: Path):
    """Test full flow: lookup -> features -> predict -> db -> retrain -> stats."""
    db_path = tmp_path / "test_pipeline.db"
    model_output_path = tmp_path / "test_retrained_model.pkl"

    # 1. Initialize DB
    init_db(db_path=db_path)

    # 2. Player lookup
    ref_df = load_reference()
    names_a = [
        "Courtois", "Van Dijk", "R. Dias", "A. Davies", "Hakimi",
        "De Bruyne", "Bellingham", "Pedri", "Vinicius Jr.", "Salah", "Haaland",
    ]
    names_b = [
        "Donnarumma", "Saliba", "Rudiger", "Robertson", "Cancelo",
        "Modric", "Kroos", "Casemiro", "Neymar Jr.", "Messi", "Kane",
    ]

    players_a = [lookup_player(n, ref_df) for n in names_a]
    players_b = [lookup_player(n, ref_df) for n in names_b]
    players_a = [p for p in players_a if p is not None]
    players_b = [p for p in players_b if p is not None]

    assert len(players_a) == 11
    assert len(players_b) == 11

    # 3. Build features & predict
    fa = build_team_features(players_a, "4-3-3")
    fb = build_team_features(players_b, "4-3-3")

    model = load_model()
    pred = predict(fa, fb, model)

    assert 0.0 <= pred["win_prob"] <= 1.0
    assert 0.0 <= pred["draw_prob"] <= 1.0
    assert 0.0 <= pred["loss_prob"] <= 1.0
    assert pred["predicted_outcome"] in ["win", "draw", "loss"]

    # 4. Save to DB
    squad_a_id = save_squad(
        "Team Alpha", "4-3-3", json.dumps(players_a, default=str), "manual", db_path=db_path
    )
    squad_b_id = save_squad(
        "Team Beta", "4-3-3", json.dumps(players_b, default=str), "manual", db_path=db_path
    )
    pred_id = save_prediction(
        squad_a_id,
        squad_b_id,
        pred["win_prob"],
        pred["draw_prob"],
        pred["loss_prob"],
        pred["predicted_scoreline"],
        "v1_initial",
        db_path=db_path,
    )

    assert squad_a_id > 0
    assert squad_b_id > 0
    assert pred_id > 0

    # 5. Log real match result
    result_id = log_real_result(
        pred_id, squad_a_id, squad_b_id, "2-1", "win", db_path=db_path
    )
    assert result_id > 0

    # 6. Verify stats
    stats = get_performance_stats(db_path=db_path)
    assert stats["total_matches"] == 1
    assert stats["accuracy"] in [0.0, 1.0]

    # 7. Retrain with custom paths
    new_version = blend_and_retrain(
        db_path=db_path,
        model_output_path=model_output_path,
    )
    assert new_version.startswith("v")
    assert model_output_path.exists()
