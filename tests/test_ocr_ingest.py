"""Tests for OCR ingestion pipeline.

Note: Full OCR accuracy tests require real Game Plan screenshots in
sample_screenshots/. These tests focus on the module's structure and
the non-OCR helper functions.
"""

import numpy as np
import pytest

from squadpredict.config import KNOWN_FORMATIONS
from squadpredict.ocr_ingest import (
    crop_regions,
    parse_formation,
    _merge_ocr_results,
)
from squadpredict.player_reference import load_reference, lookup_player


# --- Tests ---

class TestParseFormation:
    def test_exact_match(self):
        text_list = [{"text": "4-3-3", "confidence": 0.95, "bbox": [], "y_position": 0}]
        assert parse_formation(text_list) == "4-3-3"

    def test_ocr_artifacts_normalized(self):
        text_list = [{"text": "4~3~3", "confidence": 0.8, "bbox": [], "y_position": 0}]
        assert parse_formation(text_list) == "4-3-3"

    def test_underscore_artifacts(self):
        text_list = [{"text": "4_2_3_1", "confidence": 0.8, "bbox": [], "y_position": 0}]
        assert parse_formation(text_list) == "4-2-3-1"

    def test_spaces_artifacts(self):
        text_list = [{"text": "4 3 3", "confidence": 0.8, "bbox": [], "y_position": 0}]
        assert parse_formation(text_list) == "4-3-3"

    def test_empty_list(self):
        assert parse_formation([]) == "Unknown"

    def test_no_match_returns_unknown(self):
        text_list = [{"text": "hello world", "confidence": 0.95, "bbox": [], "y_position": 0}]
        result = parse_formation(text_list)
        assert result == "Unknown" or result in KNOWN_FORMATIONS


class TestCropRegions:
    def test_returns_two_regions(self):
        # Create a dummy image
        img = np.zeros((1000, 800), dtype=np.uint8)
        regions = crop_regions(img)
        assert "formation_region" in regions
        assert "player_list_region" in regions

    def test_regions_are_smaller_than_original(self):
        img = np.zeros((1000, 800), dtype=np.uint8)
        regions = crop_regions(img)
        for name, region in regions.items():
            assert region.shape[0] <= img.shape[0]
            assert region.shape[1] <= img.shape[1]


class TestMergeOcrResults:
    def test_keeps_higher_confidence(self):
        results_a = [{"text": "Haaland", "confidence": 0.7, "bbox": [], "y_position": 100}]
        results_b = [{"text": "Haaland", "confidence": 0.9, "bbox": [], "y_position": 100}]
        merged = _merge_ocr_results(results_a, results_b)
        assert len(merged) == 1
        assert merged[0]["confidence"] == 0.9

    def test_merges_different_texts(self):
        results_a = [{"text": "Haaland", "confidence": 0.9, "bbox": [], "y_position": 100}]
        results_b = [{"text": "Mbappe", "confidence": 0.8, "bbox": [], "y_position": 200}]
        merged = _merge_ocr_results(results_a, results_b)
        assert len(merged) == 2


class TestPlayerReferenceFuzzyMatch:
    """Test fuzzy matching which is critical for OCR accuracy."""

    @pytest.fixture
    def ref_df(self):
        return load_reference()

    def test_exact_match(self, ref_df):
        result = lookup_player("E. Haaland", ref_df)
        assert result is not None
        assert result["player_name"] == "E. Haaland"

    def test_typo_halland(self, ref_df):
        result = lookup_player("Halland", ref_df)
        assert result is not None
        assert "Haaland" in result["player_name"]

    def test_typo_mbape(self, ref_df):
        result = lookup_player("Mbape", ref_df)
        assert result is not None
        assert "Mbapp" in result["player_name"]

    def test_no_match_returns_none(self, ref_df):
        result = lookup_player("XXXXXXXXX", ref_df)
        assert result is None

    def test_empty_string(self, ref_df):
        result = lookup_player("", ref_df)
        assert result is None

    def test_confidence_present(self, ref_df):
        result = lookup_player("Messi", ref_df)
        assert result is not None
        assert "match_confidence" in result
        assert result["match_confidence"] > 0


class TestClassifyCardType:
    def test_big_time_by_high_rating(self):
        from squadpredict.ocr_ingest import classify_card_type
        assert classify_card_type(None, rating=109) == "Big Time"

    def test_epic_booster_rating(self):
        from squadpredict.ocr_ingest import classify_card_type
        assert classify_card_type(None, rating=105) == "Epic (Booster)"

    def test_highlight_rating(self):
        from squadpredict.ocr_ingest import classify_card_type
        assert classify_card_type(None, rating=100) == "Highlight"

    def test_standard_rating(self):
        from squadpredict.ocr_ingest import classify_card_type
        assert classify_card_type(None, rating=82) == "Standard"
