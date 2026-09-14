"""
Unit tests for Feature Extraction Engine.
"""

import pytest
import numpy as np

from smart_ew.features.schema import BandFeatureVector
from smart_ew.features.extractor import FeatureExtractor
from smart_ew.receiver.schema import ReceiverObservation


def test_initial_feature_extraction():
    extractor = FeatureExtractor()
    feat = extractor.extract_features(band_id=2, current_step=10, num_bands=5)

    assert feat.band_id == 2
    assert feat.frequency_norm == 2.0 / 4.0
    assert feat.total_scans == 0
    assert feat.hit_count == 0
    assert feat.recent_hit_rate_k5 == 0.0
    assert feat.time_since_last_scan == 10.0

    arr = feat.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10,)
    assert len(BandFeatureVector.feature_names()) == 10


def test_feature_extractor_update_and_rates():
    extractor = FeatureExtractor()

    # Add 5 observations to band 0: [Hit, Miss, Hit, Hit, Hit]
    obs_list = [
        ReceiverObservation(0, 0, True, -60.0, 20.0, 0.9, 0.0, False),
        ReceiverObservation(0, 1, False, -100.0, 0.0, 0.9, 0.0, False),
        ReceiverObservation(0, 2, True, -60.0, 20.0, 0.9, 0.0, False),
        ReceiverObservation(0, 3, True, -60.0, 20.0, 0.9, 0.0, False),
        ReceiverObservation(0, 4, True, -60.0, 20.0, 0.9, 0.0, False),
    ]

    for o in obs_list:
        extractor.update(o)

    feat = extractor.extract_features(band_id=0, current_step=5, num_bands=4)

    assert feat.total_scans == 5
    assert feat.hit_count == 4
    assert feat.miss_count == 1
    assert feat.time_since_last_scan == 1.0  # 5 - 4 = 1
    assert feat.time_since_last_hit == 1.0   # 5 - 4 = 1
    assert feat.recent_hit_rate_k5 == 4.0 / 5.0
    assert feat.overall_hit_rate == 4.0 / 5.0


def test_extract_all_features():
    extractor = FeatureExtractor()
    extractor.update(ReceiverObservation(1, 0, True, -50.0, 20.0, 0.9, 0.0, False))

    all_feats = extractor.extract_all_features(current_step=5, num_bands=3)
    assert len(all_feats) == 3
    assert 0 in all_feats and 1 in all_feats and 2 in all_feats
    assert all_feats[1].total_scans == 1
    assert all_feats[0].total_scans == 0
