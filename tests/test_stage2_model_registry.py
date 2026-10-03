"""
Tests for Stage 2 Model Registry & Singleton Lifecycle
"""

import pytest
from pathlib import Path
from src.orchestration.model_registry import ModelRegistry, compute_sha256


def test_model_registry_singleton_initialization():
    registry1 = ModelRegistry.get_instance()
    registry2 = ModelRegistry.get_instance()
    assert registry1 is registry2

    registry1.initialize_models()
    assert registry1._initialized is True
    assert registry1.detector is not None
    assert registry1.posture_predictor is not None
    assert registry1.headpose_predictor is not None


def test_model_registry_metadata():
    registry = ModelRegistry.get_instance()
    registry.initialize_models()
    meta = registry.get_metadata()

    assert "detector" in meta
    assert "posture" in meta
    assert "headpose" in meta
    assert "macro_behavior_detector" in meta

    # Verify SHA-256 digests exist and match length
    for mod_name, info in meta.items():
        assert len(info["sha256"]) == 64
        assert info["device"] is not None
        assert "precision_mode" in info


def test_compute_sha256_missing_file():
    digest = compute_sha256("non_existent_file_path.pt")
    assert digest == "FILE_NOT_FOUND"


def test_model_registry_checkpoint_identity_and_taxonomy():
    """
    Regression test asserting:
    1. loaded posture registry model_name == checkpoint['model_name']
    2. loaded headpose registry model_name == checkpoint['model_name']
    3. canonical posture class names match ['NORMAL_UPRIGHT', 'NORMAL_READ_WRITE', 'HEAD_REST_SLEEP', 'TURN_HEAD_CLEAR']
    4. headpose native support is [-99.0, 99.0]
    """
    import torch
    registry = ModelRegistry.get_instance()
    registry.initialize_models()
    meta = registry.get_metadata()

    # 1. Posture checkpoint verification
    pos_path = meta["posture"]["checkpoint_path"]
    pos_raw = torch.load(pos_path, map_location="cpu", weights_only=False)
    assert meta["posture"]["model_name"] == pos_raw["model_name"]
    assert meta["posture"]["model_name"] == "mobilenet_v3_small"
    assert meta["posture"]["actual_loaded_class"] == "MobileNetV3"
    assert meta["posture"]["num_classes"] == 4

    canonical_classes = [
        "NORMAL_UPRIGHT",
        "NORMAL_READ_WRITE",
        "HEAD_REST_SLEEP",
        "TURN_HEAD_CLEAR",
    ]
    assert meta["posture"]["class_names"] == canonical_classes
    assert pos_raw["class_names"] == canonical_classes

    # 2. Headpose checkpoint verification
    hp_path = meta["headpose"]["checkpoint_path"]
    hp_raw = torch.load(hp_path, map_location="cpu", weights_only=False)
    assert meta["headpose"]["model_name"] == hp_raw["model_name"]
    assert meta["headpose"]["model_name"] == "hopenet_yaw"
    assert meta["headpose"]["actual_loaded_class"] == "HopeNetYaw"
    assert meta["headpose"]["native_support"] == [-99.0, 99.0]
    assert meta["headpose"]["task"] == "YAW_REGRESSION"

