"""
tests/test_cbam_and_posture_models.py
Unit tests verifying CBAM attention shape safety, gradient flow,
posture model serialization, checkpoint reload, 4-class output dimension,
and fusion contract conformance.
"""

import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import pytest

from src.models.posture.cbam import CBAM, ChannelAttention, SpatialAttention
from src.models.posture.resnet_cbam import resnet18_cbam, resnet50_cbam
from src.models.posture.mobilenet_posture import mobilenet_posture
from src.models.posture.posture_classifier import (
    POSTURE_CLASSES,
    create_posture_model,
    save_posture_checkpoint,
    load_posture_checkpoint,
    PosturePredictor,
    PostureInferenceOutput,
)


def test_cbam_modules_input_output_shapes():
    """Verify ChannelAttention, SpatialAttention, and CBAM preserve spatial and channel dims."""
    batch_size = 4
    channels = 64
    height, width = 28, 28
    x = torch.randn(batch_size, channels, height, width)

    # Channel Attention
    ca = ChannelAttention(channels, ratio=16)
    ca_out = ca(x)
    assert ca_out.shape == (batch_size, channels, 1, 1), f"Unexpected CA shape: {ca_out.shape}"
    assert (ca_out >= 0.0).all() and (ca_out <= 1.0).all(), "CA outputs must be Sigmoid bounded in [0, 1]"

    # Spatial Attention
    sa = SpatialAttention(kernel_size=7)
    sa_out = sa(x)
    assert sa_out.shape == (batch_size, 1, height, width), f"Unexpected SA shape: {sa_out.shape}"
    assert (sa_out >= 0.0).all() and (sa_out <= 1.0).all(), "SA outputs must be Sigmoid bounded in [0, 1]"

    # Full CBAM block
    cbam = CBAM(channels, ratio=16, kernel_size=7)
    out = cbam(x)
    assert out.shape == x.shape, f"CBAM output shape {out.shape} does not match input shape {x.shape}"


def test_cbam_gradient_flow():
    """Verify gradients propagate properly through both channel and spatial branches of CBAM."""
    channels = 32
    cbam = CBAM(channels, ratio=8)
    x = torch.randn(2, channels, 14, 14, requires_grad=True)

    out = cbam(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient must not be None"
    assert not torch.isnan(x.grad).any(), "NaN found in input gradient"
    assert (x.grad.abs().sum() > 0).item(), "Gradients failed to flow through CBAM"

    # Verify CBAM parameters have gradients
    for name, param in cbam.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"NaN in parameter {name} gradient"


def test_resnet18_cbam_forward_and_backward():
    """Verify ResNet-18 CBAM produces exact 4-class logits and supports backprop."""
    model = resnet18_cbam(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits = model(x)

    assert logits.shape == (2, 4), f"Expected shape (2, 4), got {logits.shape}"
    loss = logits.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_resnet50_cbam_forward_shape():
    """Verify ResNet-50 CBAM produces exact 4-class logits at 224x224 and 320x320."""
    model = resnet50_cbam(num_classes=4, pretrained=False)
    for res in [224, 320]:
        x = torch.randn(1, 3, res, res)
        logits = model(x)
        assert logits.shape == (1, 4), f"Expected shape (1, 4) at resolution {res}, got {logits.shape}"


def test_resnet50_cbam_backward_finite_gradients():
    """Verify ResNet-50 CBAM produces finite gradients under backward pass without overflow."""
    model = resnet50_cbam(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient must not be None"
    assert not torch.isnan(x.grad).any(), "NaN found in input gradient"
    assert not torch.isinf(x.grad).any(), "Inf found in input gradient"
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN in {name} gradient"
            assert not torch.isinf(p.grad).any(), f"Inf in {name} gradient"


def test_mobilenet_posture_forward_shape():
    """Verify MobileNet lightweight baseline produces exact 4-class logits."""
    model = mobilenet_posture(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 4), f"Expected shape (2, 4), got {logits.shape}"


def test_posture_model_serialization_and_reload():
    """Verify model saving, checkpoint file generation, reload, and numerical consistency."""
    model = resnet18_cbam(num_classes=4, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        original_output = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_posture.pt"
        metrics = {"val_acc": 0.95, "macro_f1": 0.92}
        save_posture_checkpoint(
            model=model,
            filepath=ckpt_path,
            model_name="resnet18_cbam",
            epoch=5,
            metrics=metrics,
            extra_metadata={"resolution": 224, "representation": "TIGHT_PERSON_CROP"},
        )

        assert ckpt_path.exists(), "Checkpoint file was not created"

        reloaded_model, ckpt = load_posture_checkpoint(ckpt_path, device="cpu")
        reloaded_model.eval()

        with torch.no_grad():
            reloaded_output = reloaded_model(x)

        assert torch.allclose(original_output, reloaded_output, atol=1e-5), "Reloaded model outputs differ!"
        assert ckpt["num_classes"] == 4
        assert ckpt["class_names"] == POSTURE_CLASSES
        assert ckpt["epoch"] == 5
        assert ckpt["metrics"]["macro_f1"] == 0.92


def test_posture_predictor_fusion_contract_compliance():
    """Verify PosturePredictor produces outputs adhering strictly to configs/v4_fusion_contract.yaml."""
    model = resnet18_cbam(num_classes=4, pretrained=False)
    predictor = PosturePredictor(model=model, device="cpu")

    x = torch.randn(2, 3, 224, 224)
    student_ids = [101, 102]
    timestamps = [12.5, 12.6]
    qualities = ["GOOD", "BLURRY"]
    crop_types = ["TIGHT_PERSON_CROP", "CONTEXT_PERSON_CROP"]

    preds = predictor.predict_tensor(
        x,
        student_ids=student_ids,
        timestamps=timestamps,
        crop_qualities=qualities,
        crop_types=crop_types,
    )

    assert len(preds) == 2
    for idx, p in enumerate(preds):
        assert isinstance(p, PostureInferenceOutput)
        d = p.to_dict()
        assert d["student_id"] == student_ids[idx]
        assert d["timestamp"] == timestamps[idx]
        assert d["crop_quality"] == qualities[idx]
        assert d["crop_type"] == crop_types[idx]
        assert d["predicted_class"] in POSTURE_CLASSES
        assert 0.0 <= d["normal_upright_score"] <= 1.0
        assert 0.0 <= d["normal_read_write_score"] <= 1.0
        assert 0.0 <= d["head_rest_sleep_score"] <= 1.0
        assert 0.0 <= d["turn_head_clear_score"] <= 1.0
        # Probabilities sum to 1.0
        total_p = (
            d["normal_upright_score"]
            + d["normal_read_write_score"]
            + d["head_rest_sleep_score"]
            + d["turn_head_clear_score"]
        )
        assert abs(total_p - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {total_p}"
