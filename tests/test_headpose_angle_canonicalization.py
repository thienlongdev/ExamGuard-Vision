"""
Unit tests for Head-Pose Angle Canonicalization
Verifies mathematical properties:
- Identity mapping for valid angles in [-180, 180)
- Periodic wrapping for values beyond +/-180 (e.g., -351.2° -> +8.8°)
- Boundary condition handling at exactly -180, 0, +180
- No silent clamping (steep angles +/-100°, +/-120° are preserved)
- Strict rejection of NaN and Inf values
"""

import math
import pytest
import numpy as np
import torch

from src.models.headpose.headpose_estimator import (
    canonicalize_yaw,
    shortest_angular_difference,
    circular_mae,
    encode_yaw_sin_cos,
    decode_yaw_sin_cos,
)
from src.models.headpose.hopenet import HopeNetYaw, ResNet18Yaw


def test_canonicalize_identity_in_range():
    """Angles within [-180, 180) must remain strictly unchanged."""
    test_angles = [-180.0, -179.9, -90.0, -45.0, -0.0, 0.0, 45.0, 90.0, 105.0, 179.9]
    for angle in test_angles:
        res = canonicalize_yaw(angle)
        assert math.isclose(res, angle, abs_tol=1e-5), f"Failed for angle {angle}: got {res}"


def test_canonicalize_no_silent_clamping():
    """Steep profiles beyond +/-90 must NOT be clamped to +/-90."""
    assert math.isclose(canonicalize_yaw(105.0), 105.0, abs_tol=1e-5)
    assert math.isclose(canonicalize_yaw(-115.0), -115.0, abs_tol=1e-5)
    assert math.isclose(canonicalize_yaw(167.9), 167.9, abs_tol=1e-5)
    assert math.isclose(canonicalize_yaw(-125.1), -125.1, abs_tol=1e-5)


def test_canonicalize_wrapped_singularity():
    """Phase wrapped values from 3DMM optimization must map to correct physical angles."""
    # -351.2269° (Idx 1474) -> +8.7731°
    res1474 = canonicalize_yaw(-351.2269287)
    assert math.isclose(res1474, 8.77307, abs_tol=1e-3)

    # +187.7869° (Idx 1902) -> -172.2131°
    res1902 = canonicalize_yaw(187.78694)
    assert math.isclose(res1902, -172.21306, abs_tol=1e-3)

    # 360.0° -> 0.0°
    assert math.isclose(canonicalize_yaw(360.0), 0.0, abs_tol=1e-5)
    # -360.0° -> 0.0°
    assert math.isclose(canonicalize_yaw(-360.0), 0.0, abs_tol=1e-5)


def test_canonicalize_numpy_and_tensor():
    """Ensures arrays and tensors are canonicalized element-wise."""
    arr = np.array([-351.2269, -90.0, 0.0, 90.0, 187.7869])
    res_arr = canonicalize_yaw(arr)
    assert isinstance(res_arr, np.ndarray)
    assert math.isclose(res_arr[0], 8.7731, abs_tol=1e-3)
    assert math.isclose(res_arr[1], -90.0, abs_tol=1e-3)
    assert math.isclose(res_arr[2], 0.0, abs_tol=1e-3)
    assert math.isclose(res_arr[3], 90.0, abs_tol=1e-3)
    assert math.isclose(res_arr[4], -172.2131, abs_tol=1e-3)

    t = torch.tensor([-351.2269, -90.0, 0.0, 90.0, 187.7869])
    res_t = canonicalize_yaw(t)
    assert isinstance(res_t, torch.Tensor)
    assert math.isclose(res_t[0].item(), 8.7731, abs_tol=1e-3)


def test_canonicalize_nan_inf_rejection():
    """NaN and Inf must raise ValueError immediately."""
    with pytest.raises(ValueError):
        canonicalize_yaw(float("nan"))

    with pytest.raises(ValueError):
        canonicalize_yaw(float("inf"))

    with pytest.raises(ValueError):
        canonicalize_yaw(float("-inf"))

    with pytest.raises(ValueError):
        canonicalize_yaw(np.array([10.0, np.nan, 20.0]))

    with pytest.raises(ValueError):
        canonicalize_yaw(torch.tensor([10.0, float("inf")]))


def test_shortest_angular_difference_exact_cases():
    """
    Verifies exact required test cases:
    - target 179, pred -179 -> error 2 deg
    - target -179, pred 179 -> error 2 deg
    - target 45, pred 50 -> error 5 deg
    - target -90, pred 90 -> error 180 deg
    - target 0, pred 360-equivalent -> 0 deg after canonicalization
    """
    # target 179, pred -179 -> 2 deg
    assert math.isclose(shortest_angular_difference(prediction_deg=-179.0, target_deg=179.0), 2.0, abs_tol=1e-5)
    # target -179, pred 179 -> 2 deg
    assert math.isclose(shortest_angular_difference(prediction_deg=179.0, target_deg=-179.0), 2.0, abs_tol=1e-5)
    # target 45, pred 50 -> 5 deg
    assert math.isclose(shortest_angular_difference(prediction_deg=50.0, target_deg=45.0), 5.0, abs_tol=1e-5)
    # target -90, pred 90 -> 180 deg
    assert math.isclose(shortest_angular_difference(prediction_deg=90.0, target_deg=-90.0), 180.0, abs_tol=1e-5)
    # target 0, pred 360-equivalent -> 0 deg
    pred_360 = canonicalize_yaw(360.0)
    assert math.isclose(shortest_angular_difference(prediction_deg=pred_360, target_deg=0.0), 0.0, abs_tol=1e-5)


def test_shortest_angular_difference_signed_and_arrays():
    """Verifies signed delta behavior and array/tensor operations."""
    # pred -179, target +179 -> delta = +2 deg
    assert math.isclose(shortest_angular_difference(-179.0, 179.0, signed=True), 2.0, abs_tol=1e-5)
    # pred +179, target -179 -> delta = -2 deg
    assert math.isclose(shortest_angular_difference(179.0, -179.0, signed=True), -2.0, abs_tol=1e-5)

    # NumPy array element-wise
    preds = np.array([-179.0, 179.0, 50.0, 90.0])
    targets = np.array([179.0, -179.0, 45.0, -90.0])
    errors = shortest_angular_difference(preds, targets)
    assert isinstance(errors, np.ndarray)
    np.testing.assert_allclose(errors, [2.0, 2.0, 5.0, 180.0], atol=1e-5)

    # Torch tensor element-wise
    t_preds = torch.tensor([-179.0, 179.0, 50.0, 90.0])
    t_targets = torch.tensor([179.0, -179.0, 45.0, -90.0])
    t_errors = shortest_angular_difference(t_preds, t_targets)
    assert isinstance(t_errors, torch.Tensor)
    torch.testing.assert_close(t_errors, torch.tensor([2.0, 2.0, 5.0, 180.0]))


def test_shortest_angular_difference_nan_inf_rejection():
    """NaN and Inf must be strictly rejected."""
    with pytest.raises(ValueError):
        shortest_angular_difference(float("nan"), 0.0)

    with pytest.raises(ValueError):
        shortest_angular_difference(0.0, float("inf"))

    with pytest.raises(ValueError):
        shortest_angular_difference(np.array([10.0, np.nan]), np.array([0.0, 0.0]))

    with pytest.raises(ValueError):
        shortest_angular_difference(torch.tensor([10.0, float("inf")]), torch.tensor([0.0, 0.0]))


def test_circular_mae():
    """Verifies circular MAE across +/-180 boundary."""
    y_true = np.array([179.0, -179.0, 45.0, 0.0])
    y_pred = np.array([-179.0, 179.0, 50.0, 0.0])
    # Individual errors: 2.0, 2.0, 5.0, 0.0 -> Mean = 9.0 / 4 = 2.25
    mae = circular_mae(y_true, y_pred)
    assert math.isclose(mae, 2.25, abs_tol=1e-5)


def test_encode_decode_yaw_sin_cos():
    """Verifies unit-circle encoding and decoding across S^1."""
    test_angles = [-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 179.9]
    for angle in test_angles:
        s, c = encode_yaw_sin_cos(angle)
        # Unit circle norm check
        norm = math.sqrt(s * s + c * c)
        assert math.isclose(norm, 1.0, abs_tol=1e-5)

        # Decode check
        decoded = decode_yaw_sin_cos(s, c)
        assert math.isclose(decoded, angle, abs_tol=1e-4)

    # +179 and -179 boundary encoding proximity
    s1, c1 = encode_yaw_sin_cos(179.0)
    s2, c2 = encode_yaw_sin_cos(-179.0)
    # Cosine of angle difference should be cos(2 deg) ~ 0.99939
    cos_diff = s1 * s2 + c1 * c2
    assert math.isclose(cos_diff, math.cos(math.radians(2.0)), abs_tol=1e-4)

    # NumPy array encode/decode
    arr = np.array([-180.0, -90.0, 0.0, 90.0, 179.9])
    vec = encode_yaw_sin_cos(arr)
    assert vec.shape == (5, 2)
    decoded_arr = decode_yaw_sin_cos(vec[:, 0], vec[:, 1])
    np.testing.assert_allclose(decoded_arr, arr, atol=1e-4)

    # Tensor encode/decode
    t = torch.tensor([-180.0, -90.0, 0.0, 90.0, 179.9])
    t_vec = encode_yaw_sin_cos(t)
    assert t_vec.shape == (5, 2)
    t_dec = decode_yaw_sin_cos(t_vec[:, 0], t_vec[:, 1])
    torch.testing.assert_close(t_dec, t, atol=1e-4, rtol=1e-4)


def test_encode_decode_nan_inf_rejection():
    """Verifies encode/decode rejects NaN and Inf."""
    with pytest.raises(ValueError):
        encode_yaw_sin_cos(float("nan"))
    with pytest.raises(ValueError):
        encode_yaw_sin_cos(float("inf"))
    with pytest.raises(ValueError):
        decode_yaw_sin_cos(float("nan"), 1.0)
    with pytest.raises(ValueError):
        decode_yaw_sin_cos(0.0, float("inf"))


def test_hopenet_exact_boundary_conditions():
    """
    Verifies HopeNet exact boundary conditions on half-open interval [-99.0, +99.0):
    -99.0      -> valid (bin 0)
    -98.999    -> valid (bin 0)
    0.0        -> valid (bin 33)
    +98.999    -> valid (bin 65)
    +99.0      -> OUT OF RANGE (raises ValueError, MUST NOT be silently mapped to 65 or clamped)
    > +99.0    -> OUT OF RANGE (raises ValueError)
    < -99.0    -> OUT OF RANGE (raises ValueError)
    """
    model = HopeNetYaw(num_bins=66, min_angle=-99.0, max_angle=99.0, pretrained=False)

    # Valid angles strictly in [-99.0, +99.0)
    valid_angles = torch.tensor([-99.0, -98.999, -96.0, 0.0, 96.0, 98.999])
    bins = model.angle_to_bin(valid_angles, strict=True)
    assert bins[0].item() == 0, f"Expected -99.0 to map to bin 0, got {bins[0].item()}"
    assert bins[1].item() == 0, f"Expected -98.999 to map to bin 0, got {bins[1].item()}"
    assert bins[2].item() == 1, f"Expected -96.0 to map to bin 1, got {bins[2].item()}"
    assert bins[3].item() == 33, f"Expected 0.0 to map to bin 33, got {bins[3].item()}"
    assert bins[4].item() == 65, f"Expected 96.0 to map to bin 65, got {bins[4].item()}"
    assert bins[5].item() == 65, f"Expected +98.999 to map to bin 65, got {bins[5].item()}"

    # Exact +99.0 MUST raise ValueError (half-open upper boundary)
    with pytest.raises(ValueError, match="HopeNet range violation"):
        model.angle_to_bin(torch.tensor([99.0]), strict=True)

    # Values > +99.0 MUST raise ValueError
    with pytest.raises(ValueError, match="HopeNet range violation"):
        model.angle_to_bin(torch.tensor([99.001]), strict=True)

    with pytest.raises(ValueError, match="HopeNet range violation"):
        model.angle_to_bin(torch.tensor([105.0]), strict=True)

    with pytest.raises(ValueError, match="HopeNet range violation"):
        model.angle_to_bin(torch.tensor([167.94]), strict=True)

    # Values < -99.0 MUST raise ValueError
    with pytest.raises(ValueError, match="HopeNet range violation"):
        model.angle_to_bin(torch.tensor([-99.001]), strict=True)

    with pytest.raises(ValueError, match="HopeNet range violation"):
        model.angle_to_bin(torch.tensor([-100.0]), strict=True)

    with pytest.raises(ValueError, match="HopeNet range violation"):
        model.angle_to_bin(torch.tensor([-125.08]), strict=True)


def test_resnet18_yaw_circular_architecture():
    """
    Verifies ResNet18Yaw circular sin/cos architecture:
    - Model outputs 2 values (sin_hat, cos_hat)
    - Forward returns (raw_vector, canonical_yaw)
    - Decoded angles are in [-180, 180)
    """
    model = ResNet18Yaw(pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)

    raw_vec, canonical_yaw = model(dummy_input)
    assert raw_vec.shape == (4, 2), f"Expected shape (4, 2), got {raw_vec.shape}"
    assert canonical_yaw.shape == (4,), f"Expected shape (4,), got {canonical_yaw.shape}"

    # All outputs must be in [-180, 180)
    assert (canonical_yaw >= -180.0).all()
    assert (canonical_yaw < 180.0).all()


def test_resnet18_sin_cos_normalization_epsilon():
    """
    Verifies numerical safety of vector normalization with epsilon:
    - Near-zero and zero predicted vectors do not cause division by zero or NaN/Inf.
    """
    epsilon = 1e-7
    # Zero vector case
    zero_vec = torch.tensor([[0.0, 0.0], [1e-12, -1e-12], [1.0, 0.0]])
    norm = torch.norm(zero_vec, p=2, dim=1, keepdim=True)
    unit_vec = zero_vec / (norm + epsilon)

    assert not torch.isnan(unit_vec).any(), "Normalized vector contains NaN"
    assert not torch.isinf(unit_vec).any(), "Normalized vector contains Inf"

    # Forward through model with simulated zero output
    model = ResNet18Yaw(pretrained=False)
    with torch.no_grad():
        # Force linear weights/bias to zero to test worst-case zero logits
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        dummy_input = torch.randn(2, 3, 224, 224)
        raw_vec, canonical_yaw = model(dummy_input)

        assert not torch.isnan(raw_vec).any()
        assert not torch.isnan(canonical_yaw).any()
        assert not torch.isinf(canonical_yaw).any()


def test_aflw_pose_array_schema():
    """
    Verifies the resolved schema of raw AFLW pose arrays:
    - AFLW2000-3D.pose.npy is 1D (2000,) of signed yaw
    - AFLW2000-3D-new.pose.npy is 1D (2000,) of unsigned magnitude (disqualified)
    - AFLW_GT_crop_yaws.npy is 1D (21080,) of signed yaw
    - AFLW2000-3D.pts68.npy is 3D (2000, 3, 68) of landmark coordinates
    """
    from pathlib import Path
    configs_dir = Path("datasets/raw_v4/head_pose_aflw2000/configs")
    if not configs_dir.exists():
        pytest.skip("Raw AFLW configs directory not found")

    pose_file = configs_dir / "AFLW2000-3D.pose.npy"
    pose_new_file = configs_dir / "AFLW2000-3D-new.pose.npy"
    gt_yaws_file = configs_dir / "AFLW_GT_crop_yaws.npy"
    pts68_file = configs_dir / "AFLW2000-3D.pts68.npy"

    assert pose_file.exists(), f"Missing {pose_file}"
    pose_arr = np.load(pose_file)
    assert pose_arr.shape == (2000,), f"Expected shape (2000,), got {pose_arr.shape}"
    assert pose_arr.dtype == np.float32, f"Expected float32, got {pose_arr.dtype}"
    # Verify signed yaw (contains negative values)
    assert (pose_arr < -30.0).any(), "AFLW2000-3D.pose.npy should contain negative yaw angles"

    assert pose_new_file.exists(), f"Missing {pose_new_file}"
    pose_new_arr = np.load(pose_new_file)
    assert pose_new_arr.shape == (2000,), f"Expected shape (2000,), got {pose_new_arr.shape}"
    # Disqualified because all values are non-negative magnitude
    assert (pose_new_arr >= 0.0).all(), "AFLW2000-3D-new.pose.npy should be non-negative magnitude"

    assert gt_yaws_file.exists(), f"Missing {gt_yaws_file}"
    gt_yaws_arr = np.load(gt_yaws_file)
    assert gt_yaws_arr.shape == (21080,), f"Expected shape (21080,), got {gt_yaws_arr.shape}"
    assert gt_yaws_arr.dtype == np.float32, f"Expected float32, got {gt_yaws_arr.dtype}"

    assert pts68_file.exists(), f"Missing {pts68_file}"
    pts68_arr = np.load(pts68_file)
    assert pts68_arr.shape == (2000, 3, 68), f"Expected shape (2000, 3, 68), got {pts68_arr.shape}"


def test_yaw_array_length_consistency():
    """
    Verifies length and integrity of head pose dataset manifest:
    - Train: 16,218 records
    - Val: 2,862 records
    - Test: 2,000 records
    - Total: 23,080 records
    - All angles canonicalize cleanly to [-180, 180) without NaN/Inf
    """
    import json
    from pathlib import Path
    manifest_path = Path("datasets/v4_head_pose/manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip("v4_head_pose manifest not found")

    train_count = 0
    val_count = 0
    test_count = 0
    test_counterpart_count = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            split = rec.get("split")
            if split == "train":
                train_count += 1
            elif split == "val":
                val_count += 1
            elif split == "test":
                test_count += 1
            elif split == "test_counterpart":
                test_counterpart_count += 1

            yaw = float(rec["yaw"])
            canon = canonicalize_yaw(yaw)
            assert -180.0 <= canon < 180.0, f"Canonical yaw out of range: {canon}"

    assert train_count == 16218, f"Expected 16218 train records, got {train_count}"
    assert val_count == 2862, f"Expected 2862 val records, got {val_count}"
    assert test_count == 2000, f"Expected 2000 test records, got {test_count}"
    assert test_counterpart_count == 2000, f"Expected 2000 test_counterpart records, got {test_counterpart_count}"
    assert train_count + val_count + test_count + test_counterpart_count == 23080
    assert train_count + val_count + test_counterpart_count == 21080  # Exact AFLW-GT array length

