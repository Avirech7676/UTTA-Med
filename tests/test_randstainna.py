"""Unit tests for RandStainNA stain augmentation."""
import numpy as np
import pytest
import torch
from PIL import Image
from src.data.randstainna import RandStainNA


def test_randstainna_pil():
    aug = RandStainNA(p=1.0, delta_concentration=0.2, delta_stain=0.05)
    # Synthetic H&E patch: pink/purple background
    img = Image.new("RGB", (96, 96), color=(200, 150, 180))
    res = aug(img)
    assert isinstance(res, Image.Image)
    assert res.size == (96, 96)
    arr = np.array(res)
    assert arr.shape == (96, 96, 3)
    # Output within valid uint8 range
    assert arr.min() >= 0 and arr.max() <= 255


def test_randstainna_tensor():
    aug = RandStainNA(p=1.0, delta_concentration=0.2, delta_stain=0.05)
    tensor_img = torch.rand(3, 96, 96)
    res = aug(tensor_img)
    assert isinstance(res, torch.Tensor)
    assert res.shape == (3, 96, 96)
    assert res.min() >= 0.0 and res.max() <= 1.0


def test_randstainna_skip_probability():
    aug = RandStainNA(p=0.0)
    img = Image.new("RGB", (96, 96), color=(100, 100, 100))
    res = aug(img)
    assert np.array_equal(np.array(res), np.array(img))
