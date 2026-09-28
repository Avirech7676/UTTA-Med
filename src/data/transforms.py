"""Data transforms for Camelyon17-WILDS.
Provides standardized training augmentations and deterministic evaluation transforms.
"""
from typing import Tuple
from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_camelyon17_train_transform(image_size: int = 96) -> transforms.Compose:
    """
    Standard data augmentations for source model supervised pretraining:
    - Random horizontal and vertical flips (histopathology tissue has no canonical orientation)
    - Subtle color jitter for stain variation tolerance
    - Tensor conversion and ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_camelyon17_eval_transform(image_size: int = 96) -> transforms.Compose:
    """
    Deterministic evaluation and test-time adaptation transform:
    - Deterministic resize (96x96)
    - Tensor conversion
    - ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
