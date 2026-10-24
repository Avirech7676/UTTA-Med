"""Official models: 1-logit BCE, dropout 0.5. Official ResNet-18 architecture."""

from src.models.resnet18 import (
    OFFICIAL_DROPOUT_P,
    OFFICIAL_NUM_CLASSES,
    ResNet18Classifier,
    ResNet18UTTAMed,
    build_resnet,
    build_resnet18,
)

__all__ = [
    "OFFICIAL_DROPOUT_P",
    "OFFICIAL_NUM_CLASSES",
    "ResNet18Classifier",
    "ResNet18UTTAMed",
    "build_resnet18",
    "build_resnet",
]

