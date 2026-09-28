"""Official models: 1-logit BCE, dropout 0.5. Import ResNet18UTTAMed / ResNet50UTTAMed."""

from src.models.resnet18 import (
    OFFICIAL_DROPOUT_P,
    OFFICIAL_NUM_CLASSES,
    ResNet18Classifier,
    ResNet18UTTAMed,
    build_resnet,
    build_resnet18,
)
from src.models.resnet50 import (
    ResNet50Classifier,
    ResNet50UTTAMed,
    build_resnet50,
)

__all__ = [
    "OFFICIAL_DROPOUT_P",
    "OFFICIAL_NUM_CLASSES",
    "ResNet18Classifier",
    "ResNet18UTTAMed",
    "build_resnet18",
    "build_resnet",
    "ResNet50Classifier",
    "ResNet50UTTAMed",
    "build_resnet50",
]
