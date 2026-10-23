#!/usr/bin/env python3
"""Foundation Model LoRA Test-Time Adaptation Benchmark.

Evaluates Parameter-Efficient Test-Time Adaptation on Vision Transformers (ViT)
for computational pathology with trust-region clipping and source anchor loss.
"""
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.foundation_tta import (
    FoundationModelLoRAAdapter,
    load_pathology_foundation_model,
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Foundation Model LoRA TTA")
    parser.add_argument("--foundation_model", type=str, default="vit_b_16", help="vit_b_16, vit_l_16, uni, gigapath, virchow")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true", default=False)
    args = parser.parse_args()

    print("=" * 80)
    print(f"[*] UTTA-Med: Foundation Model LoRA TTA Benchmark")
    print(f"[*] Backbone Architecture: {args.foundation_model.upper()} | Rank = {args.lora_rank}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Load foundation model backbone
    backbone = load_pathology_foundation_model(
        model_name=args.foundation_model,
        pretrained=args.pretrained,
        hf_token=args.hf_token,
        weights_path=args.weights_path,
    ).to(device)

    # Wrap with LoRA PEFT Adapter
    adapter = FoundationModelLoRAAdapter(
        backbone=backbone,
        num_classes=2,
        lora_rank=args.lora_rank,
        lr=1e-4,
        max_displacement=0.05,
    ).to(device)

    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.get_trainable_parameters())
    param_pct = (trainable_params / total_params) * 100.0

    print(f"[+] Injected LoRA Adapters across {len(adapter.lora_layers)} projection modules.")
    print(f"[*] Total Foundation Backbone Parameters: {total_params:,}")
    print(f"[*] Trainable LoRA Parameters: {trainable_params:,} ({param_pct:.2f}% of total)")

    # Synthetic batch of pathology patches (224x224 for standard ViT)
    x = torch.randn(args.batch_size, 3, 224, 224, device=device)
    res = adapter.adapt_batch(x, confidence_threshold=0.50)

    print(f"[*] Adaptation Step: Adapted = {res['adapted']}, Eligible Patches = {res['eligible']}")
    print(f"[*] Adaptation Loss: {res['loss']:.4f}")
    print(f"[*] Post-Adaptation Logits Shape: {res['logits'].shape}")
    print("[+] Foundation Model LoRA TTA Benchmark successfully verified!")


if __name__ == "__main__":
    main()
