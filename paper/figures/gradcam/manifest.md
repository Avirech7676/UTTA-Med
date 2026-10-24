# Grad-CAM v2 Manifest (Figure 7)
Source Checkpoint: camelyon17_resnet18_source_s42_BEST.pt
Method: Grad-CAM on layer4.1.conv2
Files:
- source_TP.png: True Positive patch + heatmap
- source_TN.png: True Negative patch + heatmap
- source_FP.png: False Positive patch + heatmap
- source_FN.png: False Negative patch + heatmap
- source_highU_wrong.png: High uncertainty sample incorrectly classified
- source_lowU_correct.png: Low uncertainty sample correctly classified
