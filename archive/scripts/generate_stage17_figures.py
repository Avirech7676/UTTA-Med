import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs("results/figures", exist_ok=True)
os.makedirs("paper/figures", exist_ok=True)

# -------------------------------------------------------------------------
# FIGURE 1: UTTA-Med Methodology Flowchart
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.axis("off")

def draw_box(ax, xy, width, height, text, facecolor="#e8f4f8", edgecolor="#2b5c8f", lw=1.5, fontsize=9.5, fontweight="bold"):
    box = patches.FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.03,rounding_size=0.05",
                                facecolor=facecolor, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(box)
    ax.text(xy[0] + width/2, xy[1] + height/2, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, color="#1a202c")
    return box

def draw_arrow(ax, start, end, text="", color="#2b5c8f", lw=1.5):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15))
    if text:
        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2 + 0.03)
        ax.text(mid[0], mid[1], text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

# Draw Pipeline Nodes
draw_box(ax, (0.02, 0.72), 0.20, 0.18, "Camelyon17-WILDS\nSource Centers {0,3,4}\n(302k patches, 30 WSIs)", facecolor="#edf2f7", edgecolor="#4a5568")
draw_arrow(ax, (0.22, 0.81), (0.28, 0.81), text="")

draw_box(ax, (0.28, 0.72), 0.18, 0.18, "ResNet-18 Backbone\n(1-logit, Dropout p=0.5)\nFreezes conv & head", facecolor="#e2e8f0", edgecolor="#2b6cb0")
draw_arrow(ax, (0.46, 0.81), (0.52, 0.81), text="")

draw_box(ax, (0.52, 0.72), 0.18, 0.18, "Unlabeled Target\nStream (Hospital 2)\n85,054 patches (10 WSIs)", facecolor="#feebc8", edgecolor="#c05621")
draw_arrow(ax, (0.70, 0.81), (0.76, 0.81), text="")

draw_box(ax, (0.76, 0.72), 0.22, 0.18, "MC-Dropout (N=20 passes)\nPredictive Variance\nU(x) = Var_MC[p(y=1|x)]", facecolor="#e9d8fd", edgecolor="#6b46c1")
draw_arrow(ax, (0.87, 0.72), (0.87, 0.58), text="")

# Decision Diamond
diamond = patches.Polygon([[0.87, 0.58], [0.97, 0.46], [0.87, 0.34], [0.77, 0.46]],
                          closed=True, facecolor="#feefc3", edgecolor="#b7791f", lw=1.8)
ax.add_patch(diamond)
ax.text(0.87, 0.46, "U(x) < tau ?\n(tau = 70th %ile\nCenter 1 Val)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#744210")

# Decision branches
# YES Branch -> TTA
draw_arrow(ax, (0.77, 0.46), (0.58, 0.46), text="YES (Low-U)", color="#276749", lw=2)
draw_box(ax, (0.34, 0.37), 0.24, 0.18, "Test-Time Adaptation (TTA)\nEntropy Loss on BN gamma, beta\n(Running Stats Frozen: bn_freeze_stats)\nlr = 1e-5, k = 1 step", facecolor="#c6f6d5", edgecolor="#22543d")

# NO Branch -> Skip
draw_arrow(ax, (0.87, 0.34), (0.87, 0.18), text="NO (High-U)", color="#c53030", lw=2)
draw_box(ax, (0.76, 0.04), 0.22, 0.14, "Skip Adaptation\n(Preserve Source BN weights\nPrevent Logit Drift / Collapse)", facecolor="#fed7d7", edgecolor="#9b2c2c")

# Final Prediction
draw_arrow(ax, (0.34, 0.46), (0.18, 0.46), text="")
draw_arrow(ax, (0.76, 0.11), (0.18, 0.11), text="")
draw_box(ax, (0.02, 0.16), 0.16, 0.35, "Final Tumor\nPrediction\np(y=1|x)\n\nF1: +0.0231 (5/5)\nHarm: 0.7% vs 18.7%", facecolor="#ebf8ff", edgecolor="#2b6cb0")

plt.title("Figure 1: UTTA-Med Architecture & Selective Test-Time Adaptation Pipeline", fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig("results/figures/fig1_methodology_flowchart.png", dpi=300)
fig.savefig("paper/figures/fig1_methodology_flowchart.png", dpi=300)
plt.close()
print("Generated Figure 1: Methodology Flowchart")

# -------------------------------------------------------------------------
# FIGURE 2: Distribution Shift (Source vs OOD Val vs Target Test)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
domains = ["In-Domain Val\n(Centers 0,3,4)", "OOD Validation\n(Center 1)", "Target Test\n(Center 2)"]
auroc_vals = [0.999, 0.978, 0.930]
auroc_errs = [0.001, 0.004, 0.009]
f1_vals = [0.972, 0.920, 0.807]
f1_errs = [0.005, 0.012, 0.025]
ece_vals = [0.018, 0.050, 0.128]
ece_errs = [0.003, 0.006, 0.026]

x = np.arange(len(domains))
w = 0.25

ax.bar(x - w, auroc_vals, w, yerr=auroc_errs, capsize=4, label="Target AUROC", color="#2b5c8f", alpha=0.9)
ax.bar(x, f1_vals, w, yerr=f1_errs, capsize=4, label="F1 Score", color="#319795", alpha=0.9)
ax.bar(x + w, ece_vals, w, yerr=ece_errs, capsize=4, label="Calibration Error (ECE)", color="#dd6b20", alpha=0.9)

ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
ax.set_title("Figure 2: Clinical Distribution Shift Severity Across Camelyon17 Centers", fontsize=12, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=10, fontweight="bold")
ax.legend(frameon=True, fontsize=9.5)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, 1.1)

# Annotate shift drop
ax.annotate("AUROC Drop: -0.070\nF1 Drop: -0.165", xy=(2 - w, 0.93), xytext=(1.4, 0.75),
            arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.2), fontsize=8.5, fontweight="bold", color="#c53030")

plt.tight_layout()
fig.savefig("results/figures/fig2_distribution_shift.png", dpi=300)
fig.savefig("paper/figures/fig2_distribution_shift.png", dpi=300)
plt.close()
print("Generated Figure 2: Distribution Shift Severity")

# -------------------------------------------------------------------------
# FIGURE 3: Main Results Across 5 Seeds (Mean +- SD)
# -------------------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7.5), dpi=300)

methods = ["Source", "Tent", "EATA", "Random", "Confidence", "UTTA-Med"]
colors = ["#718096", "#e53e3e", "#805ad5", "#dd6b20", "#3182ce", "#2b6cb0"]

# 1. AUROC
auroc_m = [0.930, 0.698, 0.921, 0.697, 0.936, 0.935]
auroc_s = [0.009, 0.324, 0.026, 0.304, 0.022, 0.022]
ax1.bar(methods, auroc_m, yerr=auroc_s, capsize=4, color=colors, alpha=0.85)
ax1.set_title("Target AUROC (Higher is Better)", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 1.05)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.tick_params(axis="x", rotation=25)

# 2. F1 Score
f1_m = [0.807, 0.506, 0.809, 0.487, 0.831, 0.830]
f1_s = [0.025, 0.411, 0.031, 0.385, 0.032, 0.032]
ax2.bar(methods, f1_m, yerr=f1_s, capsize=4, color=colors, alpha=0.85)
ax2.set_title("Target F1 Score (Higher is Better)", fontsize=11, fontweight="bold")
ax2.set_ylim(0, 1.05)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.tick_params(axis="x", rotation=25)

# 3. ECE
ece_m = [0.128, 0.284, 0.152, 0.298, 0.130, 0.131]
ece_s = [0.026, 0.179, 0.024, 0.166, 0.030, 0.031]
ax3.bar(methods, ece_m, yerr=ece_s, capsize=4, color=colors, alpha=0.85)
ax3.set_title("Expected Calibration Error (Lower is Better)", fontsize=11, fontweight="bold")
ax3.set_ylim(0, 0.55)
ax3.grid(axis="y", linestyle="--", alpha=0.4)
ax3.tick_params(axis="x", rotation=25)

# 4. Harm Rate
harm_m = [0.0, 18.7, 2.5, 19.4, 0.7, 0.7]
harm_s = [0.0, 17.9, 1.5, 17.0, 0.3, 0.3]
ax4.bar(methods, harm_m, yerr=harm_s, capsize=4, color=colors, alpha=0.85)
ax4.set_title("Harmful Adaptation Rate (%) (Lower is Better)", fontsize=11, fontweight="bold")
ax4.set_ylim(0, 42)
ax4.grid(axis="y", linestyle="--", alpha=0.4)
ax4.tick_params(axis="x", rotation=25)

plt.suptitle("Figure 3: Five-Seed Benchmark Comparison (Camelyon17 Hospital 2)", fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
fig.savefig("results/figures/fig3_main_results_grid.png", dpi=300)
fig.savefig("paper/figures/fig3_main_results_grid.png", dpi=300)
plt.close()
print("Generated Figure 3: Main Results Grid")

# -------------------------------------------------------------------------
# FIGURE 4: Uncertainty/Error Monotonic Relationship (Decile Analysis)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
deciles = np.arange(1, 11)
error_rates = [0.05, 0.20, 0.40, 0.75, 1.60, 2.10, 4.00, 10.2, 23.5, 40.5]

ax.plot(deciles, error_rates, marker="o", lw=2.5, color="#805ad5", markersize=7, label="Empirical Error Rate P(error | bin)")
ax.bar(deciles, error_rates, alpha=0.25, color="#805ad5", width=0.6)

ax.set_xlabel("Predictive Uncertainty Decile (Low U -> High U)", fontsize=11, fontweight="bold")
ax.set_ylabel("Empirical Classification Error Rate (%)", fontsize=11, fontweight="bold")
ax.set_title("Figure 4: Monotonic Relationship Between Predictive Uncertainty and Error", fontsize=12, fontweight="bold", pad=12)
ax.set_xticks(deciles)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_ylim(0, 45)

ax.annotate("Tau Cutoff (70th %ile)\nExcludes high-error tail (>10% error)", xy=(7, 4.0), xytext=(4, 18),
            arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.5), fontsize=9, fontweight="bold", color="#c53030")

plt.tight_layout()
fig.savefig("results/figures/fig4_uncertainty_error_deciles.png", dpi=300)
fig.savefig("paper/figures/fig4_uncertainty_error_deciles.png", dpi=300)
plt.close()
print("Generated Figure 4: Uncertainty Error Deciles")

# -------------------------------------------------------------------------
# FIGURE 5: Coverage vs Performance / Harm Rate Curves
# -------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=300)

covs = [20, 30, 40, 50, 57.5, 70, 85, 100]
auroc_cov = [0.941, 0.946, 0.950, 0.953, 0.935, 0.928, 0.892, 0.698]
harm_utta_cov = [0.13, 0.22, 0.34, 0.52, 0.73, 1.48, 4.10, 18.75]
harm_rand_cov = [1.82, 2.95, 4.10, 5.84, 6.21, 8.45, 14.20, 18.75]

color = "#2b6cb0"
ax1.set_xlabel("Target Adaptation Coverage (%)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Target AUROC", color=color, fontsize=11, fontweight="bold")
l1 = ax1.plot(covs, auroc_cov, color=color, marker="s", lw=2.2, label="UTTA AUROC")
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim(0.65, 0.98)

ax2 = ax1.twinx()
color = "#c53030"
ax2.set_ylabel("Harmful Adaptation Rate (%)", color=color, fontsize=11, fontweight="bold")
l2 = ax2.plot(covs, harm_utta_cov, color=color, marker="o", lw=2.2, label="UTTA Harm Rate")
l3 = ax2.plot(covs, harm_rand_cov, color="#dd6b20", linestyle="--", marker="^", lw=1.8, label="Random Gate Harm Rate")
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(0, 22)

# Combined legend
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center left", frameon=True, fontsize=9.5)
plt.title("Figure 5: Selective Adaptation Trade-Off (Coverage vs. Performance & Safety)", fontsize=12, fontweight="bold", pad=12)
ax1.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
fig.savefig("results/figures/fig5_coverage_vs_performance.png", dpi=300)
fig.savefig("paper/figures/fig5_coverage_vs_performance.png", dpi=300)
plt.close()
print("Generated Figure 5: Coverage vs Performance Trade-off")
