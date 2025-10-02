#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Data from the screenshot at step 149,430
labels = [
    "Memory 50K",
    "Hybrid 50K",
    "Memory 10K",
    "Router 10K",
    "Memory 2K",
    "Memory 200",
]

values = [0.020475, 0.0089576, 0.0086141, 0.0061813, 0.0045896, 0.0025687]

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values)

# Color the bars to match the general pattern in the screenshot
colors = ["#1f77b4", "#ff7f0e", "#d62728", "#8c564b", "#e377c2", "#7f7f7f"]
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.ylabel("Loss (global_avg_loss)", fontsize=12)
plt.title("Final Loss at Step 149,430", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Add value labels on top of bars
for i, (label, value) in enumerate(zip(labels, values)):
    plt.text(i, value + 0.0002, f"{value:.6f}", ha="center", va="bottom", fontsize=9)

plt.savefig("loss_comparison.png", dpi=150, bbox_inches="tight")
print("Plot saved as loss_comparison.png")
