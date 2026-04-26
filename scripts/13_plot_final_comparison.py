# scripts/12_plot_final_comparison.py

import json
from pathlib import Path
import matplotlib.pyplot as plt

data = json.loads(Path("logs/final_comparison.json").read_text())
summary = data["summary"]

labels = ["Naive Witness", "Trained Witness"]
scores = [summary["dumb_overall"], summary["trained_overall"]]

plt.figure(figsize=(7, 5))
plt.bar(labels, scores)
plt.ylim(0, 1)
plt.ylabel("Witness Score")
plt.title("Before vs After: Resistance to Adversarial Questioning")

for i, score in enumerate(scores):
    plt.text(i, score + 0.02, f"{score:.3f}", ha="center")

Path("logs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("logs/figures/before_after_score.png", dpi=200, bbox_inches="tight")
print("Saved logs/figures/before_after_score.png")