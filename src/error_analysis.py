
import numpy as np
import matplotlib.pyplot as plt

# Day 10 final test confusion matrix
cm = np.array([
    [11, 1, 2, 0, 3, 0, 0, 0],
    [1, 12, 1, 0, 0, 0, 0, 0],
    [1, 0, 8, 0, 3, 0, 4, 1],
    [0, 0, 0, 10, 0, 0, 0, 3],
    [0, 0, 3, 0, 9, 2, 0, 0],
    [3, 1, 0, 1, 1, 11, 0, 0],
    [1, 0, 6, 1, 0, 1, 4, 0],
    [0, 0, 0, 4, 0, 0, 0, 11]
])

directions = np.arange(0, 360, 45)

# Accuracy for each stimulus direction
correct = np.diag(cm)
total = cm.sum(axis=1)
accuracy = correct / total * 100

print("Accuracy by stimulus direction:")
for direction, acc in zip(directions, accuracy):
    print(f"{direction}°: {acc:.2f}%")

print(f"\nOverall accuracy: {correct.sum() / total.sum() * 100:.2f}%")

# Plot accuracy by direction
plt.figure(figsize=(9, 5))
plt.bar(directions.astype(str), accuracy)
plt.xlabel("Stimulus Direction (degrees)")
plt.ylabel("Classification Accuracy (%)")
plt.title("Neural Decoder Accuracy by Direction")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(
    "results/figures/day11_direction_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
