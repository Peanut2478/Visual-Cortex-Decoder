import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import prepare_data, normalize_data

dataset, timestamps, dff, stim_table = load_data()
X_train_full, X_test, y_train_full, y_test = prepare_data(dff, stim_table)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)
X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

directions = np.arange(0, 360, 45)
labels = np.asarray(y_train)

mean_responses = np.array([
    np.mean(X_train[labels == direction], axis=0)
    for direction in directions
])

# Shift each neuron's tuning curve so its minimum becomes 0
shifted = mean_responses - np.min(mean_responses, axis=0, keepdims=True)
preferred_idx = np.argmax(shifted, axis=0)
opposite_idx = (preferred_idx + 4) % 8
neuron_idx = np.arange(shifted.shape[1])
preferred = shifted[preferred_idx, neuron_idx]
opposite = shifted[opposite_idx, neuron_idx]

dsi = np.divide(
    preferred - opposite,
    preferred + opposite,
    out=np.full_like(preferred, np.nan),
    where=(preferred + opposite) > 1e-8
)
valid_dsi = dsi[np.isfinite(dsi)]

print("\nRefined Direction Selectivity Analysis")
print("-" * 45)
print(f"Total neurons: {len(dsi)}")
print(f"Valid DSI neurons: {len(valid_dsi)}")
print(f"Mean DSI: {np.mean(valid_dsi):.4f}")
print(f"Median DSI: {np.median(valid_dsi):.4f}")
print(f"High selectivity (DSI > 0.5): {np.sum(valid_dsi > 0.5)}")
print(f"Low selectivity (DSI < 0.2): {np.sum(valid_dsi < 0.2)}")

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.hist(valid_dsi, bins=20, edgecolor="black")
plt.xlabel("Refined Direction Selectivity Index (DSI)")
plt.ylabel("Number of Neurons")
plt.title("Distribution of Refined Direction Selectivity")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("results/figures/refined_dsi_distribution.png", dpi=300)
plt.show()

print("\nRefined DSI histogram saved!")