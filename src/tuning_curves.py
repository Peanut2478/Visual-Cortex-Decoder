import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import prepare_data,normalize_data

dataset, timestamps, dff, stim_table = load_data()

X_train_full, X_test, y_train_full, y_test = prepare_data(
    dff, stim_table
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

X_train, X_val, X_test = normalize_data(
    X_train, X_val, X_test
)

directions = np.arange(0,360,45)
labels = np.asarray(y_train)
mean_resp = np.array([
    np.mean(X_train[labels == direction], axis=0)
    for direction in directions
])
print("Mean responses shape:", mean_resp.shape)
resp = np.maximum(mean_resp,0)
pref_idx = np.argmax(resp,axis=0)
opp_idx = (pref_idx + 4) % 8
neuron_idx  = np.arange(resp.shape[1])

preff = resp[pref_idx,neuron_idx]
opp = resp[opp_idx,neuron_idx]

dsi = np.divide(
    preff - opp,
    preff + opp,
    out=np.full_like(preff,np.nan),
    where=(preff + opp) > 1e-8
)

# get 2 highest and 2 lowrst
valid_idx = np.flatnonzero(np.isfinite(dsi))
sorted_idx = valid_idx[np.argsort(dsi[valid_idx])]
selc = np.concatenate([
    sorted_idx[-2:][::-1],
    sorted_idx[:2]
])
fig, axes = plt.subplots(2,2, figsize=(11,8))

for ax, neuron in zip(axes.flat, selc):
    tuning = mean_resp[:,neuron]
    ax.plot(
        directions,
        tuning,
        marker="o",
        linewidth=2
    )
    ax.axhline(0,color="gray", linestyle="--")
    ax.set_xticks(directions)
    ax.set_xlabel("Stimulus Direction (degrees)")
    ax.set_ylabel("Mean Normalized Response")
    ax.set_title(
        f"Neuron {neuron} | DSI = {dsi[neuron]:.3f}"
    )
    ax.grid(alpha=0.3)
plt.suptitle("Neural Direction Tuning Curves")
plt.tight_layout()

Path("results/figures").mkdir(
    parents=True, exist_ok=True
)

plt.savefig(
    "results/figures/tuning_curves.png",
    dpi=300
)

plt.show()

print("\nSelected Neurons")
print("-" * 40)

for neuron in selc:
    print(f"Neuron {neuron}: DSI = {dsi[neuron]:.4f}")

print("\nTuning curves saved!")
