import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.data_loader import load_data
from src.preprocessing import prepare_data, normalize_data

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

# Calculate DSI using training data only
directions = np.arange(0, 360, 45)
labels = np.asarray(y_train)

mean_responses = np.array([
    np.mean(X_train[labels == direction], axis=0)
    for direction in directions
])

responses = np.maximum(mean_responses, 0)

pref_idx = np.argmax(responses, axis=0)
opp_idx = (pref_idx + 4) % 8
neuron_idx = np.arange(responses.shape[1])

pref_resp = responses[pref_idx, neuron_idx]
opp_resp = responses[opp_idx, neuron_idx]

denominator = pref_resp + opp_resp

dsi = np.divide(
    pref_resp - opp_resp,
    denominator,
    out=np.full_like(denominator, np.nan),
    where=denominator > 1e-8
)

print("\nDSI Verification")
print("-" * 40)
print(f"Mean DSI: {np.nanmean(dsi):.4f}")
print(f"Median DSI: {np.nanmedian(dsi):.4f}")
print(f"Min DSI: {np.nanmin(dsi):.4f}")
print(f"Max DSI: {np.nanmax(dsi):.4f}")
print(f"Valid neurons: {np.sum(np.isfinite(dsi))}")

#cal vector_dsi
angles = np.deg2rad(directions)
vector_x = np.sum(responses * np.cos(angles)[:,None], axis=0)
vector_y = np.sum(responses * np.sin(angles)[:,None], axis=0)
tot_resp = np.sum(responses,axis=0)
vector_dsi = np.divide(
    np.sqrt(vector_x**2 + vector_y**2),
    tot_resp,
    out=np.full_like(tot_resp,np.nan),
    where=tot_resp >1e-8
)
dsi = vector_dsi
# Define neuron groups
high_dsi = np.where(np.isfinite(dsi) & (dsi > 0.5))[0]
low_dsi = np.where(np.isfinite(dsi) & (dsi < 0.2))[0]
all_neurons = np.arange(len(dsi))

groups = {
    "High DSI": high_dsi,
    "Low DSI": low_dsi,
    "All Neurons": all_neurons
}

print("\nDSI Decoder Comparison")
print("-" * 45)

results = {}

for name, indices in groups.items():

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train[:, indices], y_train)

    val_pred = model.predict(X_val[:, indices])
    #test_pred = model.predict(X_test[:, indices])

    val_acc = accuracy_score(y_val, val_pred)
    #test_acc = accuracy_score(y_test, test_pred)

    results[name] = (val_acc)

    print(f"\n{name}")
    print(f"Neurons: {len(indices)}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")


# Plot comparison
Path("results/figures").mkdir(
    parents=True, exist_ok=True
)

names = list(results.keys())
val_accuracies = [results[n] * 100 for n in names]

plt.figure(figsize=(9, 5))
plt.bar(names, val_accuracies)
plt.ylabel("Validation Accuracy (%)")
plt.title("Vector-Based DSI Decoder Comparison")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    "results/figures/vector_dsi_decoder_comparison.png",
    dpi=300
)

plt.show()

print("\nDSI decoder comparison saved!")

# Equal-size neuron comparison
print("\nEqual-Size DSI Comparison")
print("-" * 45)

rng = np.random.default_rng(42)
n_neurons = min(len(high_dsi), len(low_dsi))

high_scores = []
low_scores = []

for repeat in range(100):
    selected_high = rng.choice(
        high_dsi, size=n_neurons, replace=False
    )
    selected_low = rng.choice(
        low_dsi, size=n_neurons, replace=False
    )

    for indices, scores in [
        (selected_high, high_scores),
        (selected_low, low_scores)
    ]:
        model = LogisticRegression(
            max_iter=2000,
            random_state=42
        )
        model.fit(X_train[:, indices], y_train)

        predictions = model.predict(X_val[:, indices])
        scores.append(accuracy_score(y_val, predictions))

print(f"Neurons per group: {n_neurons}")
print(f"High DSI mean validation: {np.mean(high_scores)*100:.2f}%")
print(f"Low DSI mean validation: {np.mean(low_scores)*100:.2f}%")
print(f"Difference: {(np.mean(high_scores)-np.mean(low_scores))*100:.2f} percentage points")

plt.figure(figsize=(7, 5))
plt.boxplot(
    [high_scores, low_scores],
    tick_labels=["High DSI", "Low DSI"]
)
plt.ylabel("Validation Accuracy")
plt.title(f"Equal-Size DSI Comparison ({n_neurons} Neurons)")
plt.tight_layout()
plt.savefig(
    "results/figures/vector_dsi_equal_size_comparison.png",
    dpi=300
)
plt.show()

print("\nEqual-size comparison saved!")
