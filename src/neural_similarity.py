import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from sklearn.model_selection import train_test_split
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
X_train, X_val, X_test = normalize_data(X_train, X_val,X_test)
directions = np.arange(0,360,45)
labels = np.asarray(y_train)

mean_responses = []
for direction in directions:
    responses = X_train[labels == direction]
    mean_responses.append(np.mean(responses,axis=0))
mean_responses = np.array(mean_responses)

#cal cos sim matrix 
norms = np.linalg.norm(mean_responses, axis=1)
safe_norms = np.where(norms ==0,1, norms)
normalized_vectors = mean_responses / safe_norms[:, None]
sim = normalized_vectors @ normalized_vectors.T

print("\nOpposite-Direction Nueral Similarity")
print("-"*40)
for i in range(4):
    j = i+4
    print(
        f"{directions[i]} vs {directions[j]}:"
        f"{sim[i,j]:.4f}"
    )
plt.figure(figsize=(8,6))
plt.imshow(sim,cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Cosine Similarity")
plt.xticks(range(8),directions)
plt.yticks(range(8),directions)
plt.xlabel("Stimulus Diretion (degrees)")
plt.ylabel("Stimulus Direction (degrees)")
plt.title("Neural Pop Response Similarity")

plt.tight_layout()
Path("results/figures").mkdir(parents=True,exist_ok=True)
plt.savefig("results/figures/neural_similarity.png",dpi=300)
plt.show()
print("\nSimilarity heatmap saved!")