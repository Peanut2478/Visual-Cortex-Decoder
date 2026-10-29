import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import prepare_data, normalize_data
import matplotlib.pyplot as plt
from pathlib import Path
dataset, timestamp, dff, stim_table = load_data()
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
X_train, X_val, X_test  =normalize_data(X_train,X_val,X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

print("Linear Decoder Results")
print("----------------------")
print(f"Training Accuracy: {accuracy_score(y_train, train_preds)*100:.2f}%")
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds)*100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_test, test_preds)*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, test_preds,
    labels=np.arange(0, 360, 45)
))
# Compare test accuracy
models = ["Neural Network", "Logistic Regression"]
accuracies = [63.33, 62.50]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies)

plt.ylabel("Test Accuracy (%)")
plt.title("Neural Network vs Linear Decoder")
plt.ylim(0, 100)

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.2f}%",
        ha="center"
    )

plt.tight_layout()

Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig(
    "results/figures/day11_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
