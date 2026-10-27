from src.data_loader import load_data
from src.preprocessing import prepare_data,normalize_data
import torch 
import torch.nn as nn
import numpy as np
import random
import copy 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class NeuralDecoder(nn.Module):
    def __init__(self,input_size=207):
        super().__init__()

        self.network = nn.Sequential( nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 8))

    def forward(self,x):
        return self.network(x)

dataset, timestamps, dff, stim_table = load_data()
#extract raw neural response
X_train_full, X_test, y_train_full, y_test = prepare_data(dff, stim_table)
#split trainign data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)
#normalie 
X_train, X_val, X_test = normalize_data(X_train,X_val,X_test)

# linear testing 
linear_model = LogisticRegression(max_iter=2000)
linear_model.fit(X_train, y_train)
train_preds = linear_model.predict(X_train)
test_preds = linear_model.predict(X_test)
print("Training accuracy:",
      accuracy_score(y_train, train_preds))

print("Test accuracy:",
      accuracy_score(y_test, test_preds))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_preds))

 #convert to tensor 
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
#convert oreitntation into calss 
y_train = torch.tensor(
    (np.asarray(y_train) / 45).astype(int),
    dtype=torch.long
)
y_val = torch.tensor((np.asarray(y_val) / 45).astype(int), dtype=torch.long)
y_test = torch.tensor(
    (np.asarray(y_test) / 45).astype(int),
    dtype=torch.long
)
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)
print("Training finite:", np.isfinite(X_train).all())
print("Testing finite:", np.isfinite(X_test).all())




torch.manual_seed(42)
model = NeuralDecoder(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.001)
outputs = model(X_train)
print("Output shape:",outputs.shape)

epochs = 300
train_losses=[]
val_losses=[]
best_Val_loss = float("inf")
best_model_state = None
patience = 30
counter = 0 

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs,y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs,y_val)
    val_losses.append(val_loss.item())
    if val_loss.item() < best_Val_loss:
        best_Val_loss = val_loss.item()
        best_model_state = copy.deepcopy(model.state_dict())
        counter= 0 
    else:
        counter +=1 
    if counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    if(epoch+1)%10 ==0:
        print(f"Epoch{epoch+1}/{epochs}, Loss: {loss.item():.4f}," f"Val Loss = {val_loss.item():.4f}")

model.load_state_dict(best_model_state)
print(f"Best val loss: {best_Val_loss:.4f}")
Path("results").mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    "results/best_decoder.pt"
)

print("Best model saved!")
model.eval()


with torch.no_grad():
    train_outputs = model(X_train)
    val_outputs = model(X_val)
    train_predictions = torch.argmax(train_outputs, dim=1)
    val_predictions = torch.argmax(val_outputs, dim=1)
    train_accuracy = (
        (train_predictions == y_train).float().mean().item()
    ) * 100
    val_accuracy = (
        (val_predictions == y_val).float().mean().item()
    ) * 100

print(f"Training accuracy: {train_accuracy:.2f}%")
print(f"Validation Accuracy: {val_accuracy:.2f}%")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = torch.argmax(test_outputs, dim=1)
    test_accuracy = ((test_predictions == y_test).float().mean() * 100)
print(f"Test accuracy: {test_accuracy.item():.2f}%")

cm = confusion_matrix(
    y_test.numpy(),
    test_predictions.numpy(),
    labels=np.arange(8)
)
print("Confusion matrix:")
print(cm)

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

with torch.no_grad():
    og_preds = model(X_val).argmax(dim=1)
    base_acc = (og_preds ==y_val).float().mean().item()
print(f"Baseline val accurancy: {base_acc:.4f}")
importance_scores = []
for neuron_idx in range(X_val.shape[1]):
    drops = []
    for repeat in range(5):
        X_shuffled = X_val.clone()
        permutation = torch.randperm(X_val.shape[0])
        X_shuffled[:,neuron_idx] = (X_shuffled[permutation,neuron_idx])

        with torch.no_grad():
            preds = model(X_shuffled).argmax(dim=1)
            shuffled_acc = (preds ==y_val).float().mean().item()
        drops.append(base_acc - shuffled_acc)
    importance_scores.append(np.mean(drops))
importance_scores = np.array(importance_scores)
top_neurons = np.argsort(importance_scores)[::-1]
print("\nTop 10 most importnat neurons:")
for rank, neuron_idx in enumerate(top_neurons[:10],start=1):
    print(f"{rank}. Neuron {neuron_idx}: "
    f"Importance = {importance_scores[neuron_idx]:.4f}")

X_train_np = X_train.detach().cpu().numpy()
X_val_np = X_val.detach().cpu().numpy()
y_train_np = y_train.detach().cpu().numpy()
y_val_np = y_val.detach().cpu().numpy()

neuron_counts = [10,25,50,100,207]
results = []
for n in neuron_counts:
    # top neuron 
    selected = top_neurons[:n]
    X_train_subset = X_train_np[:,selected]
    X_val_subset = X_val_np[:,selected]

    #train a new classifier 
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_subset,y_train_np)

    #eval 
    train_acc = accuracy_score(y_train_np, clf.predict(X_train_subset))
    val_acc = accuracy_score(y_val_np,clf.predict(X_val_subset))

    results.append((n,train_acc,val_acc))
    print(
        f"{n} neurons |"
        f"Train: {train_acc:.2%}|"
        f"Val: {val_acc:.2%}"
    )

sorted_importance = importance_scores[top_neurons]
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(sorted_importance) + 1),
    sorted_importance
)
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Neuron rank")
plt.ylabel("Permutation importance")
plt.title("Neuron Importance Ranking")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/neuron_importance.png", dpi=300)
plt.show()

counts = [r[0] for r in results]
train_accs = [r[1] for r in results]
val_accs = [r[2] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(counts, train_accs, "o-", label="Training")
plt.plot(counts, val_accs, "o-", label="Validation")
plt.xlabel("Number of neurons")
plt.ylabel("Accuracy")
plt.title("Decoding Accuracy vs Number of Neurons")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/neuron_count_accuracy.png", dpi=300)
plt.show()