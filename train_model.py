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
    def __init__(self,input_size=207,dropout=0.3):
        super().__init__()

        self.network = nn.Sequential( nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

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

print("Unique training labels:", torch.unique(y_train))
print("Unique validation labels:", torch.unique(y_val))
print("Unique testing labels:", torch.unique(y_test))
print("Training label counts:", torch.bincount(y_train))
print("Validation label counts:", torch.bincount(y_val))

# param experiment 
weight_decays = [0.0001, 0.001, 0.01]
wd_results = []
for wd in weight_decays:
    print(f"\nTesting Weight Decay = {wd}")
    torch.manual_seed(42)
    test_model = NeuralDecoder(input_size=X_train.shape[1],dropout=0.5)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001, weight_decay=wd)
    best_val_loss = float("inf")
    best_state = None
    patience = 30 
    counter = 0 

    for epoch in range (300):
        test_model.train()
        optimizer.zero_grad()

        outputs = test_model(X_train)
        loss =criterion(outputs,y_train)

        loss.backward()
        optimizer.step()
        test_model.eval()

        with torch.no_grad():
            val_outputs = test_model(X_val)
            val_loss = criterion(val_outputs,y_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(test_model.state_dict())
            counter =0
        else:
            counter += 1
        if epoch < 5 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"Train Loss = {loss.item():.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"Patience Counter = {counter}"
            )    
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break 
    #restor ebst odel 
    test_model.load_state_dict(best_state)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Epochs completed: {epoch + 1}")
    test_model.eval()

    with torch.no_grad():
        train_preds = test_model(X_train).argmax(dim=1)
        val_preds = test_model(X_val).argmax(dim=1)

        train_acc = ((train_preds == y_train).float().mean().item() * 100)
        val_acc = ((val_preds == y_val).float().mean().item() * 100)
    wd_results.append((wd, train_acc, val_acc, best_val_loss))
    print(
        f"Weight Decay {wd} | "
        f"Train: {train_acc:.2f}% | "
        f"Val: {val_acc:.2f}% | "
        f"Best Val Loss: {best_val_loss:.4f}"
    )
print("\n--- Weight Decay Comparison ---")
for wd, train_acc, val_acc, val_loss in wd_results:
    print(
        f"{wd}: Train={train_acc:.2f}%, "
        f"Val={val_acc:.2f}%, "
        f"Loss={val_loss:.4f}"
    )

# Save the best hyperparameter configuration
best_wd = min(wd_results, key=lambda x: x[3])[0]
print(f"\nBest weight decay: {best_wd}")
# Retrain with selected hyperparameters
torch.manual_seed(42)
model = NeuralDecoder(
    input_size=X_train.shape[1],
    dropout=0.5
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=best_wd
)
best_val_loss = float("inf")
best_state = None
counter = 0
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
    if counter >= 30:
        break

model.load_state_dict(best_state)
model.eval()
Path("results").mkdir(exist_ok=True)
torch.save(
    model.state_dict(),
    "results/best_decoder.pt"
)
print("Best model saved!")
print(f"Best validation loss: {best_val_loss:.4f}")
''' 
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

'''
model.eval()

with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = torch.argmax(test_outputs, dim=1)

    test_accuracy = (
        (test_predictions == y_test).float().mean().item()
    ) * 100

print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

cm = confusion_matrix(
    y_test.numpy(),
    test_predictions.numpy(),
    labels=np.arange(8)
)

print("\nFinal Confusion Matrix:")
print(cm)

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Number of predictions")

plt.xticks(range(8), range(0, 360, 45))
plt.yticks(range(8), range(0, 360, 45))

plt.xlabel("Predicted Direction (degrees)")
plt.ylabel("Actual Direction (degrees)")
plt.title("Tuned Neural Decoder — Test Confusion Matrix")

plt.tight_layout()
plt.savefig(
    "results/figures/tuned_confusion_matrix.png",
    dpi=300
)
plt.show()

# Day 10: Hyperparameter tuning summary

Path("results/figures").mkdir(parents=True, exist_ok=True)

dropout_values = [0.2, 0.3, 0.4, 0.5]
dropout_acc = [70.83, 70.83, 72.92, 75.00]

learning_rates = [0.0001, 0.0005, 0.001]
lr_acc = [40.62, 75.00, 75.00]

weight_decays = [0.0001, 0.001, 0.01]
wd_acc = [76.04, 75.00, 66.67]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(dropout_values, dropout_acc, "o-")
axes[0].set_title("Dropout Tuning")
axes[0].set_xlabel("Dropout Rate")
axes[0].set_ylabel("Validation Accuracy (%)")

axes[1].plot(learning_rates, lr_acc, "o-")
axes[1].set_title("Learning Rate Tuning")
axes[1].set_xlabel("Learning Rate")
axes[1].set_ylabel("Validation Accuracy (%)")

axes[2].plot(weight_decays, wd_acc, "o-")
axes[2].set_title("Weight Decay Tuning")
axes[2].set_xlabel("Weight Decay")
axes[2].set_ylabel("Validation Accuracy (%)")

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    "results/figures/hyperparameter_tuning.png",
    dpi=300
)
plt.show()