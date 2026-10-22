from src.data_loader import load_data
from src.preprocessing import prepare_data,normalize_data
import torch 
import torch.nn as nn
import numpy as np
import random
import copy 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class NeuralDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(207,64),nn.ReLU(),nn.Dropout(p=0.3),nn.Linear(64,8))

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
model = NeuralDecoder()
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

