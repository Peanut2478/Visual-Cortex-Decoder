from src.data_loader import load_data
from src.preprocessing import prepare_data
import torch 
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

class NeuralDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(207,64),nn.ReLU(),nn.Linear(64,8))

    def forward(self,x):
        return self.network(x)

dataset, timestamps, dff, stim_table = load_data()
X_train, X_test, y_train, y_test = prepare_data(dff, stim_table)

'''
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)
print("Training finite:", np.isfinite(X_train).all())
print("Testing finite:", np.isfinite(X_test).all())
'''
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor((y_train/45).to_numpy(), dtype=torch.long)
y_test = torch.tensor((y_test/45).to_numpy(), dtype=torch.long)

model = NeuralDecoder()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
outputs = model(X_train)
print("Output shape:",outputs.shape)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs,y_train)
    loss.backward()
    optimizer.step()
    if(epoch+1)%10 ==0:
        print(f"Epoch{epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    outputs = model(X_train)
    predicted_classes = torch.argmax(outputs,dim=1)
    accuracy = (predicted_classes == y_train).float().mean() * 100
print(f"Training accuracy: {accuracy.item():.2f}%")
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

