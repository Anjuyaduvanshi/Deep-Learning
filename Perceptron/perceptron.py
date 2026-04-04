import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('placement.csv')

X = df.iloc[:, 0:2].values
y = df.iloc[:, -1].values

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Model (Perceptron)
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)  # logits

model = Perceptron()

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
epochs = 200

for epoch in range(epochs):
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Extract weights
weights = model.linear.weight.detach().numpy()[0]
bias = model.linear.bias.detach().numpy()[0]

print("Weights:", weights)
print("Bias:", bias)

# -----------------------------------
# Decision Boundary Plot 
# -----------------------------------

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# Flatten grid for prediction
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Predict
with torch.no_grad():
    logits = model(grid_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float().numpy()

Z = preds.reshape(xx.shape)

# Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.3)

# Scatter plot
plt.scatter(X[y==1][:,0], X[y==1][:,1], label="Placed (1)", marker='o')
plt.scatter(X[y==0][:,0], X[y==0][:,1], label="Not Placed (0)", marker='x')

plt.xlabel("CGPA")
plt.ylabel("Resume Score")
plt.title("Decision Regions (PyTorch Perceptron)")
plt.legend()

plt.show()
