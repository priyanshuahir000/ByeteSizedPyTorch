import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles  # Changed from make_blobs to make_circles
from sklearn.model_selection import train_test_split 
from utils import plot_decision_boundary

# Set hyperparameters
NUM_FEATURES = 2
RANDOM_SEED = 42

# Create circular non-linear binary data
X_blob, Y_blob = make_circles(n_samples=1000,
                             noise=0.03,
                             random_state=RANDOM_SEED,)  # Controls the size difference between inner and outer circle

# Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
Y_blob = torch.from_numpy(Y_blob).type(torch.float) # Use float for binary labels

# Split data
X_blob_train, X_blob_test, Y_blob_train, Y_blob_test = train_test_split(X_blob, Y_blob, test_size=0.2, random_state=RANDOM_SEED)

print(f"X_blob_train shape: {X_blob_train.shape}, Y_blob_train shape: {Y_blob_train.shape}")
print(f"X_blob_test shape: {X_blob_test.shape}, Y_blob_test shape: {Y_blob_test.shape}")

# Plot the data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=Y_blob, cmap=plt.cm.RdYlBu)
plt.title("Circular Binary Classification Data")
plt.show()

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# Create a binary classification model with more hidden units for non-linear data
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_features, hidden_units=10, output_features=1):  # Increased hidden units for more complex boundaries
        super().__init__() 
        self.layer = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    
# Create instance of the model
model = BinaryClassificationModel(input_features=NUM_FEATURES, hidden_units=10, output_features=1).to(device)

# Define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss for binary classification
# BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class
# It's more numerically stable than using a plain Sigmoid followed by a BCELoss
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.3)

# Change the data to the device
X_blob_train, Y_blob_train = X_blob_train.to(device), Y_blob_train.to(device)
X_blob_test, Y_blob_test = X_blob_test.to(device), Y_blob_test.to(device)

# Training loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    model.train()
    # Forward pass
    Y_logits = model(X_blob_train).squeeze() # Squeeze to remove extra dimension
    # Get predictions from logits (binary: 1 = probability >= 0.5, 0 = probability < 0.5)
    Y_pred = torch.round(torch.sigmoid(Y_logits))
    loss = loss_fn(Y_logits, Y_blob_train) # Compute loss
    acc = accuracy_fn(y_true=Y_blob_train, y_pred=Y_pred) # Compute accuracy
    optimizer.zero_grad() # Zero out the gradients
    loss.backward() # Backward pass or backpropagation 
    optimizer.step() # Update the weights

    # Test the model
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_blob_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, Y_blob_test)
        test_acc = accuracy_fn(y_true=Y_blob_test, y_pred=test_preds)

    # Print 
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Training loss: {loss:.5f}, Training accuracy: {acc:.2f}, Testing loss: {test_loss:.5f}, Testing accuracy: {test_acc:.2f}")

plt.figure(figsize=(10, 7))
plt.subplot(1,2,1)
plt.title("Training data")
plot_decision_boundary(model, X_blob_train, Y_blob_train)
plt.subplot(1,2,2)
plt.title("Testing data")
plot_decision_boundary(model, X_blob_test, Y_blob_test)
plt.show()
