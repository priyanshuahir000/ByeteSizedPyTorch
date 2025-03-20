import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split 
from utils import plot_decision_boundary


#Set hyperparameters
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Create multi-class data
X_blob, Y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES, # 2D data
                            centers=NUM_CLASSES, # 4 classes
                            cluster_std=1.5, # standard deviation of the clusters
                            random_state=RANDOM_SEED)

# Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
Y_blob = torch.from_numpy(Y_blob).type(torch.LongTensor)

# Split data
X_blob_train, X_blob_test, Y_blob_train, Y_blob_test = train_test_split(X_blob, Y_blob, test_size=0.2, random_state=RANDOM_SEED)

print(f"X_blob_train shape: {X_blob_train.shape}, Y_blob_train shape: {Y_blob_train.shape}")
print(f"X_blob_test shape: {X_blob_test.shape}, Y_blob_test shape: {Y_blob_test.shape}")

# Plot the data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=Y_blob, cmap=plt.cm.RdYlBu)
plt.show()

# Create device agnostic code
device= "cuda" if torch.cuda.is_available() else "cpu"

# accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# Create a multi-class classification model
class MultiClassClassificationModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__() 
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)
    
# Create instance of the model
model = MultiClassClassificationModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES, hidden_units=8).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss is used for multi-class classification
# why we use CrossEntropyLoss instead of L1Loss or MSELoss?
# CrossEntropyLoss is used for multi-class classification problems where the target labels are integers starting from 0 to C-1 where C is the number of classes.
# CrossEntropyLoss is used when the model is outputting class probabilities directly.   
# L1Loss and MSELoss are used for regression problems where the target labels are continuous values.
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.3) # lr is the learning rate and SGD stands for Stochastic Gradient Descent

# Change the data to the device
X_blob_train, Y_blob_train = X_blob_train.to(device), Y_blob_train.to(device)
X_blob_test, Y_blob_test = X_blob_test.to(device), Y_blob_test.to(device)

# Training loop
# Logits (row output of the model) -> probabilities (use torch.softmax ) -> predictions (take the argmax of the probabilities)
# Calculate the loss -> calculate the gradients -> update the weights -> zero the gradients
# Repeat for all the epochs

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    model.train()
    Y_logits = model(X_blob_train) # Forward pass
    Y_pred = torch.softmax(Y_logits, dim=1).argmax(dim=1) # Convert logits to probabilities
    loss = loss_fn(Y_logits, Y_blob_train) # Compute loss
    acc = accuracy_fn(y_true=Y_blob_train, y_pred=Y_pred) # Compute accuracy
    optimizer.zero_grad() # Zero out the gradients
    loss.backward() # Backward pass or backpropagation 
    optimizer.step() # Update the weights

    # Test the model
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, Y_blob_test)
        test_acc = accuracy_fn(y_true=Y_blob_test, y_pred=test_preds)

    # Print 
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Training loss: {loss: .4f}, Training accuracy: {acc: .2f}, Testing loss: {test_loss: .4f}, Testing accuracy: {test_acc: .2f}")

plt.figure(figsize=(10, 7))
plt.subplot(1,2,1)
plt.title("Training data")
plot_decision_boundary(model, X_blob_train, Y_blob_train)
plt.subplot(1,2,2)
plt.title("Testing data")
plot_decision_boundary(model, X_blob_test, Y_blob_test)
plt.show()
