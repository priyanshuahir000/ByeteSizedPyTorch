import torch
from torch import nn 
import matplotlib.pyplot as plt

# Create device agnostic code
device= "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create dummy data using the linear regression formula
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(1) # without unsqueeze there will be a shape mismatch because the model expects a 2D tensor
Y = weight * X + bias

# Split data
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

def plot_prediction(train_data=X_train,
                    train_labels=Y_train,
                    test_data=X_test,
                    test_labels=Y_test,
                    predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', label='Training data')

    plt.scatter(test_data, test_labels, c='g', label='Testing data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', label='Predictions')

    plt.legend(prop={'size': 14})
    plt.show() 

plot_prediction(X_train, Y_train, X_test, Y_test)

# Create a linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # use nn.linear_layer for creating model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                  out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)
model = LinearRegressionModel().to(device)
print(model, model.state_dict())

# Define loss function and optimizer
loss_function = nn.L1Loss() # same as MAE
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01) # lr is the learning rate and SGD stands for Stochastic Gradient Descent

# Training loop
torch.manual_seed(42)
epochs = 200

# put data in the target device
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)


for epoch in range(epochs):
    model.train() 
    Y_pred = model(X_train) # Forward pass
    loss = loss_function(Y_pred, Y_train) # Compute loss
    optimizer.zero_grad() # Zero out the gradients
    loss.backward() # Backward pass or backpropagation
    optimizer.step() # Update the weights

    # Testing
    model.eval() # switch to evaluation mode
    with torch.inference_mode(): # turn off gradient computation
        Y_test_pred = model(X_test)
        test_loss = loss_function(Y_test_pred, Y_test)

    # Print the loss
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Training loss: {loss.item()}, Testing loss: {test_loss.item()}")

# Make predictions
model.eval()
with torch.inference_mode():
    Y_preds = model(X_test)
print(Y_preds)

plot_prediction(predictions=Y_preds.cpu())

# Saving the model
from pathlib import Path
model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

model_name = "LinearRegression.pth"
model_save_path = model_path / model_name
torch.save(model.state_dict(), model_save_path)