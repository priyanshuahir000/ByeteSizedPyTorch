# ByteSizedPyTorch <img src="https://github.com/priyanshuahir000/ByeteSizedPyTorch/blob/main/assets/Icons/pytorch-color.svg" alt="PyTorch Logo" width="50"/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()

A collection of concise PyTorch projects that demonstrate deep learning concepts through practical implementations.

## 📚 Projects

### 1. Linear Regression Model 📈

<details>
<summary>Click to expand</summary>

#### Overview
A simple linear regression model implemented from scratch using PyTorch. This project demonstrates the fundamental workflow of creating, training, and evaluating a deep learning model.

#### Features
- 🧪 Creates synthetic data using a linear formula (y = 0.7x + 0.3)
- 🔄 Splits data into training and testing sets
- 🧠 Implements a neural network model with a single linear layer
- 📊 Visualizes predictions against actual data
- 💾 Saves the trained model for future use

#### Model Architecture
```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x):
        return self.linear_layer(x)
```

#### Results

![Linear Regression Results](https://github.com/priyanshuahir000/ByeteSizedPyTorch/blob/main/assets/Images/LinearRegression.png)
</details>

### 2. Multi-Class Classification Model 🎯

<details>
<summary>Click to expand</summary>

#### Overview
A multi-class classification model built using PyTorch to classify data points into four different classes.

#### Features
- 🌀 Uses synthetic data generated with `make_blobs`
- 📉 Splits dataset into training and testing sets
- 🧠 Implements a neural network with multiple hidden layers and ReLU activation
- 📊 Uses CrossEntropyLoss for classification
- 💾 Saves and evaluates the trained model

#### Model Architecture
```python
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
```

#### Results

![Multi Class Classification Results](https://github.com/priyanshuahir000/ByeteSizedPyTorch/blob/main/assets/Images/MultiClassClassificationModel_1.png)
![Multi Class Classification Results](https://github.com/priyanshuahir000/ByeteSizedPyTorch/blob/main/assets/Images/MultiClassClassificationModel_2.png)
</details>

## 🛠️ Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ByteSizedPyTorch.git
   cd ByteSizedPyTorch
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install requirements.txt
   ```

## 🚀 Usage

Each project is contained in its own Python file. To run a project, simply execute the corresponding file:

```bash
python LinearRegressionModel.py
python MultiClassClassificationModel.py
```

## 🐞 Future Projects

- Coming Soon 🙌

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ and PyTorch
</p>

