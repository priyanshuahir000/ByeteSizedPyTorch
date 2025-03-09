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

#### Training Process
- **Loss Function**: Mean Absolute Error (L1Loss)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.01
- **Epochs**: 200
- **Hardware Acceleration**: CUDA if available

#### Results

![Linear Regression Results](https://github.com/priyanshuahir000/ByeteSizedPyTorch/blob/main/assets/Images/LinearRegression.png)

The model achieves excellent performance in predicting the linear relationship, demonstrating the effectiveness of PyTorch's automatic differentiation and optimization capabilities.

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
```

## 🔮 Future Projects

- Comming Soon 🙌

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ and PyTorch
</p>
