# Neural Network from Scratch

This repository implements a feedforward neural network with backpropagation and mini-batch gradient descent, built from scratch using NumPy and pandas. It is designed for educational purposes to demonstrate the inner workings of neural networks, including forward propagation, cost computation, backpropagation, and weight updates. The project supports multiple datasets for binary classification tasks.

## Project Structure

- `main.py`: Main script for training, evaluating, and visualizing the neural network on various datasets.
- `backprop.py`: Contains core neural network functions, including forward propagation, backpropagation, cost calculation, and gradient computation. Also includes verification routines for backpropagation correctness.
- `data/`: Directory containing sample datasets (`loan.csv`, `raisin.csv`, `titanic.csv`, `wdbc.csv`).
- `README.md`: This documentation file.

## Installation

1. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install required libraries:**
   ```sh
   pip install pandas numpy matplotlib scikit-learn
   ```

## How to Run

- **Verify Backpropagation Implementation:**
  ```sh
  python3 backprop.py
  ```
  This runs the verification routines to check the correctness of forward and backward propagation steps.

- **Train and Evaluate Neural Network:**
  ```sh
  python3 main.py
  ```
  This script will train the neural network on the selected dataset, evaluate its performance, and plot the learning curve.

## File Descriptions

### `main.py`

- Loads and preprocesses datasets.
- Trains the neural network using mini-batch gradient descent.
- Evaluates the model using k-fold cross-validation and computes metrics.
- Plots learning curves to visualize training progress.
- Example usage for different datasets is provided as commented code.

### `backprop.py`

- Implements core neural network operations:
  - `forward_prop`: Computes activations for each layer.
  - `back_prop`: Calculates error terms (deltas) for each layer.
  - `compute_gradients`: Computes gradients for weight updates, including regularization.
  - `update_weights`: Updates weights using computed gradients and learning rate.
  - `compute_cost` and `compute_final_cost`: Calculates cost with/without regularization.
- Includes verification functions for step-by-step debugging and understanding.

## Customization

- **Change Network Architecture:**  
  Modify the `num_layers` parameter in `main.py` to set the number of neurons per layer.

- **Adjust Hyperparameters:**  
  Change `alpha` (learning rate), `lam` (regularization strength), and `num_iterations` (number of training iterations) as needed.

- **Add New Datasets:**  
  Place new CSV files in the `data` directory and update the data loading section in `main.py`.

## Example Usage

```python
# Example call in main.py:
learning_curve(X, Y, num_layers=[9, 4, 4, 1], lam=0.0, alpha=0.1, num_iterations=500)
```

## License

This project is for educational purposes and does not include a specific license.

## Contact

For questions or suggestions, please open an issue or contact the repository owner.