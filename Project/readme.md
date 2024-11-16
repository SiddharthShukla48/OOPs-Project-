# XGBoost Implementation from Scratch

## Table of Contents
* [Overview](#overview)
* [Algorithm Details](#algorithm-details)
* [Implementation](#implementation)
* [Mathematical Foundation](#mathematical-foundation)
* [Code Walkthrough](#code-walkthrough)
* [Usage Example](#usage-example)
* [Tips and Best Practices](#tips-and-best-practices)

## Overview

This is a from-scratch implementation of XGBoost (eXtreme Gradient Boosting) in Python. The implementation includes support for:

- Both regression and binary classification
- L1 and L2 regularization
- Early stopping
- Tree pruning
- Custom loss functions

## Algorithm Details

### Basic Workflow

1. **Initialization**
   - Set initial prediction (base_score)
   - For regression: mean of target values
   - For classification: log odds of positive class

2. **Boosting Process**
   ```
   For each iteration t:
       1. Calculate gradients and hessians
       2. Build new tree to predict gradients
       3. Add tree predictions * learning_rate
       4. Check early stopping criteria
   ```

3. **Tree Building**
   ```
   While node can be split:
       1. Find best split point
       2. Calculate gain
       3. Split if gain > gamma
       4. Recursively continue on child nodes
   ```

## Implementation

### Core Components

```python
class XGBoost:
    def __init__(self):
        """
        Parameters:
        - n_estimators: number of trees
        - learning_rate: step size
        - max_depth: maximum tree depth
        - min_samples_split: minimum samples for splitting
        - reg_lambda: L2 regularization
        - reg_alpha: L1 regularization
        - gamma: minimum gain for splitting
        - objective: "reg:squarederror" or "binary:logistic"
        - early_stopping_rounds: stopping criterion
        """

    def fit(self, X, y):
        """Train the model"""

    def predict(self, X):
        """Make predictions"""
```

## Mathematical Foundation

### Key Formulas

1. **Objective Function**
```
Obj(θ) = L(θ) + Ω(θ)
where:
L(θ) = Σ l(yi, ŷi)     # Training loss
Ω(θ) = γT + 1/2 λ||w||²  # Regularization
```

2. **Split Gain**
```
Gain = 1/2 * [GL²/(HL + λ) + GR²/(HR + λ) - (GL + GR)²/(HL + HR + λ)] - γ
where:
GL, GR = sum of gradients (left/right)
HL, HR = sum of hessians (left/right)
λ = L2 regularization
γ = minimum gain threshold
```

3. **Loss Functions**

For Regression:
```
Loss = (y - ŷ)²/2
Gradient = ŷ - y
Hessian = 1
```

For Binary Classification:
```
Loss = -y*log(p) - (1-y)*log(1-p)
Gradient = p - y
Hessian = p*(1-p)
where p = sigmoid(ŷ)
```

## Code Walkthrough

### 1. Initialization
```python
def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.max_depth = max_depth
```

### 2. Gradient and Hessian Calculation
```python
def _gradient_and_hessian(self, y, y_pred):
    if self.objective == "reg:squarederror":
        gradient = y_pred - y
        hessian = np.ones_like(y)
    elif self.objective == "binary:logistic":
        y_pred = 1 / (1 + np.exp(-y_pred))
        gradient = y_pred - y
        hessian = y_pred * (1 - y_pred)
    return gradient, hessian
```

### 3. Split Finding
```python
def _split(self, X, y, gradients, hessians):
    best_gain = -float("inf")
    best_split = None
    
    for feature in range(X.shape[1]):
        for threshold in np.unique(X[:, feature]):
            gain = calculate_gain(gradients, hessians, threshold)
            if gain > best_gain:
                best_gain = gain
                best_split = {"feature": feature, "threshold": threshold}
    
    return best_split
```

## Usage Example

```python
# Create synthetic data
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Initialize model
model = XGBoost(
    n_estimators=50,
    learning_rate=0.1,
    objective="binary:logistic",
    early_stopping_rounds=5
)

# Train
model.fit(X, y, X_val=X, y_val=y)

# Predict
predictions = model.predict(X)
```

## Tips and Best Practices

### Hyperparameter Tuning

1. **Learning Rate**
   - Start with small values (0.01-0.1)
   - Trade-off between training time and model quality

2. **Number of Trees**
   - More trees generally better
   - Use early stopping to find optimal number

3. **Tree Depth**
   - Controls model complexity
   - Deeper trees can overfit
   - Typical range: 3-8

### Memory Usage

The implementation stores:
- All trees in memory
- Gradients and hessians during training
- Validation error history

Memory usage scales with:
- Number of trees
- Number of samples
- Tree depth

### Performance Optimization

Time complexity:
- Training: O(n_trees * n_features * n_samples * log(n_samples))
- Prediction: O(n_trees * log(n_samples))

Space complexity:
- O(n_trees * 2^depth) for storing trees
- O(n_samples) for gradients/hessians

### Best Practices

1. **Data Preparation**
   ```python
   # Scale features
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Cross-Validation**
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X, y, cv=5)
   ```

3. **Error Monitoring**
   ```python
   plt.plot(model.validation_error_history)
   plt.title('Validation Error vs Iterations')
   plt.show()
   ```

### Common Issues and Solutions

1. **Overfitting**
   - Reduce max_depth
   - Increase min_samples_split
   - Increase reg_lambda

2. **Underfitting**
   - Increase max_depth
   - Decrease min_samples_split
   - Increase n_estimators

3. **Numerical Instability**
   - Scale features
   - Adjust learning_rate
   - Handle missing values