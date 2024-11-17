# XGBoost Implementation from Scratch

## Overview
This repository contains a pure Python implementation of the XGBoost (eXtreme Gradient Boosting) algorithm. XGBoost is a powerful and efficient implementation of gradient boosting machines that has become one of the most popular machine learning algorithms, especially for structured/tabular data.

## Features
This implementation includes several key features of the XGBoost algorithm:

* Support for both regression (`reg:squarederror`) and binary classification (`binary:logistic`)
* Gradient-based learning with first and second order gradients
* L1 and L2 regularization
* Tree pruning using the `gamma` parameter
* Early stopping functionality
* Configurable tree parameters (depth, minimum samples for splitting)

## Technical Implementation Details

### Core Components

#### 1. Tree Building (`_build_tree`):
* Implements recursive tree construction
* Uses gradient and hessian information for optimal splits
* Includes stopping criteria based on depth and minimum samples
* Calculates leaf values using the XGBoost formula: `-sum(gradients) / (sum(hessians) + λ)`

#### 2. Split Finding (`_split`):
* Implements the XGBoost split finding algorithm
* Uses the gain formula: `0.5 * [GL²/(HL + λ) + GR²/(HR + λ) - (GL + GR)²/(HL + HR + λ)] - γ`
* Includes minimum sample constraints
* Considers L2 regularization in gain calculation

#### 3. Gradient and Hessian Calculation:
* For regression:
  * Gradient = `y_pred - y`
  * Hessian = 1
* For binary classification:
  * Gradient = `p - y` where `p` is sigmoid(prediction)
  * Hessian = `p * (1-p)`

### Training Process

The training process follows these steps:

1. Initialize base predictions
2. For each boosting round:
   * Calculate gradients and hessians
   * Build a new tree to predict the negative gradients
   * Add tree predictions with learning rate
   * Update running predictions
   * Check early stopping conditions if validation data is provided

## Usage

### Basic Example

```python
from xgboost import XGBoost
import numpy as np

# Create sample data
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Initialize model
model = XGBoost(
    n_estimators=50,
    learning_rate=0.1,
    objective="binary:logistic",
    early_stopping_rounds=5
)

# Train model
model.fit(X, y, X_val=X, y_val=y)

# Make predictions
predictions = model.predict(X)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_estimators` | Number of boosting rounds | 100 |
| `learning_rate` | Step size shrinkage to prevent overfitting | 0.1 |
| `max_depth` | Maximum depth of each tree | 3 |
| `min_samples_split` | Minimum samples required to split a node | 10 |
| `reg_lambda` | L2 regularization term | 1.0 |
| `reg_alpha` | L1 regularization term | 0.0 |
| `gamma` | Minimum loss reduction for split | 0 |
| `objective` | Learning objective ("reg:squarederror" or "binary:logistic") | "reg:squarederror" |
| `early_stopping_rounds` | Stop training if validation score doesn't improve | None |

## Implementation Notes

### 1. Numerical Stability
* Uses float64 for all calculations
* Implements proper clipping for logistic objectives
* Handles edge cases in split calculations

### 2. Memory Efficiency
* Uses NumPy arrays for efficient computation
* Implements in-place updates where possible

### 3. Performance Considerations
* Implements early stopping for better efficiency
* Uses vectorized operations for gradient and prediction calculations

## Limitations

This implementation is meant for educational purposes and differs from the official XGBoost library in several ways:

* No parallel processing support
* Limited objective functions (only regression and binary classification)
* No categorical feature support
* No missing value handling
* No built-in cross-validation

## Future Improvements

Potential areas for enhancement:

* Add support for multi-class classification
* Implement parallel processing for tree building
* Add feature importance calculation
* Add categorical feature handling
* Implement missing value support
* Add more objective functions

## Requirements
* NumPy
* Python 3.x

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.