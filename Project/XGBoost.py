import numpy as np

class XGBoost:
    """
    A simplified implementation of the XGBoost algorithm supporting both regression and binary classification.
    
    XGBoost (eXtreme Gradient Boosting) is an optimized distributed gradient boosting library.
    It implements machine learning algorithms under the Gradient Boosting framework.
    This implementation includes key features like:
    - Gradient-based one-side sampling
    - Regularization (L1 and L2)
    - Tree pruning using 'gamma'
    - Early stopping
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=10, reg_lambda=1.0, reg_alpha=0.0, gamma=0, 
                 objective="reg:squarederror", early_stopping_rounds=None):
        """
        Initialize XGBoost model with the following parameters:
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds (trees) to be built
        learning_rate : float
            Step size shrinkage used to prevent overfitting. Range is [0,1]
        max_depth : int
            Maximum depth of each tree
        min_samples_split : int
            Minimum number of samples required to split an internal node
        reg_lambda : float
            L2 regularization term on weights
        reg_alpha : float
            L1 regularization term on weights
        gamma : float
            Minimum loss reduction required to make a split
        objective : str
            'reg:squarederror' for regression
            'binary:logistic' for binary classification
        early_stopping_rounds : int or None
            Training stops if validation score doesn't improve after this many rounds
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.objective = objective
        self.early_stopping_rounds = early_stopping_rounds
        self.trees = []
        self.initial_prediction = None
        self.validation_error_history = []

    def _gradient_and_hessian(self, y, y_pred):
        """
        Calculate first and second order gradients of the loss function.
        
        For regression (squared error loss):
            gradient = y_pred - y
            hessian = 1
        
        For binary classification (logistic loss):
            p = sigmoid(y_pred)
            gradient = p - y
            hessian = p * (1-p)
        """
        if self.objective == "reg:squarederror":
            # For squared error loss: L = (y - y_pred)²/2
            # First derivative: dL/dy_pred = y_pred - y
            gradient = y_pred - y
            # Second derivative: d²L/dy_pred² = 1
            hessian = np.ones_like(y)
        elif self.objective == "binary:logistic":
            # Apply sigmoid function: p = 1/(1 + e^(-y_pred))
            y_pred = 1 / (1 + np.exp(-y_pred))
            # For logistic loss: L = -y*log(p) - (1-y)*log(1-p)
            # First derivative: dL/dy_pred = p - y
            gradient = y_pred - y
            # Second derivative: d²L/dy_pred² = p*(1-p)
            hessian = y_pred * (1 - y_pred)
        else:
            raise ValueError(f"Unsupported objective function: {self.objective}")
        return gradient, hessian

    def _split(self, X, y, gradients, hessians):
        """
        Find the best split for a node in the tree.
        
        The split gain formula is:
        gain = 0.5 * [GL²/(HL + λ) + GR²/(HR + λ) - (GL + GR)²/(HL + HR + λ)] - y
        
        where:
        GL, GR = sum of gradients in left/right child
        HL, HR = sum of hessians in left/right child
        λ = L2 regularization term
        y = minimum gain needed to split
        """
        best_gain = -float("inf")
        best_split = None
        n_samples, n_features = X.shape
        
        # Try splitting on each feature
        for feature in range(n_features):
            # Get unique values in the feature to try as thresholds
            thresholds = np.unique(X[:, feature])
            
            # Try each threshold
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Skip if either child would have too few samples
                # Check if split would create valid groups
                # sum() counts True values in boolean array
                if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
                    continue
                
                # Calculate sums of gradients and hessians for both children
                G_L = np.sum(gradients[left_mask])
                H_L = np.sum(hessians[left_mask])
                G_R = np.sum(gradients[right_mask])
                H_R = np.sum(hessians[right_mask])
                
                # Calculate the gain using the XGBoost split gain formula
                gain = 0.5 * (
                    G_L**2 / (H_L + self.reg_lambda) + 
                    G_R**2 / (H_R + self.reg_lambda) - 
                    (G_L + G_R)**2 / (H_L + H_R + self.reg_lambda)
                ) - self.gamma
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {"feature": feature, "threshold": threshold, "gain": gain}
        
        return best_split

    def _build_tree(self, X, y, gradients, hessians, depth=0):
        """
        Recursively build a tree using the XGBoost algorithm.
        
        The leaf value is calculated as:
        leaf_value = -sum(gradients) / (sum(hessians) + λ)
        
        where λ is the L2 regularization term.
        """

        # Check stopping conditions:
        # 1. If we've reached maximum depth
        # 2. If we don't have enough samples to split

        if depth == self.max_depth or len(y) < self.min_samples_split:
            # calculates and return leaf value
            leaf_value = -np.sum(gradients) / (np.sum(hessians) + self.reg_lambda)
            return {"leaf_value": leaf_value}
        

        # Find the best split
        split = self._split(X, y, gradients, hessians)
        
        # If no valid split found or gain too small, make a leaf
        if split is None or split["gain"] < self.gamma:
            leaf_value = -np.sum(gradients) / (np.sum(hessians) + self.reg_lambda)
            return {"leaf_value": leaf_value}

        # Split the data
        left_mask = X[:, split["feature"]] <= split["threshold"]
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        left_tree = self._build_tree(
            X[left_mask], 
            y[left_mask], 
            gradients[left_mask], 
            hessians[left_mask], 
            depth + 1
        )
        right_tree = self._build_tree(
            X[right_mask],
            y[right_mask], 
            gradients[right_mask],
            hessians[right_mask], 
            depth + 1
        )

        return {"split": split, "left": left_tree, "right": right_tree}

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the XGBoost model to training data.
        
        The algorithm:
        1. Initialize predictions with base value
        2. For each boosting round:
            a. Calculate gradients and hessians
            b. Build a tree to predict the gradients
            c. Update predictions using learning rate
            d. Check early stopping if validation data provided
        """
        # Convert targets to float for numerical stability
        y = y.astype(np.float64)
        
        # Initialize base prediction
        if self.objective == "binary:logistic":
            # For logistic regression, use log odds of mean as initial prediction
            y_mean = np.clip(y.mean(), 1e-16, 1 - 1e-16)
            self.initial_prediction = np.log(y_mean / (1 - y_mean))
        else:
            # For regression, use mean of target
            self.initial_prediction = np.mean(y)
        
        y_pred = np.full(len(y), self.initial_prediction, dtype=np.float64)

        # Main boosting loop
        for t in range(self.n_estimators):
            # Calculate gradients and hessians
            gradients, hessians = self._gradient_and_hessian(y, y_pred)
            
            # Build new tree
            tree = self._build_tree(X, y, gradients, hessians)

            # add it to the collection of trees 
            self.trees.append(tree)
            
            # Update predictions
            y_pred += self.learning_rate * self._predict_tree(tree, X)

            # Handle early stopping if validation data provided
            if X_val is not None and y_val is not None:
                y_val = y_val.astype(np.float64)
                y_val_pred = self.predict(X_val)
                val_error = self._compute_error(y_val, y_val_pred)
                self.validation_error_history.append(val_error)

                if self.early_stopping_rounds is not None:
                    if t > 0:
                        # Get minimum error in recent rounds
                        min_recent_error = min(
                            self.validation_error_history[-self.early_stopping_rounds:]
                        )
                        # If current error is worse than recent minimum, stop
                        if val_error >= min_recent_error:
                            print(f"Early stopping at iteration {t}")
                            break

    def _predict_tree(self, tree, X):
        """
        Make predictions using a single tree.
        Recursively traverse the tree based on the split conditions.
        """
        if "leaf_value" in tree:
            return np.full(X.shape[0], tree["leaf_value"])
        
        split = tree["split"]
        # Determine which samples go left
        left_mask = X[:, split["feature"]] <= split["threshold"]

        # Initialize predictions array with zeros
        predictions = np.zeros(X.shape[0])

        # Recursively get predictions for left and right subtrees
        predictions[left_mask] = self._predict_tree(tree["left"], X[left_mask])
        predictions[~left_mask] = self._predict_tree(tree["right"], X[~left_mask])
        return predictions

    def predict(self, X):
        """
        Make predictions for all samples in X using all trees
        
        Make predictions for input data X.
        
        For regression:
            final_prediction = base_prediction + learning_rate * sum(tree_predictions)
        
        For classification:
            final_prediction = sigmoid(base_prediction + learning_rate * sum(tree_predictions))
        """
        # Start with initial prediction
        y_pred = np.full(X.shape[0], self.initial_prediction)
        
        # Add contributions from all trees
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(tree, X)
        
        # Apply sigmoid for binary classification
        if self.objective == "binary:logistic":
            y_pred = 1 / (1 + np.exp(-y_pred))
        
        return y_pred

    def _compute_error(self, y_true, y_pred):
        """
        Compute the error metric based on the objective function.
        
        For regression:
            error = mean squared error = mean((y_true - y_pred)²)
        
        For binary classification:
            error = negative log likelihood = -mean(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))
        """
        if self.objective == "reg:squarederror":
            return np.mean((y_true - y_pred)**2)
        elif self.objective == "binary:logistic":
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return None

# Example usage
if __name__ == "__main__":
    # Generate sample binary classification data
    np.random.seed(42)

    X = np.random.rand(100, 2)  # 100 samples, 2 features

      # Create binary target: 1 if sum of features > 1, else 0
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  

    # Create and train model
    model = XGBoost(
        n_estimators=50,
        learning_rate=0.1,
        objective="binary:logistic",
        early_stopping_rounds=5
    )
    
    # Fit model with validation data
model.fit(X, y, X_val=X, y_val=y)

# Make predictions on our data
predictions = model.predict(X)

# Detailed prediction analysis (first 5 samples)
print("\nDetailed prediction analysis (first 5 samples):")
correct_predictions_subset = 0  # Counter for correct predictions in the subset
for i in range(5):
    predicted_class = 1 if predictions[i] > 0.5 else 0
    true_class = y[i]
    if predicted_class == true_class:
        correct_predictions_subset += 1  # Increment correct predictions for the subset
    print(f"\nSample {i + 1}:")
    print(f"Features: [{X[i, 0]:.3f}, {X[i, 1]:.3f}]")
    print(f"True class: {true_class}")
    print(f"Predicted probability: {predictions[i]:.3f}")
    print(f"Predicted class: {predicted_class}")

# Accuracy for the first 5 samples
accuracy_subset = correct_predictions_subset / 5
print(f"\nAccuracy for the first 5 samples: {accuracy_subset:.2%}")

# Let's also evaluate the model's performance for the entire dataset
correct_predictions = sum((predictions > 0.5) == y)
accuracy = correct_predictions / len(y)
print(f"\nOverall accuracy: {accuracy:.2%}")
