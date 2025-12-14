import numpy as np
from typing import Union

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000, normalize='zscore', tol=1e-6, patience=10,reg_type = None, reg_strength = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.mean_ = None   
        self.std_ = None 
        self.normalize = normalize
        self.tol = tol
        self.patience = patience
        self.reg_type = reg_type
        if self.reg_type not in (None, 'l2'):
            raise ValueError("reg_type must be None or 'l2'")
        
        self.reg_strength = reg_strength
        self._is_fitted = False

    def fit(self, X: Union[list, np.ndarray], y: Union[list, np.ndarray]):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")

        if self.normalize == 'zscore':
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_ = np.where(self.std_ == 0, 1, self.std_)
            X_norm = (X - self.mean_) / self.std_
        elif self.normalize == 'minmax':
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            range_ = self.max_ - self.min_
            range_ = np.where(range_ == 0, 1, range_)
            X_norm = (X - self.min_) / range_
        else:
            X_norm = X.copy()
            self.mean_ = self.std_ = self.min_ = self.max_ = None
        
        X_with_bias = np.hstack([np.ones((n_samples, 1)), X_norm])
        self.w = np.zeros(n_features + 1)

        best_loss = float('inf')
        patience_counter = 0

        for i in range(self.n_iterations):
            z = X_with_bias @ self.w              
            y_pred_proba = sigmoid(z)            

            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15) 
            loss = -np.mean(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))

            if loss < best_loss - self.tol:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at iteration {i}, Log Loss: {loss:.6f}")
                    break

            error = y_pred_proba - y 
            dw = (1.0 / n_samples) * X_with_bias.T @ error

            if self.reg_type == 'l2':
                reg_grad = np.zeros_like(self.w)
                reg_grad[1:] = self.w[1:] 
                dw += 2 * self.reg_strength * reg_grad

            self.w -= self.learning_rate * dw

            if i % 1000 == 0:
                print(f"Iteration {i}, Log Loss: {loss:.6f}")

        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("Call fit() before predict()")
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]

        if self.normalize == 'zscore':
            if self.mean_ is None or self.std_ is None:
                raise ValueError("Model not fitted or normalization not applied.")
            X_norm = (X - self.mean_) / self.std_
        elif self.normalize == 'minmax':
            if self.min_ is None or self.max_ is None:
                raise ValueError("Model not fitted or normalization not applied.")
            range_ = self.max_ - self.min_
            range_ = np.where(range_ == 0, 1, range_)
            X_norm = (X - self.min_) / range_
        else:
            X_norm = X.copy()

        X_with_bias = np.hstack([np.ones((n_samples, 1)), X_norm])
        z = X_with_bias @ self.w
        y_proba = sigmoid(z)
        return (y_proba >= 0.5).astype(int)  
    
    def predict_proba(self, X):
        if not self._is_fitted:
            raise ValueError("Call fit() before predict_proba()")
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]

        if self.normalize == 'zscore':
            if self.mean_ is None or self.std_ is None:
                raise ValueError("Model not fitted or normalization not applied.")
            X_norm = (X - self.mean_) / self.std_
        elif self.normalize == 'minmax':
            if self.min_ is None or self.max_ is None:
                raise ValueError("Model not fitted or normalization not applied.")
            range_ = self.max_ - self.min_
            range_ = np.where(range_ == 0, 1, range_)
            X_norm = (X - self.min_) / range_
        else:
            X_norm = X.copy()

        X_with_bias = np.hstack([np.ones((n_samples, 1)), X_norm])
        z = X_with_bias @ self.w
        return sigmoid(z)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) 
    
    def save(self, path):
        np.savez(
            path,
            w=self.w,
            mean_=self.mean_,
            std_=self.std_,
            min_=self.min_ if hasattr(self, 'min_') else None,
            max_=self.max_ if hasattr(self, 'max_') else None,
            normalize=self.normalize,
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        model = cls(
            learning_rate=float(data['learning_rate']),
            n_iterations=int(data['n_iterations']),
            normalize=str(data['normalize'])
        )
        model.w = data['w']
        model.mean_ = data['mean_'] if 'mean_' in data else None
        model.std_ = data['std_'] if 'std_' in data else None
        model.min_ = data['min_'] if 'min_' in data else None
        model.max_ = data['max_'] if 'max_' in data else None
        model._is_fitted = True  
        return model
