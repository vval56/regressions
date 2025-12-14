"""
Сравнение весов и предсказаний с sklearn.linear_model.
"""
import numpy as np
from regressions import LinearRegression, LogisticRegression

print("=== Сравнение LinearRegression с sklearn ===")
np.random.seed(42)
X = np.random.randn(100, 3)
y_reg = X @ np.array([1.5, -2.0, 0.7]) + 0.5 + 0.1 * np.random.randn(100)

my_lr = LinearRegression(normalize=None, reg_type=None, learning_rate=0.1, n_iterations=5000)
my_lr.fit(X, y_reg)

try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    sk_lr = SklearnLR(fit_intercept=True)
    sk_lr.fit(X, y_reg)

    print(f"Твои веса:      {my_lr.w}")
    print(f"Sklearn веса:   {np.hstack([sk_lr.intercept_, sk_lr.coef_])}")
    print(f"Разница:        {np.abs(my_lr.w - np.hstack([sk_lr.intercept_, sk_lr.coef_]))}")

except ImportError:
    print("sklearn не установлен")

print("\n=== Сравнение LogisticRegression с sklearn ===")
y_clf = (X[:, 0] + X[:, 1] > 0).astype(int)

my_logr = LogisticRegression(normalize=None, reg_type=None, learning_rate=1.0, n_iterations=5000)
my_logr.fit(X, y_clf)

try:
    from sklearn.linear_model import LogisticRegression as SklearnLogR
    sk_logr = SklearnLogR(fit_intercept=True, C=1e10, solver='lbfgs')  
    sk_logr.fit(X, y_clf)

    print(f"Твои веса:      {my_logr.w}")
    print(f"Sklearn веса:   {np.hstack([sk_logr.intercept_, sk_logr.coef_[0]])}")
    print(f"Разница:        {np.abs(my_logr.w - np.hstack([sk_logr.intercept_, sk_logr.coef_[0]]))}")

except ImportError:
    print("sklearn не установлен")

print("\n Сравнение завершено. Разница должна быть < 1e-3.")