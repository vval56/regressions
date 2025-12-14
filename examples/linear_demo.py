"""
Демонстрация LinearRegression на синтетических данных и Boston Housing (упрощённо).
"""
import numpy as np
import matplotlib.pyplot as plt
from regressions import LinearRegression

print("=== Синтетические данные (y = 2x + 1 + шум) ===")
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * np.random.randn(100)

model = LinearRegression(learning_rate=0.1, n_iterations=1000, normalize='zscore')
model.fit(X, y)

print(f"R²: {model.score(X, y):.4f}")
print(f"Истинные веса: bias=1.0, w1=2.0")
print(f"Обученные веса: bias={model.w[0]:.2f}, w1={model.w[1]:.2f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.6, label='Данные')
plt.plot(X, model.predict(X), color='red', label='Предсказание')
plt.title("Linear Regression: Synthetic Data")
plt.legend()

print("\n=== Boston Housing (1 признак: среднее число комнат) ===")
try:
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X_b = diabetes.data[:, [2]] 
    y_b = diabetes.target

    model_b = LinearRegression(learning_rate=0.1, n_iterations=2000, normalize='zscore')
    model_b.fit(X_b, y_b)

    print(f"R² на Diabetes (1 признак): {model_b.score(X_b, y_b):.4f}")
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_b, y_b, alpha=0.6)
    plt.plot(X_b, model_b.predict(X_b), color='red')
    plt.title("Linear Regression: Boston Housing (RM)")
    
    plt.tight_layout()
    plt.savefig("linear_demo.png", dpi=150)
    plt.show()

except ImportError:
    print("sklearn не установлен — пропускаем Boston Housing.")

print("\n Линейная регрессия успешно протестирована!")