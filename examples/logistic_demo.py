"""
Демонстрация LogisticRegression на синтетических данных и Iris (бинарная задача).
"""
import numpy as np
import matplotlib.pyplot as plt
from regressions import LogisticRegression

print("=== Синтетические данные (бинарная классификация) ===")
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  

model = LogisticRegression(learning_rate=1.0, n_iterations=1000, normalize='zscore')
model.fit(X, y)

print(f"Accuracy: {model.score(X, y):.4f}")
print(f"Веса: {model.w}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7)
plt.title("Logistic Regression: Synthetic Data")

print("\n=== Iris Dataset (setosa vs versicolor) ===")
try:
    from sklearn.datasets import load_iris
    iris = load_iris()
    idx = iris.target != 2 
    X_iris = iris.data[idx, :2] 
    y_iris = iris.target[idx]

    model_iris = LogisticRegression(learning_rate=1.0, n_iterations=1000, normalize='zscore')
    model_iris.fit(X_iris, y_iris)

    print(f"Accuracy на Iris: {model_iris.score(X_iris, y_iris):.4f}")
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, cmap='coolwarm', alpha=0.7)
    plt.title("Logistic Regression: Iris (setosa vs versicolor)")
    plt.legend(*scatter.legend_elements(), title="Classes")

    plt.tight_layout()
    plt.savefig("logistic_demo.png", dpi=150)
    plt.show()

except ImportError:
    print("sklearn не установлен — пропускаем Iris.")

print("\n Логистическая регрессия успешно протестирована!")