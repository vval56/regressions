import numpy as np
import pytest
from regressions import LinearRegression

def test_linear_exact_fit_no_noise():
    """Тест: точное восстановление линейной зависимости без шума."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2.0 * X.flatten() + 1.0  # y = 2x + 1

    model = LinearRegression(learning_rate=0.1, n_iterations=5000, normalize=None)
    model.fit(X, y)

    assert model.score(X, y) > 0.9999
    assert abs(model.w[0] - 1.0) < 1e-2  # bias
    assert abs(model.w[1] - 2.0) < 1e-2  # вес признака


def test_linear_with_noise():
    """Тест: работа с шумом — R² должен быть высоким, но не 1.0."""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 1.5 * X.flatten() - 0.5 + 0.1 * np.random.randn(100)

    model = LinearRegression(learning_rate=0.1, n_iterations=2000, normalize='zscore')
    model.fit(X, y)

    assert 0.9 < model.score(X, y) < 1.0


def test_linear_multi_features():
    """Тест: несколько признаков."""
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
    y = X[:, 0] * 1.0 + X[:, 1] * (-2.0) + 0.5  # y = x1 - 2*x2 + 0.5

    model = LinearRegression(learning_rate=0.1, n_iterations=5000, normalize='zscore')
    model.fit(X, y)

    assert model.score(X, y) > 0.999


def test_linear_save_load():
    """Тест: сохранение и загрузка модели."""
    import os
    import uuid

    X = np.array([[1.0], [2.0], [3.0]])
    y = 3.0 * X.flatten() + 2.0

    model1 = LinearRegression(learning_rate=0.1, n_iterations=3000, normalize=None)
    model1.fit(X, y)

    tmp_path = f"/tmp/test_linear_{uuid.uuid4().hex}.npz"
    try:
        model1.save(tmp_path)
        model2 = LinearRegression.load(tmp_path)
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)
        assert np.allclose(y_pred1, y_pred2, atol=1e-6)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
