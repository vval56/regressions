import numpy as np
import pytest
from regressions import LogisticRegression

def test_logistic_perfect_separation():
    """Тест: идеальное разделение — accuracy = 1.0."""
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=1.0, n_iterations=1000, normalize=None)
    model.fit(X, y)

    assert model.score(X, y) == 1.0
    proba = model.predict_proba(X)
    assert proba[0] < 0.1
    assert proba[-1] > 0.9


def test_logistic_predict_proba_bounds():
    """Тест: вероятности в [0, 1]."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    model = LogisticRegression(learning_rate=1.0, n_iterations=1000, normalize=None)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_logistic_multi_features():
    """Тест: логистическая регрессия с 2+ признаками."""
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=1.0, n_iterations=1000, normalize=None)
    model.fit(X, y)

    assert model.score(X, y) == 1.0


def test_logistic_save_load():
    """Тест: сохранение и загрузка модели."""
    import os
    import uuid

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1])

    model1 = LogisticRegression(learning_rate=1.0, n_iterations=1000, normalize=None)
    model1.fit(X, y)

    tmp_path = f"/tmp/test_model_{uuid.uuid4().hex}.npz"
    try:
        model1.save(tmp_path)
        model2 = LogisticRegression.load(tmp_path)
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)
        assert np.array_equal(y_pred1, y_pred2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)