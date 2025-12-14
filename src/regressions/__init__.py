"""
Regresions — реализация линейной и логистической регрессии на чистом NumPy.
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression

__version__ = "0.1.0"

__all__ = [
    "LinearRegression",
    "LogisticRegression",
]