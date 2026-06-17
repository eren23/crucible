"""Tiny scorable baseline for the ERA-style code-mutation demo.

Fits a sum-of-basis-functions model to a noisy
``sin(2x) + 0.3·cos(5x)`` target in closed form (no SGD —
deliberately, so the score is stable and reproducible).

Mutation surface:
- ``ACTIVATION`` — basis function family: ``"relu"`` / ``"gelu"`` /
  ``"tanh"`` / ``"sigmoid"``. Each kernel shapes the basis.
- ``LEARNING_RATE`` — repurposed as the basis-function frequency
  multiplier; ``0.5`` is too narrow for the 2-frequency target.
- ``HIDDEN_DIM`` — number of basis functions. Underfit at 4,
  saturated by 32.

The unmutated baseline is intentionally suboptimal so simple
mutations show a measurable score lift.
"""
from __future__ import annotations

import math
import random

ACTIVATION = "relu"
LEARNING_RATE = 0.5
HIDDEN_DIM = 4
SEED = 17
N_POINTS = 256


def _activate(x: float, kind: str) -> float:
    if kind == "relu":
        return max(0.0, x)
    if kind == "gelu":
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    if kind == "tanh":
        return math.tanh(x)
    if kind == "sigmoid":
        return 1.0 / (1.0 + math.exp(-x))
    raise ValueError(f"unknown activation: {kind!r}")


def _truth(x: float) -> float:
    return math.sin(2.0 * x) + 0.3 * math.cos(5.0 * x)


def _gen_data(n: int = N_POINTS) -> list[tuple[float, float]]:
    rng = random.Random(SEED)
    return [
        (
            (x := rng.uniform(-3.14, 3.14)),
            _truth(x) + rng.gauss(0.0, 0.05),
        )
        for _ in range(n)
    ]


def _solve_least_squares(matrix: list[list[float]], targets: list[float]) -> list[float]:
    """Tiny normal-equations least-squares: (X^T X)^-1 X^T y. Pure Python."""
    n_features = len(matrix[0])
    # X^T X
    xtx = [[0.0] * n_features for _ in range(n_features)]
    for row in matrix:
        for i in range(n_features):
            for j in range(n_features):
                xtx[i][j] += row[i] * row[j]
    # X^T y
    xty = [0.0] * n_features
    for row, t in zip(matrix, targets):
        for i in range(n_features):
            xty[i] += row[i] * t
    # Ridge regularisation so the solve is stable even when basis
    # functions are nearly collinear (e.g. low HIDDEN_DIM + flat
    # activation collapses to rank-deficient).
    for i in range(n_features):
        xtx[i][i] += 1e-3
    return _gauss_solve(xtx, xty)


def _gauss_solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        # Partial pivot
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return [0.0] * n
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def train_and_score() -> float:
    """Fit the basis model in closed form and return validation MSE."""
    data = _gen_data()
    split = int(0.8 * len(data))
    train, val = data[:split], data[split:]

    # Basis centers evenly spaced over the data range.
    centers = [-3.14 + (i + 0.5) * (6.28 / HIDDEN_DIM) for i in range(HIDDEN_DIM)]

    def _features(x: float) -> list[float]:
        # LEARNING_RATE doubles as the basis frequency scale.
        return [_activate(LEARNING_RATE * (x - c), ACTIVATION) for c in centers]

    matrix = [_features(x) for x, _ in train]
    targets = [y for _, y in train]
    weights = _solve_least_squares(matrix, targets)

    sq = 0.0
    for x, y in val:
        pred = sum(w * f for w, f in zip(weights, _features(x)))
        sq += (pred - y) ** 2
    return sq / len(val)


if __name__ == "__main__":
    score = train_and_score()
    print(f"val_bpb:{score:.6f}")
