"""Tropical (max-plus) attention assignment — VAJRA's allocation primitive.

Linear assignment in the tropical semiring (R, ⊕=max, ⊗=+):

    maximise   Σ C[i, j] X[i, j]
    s.t.       Σ_j X[i, j] ≤ 1   (each friendly bids on ≤1 hostile)
               Σ_i X[i, j] ≤ 1   (each hostile receives ≤1 striker)
               X ∈ {0, 1}^{N×M}

Two solvers are exposed:

* `tropical_attention_assignment(C, beta)` — softmax-relaxed tropical attention.
  Pure NumPy, O(NM) per iteration. Acts as the differentiable / GPU-friendly
  allocator (~8 ms for 50×30 per VAJRA spec, vs ~240 ms for Hungarian).
  Used at runtime in the live console.

* `hungarian_assignment(C)` — exact reference implementation via SciPy.
  Used to validate tropical solver in tests and as the fallback when N+M ≤ 8.

The tropical solver returns the same kind of `{friendly_id: hostile_id}` map
SHIELD's greedy auction does, so it's a drop-in replacement.
"""
from __future__ import annotations

import numpy as np


def hungarian_assignment(cost_matrix: np.ndarray) -> dict[int, int]:
    """Reference assignment via scipy.optimize.linear_sum_assignment.

    Maximises the score (negates internally to match SciPy's minimise API).
    Skips pairs whose cost is non-positive — no friendly should "win" a
    hostile if its priority is zero (e.g. kill-switched bidder or hostile
    classified as decoy with all kinetics already covered).
    """
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return {}
    # Pad to square so SciPy doesn't choke; padded cells get 0 cost.
    n, m = cost_matrix.shape
    k = max(n, m)
    padded = np.zeros((k, k), dtype=np.float64)
    padded[:n, :m] = cost_matrix
    rows, cols = linear_sum_assignment(-padded)
    assignment: dict[int, int] = {}
    for r, c in zip(rows, cols, strict=True):
        if r >= n or c >= m:
            continue
        if cost_matrix[r, c] <= 0.0:
            continue
        assignment[int(r)] = int(c)
    return assignment


def tropical_attention_assignment(
    cost_matrix: np.ndarray,
    beta: float = 8.0,
    iters: int = 6,
    cap_per_friendly: int = 1,
) -> dict[int, int]:
    """Tropical-attention assignment via Sinkhorn-style soft-max iteration.

    The (max, +) semiring assignment problem is approximated by a temperature-
    sharpened doubly-stochastic projection. As beta → ∞ the soft assignment
    converges to the exact tropical (LP-relaxation) optimum, which for an
    integer-feasible cost matrix matches the Hungarian assignment.

    Args
    ----
    cost_matrix : (N, M) array of bid scores (higher = better match).
    beta        : sharpness. 4–16 trades off smoothness vs concentration.
    iters       : Sinkhorn iterations. 6 is plenty for N ≤ 64.
    cap_per_friendly : reserved for HSL-CC concurrency (currently always 1).

    Returns
    -------
    dict {friendly_idx → hostile_idx} — winners. Pairs with priority ≤ 0
    are dropped (a kill-switched striker shouldn't claim anything).
    """
    if cost_matrix.size == 0:
        return {}
    C = np.asarray(cost_matrix, dtype=np.float64)
    n, m = C.shape

    # Sharpen and exponentiate to enter log-space probabilistic form.
    # Subtract row max for numerical stability.
    logits = beta * (C - C.max())
    P = np.exp(logits)

    # Sinkhorn-style alternating normalisation toward a (≤1 row, ≤1 col)
    # constraint set. For a rectangular matrix, normalising to row-sum=1
    # and col-sum=1 gives the LP-relaxation of the assignment.
    for _ in range(iters):
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        P = P / row_sums
        col_sums = P.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums > 1e-12, col_sums, 1.0)
        P = P / col_sums

    # Greedy decode of the doubly-stochastic-ish matrix: repeatedly pick
    # the highest remaining (i, j) until exhaustion. With sharp beta the
    # picks coincide with the LP-optimal assignment.
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    assignment: dict[int, int] = {}
    flat = np.argsort(P.flatten())[::-1]
    for idx in flat:
        i, j = divmod(int(idx), m)
        if i in used_rows or j in used_cols:
            continue
        if C[i, j] <= 0.0:
            continue
        assignment[i] = j
        used_rows.add(i)
        used_cols.add(j)
        if len(used_rows) >= n or len(used_cols) >= m:
            break
    return assignment


def assignment_score(cost_matrix: np.ndarray, assignment: dict[int, int]) -> float:
    """Sum of selected cost-matrix entries — used by tests to compare solvers."""
    if not assignment:
        return 0.0
    return float(sum(cost_matrix[i, j] for i, j in assignment.items()))
