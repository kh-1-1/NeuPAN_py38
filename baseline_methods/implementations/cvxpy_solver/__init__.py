"""
CVXPY Solver - Ground Truth Baseline

This module implements a convex optimization solver using CVXPY for solving
the dual variable prediction problem. It serves as a ground truth baseline
for comparing other methods.

The solver solves the following optimization problem:
    maximize: mu^T (G p - h)
    subject to: mu >= 0
                ||G^T @ mu||_2 <= 1
"""

from .solver import CVXPYSolver

__all__ = ['CVXPYSolver']

