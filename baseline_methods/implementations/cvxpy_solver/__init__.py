"""
CVXPY Solver - Ground Truth Baseline

This module implements a convex optimization solver using CVXPY for solving
the dual variable prediction problem. It serves as a ground truth baseline
for comparing other methods.

The solver solves the following optimization problem:
    minimize: ||mu||^2 + ||lambda||^2
    subject to: mu >= 0
                ||G^T @ mu||_2 <= 1
                distance_constraint(lambda, point_cloud)
"""

from .solver import CVXPYSolver

__all__ = ['CVXPYSolver']

