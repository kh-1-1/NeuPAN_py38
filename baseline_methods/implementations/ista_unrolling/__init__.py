"""
ISTA_UNROLLING - ISTA Algorithm Unrolling

Algorithm unrolling method using ISTA (Iterative Shrinkage/Thresholding Algorithm).
Implemented via DeepInverse library.

Reference: Gregor & LeCun, "Learning Fast Approximations of Sparse Coding", ICML 2010
"""

from .model import ISTAUnrolling

__all__ = ['ISTAUnrolling']

