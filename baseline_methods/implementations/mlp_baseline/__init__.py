"""
MLP Baseline - Standard Multi-Layer Perceptron

A simple black-box neural network baseline that maps point clouds
directly to dual variables without any structural constraints.
"""

from .model import MLPBaseline

__all__ = ['MLPBaseline']

