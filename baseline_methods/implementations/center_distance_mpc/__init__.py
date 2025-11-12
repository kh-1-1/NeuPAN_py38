"""
CENTER_DISTANCE_MPC - 中心点距离方法

A traditional method that uses the center point distance to approximate
the dual variables for MPC-based navigation.
"""

from .solver import CenterDistanceMPC

__all__ = ['CenterDistanceMPC']

