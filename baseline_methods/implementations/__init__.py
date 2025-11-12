"""
Baseline Methods Implementations for PDPL-Net

This package contains implementations of 12 baseline methods for comparison with PDPL-Net.

All methods follow a unified interface:
- Input: point_cloud (N, 2) - point cloud coordinates
- Output: mu (E, N), lam (3, N) - dual variables

Methods:
1. CVXPY_SOLVER - Convex optimization solver (ground truth)
2. MLP_BASELINE - Standard multi-layer perceptron
3. CENTER_DISTANCE_MPC - Center point distance method
4. POINT_TRANSFORMER_V3 - Point Transformer V3 (CVPR 2024)
5. POINTNET_PLUSPLUS - PointNet++ (NeurIPS 2017)
6. CVXPYLAYERS_SOLVER - Differentiable convex optimization
7. ISTA_UNROLLING - ISTA algorithm unrolling (via DeepInverse)
8. ADMM_UNROLLING - ADMM algorithm unrolling (via DeepInverse)
9. FISTA_UNROLLING - FISTA algorithm unrolling (via DeepInverse)
10. PHYSICS_INFORMED_PROJECTION - Physics-informed hard projection
11. ESDF_MPC - ESDF-based MPC (optional)
12. NEUPAN - NeuPAN baseline (existing implementation)
"""

__version__ = "1.0.0"
__author__ = "PDPL-Net Team"

# Import all baseline methods
try:
    from .cvxpy_solver import CVXPYSolver
except ImportError:
    CVXPYSolver = None

try:
    from .mlp_baseline import MLPBaseline
except ImportError:
    MLPBaseline = None

try:
    from .center_distance_mpc import CenterDistanceMPC
except ImportError:
    CenterDistanceMPC = None

try:
    from .point_transformer_v3 import PointTransformerV3
except ImportError:
    PointTransformerV3 = None

try:
    from .pointnet_plusplus import PointNetPlusPlus
except ImportError:
    PointNetPlusPlus = None

try:
    from .cvxpylayers import CvxpyLayersSolver
except ImportError:
    CvxpyLayersSolver = None

try:
    from .ista_unrolling import ISTAUnrolling
except ImportError:
    ISTAUnrolling = None

try:
    from .admm_unrolling import ADMMUnrolling
except ImportError:
    ADMMUnrolling = None

try:
    from .deepinverse import DeepInverseUnrolling
except ImportError:
    DeepInverseUnrolling = None

try:
    from .esdf_mpc import ESDFMPCSolver
except ImportError:
    ESDFMPCSolver = None

try:
    from .physics_informed_hard_proj import PhysicsInformedHardProj
except ImportError:
    PhysicsInformedHardProj = None

__all__ = [
    'CVXPYSolver',
    'MLPBaseline',
    'CenterDistanceMPC',
    'PointTransformerV3',
    'PointNetPlusPlus',
    'CvxpyLayersSolver',
    'ISTAUnrolling',
    'ADMMUnrolling',
    'DeepInverseUnrolling',
    'ESDFMPCSolver',
    'PhysicsInformedHardProj',
]

