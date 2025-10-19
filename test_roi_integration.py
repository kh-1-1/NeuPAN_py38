"""
Test script to verify ROI integration
Tests:
1. ROI disabled (default) - should not affect existing behavior
2. ROI enabled - should filter points and log statistics
"""

import numpy as np
import yaml
from neupan import neupan

def test_roi_disabled():
    """Test that ROI disabled does not affect existing behavior"""
    print("\n=== Test 1: ROI Disabled (Default) ===")
    
    # Load planner with ROI disabled
    planner = neupan.init_from_yaml('example/LON/planner.yaml')
    
    # Verify ROI is disabled
    assert not planner.roi_enabled, "ROI should be disabled by default"
    print("✓ ROI is disabled")
    
    # Create dummy state and points
    state = np.array([[0.0], [20.0], [0.0]])  # x, y, theta
    points = np.random.randn(2, 100) * 5 + np.array([[5.0], [20.0]])  # random points around robot
    
    # Run forward pass
    action, info = planner.forward(state, points)
    
    # Verify no ROI info in output
    assert 'roi' not in info, "ROI info should not be present when disabled"
    print("✓ No ROI info in output")
    print(f"✓ Action shape: {action.shape}")
    print(f"✓ Info keys: {list(info.keys())}")
    
    print("✓ Test 1 PASSED: ROI disabled works correctly\n")


def test_roi_enabled():
    """Test that ROI enabled filters points and logs statistics"""
    print("\n=== Test 2: ROI Enabled ===")

    # Load planner normally first
    planner = neupan.init_from_yaml('example/LON/planner.yaml')

    # Manually enable ROI for testing
    planner.roi_enabled = True
    from neupan.blocks.roi_selector import ROISelector, ROIConfig
    roi_cfg = ROIConfig(enabled=True)
    planner.roi_selector = ROISelector(roi_cfg)
    
    # Verify ROI is enabled
    assert planner.roi_enabled, "ROI should be enabled"
    print("✓ ROI is enabled")
    
    # Create dummy state and points
    state = np.array([[0.0], [20.0], [0.0]])
    points = np.random.randn(2, 200) * 10 + np.array([[5.0], [20.0]])
    
    # Run forward pass
    action, info = planner.forward(state, points)
    
    # Verify ROI info is present
    assert 'roi' in info, "ROI info should be present when enabled"
    print("✓ ROI info present in output")
    
    # Check ROI statistics
    roi_info = info['roi']
    assert 'strategy' in roi_info, "ROI strategy should be logged"
    assert 'n_in' in roi_info, "Input point count should be logged"
    assert 'n_roi' in roi_info, "Output point count should be logged"
    
    print(f"✓ ROI Strategy: {roi_info['strategy']}")
    print(f"✓ Input points: {roi_info['n_in']}")
    print(f"✓ Filtered points: {roi_info['n_roi']}")
    print(f"✓ Compression ratio: {roi_info['n_roi'] / max(roi_info['n_in'], 1):.2%}")
    
    # Verify action is still valid
    assert action.shape == (2, 1), "Action shape should be (2, 1)"
    print(f"✓ Action shape: {action.shape}")
    
    print("✓ Test 2 PASSED: ROI enabled works correctly\n")


def test_roi_strategies():
    """Test different ROI strategies"""
    print("\n=== Test 3: ROI Strategies ===")
    
    # Test wedge strategy (should always work with heading)
    print("\n--- Testing Wedge Strategy ---")
    config = {
        'receding': 10,
        'step_time': 0.1,
        'ref_speed': 4.0,
        'device': 'cpu',
        'robot_kwargs': {'kinematics': 'diff', 'max_speed': [8, 1], 'max_acce': [8, 3], 'length': 1.6, 'width': 2.0},
        'ipath_kwargs': {'waypoints': [[0, 20, 0], [75, 20, 0]], 'curve_style': 'line'},
        'pan_kwargs': {'iter_num': 2, 'dune_max_num': 100, 'nrmp_max_num': 10, 'dune_checkpoint': None},
        'adjust_kwargs': {'q_s': 0.1, 'p_u': 2.0, 'eta': 10, 'd_max': 0.2, 'd_min': 0.01},
        'train_kwargs': {},
        'roi_kwargs': {
            'enabled': True,
            'strategy_order': ['wedge'],  # Only wedge
            'wedge': {'fov_deg': 60.0, 'r_max_m': 8.0},
            'guardrail': {'n_min': 10, 'n_max': 500},
        }
    }
    
    planner = neupan(**config)
    state = np.array([[0.0], [20.0], [0.0]])
    
    # Create points in front and behind
    points_front = np.random.randn(2, 50) * 2 + np.array([[5.0], [20.0]])  # in front
    points_behind = np.random.randn(2, 50) * 2 + np.array([[-5.0], [20.0]])  # behind
    points = np.hstack([points_front, points_behind])
    
    action, info = planner.forward(state, points)
    
    assert info['roi']['strategy'] == 'wedge', "Should use wedge strategy"
    print(f"✓ Wedge strategy selected")
    print(f"✓ Filtered {info['roi']['n_in']} → {info['roi']['n_roi']} points")
    
    print("✓ Test 3 PASSED: ROI strategies work correctly\n")


if __name__ == '__main__':
    print("=" * 60)
    print("ROI Integration Test Suite")
    print("=" * 60)
    
    try:
        test_roi_disabled()
        test_roi_enabled()
        test_roi_strategies()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

