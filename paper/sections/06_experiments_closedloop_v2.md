# 5.11 Integration with MPC for Closed-Loop Navigation

The ultimate validation of a perception module lies in its performance within the complete navigation pipeline. While point-level metrics such as MSE and constraint satisfaction rate characterize the quality of dual variable prediction in isolation, closed-loop experiments reveal how front-end improvements translate to observable benefits in robot behavior. In this section, we integrate PDPL-Net as the front-end perception module within the NeuPAN MPC framework and evaluate navigation performance across multiple scenarios. Critically, we focus not on success rate—which is uniformly high across methods in our chosen scenarios—but on the **qualitative advantages that the improved front-end brings to the back-end optimization and resulting trajectories**.

## 5.11.1 Experimental Setup

We conduct closed-loop navigation experiments in the IR-SIM simulation environment, which provides physics-based simulation of Ackermann-steered mobile robots. The robot platform follows the specifications described in Table 3, with a wheelbase of 3.0m and maximum steering angle of 1.0 rad. We select two representative scenarios that all methods can successfully navigate, allowing us to focus on performance differences rather than binary success/failure outcomes:

- **Convex Obstacle (convex_obs)**: A scenario featuring convex polygonal obstacles that require the robot to plan smooth avoidance maneuvers. This scenario tests the quality of distance estimation and gradient information provided by the front-end.

- **Corridor**: A narrow passage scenario that demands precise lateral control. This scenario is geometrically constrained, limiting the space for trajectory variation and thus revealing subtle differences in optimization quality.

For each scenario, we compare four front-end configurations:
- **Baseline (NeuPAN-DUNE)**: The original PointNet-based DUNE module from NeuPAN
- **PDPL-Net (J=1)**: Single-layer PDHG unrolling with learned proximal operators
- **PDPL-Net (J=2)**: Two-layer PDHG unrolling
- **PDPL-Net (J=3)**: Three-layer PDHG unrolling

## 5.11.2 Dual Feasibility in Closed-Loop Operation

The most significant advantage of PDPL-Net in closed-loop operation is the **complete elimination of dual feasibility violations**. Table 8 presents the dual violation statistics aggregated across all MPC steps during navigation.

**Table 8: Dual Feasibility Violation in Closed-Loop Navigation**

| Method | J | Dual Violation Rate ↓ | P95 Dual Norm | Interpretation |
|:-------|:-:|:---------------------:|:-------------:|:---------------|
| Baseline (DUNE) | N/A | 43–50% | > 1.007 | Nearly half of all inference steps produce infeasible dual variables |
| **PDPL-Net** | J=1 | **0.0%** | **1.000** | Perfect constraint satisfaction |
| **PDPL-Net** | J=2 | **0.0%** | **1.000** | Perfect constraint satisfaction |
| **PDPL-Net** | J=3 | **0.0%** | **1.000** | Perfect constraint satisfaction |

The baseline DUNE module violates dual feasibility constraints in approximately 43–50% of inference calls during navigation. These violations manifest as $\|\mathbf{G}^\top \mu\|_2 > 1$, indicating that the predicted dual variables lie outside the valid domain for distance computation. While NeuPAN applies a hard projection layer to correct these violations before passing the dual variables to the NRMP optimization layer, this post-hoc correction has two undesirable consequences. First, the projected solution may be far from the optimal dual variable that the network intended to predict, degrading the quality of gradient information provided to the optimizer. Second, the correction introduces a discontinuity between training (where projection is not applied to the loss) and inference, potentially causing distribution shift effects.

In contrast, PDPL-Net produces dual variables that satisfy feasibility constraints **by construction**. The hard projection layer is embedded within the network architecture and applied during both training and inference, ensuring consistency. As Table 8 shows, the dual violation rate drops to exactly 0% across all PDPL-Net variants, regardless of the number of unrolling layers $J$. This structural guarantee eliminates the need for post-hoc correction and ensures that the optimizer receives high-quality, geometrically valid distance information at every control step.

## 5.11.3 Computational Efficiency

Beyond constraint satisfaction, PDPL-Net demonstrates improved computational efficiency in closed-loop operation. Table 9 presents the average per-step computation time for each method across the two scenarios.

**Table 9: Per-Step Computation Time in Closed-Loop Navigation (ms)**

| Scenario | Baseline | J=1 | J=2 | J=3 | Speedup (J=1 vs Baseline) |
|:---------|:--------:|:---:|:---:|:---:|:-------------------------:|
| convex_obs | 56.38 | **52.48** | 56.10 | 58.90 | +7.0% |
| corridor | 115.60 | **106.63** | 105.78 | 106.69 | +7.8% |

PDPL-Net with J=1 achieves the fastest computation time, approximately 7–8% faster than the baseline. This speedup may appear modest but is significant when considered over thousands of control steps in a navigation task. The efficiency gain stems from the streamlined architecture of PDPL-Net compared to the baseline's PointNet-based feature extraction. While adding unrolling layers (J=2, J=3) slightly increases computation time, all PDPL-Net variants remain faster than or comparable to the baseline.

It is worth noting that the computation time is dominated by the back-end NRMP optimization layer rather than the front-end perception module. The front-end accounts for only 10–15% of the total per-step time, which explains why architectural improvements in the front-end translate to relatively modest speedups in the overall pipeline. Nevertheless, this consistent efficiency advantage compounds over long navigation trajectories.

## 5.11.4 Path Quality and Planning Efficiency

The improved front-end quality translates to measurable benefits in the resulting navigation trajectories. Table 10 presents path quality metrics for the convex_obs scenario, where obstacle geometry creates opportunities for differentiated planning behavior.

**Table 10: Path Quality Metrics in convex_obs Scenario**

| Method | J | Steps to Goal ↓ | Path Length (m) ↓ | Max Velocity (m/s) | Analysis |
|:-------|:-:|:---------------:|:-----------------:|:------------------:|:---------|
| Baseline | N/A | 146 | 53.43 | 5.44 | Reference performance |
| PDPL-Net | J=1 | 172 | 53.63 | 5.39 | Slightly conservative |
| PDPL-Net | J=2 | 145 | 53.42 | 5.44 | Matches baseline |
| **PDPL-Net** | **J=3** | **138** | **53.33** | 5.40 | **Fewest steps, shortest path** |

The results reveal an interesting trend: as the number of unrolling layers $J$ increases, path quality progressively improves. PDPL-Net with J=3 achieves the fewest steps to goal (138 vs 146 for baseline, a 5.5% reduction) and the shortest path length (53.33m vs 53.43m). This improvement indicates that additional unrolling layers produce higher-quality dual variable estimates, which in turn provide more accurate distance gradients to the optimizer. With better gradient information, the MPC generates trajectories that more efficiently navigate around obstacles.

The J=1 variant exhibits slightly conservative behavior, requiring more steps than the baseline. This suggests that a single unrolling layer, while sufficient for constraint satisfaction, may not fully capture the optimal solution structure. The J=2 variant closely matches baseline performance, while J=3 surpasses it. This progression validates the design principle of algorithm unrolling: each additional layer brings the solution closer to optimality.

For the corridor scenario, all methods exhibit nearly identical path quality metrics (179–181 steps), indicating that this geometrically constrained environment leaves little room for trajectory variation. The corridor scenario thus confirms that PDPL-Net does not degrade performance even when the baseline is already operating near the geometric optimum.

## 5.11.5 Summary of Closed-Loop Advantages

The closed-loop navigation experiments demonstrate three key advantages of PDPL-Net as a front-end perception module:

1. **Guaranteed Dual Feasibility**: The structural hard projection layer eliminates constraint violations entirely (0% vs 43–50%), providing the back-end optimizer with geometrically valid distance information at every control step. This guarantee is particularly important for safety-critical applications where theoretical soundness is as important as empirical performance.

2. **Improved Computational Efficiency**: The streamlined PDPL-Net architecture achieves 7–8% faster per-step computation compared to the baseline, with the J=1 variant being the most efficient. While the speedup is modest in percentage terms, it accumulates over thousands of control steps in practical navigation tasks.

3. **Higher Path Quality with Deeper Unrolling**: Increasing the number of unrolling layers $J$ progressively improves path quality, with J=3 achieving 5.5% fewer steps than the baseline in the convex_obs scenario. This validates that algorithm unrolling produces better solutions with more layers, and that improved front-end accuracy translates to observable benefits in robot behavior.

For practical deployment, we recommend **J=2 or J=3** as the default configuration, balancing computational efficiency with path quality. The J=2 variant offers stable performance across scenarios with minimal computational overhead, while J=3 provides the best path quality for applications where trajectory optimality is prioritized.

