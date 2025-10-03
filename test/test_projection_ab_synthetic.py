import torch
import numpy as np

from neupan.blocks.pan import PAN
from neupan.robot.robot import robot as Robot
from neupan.configuration import to_device
import os


def run_once(projection: str = 'hard', n_points: int = 512, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Construct a simple robot and PAN with given projection setting
    rob = Robot(kinematics='acker', length=4.6, width=1.6, wheelbase=3)

    ckpt = 'example/model/acker_robot_default/model_5000.pth'
    has_ckpt = os.path.exists(ckpt)

    pan = PAN(
        receding=10,
        step_time=0.2,
        robot=rob,
        dune_max_num=n_points,
        nrmp_max_num=10,
        dune_checkpoint=ckpt if has_ckpt else None,
        adjust_kwargs=dict(eta=15.0, d_max=1.0, d_min=0.1),
        train_kwargs=dict(
            projection=projection,
            monitor_dual_norm=True,
            unroll_J=0,
            se2_embed=False,
            direct_train=not has_ckpt,  # allow running without checkpoint
        ),
    )

    # Synthetic obs points and nominal states
    obs = to_device(torch.randn(2, n_points) * 10.0)
    nom_s = to_device(torch.zeros(3, 11))

    pf, Rl, opl = pan.generate_point_flow(nom_s, obs)

    # Call DUNE directly to avoid running the full PAN forward loop
    mu_list, lam_list, _ = pan.dune_layer(pf, Rl, opl)

    dl = pan.dune_layer
    pre_violation = getattr(dl, 'dual_norm_violation_rate', None)
    pre_p95 = getattr(dl, 'dual_norm_p95', None)
    pre_excess = getattr(dl, 'dual_norm_max_excess_pre', None)
    post_excess = getattr(dl, 'dual_norm_max_excess_post', None)

    post_max_norm = 0.0
    for lam in lam_list:
        if lam is None or lam.numel() == 0:
            continue
        # Since R is orthonormal, ||lam|| = ||G^T mu|| after projection
        post_max_norm = max(post_max_norm, torch.norm(lam, dim=0).max().item())

    return pre_violation, pre_p95, post_max_norm, pre_excess, post_excess


def main():
    print('Synthetic A/B test for DUNE projection')

    num_trials = 5

    for proj in ['hard', 'none']:
        records = []
        print(f'projection={proj}')
        for trial in range(num_trials):
            pre_v, pre_p95, post_mx, pre_exc, post_exc = run_once(
                projection=proj,
                n_points=512,
                seed=trial,
            )
            records.append((pre_v, pre_p95, post_mx, pre_exc, post_exc))
            print(f'  trial {trial:02d} | pre-viol {pre_v} | pre-p95 {pre_p95} | post-max {post_mx} | pre-excess {pre_exc} | post-excess {post_exc}')

        valid_pre_exc = [r[3] for r in records if r[3] is not None]
        valid_post_exc = [r[4] for r in records if r[4] is not None]
        avg_pre_exc = sum(valid_pre_exc) / len(valid_pre_exc) if valid_pre_exc else 0.0
        avg_post_exc = sum(valid_post_exc) / len(valid_post_exc) if valid_post_exc else 0.0

        print(f'  avg pre-proj excess: {avg_pre_exc}')
        print(f'  avg post-proj excess: {avg_post_exc}')

    print('\nExpectation:')
    print('- With projection=hard, post-proj max_norm should be <= 1.0')
    print('- With projection=none, post-proj max_norm may be > 1.0')


if __name__ == '__main__':
    main()
