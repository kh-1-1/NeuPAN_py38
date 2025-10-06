"""
DUNE (Deep Unfolded Neural Encoder) is the core class of the PAN class. It maps the point flow to the latent distance space: mu and lambda. 

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
"""


import torch
import os
from math import inf
from neupan.blocks import ObsPointNet, DUNETrain
from neupan.blocks.learned_prox import ProxHead
from neupan.blocks.pdhg_unroll import PDHGUnroll, PDHGUnrollPerStep
from neupan.configuration import np_to_tensor, to_device
from neupan.util import time_it, file_check, repeat_mk_dirs
from typing import Optional
import sys
class DUNE(torch.nn.Module):

    def __init__(self, receding: int=10, checkpoint =None, robot=None, dune_max_num: int=100, train_kwargs: dict=dict()) -> None:
        super(DUNE, self).__init__()
  
        self.T = receding
        self.max_num = dune_max_num

        self.robot = robot

        self.G = np_to_tensor(robot.G)
        self.h = np_to_tensor(robot.h)
        self.edge_dim = self.G.shape[0]
        self.state_dim = self.G.shape[1]

        # configuration flags (with safe defaults)
        train_kwargs = train_kwargs or dict()
        self.projection = train_kwargs.get('projection', 'hard')  # 'hard' | 'none' | 'learned'
        self.monitor_dual_norm = train_kwargs.get('monitor_dual_norm', True)
        self.unroll_J = int(train_kwargs.get('unroll_J', 0))  # PDHG steps (0=disabled)
        self.se2_embed = bool(train_kwargs.get('se2_embed', False))

        # ObsPointNet with optional SE(2) embedding (backward compatible default False)
        self.model = to_device(ObsPointNet(2, self.edge_dim, se2_embed=self.se2_embed))

        # optional learned proximal head (for 'learned' projection)
        self.prox_head = None
        if self.projection == 'learned':
            try:
                self.prox_head = to_device(ProxHead(self.edge_dim, hidden=32))
            except Exception:
                self.prox_head = None

        # PDHG-Unroll module (Stage 1: fixed step sizes)
        self.pdhg_unroll = None
        if self.unroll_J > 0:
            pdhg_tau = float(train_kwargs.get('pdhg_tau', 0.5))
            pdhg_sigma = float(train_kwargs.get('pdhg_sigma', 0.5))
            pdhg_learnable = bool(train_kwargs.get('pdhg_learnable', False))
            pdhg_per_step = bool(train_kwargs.get('pdhg_per_step', False))

            if pdhg_per_step:
                # Stage 4: per-step learnable (advanced)
                self.pdhg_unroll = to_device(
                    PDHGUnrollPerStep(self.edge_dim, J=self.unroll_J,
                                     tau_init=pdhg_tau, sigma_init=pdhg_sigma)
                )
            else:
                # Stage 1-3: global step sizes (fixed or learnable)
                self.pdhg_unroll = to_device(
                    PDHGUnroll(self.edge_dim, J=self.unroll_J,
                              tau=pdhg_tau, sigma=pdhg_sigma,
                              learnable_steps=pdhg_learnable)
                )

        self.dual_norm_violation_rate = None
        self.dual_norm_p95 = None
        self.dual_norm_max_excess_pre = None
        self.dual_norm_max_excess_post = None
        self.load_model(checkpoint, train_kwargs)

        self.obstacle_points = None
        self.min_distance = inf


        
    @time_it('- dune forward')
    def forward(self, point_flow, R_list, obs_points_list=[]):

        '''
        map point flow to the latent distance features: lam, mu

        Args:
            point_flow: point flow under the robot coordinate, list of (state_dim, num_points); list length: T+1
            R_list: list of Rotation matrix, list of (2, 2), used to generate the lam from mu; list length: T
            obstacle_points: tensor of shape (2, num_points), global coordinate; 

        Returns: 
            lam_list: list of lam tensor, each element is a tensor of shape (state_dim, num_points); list length: T+1
            mu_list: list of mu tensor, each element is a tensor of shape (edge_number, num_points); list length: T+1
            sort_point_list: list of point tensor, each element is a tensor of shape (state_dim, num_points); list length: T+1; 
        '''

        # basic consistency checks
        assert isinstance(point_flow, list) and isinstance(R_list, list), "point_flow and R_list must be lists"
        assert len(R_list) == len(point_flow), f"R_list length {len(R_list)} != point_flow length {len(point_flow)}"
        if isinstance(obs_points_list, list) and len(obs_points_list) > 0:
            assert len(obs_points_list) == len(point_flow), f"obs_points_list length {len(obs_points_list)} != point_flow length {len(point_flow)}"

        mu_list, lam_list, sort_point_list = [], [], []
        self.obstacle_points = obs_points_list[0] # current obstacle points considered in the dune at time 0

        total_points = torch.hstack(point_flow)

        # map the point flow to the latent distance features mu
        with torch.no_grad():
            total_mu = self.model(total_points.T).T  # [E, N]

            # PDHG-Unroll refinement (before learned_prox/hard projection)
            if self.pdhg_unroll is not None:
                mu_pre = total_mu.detach()
                a_total = (self.G @ total_points - self.h)  # [E, N_total]
                total_mu = self.pdhg_unroll(total_mu, a_total, self.G)

                # profiling: mu statistics pre/post PDHG and timing
                try:
                    v_pre = self.G.t() @ mu_pre
                    v_post = self.G.t() @ total_mu
                    pre_mean_norm = torch.norm(v_pre, dim=0).mean().item()
                    post_mean_norm = torch.norm(v_post, dim=0).mean().item()
                    delta_mu_l2_mean = torch.norm(total_mu - mu_pre, dim=0).mean().item()
                except Exception:
                    pre_mean_norm = None
                    post_mean_norm = None
                    delta_mu_l2_mean = None

                timing = getattr(self.pdhg_unroll, 'last_timing', {}) or {}
                self.pdhg_profile = {
                    'pre_mean_dual_norm': pre_mean_norm,
                    'post_mean_dual_norm': post_mean_norm,
                    'delta_mu_l2_mean': delta_mu_l2_mean,
                    'total_time': timing.get('total'),
                    'per_step_times': timing.get('per_step'),
                }

            # Optional learned proximal refinement before hard projection
            if self.projection == 'learned' and self.prox_head is not None:
                try:
                    total_mu = self.prox_head(total_mu, self.G)
                except Exception:
                    # fallback: skip prox if any shape/device issue
                    pass

            # monitor dual feasibility on pre-hard values (post-prox if learned)
            if self.monitor_dual_norm or (self.projection in ('hard', 'learned')):
                v_pre = (self.G.T @ total_mu)
                norms_pre = torch.norm(v_pre, dim=0)
                if norms_pre.numel() > 0:
                    try:
                        p95 = torch.quantile(norms_pre, 0.95).item()
                    except Exception:
                        # fallback: take kth largest where k ~= 0.95*N
                        k = max(1, int(0.95 * norms_pre.numel()))
                        p95 = torch.topk(norms_pre, k, largest=True).values.min().item()
                    self.dual_norm_violation_rate = (norms_pre > 1.0).float().mean().item()
                    self.dual_norm_p95 = p95
                    self.dual_norm_max_excess_pre = torch.clamp(norms_pre - 1.0, min=0.0).max().item()
                else:
                    self.dual_norm_violation_rate = 0.0
                    self.dual_norm_p95 = 0.0
                    self.dual_norm_max_excess_pre = 0.0

            # Apply hard projection for 'hard' and 'learned' modes
            if self.projection in ('hard', 'learned'):
                # clamp mu >= 0
                total_mu.clamp_(min=0.0)
                # project columns to satisfy ||G^T mu||_2 <= 1
                v = (self.G.T @ total_mu)
                v_norm = torch.norm(v, dim=0, keepdim=True)
                mask = (v_norm > 1.0).float()
                # avoid div by zero by clamping
                denom = v_norm.clamp(min=1.0)
                scale = mask / denom + (1.0 - mask)
                total_mu = total_mu * scale

            if self.monitor_dual_norm or (self.projection in ('hard', 'learned')):
                v_post = (self.G.T @ total_mu)
                norms_post = torch.norm(v_post, dim=0)
                if norms_post.numel() > 0:
                    self.dual_norm_max_excess_post = torch.clamp(norms_post - 1.0, min=0.0).max().item()
                else:
                    self.dual_norm_max_excess_post = 0.0
        
        for index in range(self.T+1):
            num_points = point_flow[index].shape[1]
            mu = total_mu[:, index*num_points : (index+1)*num_points]
            R = R_list[index]
            p0 = point_flow[index]
            lam = (- R @ self.G.T @ mu)

            if mu.ndim == 1:
                mu = mu.unsqueeze(1)
                lam = lam.unsqueeze(1)

            distance = self.cal_objective_distance(mu, p0)

            if index == 0: 
                self.min_distance = torch.min(distance) 
            
            sort_indices = torch.argsort(distance)

            mu_list.append(mu[:, sort_indices])
            lam_list.append(lam[:, sort_indices])
            sort_point_list.append(obs_points_list[index][:, sort_indices])

        return mu_list, lam_list, sort_point_list


    def cal_objective_distance(self, mu: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:

        '''
        input: 
            mu: (edge_dim, num_points)
            p0: (state_dim, num_points)   
        output:
            distance:  mu.T (G @ p0 - h),  (num_points,)
        ''' 

        temp = (self.G @ p0 - self.h).T.unsqueeze(2)
        muT = mu.T.unsqueeze(1)

        distance = torch.squeeze(torch.bmm(muT, temp)) 

        if distance.ndim == 0:
            distance = distance.unsqueeze(0)

        return distance
    


    def load_model(self, checkpoint: Optional[str]=None, train_kwargs: Optional[dict]=None):

        '''
        checkpoint: pth file path of the model
        '''

        try:
            if checkpoint is None:
                raise FileNotFoundError

            self.abs_checkpoint_path = file_check(checkpoint)
            state = torch.load(self.abs_checkpoint_path, map_location=torch.device('cpu'))
            if isinstance(state, dict) and ('model' in state or 'prox_head' in state or 'pdhg_unroll' in state):
                # composite checkpoint
                self.model.load_state_dict(state['model'] if 'model' in state else state)
                if self.projection == 'learned' and self.prox_head is not None and 'prox_head' in state:
                    try:
                        self.prox_head.load_state_dict(state['prox_head'])
                    except Exception:
                        pass
                # Load PDHG-Unroll state (only if learnable parameters exist)
                if self.pdhg_unroll is not None and 'pdhg_unroll' in state:
                    try:
                        self.pdhg_unroll.load_state_dict(state['pdhg_unroll'])
                    except Exception:
                        pass
            else:
                # legacy: plain state_dict for model
                self.model.load_state_dict(state)
            to_device(self.model)
            self.model.eval()
            if self.projection == 'learned' and self.prox_head is not None:
                to_device(self.prox_head)
                self.prox_head.eval()

        except FileNotFoundError:

            if train_kwargs is None or len(train_kwargs) == 0:
                print('No train kwargs provided. Default value will be used.')
                train_kwargs = dict()

            direct_train = train_kwargs.get('direct_train', False)

            if direct_train:
                print('train or test the model directly.')
                return

            if self.ask_to_train():
                self.train_dune(train_kwargs)

                if self.ask_to_continue():
                    state = torch.load(self.full_model_name, map_location=torch.device('cpu'))
                    if isinstance(state, dict) and ('model' in state or 'prox_head' in state or 'pdhg_unroll' in state):
                        self.model.load_state_dict(state['model'] if 'model' in state else state)
                        if self.projection == 'learned' and self.prox_head is not None and 'prox_head' in state:
                            try:
                                self.prox_head.load_state_dict(state['prox_head'])
                            except Exception:
                                pass
                        if self.pdhg_unroll is not None and 'pdhg_unroll' in state:
                            try:
                                self.pdhg_unroll.load_state_dict(state['pdhg_unroll'])
                            except Exception:
                                pass
                    else:
                        self.model.load_state_dict(state)
                    to_device(self.model)
                    self.model.eval()
                    if self.projection == 'learned' and self.prox_head is not None:
                        to_device(self.prox_head)
                        self.prox_head.eval()
                else:
                    print('You can set the new model path to the DUNE class to use the trained model.')

            else:
                print('Can not find checkpoint. Please check the path or train first.')
                raise FileNotFoundError


    def train_dune(self, train_kwargs):

        model_name = train_kwargs.get("model_name", self.robot.name)

        # Build checkpoint directory robustly (Windows-friendly) and ensure parent exists
        model_root = os.path.join(sys.path[0], 'model')
        try:
            os.makedirs(model_root, exist_ok=True)
        except Exception:
            pass

        checkpoint_path = os.path.join(model_root, model_name)
        checkpoint_path = repeat_mk_dirs(checkpoint_path)

        # pass optional prox_head and pdhg_unroll to trainer
        prox_head = self.prox_head if self.projection == 'learned' else None
        pdhg_unroll = self.pdhg_unroll if self.unroll_J > 0 else None
        self.train_model = DUNETrain(self.model, self.G, self.h, checkpoint_path,
                                     prox_head=prox_head, pdhg_unroll=pdhg_unroll)
        self.full_model_name = self.train_model.start(**train_kwargs)
        print('Complete Training. The model is saved in ' + self.full_model_name)

    def ask_to_train(self):
        
        while True:
            choice = input("Do not find the DUNE model; Do you want to train the model now, input Y or N:").upper()
            if choice == 'Y':
                return True
            elif choice == 'N':
                print('Please set the your model path for the DUNE layer.')
                sys.exit()
            else:
                print("Wrong input, Please input Y or N.")


    def ask_to_continue(self):
        
        while True:
            choice = input("Do you want to continue the case running, input Y or N:").upper()
            if choice == 'Y':
                return True
            elif choice == 'N':
                print('exit the case running.')
                sys.exit()
            else:
                print("Wrong input, Please input Y or N.")


    @property
    def points(self):
        '''
        point considered in the dune layer
        '''

        return self.obstacle_points

