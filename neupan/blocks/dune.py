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
from neupan.blocks.learned_prox import ProxHead
from neupan.blocks.obs_point_net import ObsPointNet
from neupan.blocks.flexible_pdhg import FlexiblePDHGFront
from neupan.blocks.dune_train import DUNETrain
from neupan.configuration import np_to_tensor, to_device
from neupan.util import time_it, file_check, repeat_mk_dirs
from typing import Optional
import sys
class DUNE(torch.nn.Module):

    def __init__(self, receding: int=10, checkpoint =None, robot=None, dune_max_num: int=100, train_kwargs: dict=dict(),
                 use_directional_sampling: bool=False, key_directions: list=None, nearest_num: int=2) -> None:
        super(DUNE, self).__init__()

        self.T = receding
        self.max_num = dune_max_num

        self.robot = robot

        self.G = np_to_tensor(robot.G)
        self.h = np_to_tensor(robot.h)
        self.edge_dim = self.G.shape[0]
        self.state_dim = self.G.shape[1]

        # Directional sampling configuration
        self.use_directional_sampling = use_directional_sampling
        self.key_directions = key_directions if key_directions else []
        self.nearest_num = nearest_num

        # configuration flags (with safe defaults)
        train_kwargs = train_kwargs or dict()
        self.projection = train_kwargs.get('projection', 'hard')  # 'hard' | 'none' | 'learned'
        self.monitor_dual_norm = train_kwargs.get('monitor_dual_norm', True)
        self.se2_embed = bool(train_kwargs.get('se2_embed', False))

        # Select front-end model
        front_name = str(train_kwargs.get('front', 'obs_point')).lower()
        if front_name in ('flex_pdhg', 'flexible_pdhg', 'flex', 'flexible'):
            # New Flexible PDHG front-end
            self.front_type = 'flex_pdhg'
            self.unroll_J = 0  # keep legacy field for compatibility

            front_J = int(train_kwargs.get('front_J', 1))
            front_hidden = int(train_kwargs.get('front_hidden', 32))
            front_learned = bool(train_kwargs.get('front_learned', True))
            front_tau = float(train_kwargs.get('front_tau', 0.5))
            front_sigma = float(train_kwargs.get('front_sigma', 0.5))
            residual_scale = float(train_kwargs.get('front_residual_scale', 0.5))
            front_precond = str(train_kwargs.get('front_precond', 'none')).lower()
            use_precond = front_precond in ('row', 'rows', 'true', 'yes', '1')
            learn_steps = bool(train_kwargs.get('front_learn_steps',
                                                train_kwargs.get('front_learnable_steps', False)))
            front_tau_min = float(train_kwargs.get('front_tau_min', 0.05))
            front_tau_max = float(train_kwargs.get('front_tau_max', 0.99))
            front_sigma_min = float(train_kwargs.get('front_sigma_min', 0.05))
            front_sigma_max = float(train_kwargs.get('front_sigma_max', 0.99))

            self.model = to_device(FlexiblePDHGFront(
                input_dim=2,
                E=self.edge_dim,
                G=self.G,
                h=self.h,
                hidden=front_hidden,
                J=front_J,
                se2_embed=self.se2_embed,
                use_learned_prox=front_learned,
                residual_scale=residual_scale,
                tau=front_tau,
                sigma=front_sigma,
                use_precond=use_precond,
                learnable_steps=learn_steps,
                tau_min=front_tau_min,
                tau_max=front_tau_max,
                sigma_min=front_sigma_min,
                sigma_max=front_sigma_max,
            ))
        else:
            # Default: ObsPointNet with optional SE(2) embedding
            self.model = to_device(ObsPointNet(2, self.edge_dim, se2_embed=self.se2_embed))
            self.front_type = 'obs_point'

        # optional learned proximal head (for 'learned' projection)
        self.prox_head = None
        if self.projection == 'learned':
            try:
                self.prox_head = to_device(ProxHead(self.edge_dim, hidden=32))
            except Exception:
                self.prox_head = None


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

            # Directional sampling: key directions + nearest points
            if self.use_directional_sampling and num_points > 0:
                # Step 1: Extract points from all key directions
                key_indices_list = []
                for key_dir in self.key_directions:
                    center_idx = int(key_dir.get('center_index', 50))
                    window_size = int(key_dir.get('window_size', 5))

                    # Robust clamping: ensure center and window are valid for num_points
                    if window_size <= 0:
                        continue
                    window_size = min(window_size, int(num_points))
                    if window_size <= 0:
                        continue

                    # Clamp center into valid index range
                    center_idx = max(0, min(int(center_idx), int(num_points) - 1))

                    half_before = (window_size - 1) // 2
                    # Choose start so that we get exactly window_size points and stay in bounds
                    start_idx = center_idx - half_before
                    start_idx = max(0, min(start_idx, int(num_points) - window_size))
                    end_idx = start_idx + window_size

                    # Now construct contiguous window [start_idx, end_idx)
                    window_indices = torch.arange(int(start_idx), int(end_idx), dtype=torch.long)

                    key_indices_list.append(window_indices)

                # Concatenate all key direction indices
                if key_indices_list:
                    key_indices = torch.cat(key_indices_list)
                else:
                    key_indices = torch.tensor([], dtype=torch.long)

                # Step 2: Select nearest points from remaining points
                remaining_mask = torch.ones(num_points, dtype=torch.bool)
                remaining_mask[key_indices] = False
                remaining_indices = torch.where(remaining_mask)[0]

                if len(remaining_indices) > 0 and self.nearest_num > 0:
                    remaining_distances = distance[remaining_indices]
                    sorted_remaining = torch.argsort(remaining_distances)
                    nearest_indices = remaining_indices[sorted_remaining[:self.nearest_num]]
                else:
                    nearest_indices = torch.tensor([], dtype=torch.long)

                # Step 3: Concatenate (key direction points + nearest points)
                sort_indices = torch.cat([key_indices, nearest_indices])

                # Limit to max_num if needed
                if len(sort_indices) > self.max_num:
                    sort_indices = sort_indices[:self.max_num]
            else:
                # Original logic: sort all points by distance
                sort_indices = torch.argsort(distance)
                sort_indices = sort_indices[:self.max_num]

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
            # Prefer safe loading in newer PyTorch; gracefully fallback if unsupported
            try:
                state = torch.load(
                    self.abs_checkpoint_path,
                    map_location=torch.device('cpu'),
                    weights_only=True,
                )
            except TypeError:
                state = torch.load(self.abs_checkpoint_path, map_location=torch.device('cpu'))

            # Helper: adapt legacy FlexiblePDHG checkpoints (tau/sigma -> tau_buf/sigma_buf)
            def _compat_state_dict(sd: dict) -> dict:
                try:
                    from neupan.blocks.flexible_pdhg import FlexiblePDHGFront as _F
                    is_flex = isinstance(self.model, _F)
                except Exception:
                    is_flex = False
                if not (isinstance(sd, dict) and is_flex):
                    return sd
                # Map legacy scalar step-size keys
                if 'tau' in sd and 'tau_buf' not in sd and hasattr(self.model, 'tau_buf'):
                    sd['tau_buf'] = sd['tau']
                if 'sigma' in sd and 'sigma_buf' not in sd and hasattr(self.model, 'sigma_buf'):
                    sd['sigma_buf'] = sd['sigma']
                return sd

            if isinstance(state, dict) and ('model' in state or 'prox_head' in state):
                # composite checkpoint
                model_sd = state['model'] if 'model' in state else state
                model_sd = _compat_state_dict(model_sd)
                load_result = self.model.load_state_dict(model_sd, strict=False)
                # load prox head non-strict if present
                if self.projection == 'learned' and self.prox_head is not None and 'prox_head' in state:
                    try:
                        self.prox_head.load_state_dict(state['prox_head'], strict=False)
                    except Exception:
                        pass
            else:
                # legacy: plain state_dict for model
                model_sd = _compat_state_dict(state if isinstance(state, dict) else dict())
                self.model.load_state_dict(model_sd, strict=False)
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
                    def _compat_state_dict_train(sd: dict) -> dict:
                        # local wrapper to avoid name capture issues
                        try:
                            from neupan.blocks.flexible_pdhg import FlexiblePDHGFront as _F
                            is_flex = isinstance(self.model, _F)
                        except Exception:
                            is_flex = False
                        if not (isinstance(sd, dict) and is_flex):
                            return sd
                        if 'tau' in sd and 'tau_buf' not in sd and hasattr(self.model, 'tau_buf'):
                            sd['tau_buf'] = sd['tau']
                        if 'sigma' in sd and 'sigma_buf' not in sd and hasattr(self.model, 'sigma_buf'):
                            sd['sigma_buf'] = sd['sigma']
                        return sd
                    if isinstance(state, dict) and ('model' in state or 'prox_head' in state):
                        model_sd = _compat_state_dict_train(state['model'] if 'model' in state else state)
                        self.model.load_state_dict(model_sd, strict=False)
                        if self.projection == 'learned' and self.prox_head is not None and 'prox_head' in state:
                            try:
                                self.prox_head.load_state_dict(state['prox_head'], strict=False)
                            except Exception:
                                pass
                    else:
                        model_sd = _compat_state_dict_train(state if isinstance(state, dict) else dict())
                        self.model.load_state_dict(model_sd, strict=False)
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

        # pass optional prox_head to trainer
        prox_head = self.prox_head if self.projection == 'learned' else None
        self.train_model = DUNETrain(self.model, self.G, self.h, checkpoint_path,
                                     prox_head=prox_head)
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

