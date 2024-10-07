import torch
import numpy as np
from forward_locomotion_sds.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class SDSReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_reward(self):
        env = self.env  # Necessary line to access environment parameters
    
        # Velocity reward: Encourage forward velocity close to 2 m/s
        target_velocity = 2.0
        vel_reward = -torch.abs(env.base_lin_vel[:, 0] - target_velocity)
    
        # Height reward: Encourage height around 0.34 meters
        target_height = 0.34
        height_reward = -torch.abs(env.root_states[:, 2] - target_height)
    
        # Orientation reward: Encourage upright orientation
        upright_reward = torch.dot(env.projected_gravity, env.gravity_vec.T).view(-1)
    
        # Contact pattern reward: Encourage walking gait
        fl_contact = env.contact_forces[:, 4, 2] > 0
        fr_contact = env.contact_forces[:, 8, 2] > 0
        rl_contact = env.contact_forces[:, 12, 2] > 0
        rr_contact = env.contact_forces[:, 16, 2] > 0
    
        walk_pattern_reward = (fl_contact & ~fr_contact & ~rl_contact & rr_contact |
                              ~fl_contact & fr_contact & rl_contact & ~rr_contact).float()
    
        # Action rate penalty: Minimize change in actions
        action_rate_penalty = -torch.sum(torch.square(env.actions - env.last_actions), dim=1)
    
        # Joint limits penalty: Encourage staying within joint limits
        joint_limits_penalty = -torch.sum((env.dof_pos - env.default_dof_pos).abs() > env.dof_pos_limits[:, 1], dim=1).float()
    
        # Combine rewards
        total_reward = 2.0 * vel_reward + 1.0 * height_reward + 1.0 * upright_reward + \
                       0.5 * walk_pattern_reward + 0.1 * action_rate_penalty + 0.1 * joint_limits_penalty
    
        reward_components = {
            "vel_reward": vel_reward,
            "height_reward": height_reward,
            "upright_reward": upright_reward,
            "walk_pattern_reward": walk_pattern_reward,
            "action_rate_penalty": action_rate_penalty,
            "joint_limits_penalty": joint_limits_penalty
        }
    
        return total_reward, reward_components
    


