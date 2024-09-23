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
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Velocity component
        vel_error = torch.abs(env.base_lin_vel[:, 0] - 2.0)
        velocity_reward = torch.exp(-2 * vel_error)  # Keep same scaling for precision
        
        # Height component
        height_error = torch.abs(env.root_states[:, 2] - 0.34)
        height_reward = torch.exp(-height_error / 2)  # Slightly decrease the impact
        
        # Orientation component
        orientation_error = torch.abs(env.projected_gravity[:, 2] - 1.0)
        orientation_reward = torch.exp(-orientation_error)

        # Contact pattern component
        FL_contact = env.contact_forces[:, 4, 2] > 0
        FR_contact = env.contact_forces[:, 8, 2] > 0
        RL_contact = env.contact_forces[:, 12, 2] > 0
        RR_contact = env.contact_forces[:, 16, 2] > 0
        
        gait_pattern = (FL_contact & RL_contact) | (FR_contact & RR_contact)
        contact_pattern_reward = 0.1 * gait_pattern.float()
        
        # Ensure at least two feet are in contact with the ground
        # Rewriting to consider more realistic contact constraints
        contact_sum = (FL_contact.float() + FR_contact.float() + RL_contact.float() + RR_contact.float())
        two_feet_contact_reward = torch.where(contact_sum == 2, torch.tensor(0.05, device=env.device), torch.tensor(0.0, device=env.device))
        
        # Stability component: minimal action rate
        action_rate_reward = torch.exp(-0.02 * torch.sum(torch.abs(env.actions - env.last_actions), dim=-1))  # Further decreased scaling
        
        # Stay within DOF limits
        within_dof_limits = ((env.dof_pos >= env.dof_pos_limits[:, 0]) & (env.dof_pos <= env.dof_pos_limits[:, 1])).all(dim=1)
        dof_limits_reward = torch.where(within_dof_limits, torch.tensor(0.05, device=env.device), torch.tensor(0.0, device=env.device))

        # Total reward
        total_reward = (
            0.25 * velocity_reward + 
            0.1 * height_reward +  # Decreased weight
            0.1 * orientation_reward +
            contact_pattern_reward +
            two_feet_contact_reward +
            0.05 * action_rate_reward +  # Further decreased weight
            0.05 * dof_limits_reward
        )
        
        reward_components = {
            "velocity_reward": velocity_reward,
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "contact_pattern_reward": contact_pattern_reward,
            "two_feet_contact_reward": two_feet_contact_reward,
            "action_rate_reward": action_rate_reward,
            "dof_limits_reward": dof_limits_reward
        }
        
        return total_reward, reward_components
    



