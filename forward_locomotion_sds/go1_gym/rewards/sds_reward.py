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
        
        # Reward component for maintaining desired forward velocity
        target_velocity = 1.5  # m/s
        forward_velocity = torch.norm(env.base_lin_vel[:, 0] - target_velocity, dim=-1)
        reward_velocity = torch.exp(-torch.square(forward_velocity))
        
        # Reward component for maintaining torso height around 0.34 meters
        target_height = 0.34
        torso_height_diff = torch.abs(env.root_states[:, 2] - target_height)
        reward_torso_height = torch.exp(-torso_height_diff)
        
        # Reward component for maintaining correct orientation (perpendicular to gravity)
        orientation_error = torch.norm(env.projected_gravity[:, :2], dim=-1)
        reward_orientation = torch.exp(-orientation_error)
        
        # Reward component for Trot gait pattern
        contact_foot_pairs = [(env.contact_forces[:, 4, :], env.contact_forces[:, 12, :]),
                              (env.contact_forces[:, 8, :], env.contact_forces[:, 16, :])]
        
        reward_gait = 0.0
        for pair in contact_foot_pairs:
            contacts = torch.norm(pair[0], dim=-1) > 0, torch.norm(pair[1], dim=-1) > 0
            correct_gait = torch.logical_and(contacts[0], contacts[1]).float()
            incorrect_gait = torch.logical_xor(contacts[0], contacts[1]).float()
            reward_gait += correct_gait - incorrect_gait
    
        # Normalize the gait reward
        reward_gait /= len(contact_foot_pairs)
        
        # Reward component for minimizing action magnitude
        reward_action_efficiency = torch.exp(-torch.norm(env.actions, dim=-1))
        
        # Penalty for joint positions and velocities exceeding limits
        joint_pos_error = torch.abs(env.dof_pos - env.default_dof_pos)
        joint_vel_error = torch.abs(env.dof_vel)
    
        joint_pos_limits = torch.maximum(env.dof_pos_limits[:, 1] - env.default_dof_pos, env.default_dof_pos - env.dof_pos_limits[:, 0])
        pos_violations = torch.maximum(joint_pos_error - joint_pos_limits, torch.zeros_like(joint_pos_error))
        
        reward_dof_limits = torch.exp(-torch.norm(pos_violations, dim=-1))
    
        # Combining all reward components
        total_reward = (
            0.4 * reward_velocity +  # Prioritize reaching the desired forward velocity
            0.2 * reward_torso_height +  # Maintaining the torso height
            0.2 * reward_orientation +  # Correct orientation
            0.1 * reward_gait +  # Trot gait pattern
            0.05 * reward_action_efficiency +  # Minimize action magnitude
            0.05 * reward_dof_limits  # Stay within movements DOF limits
        )
        
        # Individual reward components dictionary
        reward_components = {
            "reward_velocity": reward_velocity,
            "reward_torso_height": reward_torso_height,
            "reward_orientation": reward_orientation,
            "reward_gait": reward_gait,
            "reward_action_efficiency": reward_action_efficiency,
            "reward_dof_limits": reward_dof_limits
        }
        
        # Ensuring all tensors are on the same device
        total_reward = total_reward.to(env.device)
        for key, value in reward_components.items():
            reward_components[key] = value.to(env.device)
        
        return total_reward, reward_components
    


