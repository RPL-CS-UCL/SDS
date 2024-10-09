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
    
        # Velocity reward: Encourage forward movement at 2.5 m/s in the x direction
        target_velocity = 2.5
        velocity_reward = -torch.square(env.base_lin_vel[:, 0] - target_velocity)
    
        # Height reward: Encourage maintaining the torso height around 0.34 meters
        target_height = 0.34
        height_reward = -torch.square(env.root_states[:, 2] - target_height)
    
        # Orientation reward: Penalize deviation from perpendicular alignment to gravity
        orientation_penality = -torch.square(env.projected_gravity[:,2] - 1.0)
    
        # Contact pattern reward: Encourage a Trot gait where diagonal pairs of legs move together
        FL_contact = (env.contact_forces[:, 4, 2] > 0).float()
        FR_contact = (env.contact_forces[:, 8, 2] > 0).float()
        RL_contact = (env.contact_forces[:, 12, 2] > 0).float()
        RR_contact = (env.contact_forces[:, 16, 2] > 0).float()
        
        diagonal_contact = (FL_contact + RR_contact) / 2 * (1 - (FR_contact + RL_contact) / 2)
        contact_reward = diagonal_contact
    
        # Penalty for excessive joint actions
        action_penalty = -torch.sum(torch.square(env.actions - env.last_actions), dim=1)
    
        # Combining all rewards
        total_reward = velocity_reward + height_reward + orientation_penality + contact_reward + action_penalty
    
        return total_reward, {
            "velocity_reward": velocity_reward,
            "height_reward": height_reward,
            "orientation_penality": orientation_penality,
            "contact_reward": contact_reward,
            "action_penalty": action_penalty
        }
    


