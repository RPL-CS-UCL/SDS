import torch
import numpy as np
from forward_locomotion_sds.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class SDSReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

# INSERT SDS REWARD HERE
    

