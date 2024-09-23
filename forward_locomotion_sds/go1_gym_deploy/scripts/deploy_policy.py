import glob
import pickle as pkl
import lcm
import sys
import os
import argparse

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner, NextPolicyException, PrevPolicyException
from go1_gym_deploy.envs.legged_robot_config import Cfg
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def run_policy(label, se, max_steps, max_vel, max_yaw_vel, experiment_name):

    Cfg.commands = Cfg.commands_original
    Cfg.rewards = Cfg.rewards_sds
    Cfg.domain_rand = Cfg.domain_rand_off

    # prepare environment
    config_go1(Cfg)

    cfg = Cfg

    control_dt = 0.05

    ## Change according to skill
    action_scales = [0.12,0.16,0.2,0.12]
    decimations = [4,4,4,1]

    # Find the label and its index where the last word matches the experiment name (robot name)
    matching_label_with_index = next(((i, label) for i, label in enumerate(labels) if label.split("/")[-1] == experiment_name), None)

    if matching_label_with_index is None:
        raise ValueError(f"No matching checkpoint found for experiment name: {experiment_name}")
    index, label = matching_label_with_index

    cfg.control.action_scale = action_scales[index]
    cfg.control.decimation = decimations[index]


    print("Experiment Name: ",experiment_name )
    print("Control DT: ",control_dt)
    print("Action Scale: ",cfg.control.action_scale)
    print("Decimation: ",cfg.control.decimation)
    print(f'Max steps: {max_steps}')

    
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    logdir = label
    
    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = LCMAgent(cfg, se, command_profile)
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy(logdir)

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    log_prefix = "_".join(label.split("/")[-2:])
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                        log_root=os.path.join(root, experiment_name), log_prefix=log_prefix)
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    deployment_runner.run(max_steps=max_steps, logging=True)

def run(labels, experiment_name, max_steps, max_vel=1.0, max_yaw_vel=1.0):
    se = StateEstimator(lc)
    se.spin()

    idx = 0

    while True:
        try:
            print()
            print(f"Running policy idx {idx}")
            run_policy(labels[idx], se, max_steps, max_vel, max_yaw_vel, experiment_name)
        except NextPolicyException:
            idx = (idx + 1) % len(labels)
        except PrevPolicyException:
            idx = (idx - 1) % len(labels)

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def config_go1(Cnfg):
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.34]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
    }

    _ = Cnfg.control
    _.control_type = 'P'
    _.stiffness = {'joint': 20.}  # [N*m/rad]
    _.damping = {'joint': 0.5}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_constrained.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = ["base","thigh"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    _.flip_visual_attachments = False
    _.fix_base_link = False

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.34

    _ = Cnfg.rewards.scales
    _.torques = -0.0001
    _.action_rate = -0.01
    _.dof_pos_limits = -10.0
    _.orientation = -5.
    _.base_height = -30.

    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 50

    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False

    _ = Cnfg.env
    _.num_observations = 39
    _.observe_vel = False
    # _.num_envs = 2
    # _.camera_mode = "corner"
    _.camera_mode = "sideways"
    _.num_envs = 4000

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = False
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.6, 0.6]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1, 1]

    _ = Cnfg.domain_rand
    _.rand_interval_s = 6


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run deployment policy with robot name")
    parser.add_argument("--skill", type=str, default="trot", help="Name of the skill")
    parser.add_argument("--max_steps", type=int, default=1000, help="Number of total steps")
    args = parser.parse_args()


    labels = [
        # Put paths to your policy here
        "../../runs/real_checkpoints/trot",
        "../../runs/real_checkpoints/hop",
        "../../runs/real_checkpoints/bound",
        "../../runs/real_checkpoints/pace"
    ]

    experiment_name= args.skill
    max_steps = args.max_steps

    print("Deploying policy: ", experiment_name)

    run(labels, experiment_name=experiment_name, max_steps =max_steps, max_vel=3.5, max_yaw_vel=5.0)
