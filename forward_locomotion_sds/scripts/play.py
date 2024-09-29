import isaacgym

assert isaacgym
import torch
import numpy as np
import argparse
import shutil
import pickle as pkl

from forward_locomotion_sds.go1_gym.envs import *
from forward_locomotion_sds.go1_gym.envs.base.legged_robot_config import Cfg
from forward_locomotion_sds.go1_gym.envs.go1.go1_config import config_go1
from forward_locomotion_sds.go1_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

from utils.contact_plot import plot_foot_contacts


# def configure_camera_pose(env):
#     env.

def load_env(label, headless=False, dr_config=None, save_video=True,evaluation=False):
    # Will be overwritten by the loaded config from parameters.pkl
    Cfg.commands = Cfg.commands_original
    Cfg.rewards = Cfg.rewards_sds
    Cfg.domain_rand = Cfg.domain_rand_off

    # prepare environment
    config_go1(Cfg)

    from ml_logger import logger

    Cfg.commands.command_curriculum = False

    if dr_config == "original":
        Cfg.domain_rand = Cfg.domain_rand_original
    elif dr_config == "sds":
        Cfg.domain_rand = Cfg.domain_rand_sds
    elif dr_config == "off":
        Cfg.domain_rand = Cfg.domain_rand_off
    
    Cfg.env.record_video = save_video
    Cfg.env.num_recording_envs = 1 if save_video else 0
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    
    Cfg.sim.dt = 0.002
    
    if evaluation:
        Cfg.env.episode_length_s = 100

    from forward_locomotion_sds.go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from forward_locomotion_sds.go1_gym_learn.ppo.actor_critic import ActorCritic

    actor_critic = ActorCritic(
        num_obs=Cfg.env.num_observations,
        num_privileged_obs=Cfg.env.num_privileged_obs,
        num_obs_history=Cfg.env.num_observations * \
                        Cfg.env.num_observation_history,
        num_actions=Cfg.env.num_actions)

    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    policy = actor_critic.act_inference

    return env, policy


def play_mc(iterations=1000, headless=True, label=None, dr_config=None, verbose=False, save_video=False, save_contact=False, evaluation=False):
    from ml_logger import logger

    import os

    logger.configure(label)
    
    save_contact_sequence = []

    env, policy = load_env(label, headless=headless, dr_config=dr_config, save_video=save_video, evaluation=evaluation)
    
    num_eval_steps = iterations
    
    torso_height = np.zeros(num_eval_steps)
    measured_global_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * 2.0
    joint_positions = np.zeros((num_eval_steps, 12))
    joint_velocities = np.zeros((num_eval_steps, 12))
    torques = np.zeros((num_eval_steps, 12))

    if save_video:
        import imageio
        mp4_writer = imageio.get_writer('locomotion.mp4', fps=50)
        
    obs = env.reset()

    starting_pos = env.root_states[0, :3].cpu().numpy()
    for i in tqdm(range(num_eval_steps)):
        env.commands[:, :] = 0.0
        env.commands[:, 0] = 2.0
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        save_contact_sequence.append(info["contact_states"])
        if verbose:
            print(f'linear velocity: {info["body_global_linear_vel"]}')
            print(f"distance traveled (x): {(env.root_states[0, 0].cpu().numpy() - starting_pos)[0]}")

        joint_positions[i] = env.dof_pos[0, :].cpu()
        joint_velocities[i] = env.dof_vel[0, :].cpu()
        torques[i] = env.torques[0, :].detach().cpu()
        torso_height[i] = env.root_states[0,2].detach().cpu()

        if save_video:
            img = env.render(mode='rgb_array')
            mp4_writer.append_data(img)
        
        if not evaluation:
            if env.num_resets.detach().cpu().item() > 0:
                raise RuntimeError
    
    resets = env.num_resets.detach().cpu().item()
    print("Number of Resets: ",resets)
    if save_contact:
        plot_foot_contacts(save_contact_sequence,save_root=logger.prefix,evaluation=evaluation)
    
    if save_video:
        mp4_writer.close()
        video_dir_path = os.path.join(label, f"{logger.prefix}/videos")
        if not os.path.exists(video_dir_path):
            os.makedirs(video_dir_path)
        shutil.move("locomotion.mp4", os.path.join(video_dir_path, "play.mp4"))
        
        
        data_dict = {
            "Torso Height": {
                "data":torso_height,
                "y_label":"Height (m)"
            },
            # "Velocity":{
            #     "data":target_x_vels,
            #     "y_label":"Velocity (m/s)"
            # },
            "Resets":{
                "data":resets
            }
        }
        
        with open(os.path.join(label, f"{logger.prefix}",'eval_data.pkl'), 'wb') as file:
            pkl.dump(data_dict, file)
        
        np.savez(os.path.join(label, f"{logger.prefix}","eval_config.npz"),num_eval_steps=num_eval_steps,dt=env.dt)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--dr-config", type=str, choices=["original", "sds", "off"])
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--save_contact", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args()

    play_mc(iterations=args.iterations, headless=args.headless, label=args.run, dr_config=args.dr_config, verbose=args.verbose, save_video=not args.no_video, save_contact = args.save_contact, evaluation = args.evaluation)