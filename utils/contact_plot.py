import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_foot_contacts(act_foot_contacts, save_root,title='Contact Sequence',evaluation=False):
    act_foot_contacts = np.array(act_foot_contacts)
    act_foot_contacts = act_foot_contacts.squeeze(axis=1)
    
    START_TIME = int(act_foot_contacts.shape[0]/3)
    if evaluation:
        END_TIME = min(act_foot_contacts.shape[0] - int(3*act_foot_contacts.shape[0]/6),START_TIME+250)
    else:
        END_TIME = act_foot_contacts.shape[0] - int(act_foot_contacts.shape[0]/3)

    time = np.arange(act_foot_contacts.shape[0])
    time = time[START_TIME:END_TIME]
    
    foot_contacts = act_foot_contacts[START_TIME:END_TIME]

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))

    ax.set_ylim((-0.5, 3.5))
    
    theme_brown = (179 / 256, 144 / 256, 117 / 256, 1)
    google_red = (219 / 256, 68 / 256, 55 / 256, 1)
    light_brown = (199 / 256, 111 / 256, 44/256, 1)
    google_green = (15 / 256, 157 / 256, 88 / 256, 1)
    brown = (216 / 256, 131 / 256, 46 / 256, 1)
    
    foot_names = ['FL', 'RL', 'RR', 'FR']
    foot_colors = [theme_brown, light_brown, light_brown, theme_brown]
    default_color = 'darkblue'
    foot_contacts = np.array(foot_contacts)
    
    foot_contacts = foot_contacts[:,[0,2,3,1]]
        
    ax.set_yticks([3,2,1,0])
    ax.set_yticklabels(foot_names)
    for i in range(4):
        # Select timesteps where foot is on the ground
        ground_idx = foot_contacts[:, i] == 1
        ax.set_title(title)
        ax.axhline(y=i+0.5, color='black', linestyle='--')
        if evaluation:
            ax.fill_between(time, i-0.3, i+0.3, where=ground_idx, color=foot_colors[i])
        else:
            ax.fill_between(time, i-0.3, i+0.3, where=ground_idx, color=default_color)
    
    ax.xaxis.set_visible(False)
    
    if evaluation:
        plt.tight_layout()
    
    plt.savefig(os.path.join(save_root,"contact_sequence.png"))
