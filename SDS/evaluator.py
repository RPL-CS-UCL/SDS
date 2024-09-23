import re
import os
import numpy as np
import subprocess
from matplotlib import pyplot as plt
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

eval_iter = 10
eval_steps = 7500
ROOT_DIR = os.path.join(os.getcwd(),"..")
play_script_dir = os.path.join(ROOT_DIR,"forward_locomotion_sds/scripts/play.py")
eval_path = os.path.join(ROOT_DIR,"results/checkpoints")

def do_plot(metrics:dict,eval_cfg,chkpt_dir):
    num_eval_steps = eval_cfg["num_eval_steps"]
    dt = eval_cfg["dt"]

    fig, axs = plt.subplots(len(metrics)-1, 1, figsize=(12, 10))
    
    i = 0
    for metric in metrics.keys():
        if metric == "Resets":
            continue
        metric_dic = metrics[metric]
        data = np.array([iter["data"] for iter in metric_dic])
        y_label = metric_dic[0]["y_label"]
        
        mean_values = np.mean(data, axis=0)
        std_values = np.std(data, axis=0)
        
        if len(metrics)-1 == 1:
            axs.plot(np.linspace(0, num_eval_steps * dt, num_eval_steps), mean_values, color='black', linestyle="-", label="Measured")
            axs.fill_between(np.linspace(0, num_eval_steps * dt, num_eval_steps), mean_values - std_values, mean_values + std_values, color='blue', alpha=0.2, label='Std Dev')
            axs.legend()
            axs.set_title(metric)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel(y_label)
        else:
            axs[i].plot(np.linspace(0, num_eval_steps * dt, num_eval_steps), mean_values, color='black', linestyle="-", label="Measured")
            axs[i].fill_between(np.linspace(0, num_eval_steps * dt, num_eval_steps), mean_values - std_values, mean_values + std_values, color='blue', alpha=0.2, label='Std Dev')
            axs[i].legend()
            axs[i].set_title(metric)
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel(y_label)
        
        i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(chkpt_dir, "eval_plot.png"))

def do_evaluation(chkpt, iter=eval_iter):
    metrics = {}
    run_chkpt_dir = os.path.join(eval_path, f"{chkpt}")
    eval_data_dir = os.path.join(run_chkpt_dir, "eval_data.pkl")
    eval_cfg_dir = os.path.join(run_chkpt_dir, "eval_config.npz")
    play_script = f"python -u {play_script_dir} --run {run_chkpt_dir} --dr-config sds --headless --save_contact --iterations {eval_steps} --evaluation"
    
    for _ in range(iter):
        subprocess.run(play_script.split(" "))
        with open(eval_data_dir, 'rb') as file:
            eval_data = pickle.load(file)
            
        for metric in eval_data.keys():
            if metric not in metrics:
                metrics[metric] = [eval_data[metric]]
            else:
                metrics[metric].append(eval_data[metric])
                
    eval_cfg = np.load(eval_cfg_dir)
    resets = np.mean([iter["data"] for iter in metrics["Resets"]])
    
    with open(os.path.join(run_chkpt_dir, "final_eval_data.pkl"), 'wb') as file:
        pickle.dump(metrics, file)
    
    do_plot(metrics, eval_cfg, run_chkpt_dir)
    
    return resets

if __name__ == "__main__":
    all_resets = []
    chkpts = []
    for chkpt in os.listdir(eval_path):
        chkpts.append(chkpt)  # Corrected this line
        resets = do_evaluation(chkpt)
        all_resets.append(resets)
        
    fig, ax = plt.subplots()

    ax.bar(chkpts, all_resets, label='Resets per Class')
    ax.set_title('Resets per Class')
    ax.set_ylabel('Number of Resets')
    ax.legend()
    
    plt.savefig(os.path.join(eval_path, '../resets_per_class_chart.png'))
    plt.close()
