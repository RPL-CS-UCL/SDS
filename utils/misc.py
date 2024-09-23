import subprocess
import os
import json
import logging
import re
from utils.extract_task_code import file_to_string
import openai
import time

def gpt_query(sample,messages,temperature,model):
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    chunk_size = 4

    while True:
        if total_samples >= sample:
            break
        for attempt in range(3):
            try:
                response_cur = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=chunk_size
                )
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        responses.extend(response_cur["choices"])
        prompt_tokens = response_cur["usage"]["prompt_tokens"]
        total_completion_token += response_cur["usage"]["completion_tokens"]
        total_token += response_cur["usage"]["total_tokens"]
    
    return responses,prompt_tokens,total_completion_token,total_token

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    # Note: if this line breaks, you can provide an absolute path to gpustat instead
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])
    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def extract_training_log_dir(file_path):
    with open(file_path,mode="r") as f:
        for line in f:
            dashboard_match = re.match(r"Dashboard: http://app.dash.ml/(.+)",line)
            if dashboard_match:
                return dashboard_match.group(1)

def block_until_training(rl_filepath, success_keyword, failure_keyword, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on

    while True:
        rl_log = file_to_string(rl_filepath)
        if "running" in rl_log or "Traceback" in rl_log:
            if log_status and "running" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully trained!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break


def construct_run_log(stdout_str):
    run_log = {}
    lines = stdout_str.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("│") and line.endswith("│"):
            line = line[1:-1].split("│")
            key, val = line[0].strip(), line[1].strip()
            if key == "timesteps" or key == "iterations":
                key = key + "/"
            elif "train/episode/rew" in key:
                key = key.split("/")[2]
            elif key == "train/episode/episode length/mean":
                key = "episode length"

            run_log[key] = run_log.get(key, []) + [float(val)]
    run_log["gpt_reward"] = []
    run_log["gt_reward"] = []
    for i in range(len(run_log["episode length"])):
        cur_sum = 0
        for key in run_log:
            if "rew " in key:
                cur_sum += run_log[key][i]
        run_log["gpt_reward"].append(cur_sum)
        run_log["gt_reward"].append(cur_sum)
    return run_log