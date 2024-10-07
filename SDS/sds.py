import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import re
import subprocess
from pathlib import Path
import shutil
import torch
from utils.misc import * 
from utils.extract_task_code import *
from utils.vid_utils import create_grid_image,encode_image,save_grid_image
from utils.easy_vit_pose import vitpose_inference
import cv2
import openai
import os
from agents import SUSGenerator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

SDS_ROOT_DIR = os.getcwd()
ROOT_DIR = f"{SDS_ROOT_DIR}/.."
openai.api_key = os.getenv("OPENAI_API_KEY")

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {SDS_ROOT_DIR}")
    logging.info(f"Running for {cfg.iteration} iterations")
    logging.info(f"Training each RF for: {cfg.train_iterations} iterations")
    logging.info(f"Generating {cfg.sample} reward function samples per iteration")

    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info(f"Imitation Task: {cfg.task.description}")

    env_name = cfg.env_name.lower()

    task_rew_file = f'{ROOT_DIR}/{env_name}/{cfg.reward_template_file}'
    task_obs_file = f'{SDS_ROOT_DIR}/envs/{env_name}.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_rew_code_string = file_to_string(task_rew_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ROOT_DIR}/{env_name}/{cfg.reward_output_file}"

    # Loading all text prompts
    prompt_dir = f'{SDS_ROOT_DIR}/prompts'
    initial_reward_engineer_system = file_to_string(f'{prompt_dir}/initial_reward_engineer_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_reward_engineer_user = file_to_string(f'{prompt_dir}/initial_reward_engineer_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signatures/{env_name}.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    initial_task_evaluator_system = file_to_string(f'{prompt_dir}/initial_task_evaluator_system.txt')
    
    demo_video_name = cfg.task.video
    video_do_crop = cfg.task.crop
    logging.info(f"Demonstration Video: {demo_video_name}, Crop Option: {cfg.task.crop_option}")
    gt_frame_grid = create_grid_image(f'{ROOT_DIR}/videos/{demo_video_name}',grid_size=(cfg.task.grid_size,cfg.task.grid_size),crop=video_do_crop,crop_option=cfg.task.crop_option)
    save_grid_image(gt_frame_grid,"gt_demo.png")
    
    annotated_video_path = vitpose_inference(f'{ROOT_DIR}/videos/{demo_video_name}',f"{workspace_dir}/pose-estimate/gt-pose-estimate")
    gt_annotated_frame_grid = create_grid_image(annotated_video_path,grid_size=(cfg.task.grid_size,cfg.task.grid_size),crop=video_do_crop,crop_option=cfg.task.crop_option)
    save_grid_image(gt_annotated_frame_grid,"gt_demo_annotated.png")
    
    eval_script_dir = os.path.join(ROOT_DIR,"forward_locomotion_sds/scripts/play.py")
    
    encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo.png')
    
    sus_generator = SUSGenerator(cfg,prompt_dir)
    SUS_prompt = sus_generator.generate_sus_prompt(encoded_gt_frame_grid)
    
    initial_reward_engineer_system = initial_reward_engineer_system.format(task_reward_signature_string=reward_signature,task_obs_code_string=task_obs_code_string) + code_output_tip

    initial_reward_engineer_user = initial_reward_engineer_user.format(sus_string=SUS_prompt,task_obs_code_string=task_obs_code_string)
    
    initial_task_evaluator_system = initial_task_evaluator_system.format(sus_string=SUS_prompt)


    reward_query_messages = [
        {"role": "system", "content": initial_reward_engineer_system}, 
        {"role": "user", "content": [
          {
            "type": "text",
            "text": initial_reward_engineer_user 
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/png;base64,{encoded_gt_frame_grid}",
              "detail": cfg.image_quality
            }
          }
        ]
        }
    ]

    os.mkdir(f"{workspace_dir}/training_footage")
    os.mkdir(f"{workspace_dir}/contact_sequence")

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    # execute_rates = []
    best_code_paths = []
    max_reward_code_path = None 
    
    best_footage = None
    best_contact = None

    for iter in range(cfg.iteration):

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        responses,prompt_tokens,total_completion_token,total_token = gpt_query(cfg.sample,reward_query_messages,cfg.temperature,cfg.model)
        
        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        code_runs = [] 
        rl_runs = []
        footage_grids_dir = []
        contact_pattern_dirs = []
        
        successful_runs_index = []
        
        eval_success = False

        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            # print(response_cur)
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            lines = [" "*4 + line for line in lines]
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    break
            
            
            def ensure_doubly_indented(code_str):
                lines = code_str.splitlines()
                
                base_indentation = len(lines[0]) - len(lines[0].lstrip())

                def adjust_indentation(line):
                    stripped_line = line.lstrip()
                    current_indentation = len(line) - len(stripped_line)
                    
                    return " "*(4+current_indentation-base_indentation) + stripped_line

                adjusted_lines = []
                for i, line in enumerate(lines):
                    if i == 0 and line.strip().startswith('def'):
                        adjusted_lines.append(adjust_indentation(line))
                    else:
                        adjusted_lines.append(adjust_indentation(line))
                
                # Join the lines back into a single string with newline characters
                adjusted_code = '\n'.join(adjusted_lines)
                
                return adjusted_code
            
            code_string = ensure_doubly_indented(code_string)
            code_runs.append(code_string)
                    
            # Add the SDS Reward Signature to the environment code
            cur_task_rew_code_string = task_rew_code_string.replace("# INSERT SDS REWARD HERE", code_string)

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(cur_task_rew_code_string + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()
            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                command = f"python -u {ROOT_DIR}/{env_name}/{cfg.train_script} --iterations {cfg.train_iterations} --dr-config off --reward-config sds --no-wandb"
                command = command.split(" ")
                process = subprocess.run(command, stdout=f, stderr=f)
            training_success = block_until_training(rl_filepath, success_keyword=cfg.success_keyword, failure_keyword=cfg.failure_keyword,
                                 log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)
            
            if training_success:
                training_log_dir = extract_training_log_dir(rl_filepath)
                full_training_log_dir =  os.path.join(f'{ROOT_DIR}/{env_name}/runs/',training_log_dir)
                contact_pattern_dir = os.path.join(full_training_log_dir,"contact_sequence.png")
                eval_script = f"python -u {eval_script_dir} --run {full_training_log_dir} --dr-config sds --headless --save_contact"
                training_footage_dir = os.path.join(full_training_log_dir,"videos")
                
                try:
                    subprocess.run(eval_script.split(" "))
                    
                    annotated_video_path = vitpose_inference(os.path.join(training_footage_dir,"play.mp4"),f"{workspace_dir}/pose-estimate/sample-pose-estimate")
                    training_frame_grid = create_grid_image(annotated_video_path,training_fixed_length=True)
                    # save_grid_image(training_annotated_frame_grid,f"training_footage/training_frame_{iter}_{response_id}_annotated.png")
                    
                    footage_grid_save_dir = f"training_footage/training_frame_{iter}_{response_id}.png"
                    save_grid_image(training_frame_grid,footage_grid_save_dir)
                    
                    contact_sequence_save_dir = f"{workspace_dir}/contact_sequence/contact_sequence_{iter}_{response_id}.png"
                    
                    shutil.copy(contact_pattern_dir, contact_sequence_save_dir)
                    
                    footage_grids_dir.append(footage_grid_save_dir)
                    
                    contact_pattern_dirs.append(contact_sequence_save_dir)    
                    
                    successful_runs_index.append(response_id)
                    
                    eval_success = True

                except:
                    # No footages saved due to reward run time error
                    logging.info(f"Iteration {iter}: Code Run {response_id} Failed to Evaluate, Not evaluated")
            else:
                logging.info(f"Iteration {iter}: Code Run {response_id} Unstable, Not evaluated")
        
        # Repeat the iteration if all code generation failed
        if not eval_success and cfg.sample != 1:
            # execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code evaluation failed! Repeat this iteration from the current message checkpoint!")
            continue
        

        code_feedbacks = []
        contents = []
        reward_correlations = []
        code_paths = []
        
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read()
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                run_log = construct_run_log(stdout_str)
                
                train_iterations = np.array(run_log['iterations/']).shape[0]
                epoch_freq = max(int(train_iterations // 10), 1)
                
                epochs_per_log = 10
                content += policy_feedback.format(epoch_freq=epochs_per_log*epoch_freq)
                
                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in run_log and "gpt_reward" in run_log:
                    gt_reward = np.array(run_log["gt_reward"])
                    gpt_reward = np.array(run_log["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in sorted(run_log.keys()):
                    if "/" not in metric:
                        metric_cur = ['{:.2f}'.format(x) for x in run_log[metric][::epoch_freq]]
                        metric_cur_max = max(run_log[metric])
                        metric_cur_mean = sum(run_log[metric]) / len(run_log[metric])

                        metric_cur_min = min(run_log[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            metric_name = metric 
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
               
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback

                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue
        
        
        def compute_similarity_score_gpt(footage_grids_dir,contact_pattern_dirs):
            
            evaluator_query_content = [
            {
                        "type": "text",
                        "text": "You will be rating the following images:"
                    } 
            ]
            
            for footage_dir in footage_grids_dir:
                
                encoded_footage = encode_image(footage_dir)
            
                evaluator_query_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_footage}",
                        "detail": cfg.image_quality
                    }
                    }
                )
        

            contact_evaluator_query_content = [
            {
                        "type": "text",
                        "text": "They have the following corresponding foot contact sequence plots, where FR means Front Right Foot, FL means Front Left Foot, RR means Rear Right Foot and RL means Rear Right Foot"
                    } 
            ]
            
            for contact_dir in contact_pattern_dirs:
                
                encoded_contact = encode_image(contact_dir)
            
                contact_evaluator_query_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_contact}",
                        "detail": cfg.image_quality
                    }
                    }
                )
        
            if best_footage is not None:
                evaluator_query_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{best_footage}",
                        "detail": cfg.image_quality
                    }
                    }
                )
                
                contact_evaluator_query_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{best_contact}",
                        "detail": cfg.image_quality
                    }
                    }
                )
                
                successful_runs_index.append(-1)
            
            if cfg.task.use_annotation:
                encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo_annotated.png')
            else:
                encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo.png')
            
            evaluator_query_messages = [
                {"role": "system", "content": initial_task_evaluator_system},
                {"role" : "user", "content":
                    [
                        {
                            "type": "text",
                            "text": "Here is the image demonstrating the ground truth task"
                        },
                        {
                            
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_gt_frame_grid}",
                                "detail": cfg.image_quality
                            }
                        }
                    ]
                },
                None,
                None
            ]
            
            evaluator_query_messages[2] ={"role" : "user", "content": evaluator_query_content}
            
            evaluator_query_messages[3] ={"role" : "user", "content": contact_evaluator_query_content}
            
            logging.info("Evaluating...")
            eval_responses,_,_,_ = gpt_query(1,evaluator_query_messages,cfg.temperature,cfg.model)
        
            eval_responses = eval_responses[0]["message"]["content"]
            
            scores_re = re.findall(r'\[([^\]]*)\](?!.*\[)',eval_responses)
            scores_re = scores_re[-1]

            scores = [float(x) for x in scores_re.split(",")]
            
            if len(scores) == 1:
                logging.info(f"Best Sample Index: {0}")
                return 0,True
            else:
                best_idx_in_successful_runs = np.argmax(scores)
                second_best_idx_in_successful_runs = np.argsort(scores)[-2]
                
                best_idx = successful_runs_index[best_idx_in_successful_runs]
                second_best_idx = successful_runs_index[second_best_idx_in_successful_runs]
                # logging.info(f"Iteration {iter}: Prompt Tokens: {eval_prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

                logging.info(f"Best Sample Index: {best_idx}, Second Best Sample Index: {second_best_idx}")

                with open(f'evaluator_query_messages_{iter}.json', 'w') as file:
                    json.dump(evaluator_query_messages + [{"role": "assistant", "content": eval_responses}], file, indent=4)
            
                if best_idx == -1:
                    # Best sample is the previous best footage
                    return second_best_idx,False
                
                return best_idx,True
        
        best_sample_idx,improved = compute_similarity_score_gpt(footage_grids_dir,contact_pattern_dirs)

        best_content = contents[best_sample_idx]

        
        if improved:
            logging.info(f"Iteration {iter}: A better reward function has been generated")
            max_reward_code_path = code_paths[best_sample_idx]
            best_footage = encode_image(f'{workspace_dir}/training_footage/training_frame_{iter}_{best_sample_idx}.png')
            
            rl_filepath = f"env_iter{iter}_response{best_sample_idx}.txt"
            training_log_dir = extract_training_log_dir(rl_filepath)
            full_training_log_dir =  os.path.join(f'{ROOT_DIR}/{env_name}/runs/',training_log_dir)
            contact_pattern_dir = os.path.join(full_training_log_dir,"contact_sequence.png")
            
            best_contact = encode_image(contact_pattern_dir)

        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
            
            
        if len(reward_query_messages) == 2:
            reward_query_messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
            reward_query_messages += [{"role": "user", "content": best_content}]
        else:
            assert len(reward_query_messages) == 4
            reward_query_messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
            reward_query_messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('reward_query_messages.json', 'w') as file:
            json.dump(reward_query_messages, file, indent=4)
    
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Best Reward Code Path: {max_reward_code_path}")

    best_reward = file_to_string(max_reward_code_path)
    with open(output_file, 'w') as file:
        file.writelines(best_reward + '\n')
    
    # Get run directory of best-performing policy
    with open(max_reward_code_path.replace(".py", ".txt"), "r") as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith("Dashboard: "):
            run_dir = line.split(": ")[1].strip()
            run_dir = run_dir.replace("http://app.dash.ml/", f"{ROOT_DIR}/{env_name}/runs/")
            logging.info("Best policy run directory: " + run_dir)

if __name__ == "__main__":
    main()