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
from utils.vid_utils import create_grid_image,encode_image,save_grid_image,gen_placehold_image
from utils.easy_vit_pose import vitpose_inference
import cv2
import openai
import os
from agents import GaitAnalyser

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

EUREKA_ROOT_DIR = os.getcwd()
ROOT_DIR = f"{EUREKA_ROOT_DIR}/.."
openai.api_key = os.getenv("OPENAI_API_KEY")


class ENV_CONFIG():
    task = "Forward Locomotion"
    env_name = "forward_locomotion_sds"

    train_script = "scripts/train.py"
    reward_template_file = "go1_gym/rewards/sds_reward_template.py"
    reward_output_file = "go1_gym/rewards/sds_reward.py"

    # train_iterations = 200
    train_iterations = 1000
    success_keyword = "running"
    failure_keyword = "Traceback"




@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")
    logging.info(f"Running for {cfg.iteration} iterations")
    logging.info(f"Generating {cfg.sample} reward function samples per iteration")

    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info(f"Imitation Task: {cfg.task.description}")

    env_name = ENV_CONFIG.env_name.lower()

    task_rew_file = f'{ROOT_DIR}/{env_name}/{ENV_CONFIG.reward_template_file}'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_name}.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_rew_code_string = file_to_string(task_rew_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ROOT_DIR}/{env_name}/{ENV_CONFIG.reward_output_file}"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/prompts'
    initial_reward_engineer_system = file_to_string(f'{prompt_dir}/initial_reward_engineer_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_reward_engineer_user = file_to_string(f'{prompt_dir}/initial_reward_engineer_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signatures/{env_name}.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    SUS_prompt = file_to_string(f'{prompt_dir}/SUS.txt')
    initial_task_evaluator_system = file_to_string(f'{prompt_dir}/initial_task_evaluator_system.txt')
    
    demo_video_name = cfg.task.video
    video_do_crop = cfg.task.crop
    logging.info(f"Demonstration Video: {demo_video_name}, Crop Option: {cfg.task.crop_option}")
    gt_frame_grid = create_grid_image(f'{ROOT_DIR}/videos/{demo_video_name}',grid_size=(cfg.task.grid_size,cfg.task.grid_size),crop=video_do_crop,crop_option=cfg.task.crop_option)
    save_grid_image(gt_frame_grid,"gt_demo.png")
    
    # annotated_video_path = vitpose_inference(f'{ROOT_DIR}/videos/{demo_video_name}',f"{workspace_dir}/pose-estimate/gt-pose-estimate")
    # gt_annotated_frame_grid = create_grid_image(annotated_video_path,grid_size=(cfg.task.grid_size,cfg.task.grid_size),crop=video_do_crop,crop_option=cfg.task.crop_option)
    # save_grid_image(gt_annotated_frame_grid,"gt_demo_annotated.png")
    
    encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo.png')
    
    gait_analyser = GaitAnalyser(cfg,prompt_dir)
    gait_response = gait_analyser.analyse(encoded_gt_frame_grid)
    
    SUS_prompt = SUS_prompt.format(gait_analysis=gait_response)
    

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

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    # execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    best_footage = None
    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        responses,prompt_tokens,total_completion_token,total_token = gpt_query(cfg.sample,reward_query_messages,cfg.temperature,cfg.model)

        # Loading pre-queried reward functions
        reward_dir = os.path.join(EUREKA_ROOT_DIR,"saved_rewards")
        response_dir = os.path.join(reward_dir,"responses.pt")
        prompt_token_dir = os.path.join(reward_dir,"prompt_tokens.pt")
        total_completion_token_dir = os.path.join(reward_dir,"total_completion_token.pt")
        total_token_dir = os.path.join(reward_dir,"total_token.pt")

        torch.save(responses,response_dir)
        torch.save(prompt_tokens,prompt_token_dir)
        torch.save(total_completion_token,total_completion_token_dir)
        torch.save(total_token,total_token_dir)
        

        # responses = torch.load(response_dir)
        # prompt_tokens = torch.load(prompt_token_dir)
        # total_completion_token = torch.load(total_completion_token_dir)
        # total_token = torch.load(total_token_dir)

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        code_runs = [] 
        rl_runs = []
        footage_grids_dir = []

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
                command = f"python -u {ROOT_DIR}/{env_name}/{ENV_CONFIG.train_script} --iterations {ENV_CONFIG.train_iterations} --dr-config off --reward-config sds --no-wandb"
                command = command.split(" ")
                process = subprocess.run(command, stdout=f, stderr=f)
            block_until_training(rl_filepath, success_keyword=ENV_CONFIG.success_keyword, failure_keyword=ENV_CONFIG.failure_keyword,
                                 log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)
            
            try:
                training_log_dir = extract_training_log_dir(rl_filepath)
                training_footage_dir = os.path.join(f'{ROOT_DIR}/{env_name}/runs/',training_log_dir,"videos")

                files_with_ctime = [(file, os.path.getctime(os.path.join(training_footage_dir, file))) for file in os.listdir(training_footage_dir)]
                latest_footage = max(files_with_ctime, key=lambda x: x[1])[0]
            
                
                # training_frame_grid = create_grid_image(os.path.join(training_footage_dir,latest_footage),training_fixed_length=True)
                # save_grid_image(training_frame_grid,f"training_footage/training_frame_{iter}_{response_id}.png")
                
                annotated_video_path = vitpose_inference(os.path.join(training_footage_dir,latest_footage),f"{workspace_dir}/pose-estimate/sample-pose-estimate")
                training_frame_grid = create_grid_image(annotated_video_path,training_fixed_length=True)
                # save_grid_image(training_annotated_frame_grid,f"training_footage/training_frame_{iter}_{response_id}_annotated.png")

            except:
                # No footages saved due to reward run time error
                training_frame_grid = gen_placehold_image()
            
            footage_grid_save_dir = f"training_footage/training_frame_{iter}_{response_id}.png"
            save_grid_image(training_frame_grid,footage_grid_save_dir)
            
            footage_grids_dir.append(footage_grid_save_dir)
        
        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        # successes = []
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
                # successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                run_log = construct_run_log(stdout_str)
                # print("Run Log:")
                # print(run_log)
                
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
                        # if "consecutive_successes" == metric:
                        #     successes.append(metric_cur_max)
                        metric_cur_min = min(run_log[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            metric_name = metric 
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        # else:
                        #     # Provide ground-truth score when success rate not applicable
                        #     if "consecutive_successes" not in run_log:
                        #         content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback
                # successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            # execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue
        
        
        def compute_similarity_score_gpt(footage_grids_dir):
            
            evaluator_query_content = [
            {
                        "type": "text",
                        "text": "Please now rate the following images:"
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
                None
            ]
            
            evaluator_query_messages[2] ={"role" : "user", "content": evaluator_query_content}
            
            logging.info("Evaluating...")
            eval_responses,eval_prompt_tokens,eval_total_completion_token,eval_total_token = gpt_query(1,evaluator_query_messages,cfg.temperature,cfg.model)
        
            eval_response_dir = os.path.join(reward_dir,"eval_responses.pt") 
            torch.save(eval_responses,eval_response_dir)
            # eval_responses = torch.load(eval_response_dir)
        
            eval_responses = eval_responses[0]["message"]["content"]
            
            scores_re = re.findall(r'\[([^\]]*)\](?!.*\[)',eval_responses)
            scores_re = scores_re[-1]

            scores = [int(x) for x in scores_re.split(",")]
            best_idx = np.argmax(scores)
            # logging.info(f"Iteration {iter}: Prompt Tokens: {eval_prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

            logging.info(f"Best Sample Index: {best_idx}, Second Best Sample Index: {np.argsort(scores)[-2]}")

            with open('evaluator_query_messages.json', 'w') as file:
                json.dump(evaluator_query_messages + [{"role": "assistant", "content": eval_responses}], file, indent=4)
        
            if best_idx + 1 > cfg.sample:
                # Best sample is the previous best footage
                return np.argsort(scores)[-2],False
            
            return best_idx,True
        
        best_sample_idx,improved = compute_similarity_score_gpt(footage_grids_dir)

        # Select the best code sample based on the success rate
        # best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]
        
        # max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        # execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

        # # Update the best Eureka Output
        # if max_success > max_success_overall:
        #     max_success_overall = max_success
        #     max_success_reward_correlation_overall = max_success_reward_correlation
        #     max_reward_code_path = code_paths[best_sample_idx]
        
        if improved:
            logging.info(f"Iteration {iter}: A better reward function has been generated")
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]
            best_footage = encode_image(f'{workspace_dir}/training_footage/training_frame_{iter}_{best_sample_idx}.png')

        # execute_rates.append(execute_rate)
        # max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        # logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
            
        # # Plot the success rate
        # fig, axs = plt.subplots(2, figsize=(6, 6))
        # fig.suptitle(f'{task}')

        # x_axis = np.arange(len(max_successes))

        # axs[0].plot(x_axis, np.array(max_successes))
        # axs[0].set_title("Max Success")
        # axs[0].set_xlabel("Iteration")

        # axs[1].plot(x_axis, np.array(execute_rates))
        # axs[1].set_title("Execute Rate")
        # axs[1].set_xlabel("Iteration")

        # fig.tight_layout(pad=3.0)
        # plt.savefig('summary.png')
        # np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

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