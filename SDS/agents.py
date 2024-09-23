import logging
from utils.misc import gpt_query

class Conversation():
    def __init__(self,system_prompt:str) -> None:
        self.messages = [
            {
            "role": "system",
            "content": [
                {
                "text": system_prompt,
                "type": "text"
                }
            ]
            }
        ]
        self.conversation_completion_tokens = 0
        self.conversation_prompt_tokens = 0
        self.converstation_total_tokens = 0
    
    def modify_system_prompt(self,new_system_prompt):
        self.messages[0]["content"][0]["text"] = new_system_prompt
    
    def add_user_content(self,content:list):
        self.messages.append(
            {
            "role": "user",
            "content": content
            }
        )
    
    
    def add_assistant_content(self,prompt):
        self.messages.append(
            {
            "role": "assistant",
            "content": [
                {
                "text": prompt,
                "type": "text"
                }
            ]
            }
        )
    
    def add_usage(self,usage):
        self.conversation_prompt_tokens += usage.prompt_tokens
        self.conversation_completion_tokens += usage.completion_tokens
        self.converstation_total_tokens += usage.total_tokens
    
    def get_message(self):
        return self.messages
    
    def get_last_content(self):
        return 

class Agent():
    def __init__(self, system_prompt_file,cfg):
        with open(system_prompt_file,"r") as f:
            self.system_prompt = f.read()
        self.cfg = cfg
        self.conversation = Conversation(self.system_prompt)
        self.last_assistant_content = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Activated Agent {self.__class__.__name__}")
        self.sample = 1
        self.model = cfg.model
        self.temperature = cfg.temperature
    
    def get_conversation(self):
        return self.conversation
    
    def prepare_user_content(self,contents:list):
        full_content = []
        
        for content in contents:
            if content["type"] == "text":
                 full_content.append(
                    {
                    "text": content["data"],
                    "type": "text"
                    }
                )
            elif content["type"] == "image_uri":
                full_content.append(
                    {
                    "type": "image_url",
                    "image_url": 
                        {
                        "url": f"data:image/png;base64,{content['data']}",
                        "detail": "high"
                        }
                    }
                )
            else:
                full_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": content["data"],
                        "detail": "high"
                        }
                    }
                )
        
        self.conversation.add_user_content(full_content)
    
    def query(self):
        responses,_,_,_ = gpt_query(sample=self.sample,temperature=self.temperature,model=self.model,messages=self.conversation.get_message())
        assistant_content = responses[0]["message"]["content"]
        self.conversation.add_assistant_content(assistant_content)
        self.last_assistant_content = assistant_content
        return assistant_content
    
    def obtain_results(self):
        return self.last_assistant_content

class ContactSequenceAnalyser(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/contact_sequence_system.txt"
        super().__init__(system_prompt_file, cfg)
    def analyse(self,encoded_frame_grid):
        self.prepare_user_content([{"type":"image_uri","data":encoded_frame_grid}])
        return self.query()

class GaitAnalyser(Agent):
    def __init__(self, cfg, prompt_dir):
        self.prompt_dir = prompt_dir
        system_prompt_file=f"{prompt_dir}/gait_pattern_system.txt"
        super().__init__(system_prompt_file, cfg)
    
    def analyse(self,encoded_frame_grid):
        contact_sequence_analyser = ContactSequenceAnalyser(self.cfg,self.prompt_dir)
        contact_pattern = contact_sequence_analyser.analyse(encoded_frame_grid)
        
        # print("Contact Pattern: \n",contact_pattern)
        
        # print("="*50)
        
        self.prepare_user_content([{"type":"image_uri","data":encoded_frame_grid},{"type":"text","data":f"For the provided sequential frames, you are provided with the following feet contact pattern: {contact_pattern}"}])
        gait_pattern_response = self.query()
        # print(gait_pattern_response)

        return gait_pattern_response
    
if __name__ == "__main__":
    import hydra
    import logging 
    import os
    from pathlib import Path
    import shutil
    from utils.misc import * 
    from utils.extract_task_code import *
    from utils.vid_utils import create_grid_image,encode_image,save_grid_image,gen_placehold_image
    # from utils.easy_vit_pose import vitpose_inference
    import openai
    import os
    from agents import GaitAnalyser

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    EUREKA_ROOT_DIR = os.getcwd()
    ROOT_DIR = f"{EUREKA_ROOT_DIR}/.."
    #openai.api_key = REPLACE WITH OPENAI API KEY 
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
        # task_rew_code_string = file_to_string(task_rew_file)
        # task_obs_code_string = file_to_string(task_obs_file)
        # output_file = f"{ROOT_DIR}/{env_name}/{ENV_CONFIG.reward_output_file}"

        # Loading all text prompts
        prompt_dir = f'{EUREKA_ROOT_DIR}/prompts'
        # initial_reward_engineer_system = file_to_string(f'{prompt_dir}/initial_reward_engineer_system.txt')
        # code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
        # code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
        # initial_reward_engineer_user = file_to_string(f'{prompt_dir}/initial_reward_engineer_user.txt')
        # reward_signature = file_to_string(f'{prompt_dir}/reward_signatures/{env_name}.txt')
        # policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
        # execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
        # SUS_prompt = file_to_string(f'{prompt_dir}/SUS.txt')
        # initial_task_evaluator_system = file_to_string(f'{prompt_dir}/initial_task_evaluator_system.txt')
        
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
        
    main()