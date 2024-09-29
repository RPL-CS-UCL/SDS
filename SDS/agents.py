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
        self.temperature = 0.8
    
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
    
class TaskDescriptor(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/task_descriptor_system.txt"
        super().__init__(system_prompt_file, cfg)
    def analyse(self,encoded_frame_grid):
        self.prepare_user_content([{"type":"image_uri","data":encoded_frame_grid}])
        return self.query()

class ContactSequenceAnalyser(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/contact_sequence_system.txt"
        super().__init__(system_prompt_file, cfg)
    def analyse(self,encoded_frame_grid):
        self.prepare_user_content([{"type":"image_uri","data":encoded_frame_grid}])
        
        contact_sequence = self.query()
        
        self.prepare_user_content([{"type":"text","data":"Revise the contact sequence that you just generated with the provided image containing sequential frames of a video. Check frame by frame to make sure it is correct. If it is, give your reasoning, if not fix the error."}])
        
        return self.query()
    
class TaskRequirementAnalyser(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/task_requirement_system.txt"
        super().__init__(system_prompt_file, cfg)
    def analyse(self,encoded_frame_grid):
        self.prepare_user_content([{"type":"image_uri","data":encoded_frame_grid}])
        
        return self.query()

class GaitAnalyser(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/gait_pattern_system.txt"
        self.prompt_dir = prompt_dir
        super().__init__(system_prompt_file, cfg)
    
    def analyse(self,encoded_frame_grid,contact_pattern):
        self.prepare_user_content([{"type":"image_uri","data":encoded_frame_grid},{"type":"text","data":f"For the provided sequential frames, you are provided with a likely corresponding feet contact pattern: {contact_pattern}"}])
        gait_pattern_response = self.query()

        return gait_pattern_response
    
class SUSGenerator(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/SUS_generation_prompt.txt"
        self.prompt_dir = prompt_dir
        super().__init__(system_prompt_file, cfg)
    
    def generate_sus_prompt(self,encoded_gt_frame_grid):
        task_descriptor = TaskDescriptor(self.cfg,self.prompt_dir)
        task_description = task_descriptor.analyse(encoded_gt_frame_grid)   
        
        contact_sequence_analyser = ContactSequenceAnalyser(self.cfg,self.prompt_dir)
        contact_pattern = contact_sequence_analyser.analyse(encoded_gt_frame_grid)
        
        gait_analyser = GaitAnalyser(self.cfg,self.prompt_dir)
        gait_response = gait_analyser.analyse(encoded_gt_frame_grid,contact_pattern)
        
        task_requirement_analyser = TaskRequirementAnalyser(self.cfg,self.prompt_dir)
        task_requirement_response = task_requirement_analyser.analyse(encoded_gt_frame_grid)
        
        self.prepare_user_content([{"type":"text","data":task_description},{"type":"text","data":gait_response},{"type":"text","data":task_requirement_response}])
        
        sus_prompt = self.query()
        
        return sus_prompt