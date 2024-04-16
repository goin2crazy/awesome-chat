from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

class ChatModel():
    @staticmethod
    def join_dialogue(d):
      s = ""

      for message in d:
        if message['role'] == 'user':
          s += f"You: {message['content']}\n"
        else:
          s += f"[CHARACTER]: {message['content']}\n"

      return s

    def __init__(cfg, preset, chat_summarization_fn, **generation_parameters)->None:
        # requires transformers and peft installed libs
        pygmalion_config = PeftConfig.from_pretrained(preset)

        pygmalion_model = AutoModelForCausalLM.from_pretrained(pygmalion_config.base_model_name_or_path)
        cfg.tokenizer = AutoTokenizer.from_pretrained(pygmalion_config.base_model_name_or_path)

        model = PeftModel.from_pretrained(pygmalion_model, preset)
        
        cfg.summ = chat_summarization_fn
        cfg.generation_params = generation_parameters
        
        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg.model = model.to(cfg.device)

    def __call__(self, d, messages_split:int = 6, **generation_params):
        print("0/1 Summarization process..")
        s_text = self.join_dialogue(d[:-messages_split])
        s = self.summ(s_text)
        
        text = f"*HISTORY: {s}* {self.join_dialogue(d[-messages_split:])}"
        print(f"1/1 Summarized Dialogue: {text}")
        
        tokens = self.tokenizer(text, return_tensors = 'pt', truncation=True, padding=True).to(self.device)
        if len(generation_params):
            gen = self.model.generate(**tokens, **generation_params)
        else:
            gen = self.model.generate(**tokens, **self.generation_params)

        answer_dialogue = sum(self.tokenizer.batch_decode(gen))
        answer_message = answer_dialogue.replace(text.strip(), '')
        
        return answer_message, answer_dialogue