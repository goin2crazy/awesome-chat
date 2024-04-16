from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import torch
from torch.nn import DataParallel

class SumModel():
  def __init__(cfg, model_preset, **generation_parameters)->None:
    # requires transformers and peft installed libs

    cfg.model_preset = model_preset
    cfg.generation_params = generation_parameters

    config = PeftConfig.from_pretrained(cfg.model_preset)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    cfg.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    cfg.lora_model = PeftModel.from_pretrained(model, cfg.model_preset)

    cfg.lora_model.print_trainable_parameters()

    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.lora_model.model = DataParallel(cfg.lora_model.model)

  def __call__(self, text, **generation_params):
    tokens = self.tokenizer(text, return_tensors = 'pt', truncation=True, padding=True).to(self.device)
    if len(generation_params):
      gen = self.lora_model.generate(**tokens, **generation_params)
    else:
      gen = self.lora_model.generate(**tokens, **self.generation_params)

    return self.tokenizer.batch_decode(gen)