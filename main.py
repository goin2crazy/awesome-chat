from summ import SumModel
from chat import ChatModel
from typing import Iterable

class MinimalInterface():
    def __init__(self, 
                 summ_preset = 'doublecringe123/bardt-large-cnn-dialoguesum-booksum-lora', 
                 chat_preset = "doublecringe123/pygmalion-dialogsum-empathetic_dialogues_llm", 
                 summ_params = None, 
                 chat_params = None, 
                 messages_split = 6, 
                 ): 
    
        if summ_params == None: 
            summ_params = {  'max_length' : 96,
                            'min_length' : 26,
                            'do_sample' : True,
                            'temperature' : 0.9,
                            'num_beams' : 8,
                            'repetition_penalty': 2.}

        sum_model = SumModel(model_preset = summ_preset, **summ_params)
        summ = lambda t: sum_model(t)[0]

        if chat_params == None: 
            chat_params = {
                'max_new_tokens' : 50, 
                'min_new_tokens': 5, 
                'do_sample' : True, 
                'temperature' : 0.9, 
                'repetition_penalty': 2.
            }

        self.chat_model = ChatModel(preset = chat_preset, 
                            chat_summarization_fn = summ,  
                            **chat_params)
        
        self.ms = messages_split
        self.history = []

    def __call__(self, message_or_messages) -> str:
        """
        Predict and adds new message to history
        
        message_or_messages: 
            Assuming that it is message: str. For example: "Hey, can you tell pizza recipe?~" 

            Or dialogue: list[dict()], it have to look like: 
                [
                {"content": "Hey, can you tell margarita pizza recipe?~", "role": "user"}, 
                {"content": " ... ", "role": "bot"}
                ]
        """

        if type(message_or_messages) == str: 
            self.history += [{'content': message_or_messages, 'role': 'user'}]

        if type(message_or_messages) == Iterable: 
            self.history + message_or_messages

        answer_message, answer_dialogue = self.chat_model(self.history, self.ms)
        # adding to history 
        self.history += [{'content': (answer_message
                                      .replace('[CHARACTER]', '')
                                      .replace('<|endoftext|>', '')), 
                            'role': 'bot'}]

        return answer_message, answer_dialogue


    