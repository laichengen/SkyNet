import torch
from transformers import AutoTokenizer,  AutoModelForCausalLM
from qwen_vl_utils import process_vision_info


class BaseModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "checkpoints/Qwen2-Coder-0.5B-Instruct",  device_map="cuda", torch_dtype=torch.bfloat16, load_in_8bit=True)
        self.processor = AutoTokenizer.from_pretrained("checkpoints/Qwen2-Coder-0.5B-Instruct")

    def inference(self, text):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.processor([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response