import json
import sys

import torch
from datasets import Dataset
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer

from model.rlhf.loss import KPairwiseLoss


class GPTCriticModel(torch.nn.Module):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPT2Model.from_pretrained(model_path)
        self.score_head = nn.Linear(768, 1, bias=False)
        self.criterion = KPairwiseLoss()

    def forward_critic(self, input_ids, attention_mask, num_actions=0):
        embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        score = self.score_head(embeddings.last_hidden_state).mean(dim=1)
        score = score * attention_mask
        score = score[:, -num_actions:].mean(dim=1)
        return score  # (B, 1)


