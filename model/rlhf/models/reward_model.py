import json
import sys

import torch
from datasets import Dataset
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer

from model.rlhf.loss import KPairwiseLoss


class GPTRewardModel(torch.nn.Module):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPT2Model.from_pretrained(model_path)
        self.score_head = nn.Linear(768, 1, bias=False)
        self.criterion = KPairwiseLoss()

    def freeze_weights(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward_reward(self, input_ids, attention_mask):
        if len(input_ids.size()) > 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
        # input_ids = torch.tensor(input_ids)
        # attention_mask = torch.tensor(attention_mask)
        embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        score = self.score_head(embeddings.last_hidden_state).mean(dim=1)
        return score

    def forward(self, input_ids, attention_mask):
        if len(input_ids.size()) > 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
        # input_ids = torch.tensor(input_ids)
        # attention_mask = torch.tensor(attention_mask)
        embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        score = self.score_head(embeddings.last_hidden_state).mean(dim=1)
        score = torch.transpose(score, 0, 1)
        loss = self.criterion(score)
        return {'loss':loss}

