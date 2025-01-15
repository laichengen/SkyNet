import torch.nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F


class GPTActorModel(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.actor = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, pad_token="<|endoftext|>")

    def freeze_weights(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward_actor(self,
                      x,
                      attention_mask,
                      num_actions=1):
        """
        x (B, T)
        """
        outputs = self.actor.forward(
            input_ids=x, attention_mask=attention_mask)  # logits = (B, T, voca_size)
        logits = outputs.logits
        log_prob_all_vocab = F.log_softmax(logits[:, :-1, :],
                                           dim=2)  # (B, T-1, vocab_size)
        # no need to know the logits of last token because we don't have the index of that token in x
        index = x[:, 1:].unsqueeze(-1)  # (B, T-1, 1)
        log_prob_output = log_prob_all_vocab.gather(
            dim=2,
            index=index)  # teacher-forcing, get the prob of each gt token
        return log_prob_output[:, -num_actions:, 0]  # (B, T)

    def batch_generate(self, input_ids, attention_mask):
        B, T = input_ids.size()
        completions = self.actor.generate(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = torch.where(completions != self.tokenizer.pad_token_id,
                                     torch.ones_like(completions),
                                     torch.zeros_like(completions))
        action_mask = torch.ones_like(completions, dtype=torch.bool)
        action_mask[:, :T] = 0.0
        action_mask = action_mask[:, 1:]
        # we can only take the minimum among all instances in this batch as common num_actions
        num_actions = completions.size(1) - T
        return completions, attention_mask, num_actions, action_mask[:, -num_actions:]
