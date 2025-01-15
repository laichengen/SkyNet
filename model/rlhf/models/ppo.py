from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM

from model.rlhf.loss import PolicyLoss, ValueLoss
from model.rlhf.models.actor_model import GPTActorModel
from model.rlhf.models.critic_model import GPTCriticModel
from model.rlhf.models.reference_model import GPTRefModel
from model.rlhf.models.reward_model import GPTRewardModel


@dataclass
class Experience:
    completion: torch.Tensor
    actor_log_probs: torch.Tensor
    attention_mask: torch.Tensor
    kl_penalized_reward: torch.Tensor
    advantage: torch.Tensor
    num_actions: int
    estimated_kl: torch.Tensor
    values: torch.Tensor
    action_mask: torch.Tensor


class PPO(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.actor = GPTActorModel(path)
        self.reference = GPTRefModel(path)
        self.reference.freeze_weights()
        self.reward = GPTRewardModel(path)
        self.reward.freeze_weights()
        self.critic = GPTCriticModel(path)
        self.kl_beta = 0.1
        self.ppo_epoches = 2
        self.actor_criterion = PolicyLoss()
        self.critic_criterion = ValueLoss()

    def make_experience(self, input_ids, attention_mask):
        completion, attention_mask, num_actions, action_mask = self.actor.batch_generate(input_ids, attention_mask)
        actor_log_probs = self.actor.forward_actor(
            completion,
            attention_mask,  # (B, num_actions)
            num_actions)
        reference_log_probs = self.reference.forward_actor(
            completion,
            attention_mask,  # (B, num_actions)
            num_actions)
        values = self.critic.forward_critic(completion,
                                            attention_mask, num_actions).view(-1, 1)
        reward = self.reward.forward_reward(completion, attention_mask)  # (B, 1)
        kl_penalized_reward, estimated_kl = self.kl_penalized_reward(
            reward, actor_log_probs, reference_log_probs)
        advantage = kl_penalized_reward - values
        return Experience(
            completion, actor_log_probs, attention_mask, kl_penalized_reward, advantage, num_actions, estimated_kl,
            values, action_mask)

    def kl_penalized_reward(self, reward, log_prob_rl, log_prob_sft, action_mask=None):
        # log(π_RL(y|x) / π_SFL(y|x)) = log(π_RL(y|x)) - log(π_SFL(y|x))
        ratio = log_prob_rl - log_prob_sft
        # k3 in http://joschu.net/blog/kl-approx.html
        estimated_kl = (torch.exp(ratio) - 1) - ratio
        if action_mask:
            estimated_kl = estimated_kl * action_mask
            estimated_kl.sum(dim=1) / action_mask.sum(dim=1)
        estimated_kl = estimated_kl.mean(
            dim=1, keepdim=True)  # estimated_kl -> (B, 1)
        return reward - self.kl_beta * estimated_kl, estimated_kl

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        experience = self.make_experience(input_ids, attention_mask)
        total_loss = 0
        for _ in range(self.ppo_epoches):
            curr_actor_log_probs = self.actor.forward_actor(
                experience.completion, experience.attention_mask, experience.num_actions)

            actor_loss = self.actor_criterion(curr_actor_log_probs,
                                              experience.actor_log_probs,
                                              experience.advantage,
                                              experience.action_mask)
            new_values = self.critic.forward_critic(
                                    experience.completion, experience.attention_mask, experience.num_actions).view(-1, 1)
            critic_loss = self.critic_criterion(new_values, experience.kl_penalized_reward, experience.values,
                                                experience.action_mask)
            current_loss = actor_loss + critic_loss
            total_loss += current_loss
        return {'loss': total_loss}