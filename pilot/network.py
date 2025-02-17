import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from peft import get_peft_model, get_peft_model_state_dict
from torch.distributions import Categorical
from transformers import PretrainedConfig, PreTrainedModel


class GRPONetwork(nn.Module):
  def __init__(self, model, device="cpu"):
    super().__init__()
    self.model = model.to(device)
    self.device = device

  def forward(self, input_ids, attention_mask=None):
    input_ids = input_ids.to(self.device)
    if attention_mask is not None:
      attention_mask = attention_mask.to(self.device)
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False)
    logits = outputs.logits[:, -1, :]
    return logits


class GroupBuffer:
  def __init__(self, max_size=5, device="cpu"):
    self.max_size = max_size
    self.policies = []
    self.returns = []
    self.device = device

  def add(self, policy_state_dict, avg_return):
    if len(self.policies) >= self.max_size:
      self.policies.pop(0)
      self.returns.pop(0)
    self.policies.append(policy_state_dict)
    self.returns.append(avg_return)

  def calculate_relative_advantage(self, rewards):
    if not rewards:
      return []
    group_mean = np.mean(rewards)
    group_std = np.std(rewards) + 1e-8
    return (np.array(rewards) - group_mean) / group_std

  def mean_return(self):
    return sum(self.returns) / len(self.returns) if self.returns else 0


def calculate_kl_divergence(new_logits, ref_logits):
  dist_new = Categorical(logits=new_logits)
  dist_ref = Categorical(logits=ref_logits)
  kl_div = torch.distributions.kl.kl_divergence(dist_new, dist_ref)
  return kl_div


def test_group_buffer(device="cpu"):
  buffer = GroupBuffer(max_size=3)

  # Add mock policies and returns
  buffer.add({"layer1": torch.tensor([1.0], device=device)}, 1.0)
  buffer.add({"layer1": torch.tensor([2.0], device=device)}, 2.0)
  buffer.add({"layer1": torch.tensor([3.0], device=device)}, 3.0)

  assert len(buffer.policies) == 3, f"Expected 3 policies, got {len(buffer.policies)}"
  assert len(buffer.returns) == 3, f"Expected 3 returns, got {len(buffer.returns)}"
  assert buffer.mean_return() == 2.0, f"Expected mean return 2.0, got {buffer.mean_return()}"

  # Add another policy, causing the oldest one to be removed
  buffer.add({"layer1": torch.tensor([4.0], device=device)}, 4.0)
  assert len(buffer.policies) == 3, f"Expected buffer size 3, got {len(buffer.policies)}"
  assert buffer.returns == [2.0, 3.0, 4.0], f"Unexpected returns list: {buffer.returns}"

  # Test relative advantage calculation
  rewards = [2.0, 4.0, 6.0]
  advantages = buffer.calculate_relative_advantage(rewards)
  assert len(advantages) == 3, f"Expected 3 advantages, got {len(advantages)}"
  assert np.isclose(np.mean(advantages), 0), f"Expected mean advantage 0, got {np.mean(advantages)}"
  assert np.isclose(np.std(advantages), 1), f"Expected std 1, got {np.std(advantages)}"

  print("All GroupBuffer tests passed.")


def test_grpo_network(device="cpu"):

  class DummyLlamaConfig(PretrainedConfig):
    def __init__(self, vocab_size=32000, hidden_size=128, num_hidden_layers=2, num_attention_heads=2, **kwargs):
      super().__init__(**kwargs)
      self.vocab_size = vocab_size
      self.hidden_size = hidden_size
      self.num_hidden_layers = num_hidden_layers
      self.num_attention_heads = num_attention_heads

  class DummyLlamaModel(PreTrainedModel):
    config_class = DummyLlamaConfig

    def __init__(self, config):
      super().__init__(config)
      self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
      self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False):
      x = self.embedding(input_ids)
      logits = self.fc(x)
      return type("Output", (object,), {"logits": logits})

  config = DummyLlamaConfig()
  dummy_model = DummyLlamaModel(config)
  grpo_model = GRPONetwork(dummy_model, device=device)

  batch_size = 2
  seq_length = 5
  input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

  output = grpo_model(input_ids)
  assert output.shape == (batch_size, config.vocab_size), f"Expected shape {(batch_size, config.vocab_size)}, got {output.shape}"

  print("Test passed: GRPONetwork produces the expected output shape.")


if __name__ == "__main__":
  # Choose device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  test_grpo_network(device=device)
  test_group_buffer(device=device)
