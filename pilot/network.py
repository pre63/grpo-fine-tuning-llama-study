import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.distributions import Categorical
from transformers import PretrainedConfig, PreTrainedModel


def compute_loss(logits, targets):
  return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def calculate_kl_divergence(new_logits, ref_logits):
  device = new_logits.device
  ref_logits = ref_logits.to(device)
  dist_new = Categorical(logits=new_logits)
  dist_ref = Categorical(logits=ref_logits)
  kl_div = torch.distributions.kl.kl_divergence(dist_new, dist_ref).to(device)
  return kl_div


class GRPONetwork(nn.Module):
  def __init__(self, model, device="cpu"):
    super().__init__()
    self.model = model.to(device)
    self.device = device

  def forward(self, input_ids, attention_mask=None):
    input_ids = input_ids.to(self.device)
    # Call the underlying model with the expected keyword "tokens".
    outputs = self.model(tokens=input_ids)
    logits = outputs.logits[:, -1, :].to(self.device)
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
    self.policies.append({k: v.to(self.device) for k, v in policy_state_dict.items()})
    self.returns.append(avg_return)

  def calculate_relative_advantage(self, rewards):
    if not rewards:
      return []
    group_mean = np.mean(rewards)
    group_std = np.std(rewards) + 1e-8
    return (np.array(rewards) - group_mean) / group_std

  def mean_return(self):
    return sum(self.returns) / len(self.returns) if self.returns else 0


# Dummy model for testing QLoRA integration.
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
    # We'll use a single linear layer as the "head" that we want to adapt.
    self.fc = nn.Linear(config.hidden_size, config.vocab_size)

  def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False):
    device = input_ids.device
    x = self.embedding(input_ids.to(device))
    # For simplicity, we assume a single timestep output.
    logits = self.fc(x).to(device)
    return type("Output", (object,), {"logits": logits})


def test_q_lora_grpo_network(device="cpu"):
  config = DummyLlamaConfig()
  base_model = DummyLlamaModel(config)

  # Define a LoRA configuration.
  # Note: The 'quantize_base' flag is not supported by LoraConfig.
  lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["fc"], lora_dropout=0.0, bias="none")  # Apply LoRA to the linear head.

  # Wrap the base model with PEFT to inject LoRA adapters.
  qlora_model = get_peft_model(base_model, lora_config)

  # For testing purposes, print trainable parameters.
  trainable = [n for n, p in qlora_model.named_parameters() if p.requires_grad]
  print("Trainable parameters:", trainable)

  # Wrap the LoRA-enabled model with GRPONetwork.
  grpo_model = GRPONetwork(qlora_model, device=device)

  batch_size = 2
  seq_length = 5
  input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)

  output = grpo_model(input_ids)
  expected_shape = (batch_size, config.vocab_size)
  assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

  print("Test passed: GRPONetwork with QLoRA model produces the expected output shape.")


if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  test_q_lora_grpo_network(device=device)
