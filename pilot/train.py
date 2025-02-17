import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.distributions import Categorical
from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # Remove this line to enable GPU training

from network import GroupBuffer, GRPONetwork, calculate_kl_divergence


def load_and_split_dataset():
  data = load_dataset("cais/hle")
  data = data["test"].train_test_split(test_size=0.2)
  train_data = data["test"].train_test_split(test_size=0.125)
  val_data = train_data["test"]
  train_data = train_data["train"]
  test_data = data["test"]
  return train_data, val_data, test_data


def tokenize_function(example, tokenizer):
  return tokenizer(example["question"], truncation=True, padding="max_length", max_length=128)


@torch.no_grad()
def generate_text(policy_model, tokenizer, input_ids, max_new_tokens=50):
  input_ids = input_ids.to(device)
  for _ in range(max_new_tokens):
    logits = policy_model(input_ids)
    dist = Categorical(logits=logits)
    next_token = dist.sample()
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
  return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def reward_function(generated_text):
  if "correct" in generated_text.lower():
    return 1.0
  return -0.5


def data_collator(batch):
  input_ids = torch.tensor([f["input_ids"] for f in batch], dtype=torch.long, device=device)
  attention_mask = torch.tensor([f["attention_mask"] for f in batch], dtype=torch.long, device=device)
  return {"input_ids": input_ids, "attention_mask": attention_mask}


def train_grpo_llama():
  train_data, val_data, test_data = load_and_split_dataset()
  tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
  tokenizer.pad_token = tokenizer.eos_token

  train_data = train_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
  val_data = val_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
  test_data = test_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)

  # BitsAndBytes 4-bit quantization configuration
  bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, llm_int8_enable_fp32_cpu_offload=True)

  # Load policy model with quantization
  policy_llama = LlamaForCausalLM.from_pretrained(
    "huggyllama/llama-7b", quantization_config=bnb_config, device_map={"": device}  # Ensure proper device mapping
  )

  peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
  policy_llama = get_peft_model(policy_llama, peft_config)
  policy_model = GRPONetwork(policy_llama, device=device).eval()

  # Load reference model
  reference_llama = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b", device_map={"": device})
  reference_model = GRPONetwork(reference_llama, device=device).eval()

  ref_sd = get_peft_model_state_dict(policy_model.model)
  optimizer = optim.Adam(policy_model.parameters(), lr=1e-5)
  group_buffer = GroupBuffer(max_size=5, device=device)

  gamma = 0.99
  epsilon = 0.2
  beta = 0.02
  epochs = 3
  steps_per_group = 4
  batch_size = 1
  dataset_iter = iter(train_data)

  # Ensure all bnb layers are properly initialized on the device
  for module in policy_model.modules():
    if isinstance(module, bnb.nn.LinearFP4):
      module.to(device)

  for epoch in range(epochs):
    print("New Epoch:", epoch)
    reference_model.load_state_dict(ref_sd, strict=False)
    group_returns = []
    input_batches = []

    for _ in range(steps_per_group):
      try:
        batch = [next(dataset_iter) for __ in range(batch_size)]
      except StopIteration:
        dataset_iter = iter(train_data)
        batch = [next(dataset_iter) for __ in range(batch_size)]

      collated = data_collator(batch)
      input_ids = collated["input_ids"]
      attention_mask = collated["attention_mask"]

      with torch.no_grad():
        ref_logits = reference_model(input_ids, attention_mask)
        logits = policy_model(input_ids, attention_mask)

      dist = Categorical(logits=logits)
      actions = dist.sample()
      log_probs_old = dist.log_prob(actions)
      text_gens = [generate_text(policy_model, tokenizer, input_ids[i].unsqueeze(0)) for i in range(input_ids.size(0))]

      rewards = torch.tensor([reward_function(txt) for txt in text_gens], dtype=torch.float, device=device)
      group_returns.append(rewards.mean().item())

      input_batches.append((input_ids, attention_mask, actions, log_probs_old, rewards, ref_logits))

      del ref_logits, logits
      torch.cuda.empty_cache()

    advantages = torch.tensor(group_buffer.calculate_relative_advantage(group_returns), dtype=torch.float, device=device)

    for i, (inp, attn, acts, logp_old, rews, ref_logits) in enumerate(input_batches):
      advantage = advantages[i].expand_as(acts)
      logits = policy_model(inp, attn)
      dist_new = Categorical(logits=logits)
      log_probs_new = dist_new.log_prob(acts)

      ratio = torch.exp(log_probs_new - logp_old)
      surr1 = ratio * advantage
      surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
      policy_loss = -torch.min(surr1, surr2).mean()

      kl_loss = beta * calculate_kl_divergence(logits.detach(), ref_logits).mean()
      loss = policy_loss + kl_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      del logits, dist_new, log_probs_new, ratio, surr1, surr2, policy_loss, kl_loss, loss
      torch.cuda.empty_cache()

    group_avg = np.mean(group_returns)
    group_buffer.add(get_peft_model_state_dict(policy_model.model), group_avg)
    if group_avg > 0:
      ref_sd = get_peft_model_state_dict(policy_model.model)

  policy_llama.save_pretrained("grpo_llama_final")
  torch.cuda.empty_cache()


if __name__ == "__main__":
  train_grpo_llama()
