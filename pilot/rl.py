import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from hle.dataset import load_and_split_dataset


def apply_chat_template(example):
  messages = [{"role": "user", "content": example["question"]}, {"role": "assistant", "content": example["answer"]}]
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  return {"prompt": prompt}


def tokenize_function(example):
  tokens = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128)
  tokens["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in tokens["input_ids"]]
  return tokens


def reward_fn(prompts, completions, **kwargs):
  return [1.0] * len(completions)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
  args = parser.parse_args()
  device_map = "cpu" if args.cpu else "auto"

  model_id = "meta-llama/Llama-3.2-1B-Instruct"
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map=device_map)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token

  train_data, val_data, test_data = load_and_split_dataset(test_size=0.05, val_size=0.1)

  train_data = train_data.map(apply_chat_template)
  train_data = train_data.map(tokenize_function)
  train_data = train_data.remove_columns(["question", "answer"])

  test_data = test_data.map(apply_chat_template)
  test_data = test_data.map(tokenize_function)
  test_data = test_data.remove_columns(["question", "answer"])

  config = GRPOConfig(output_dir="./results", learning_rate=1e-5, per_device_train_batch_size=2, num_train_epochs=2, num_generations=2)

  trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, train_dataset=train_data, eval_dataset=test_data, processing_class=tokenizer, args=config)

  trainer.train()
  model.save_pretrained("./fine-tuned-model")
  tokenizer.save_pretrained("./fine-tuned-model")
  print("Model and tokenizer saved.")
