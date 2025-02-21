import argparse
import os

import torch
from trl import GRPOTrainer

from grpo.config import get_config
from grpo.conversation import tokenize_example
from grpo.hardware import get_device_map
from grpo.model import apply_lora, load_model_and_processor
from grpo.reward import compute_reward
from hle.dataset import load_and_split_dataset

# Set PyTorch memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
  parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
  args = parser.parse_args()

  device_map = get_device_map(args.cpu)
  print(f"Using device_map: {device_map}")

  model, processor = load_model_and_processor(args.model, device_map)
  model = apply_lora(model)

  train_data, val_data, test_data = load_and_split_dataset(test_size=0.05, val_size=0.1)
  train_data = train_data.map(lambda x: tokenize_example(x, processor)).remove_columns(["question", "image"])
  test_data = test_data.map(lambda x: tokenize_example(x, processor)).remove_columns(["question", "image"])

  config = get_config()
  trainer_args = {
    "model": model,
    "reward_funcs": lambda prompts, completions, **kw: compute_reward(prompts, completions, kw.get("ground_truths", [d["answer"] for d in train_data])),
    "train_dataset": train_data,
    "eval_dataset": test_data,
    "processing_class": processor,
    "args": config,
  }
  trainer = GRPOTrainer(**trainer_args)

  trainer.train()
  model.save_pretrained("./fine-tuned-model")
  processor.save_pretrained("./fine-tuned-model")
  print("Model and processor saved.")
