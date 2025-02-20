import argparse
import base64
import io

import torch
from PIL import Image as PIL_Image
from sentence_transformers import SentenceTransformer, util
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from trl import GRPOConfig, GRPOTrainer

from hle.dataset import load_and_split_dataset

# Global processor and embedder
processor = None
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model for embeddings


def apply_chat_template(example):
  conversation = []
  if example.get("image", "").strip():
    # Decode base64 string to PIL Image
    image_data = base64.b64decode(example["image"])
    image = PIL_Image.open(io.BytesIO(image_data)).convert("RGB")
    conversation.append(
      {
        "role": "user",
        "content": [
          {"type": "image", "data": image},  # Pass PIL Image
          {"type": "text", "text": example["question"]},
        ],
      }
    )
  else:
    conversation.append(
      {
        "role": "user",
        "content": [{"type": "text", "text": example["question"]}],
      }
    )
  conversation.append(
    {
      "role": "assistant",
      "content": [{"type": "text", "text": example["answer"]}],
    }
  )
  prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
  return {"prompt": prompt, "answer": example["answer"]}


def tokenize_function(example):
  tokens = processor(example["prompt"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
  input_ids = tokens["input_ids"][0].tolist()
  labels = [-100 if token == processor.pad_token_id else token for token in input_ids]
  tokens["input_ids"] = input_ids
  tokens["labels"] = labels
  tokens["answer"] = example["answer"]  # Pass answer through
  return tokens


def reward_fn(prompts, completions, **kwargs):
  # Extract ground truth answers from kwargs
  ground_truths = kwargs.get("ground_truths", [None] * len(completions))
  rewards = []

  for completion, truth in zip(completions, ground_truths):
    if truth:
      # Compute embeddings for completion and ground truth
      emb1 = embedder.encode(completion, convert_to_tensor=True)
      emb2 = embedder.encode(truth, convert_to_tensor=True)
      # Cosine similarity (range: -1 to 1)
      similarity = util.cos_sim(emb1, emb2).item()
      # Normalize to [0, 1]
      reward = (similarity + 1) / 2
    else:
      reward = 1.0  # Default if no ground truth
    rewards.append(reward)
  return rewards


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
  args = parser.parse_args()
  device_map = "cpu" if args.cpu else "auto"

  model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
  model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32, device_map=device_map, trust_remote_code=True)
  processor = MllamaProcessor.from_pretrained(model_id)
  processor.pad_token = processor.eos_token

  train_data, val_data, test_data = load_and_split_dataset(test_size=0.05, val_size=0.1)

  # Process datasets and keep the answer column
  train_data = train_data.map(apply_chat_template)
  train_data = train_data.map(tokenize_function)
  train_data = train_data.remove_columns(["question", "image"])  # Keep "answer"

  test_data = test_data.map(apply_chat_template)
  test_data = test_data.map(tokenize_function)
  test_data = test_data.remove_columns(["question", "image"])  # Keep "answer"

  config_args = {
    "output_dir": "./results",
    "learning_rate": 1e-5,
    "per_device_train_batch_size": 2,
    "num_train_epochs": 2,
    "num_generations": 2,
    "vllm_device": device_map,
    "no_cuda": args.cpu,
  }

  config = GRPOConfig(**config_args)

  trainer_args = {
    "model": model,
    "reward_funcs": lambda p, c, **kw: reward_fn(p, c, ground_truths=[d["answer"] for d in train_data], **kw),
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
