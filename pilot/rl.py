import argparse
import base64
import io

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image as PIL_Image
from sentence_transformers import SentenceTransformer, util
from transformers import MllamaForConditionalGeneration, MllamaProcessor, TorchAoConfig
from trl import GRPOConfig, GRPOTrainer

from hle.dataset import load_and_split_dataset

processor = None
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def tokenize_function(example):
  conversation = []
  images = []
  if example.get("image", "").strip():
    try:
      image_str = example["image"]
      if "," in image_str:
        image_str = image_str.split(",", 1)[1]
      image_data = base64.b64decode(image_str)
      image = PIL_Image.open(io.BytesIO(image_data)).convert("RGB")
      conversation.append(
        {
          "role": "user",
          "content": [
            {"type": "image"},
            {"type": "text", "text": example["question"]},
          ],
        }
      )
      images.append(image)
    except Exception as e:
      print(f"Error decoding image for question '{example['question']}': {e}")
      conversation.append(
        {
          "role": "user",
          "content": [{"type": "text", "text": example["question"]}],
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
  tokens = processor(text=prompt, images=images if images else None, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
  input_ids = tokens["input_ids"][0].tolist()
  labels = [-100 if token == processor.tokenizer.pad_token_id else token for token in input_ids]
  return {"input_ids": input_ids, "labels": labels, "answer": example["answer"]}


def reward_fn(prompts, completions, **kwargs):
  ground_truths = kwargs.get("ground_truths", [None] * len(completions))
  rewards = []
  for completion, truth in zip(completions, ground_truths):
    if truth:
      emb1 = embedder.encode(completion, convert_to_tensor=True)
      emb2 = embedder.encode(truth, convert_to_tensor=True)
      similarity = util.cos_sim(emb1, emb2).item()
      reward = (similarity + 1) / 2
    else:
      reward = 1.0
    rewards.append(reward)
  return rewards


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
  args = parser.parse_args()

  use_mps = torch.backends.mps.is_available() and not args.cpu
  device_map = "mps" if use_mps else "cpu"
  print(f"Using device_map: {device_map}")

  model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
  lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

  if use_mps:
    quantization_config = TorchAoConfig(quant_type="int4_weight_only", group_size=128)
    model = MllamaForConditionalGeneration.from_pretrained(
      model_id, torch_dtype=torch.bfloat16, device_map=device_map, quantization_config=quantization_config, trust_remote_code=True
    )
  else:
    print("Warning: QLoRA with int4_weight_only requires GPU/MPS. Falling back to float16 + LoRA on CPU.")
    model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True)
  model = get_peft_model(model, lora_config)

  processor = MllamaProcessor.from_pretrained(model_id)
  if hasattr(processor, "tokenizer"):
    if processor.tokenizer.pad_token is None:
      processor.tokenizer.pad_token = processor.tokenizer.bos_token or "<pad>"
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
    processor.pad_token = processor.tokenizer.pad_token
    # Expose pad_token_id at top level for GRPOTrainer
    processor.pad_token_id = processor.tokenizer.pad_token_id

  train_data, val_data, test_data = load_and_split_dataset(test_size=0.05, val_size=0.1)
  train_data = train_data.map(tokenize_function)
  train_data = train_data.remove_columns(["question", "image"])
  test_data = test_data.map(tokenize_function)
  test_data = test_data.remove_columns(["question", "image"])

  config_args = {
    "output_dir": "./results",
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 2,
    "num_train_epochs": 2,
    "num_generations": 2,
    "max_prompt_length": 128,
    "max_completion_length": 128,
    "temperature": 0.7,
    "beta": 0.1,
    "remove_unused_columns": False,
    "log_completions": True,
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
