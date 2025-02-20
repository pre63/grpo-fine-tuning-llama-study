import argparse
import base64
import io
import os
from typing import Dict, List, Optional, Union

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image as PIL_Image
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MllamaForConditionalGeneration, MllamaProcessor
from trl import GRPOConfig, GRPOTrainer

from hle.dataset import load_and_split_dataset

# Set PyTorch memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() if torch.cuda.is_available() else None
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def decode_base64_image(image_str: str) -> Optional[PIL_Image.Image]:
  try:
    if "," in image_str:
      image_str = image_str.split(",", 1)[1]
    image_data = base64.b64decode(image_str)
    return PIL_Image.open(io.BytesIO(image_data)).convert("RGB")
  except Exception as e:
    print(f"Error decoding image: {e}")
    return None


def build_conversation(example: Dict, is_vision_model: bool) -> List[Dict]:
  user_content = [{"type": "text", "text": example["question"]}]
  if is_vision_model and example.get("image", "").strip():
    image = decode_base64_image(example["image"])
    user_content = [{"type": "image"}] + user_content if image else user_content
  return [
    {"role": "user", "content": user_content},
    {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
  ]


def tokenize_prompt(processor, conversation: List[Dict], images: List[PIL_Image.Image]) -> Dict:
  prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
  is_vision_model = "vision" in processor.model_name.lower()
  tokens = (
    processor(text=prompt, images=images if is_vision_model and images else None, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    if is_vision_model
    else processor(prompt, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
  )
  input_ids = tokens["input_ids"][0].tolist()
  labels = [-100 if token == processor.pad_token_id else token for token in input_ids]
  return {"input_ids": input_ids, "labels": labels, "answer": conversation[1]["content"][0]["text"]}


def tokenize_example(example: Dict, processor) -> Dict:
  is_vision_model = "vision" in processor.model_name.lower()
  images = [decode_base64_image(example["image"]) or None] if is_vision_model and example.get("image", "").strip() else []
  conversation = build_conversation(example, is_vision_model)
  return tokenize_prompt(processor, conversation, [img for img in images if img])


def compute_reward(prompts: List[str], completions: List[str], ground_truths: List[Optional[str]]) -> List[float]:
  rewards = []
  for prompt, completion, truth in zip(prompts, completions, ground_truths or [None] * len(completions)):
    reward = (
      1.0 if not truth else (util.cos_sim(embedder.encode(completion, convert_to_tensor=True), embedder.encode(truth, convert_to_tensor=True)).item() + 1) / 2
    )
    rewards.append(reward)
  return rewards


def get_device_map(cpu: bool) -> str:
  use_cuda = torch.cuda.is_available() and not cpu
  use_mps = torch.backends.mps.is_available() and not cpu and not use_cuda
  return "cuda" if use_cuda else "mps" if use_mps else "cpu"


def load_model_and_processor(model_id: str, device_map: str) -> tuple:
  is_vision_model = "vision" in model_id.lower()
  quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
  model_class = MllamaForConditionalGeneration if is_vision_model else AutoModelForCausalLM
  processor_class = MllamaProcessor if is_vision_model else AutoTokenizer

  model = model_class.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device_map in ["cuda", "mps"] else torch.float16,
    device_map=device_map,
    quantization_config=quantization_config if device_map in ["cuda", "mps"] else None,
    trust_remote_code=True,
  )
  processor = processor_class.from_pretrained(model_id)

  if device_map not in ["cuda", "mps"]:
    print("Warning: QLoRA with bitsandbytes requires GPU/MPS. Using float16 + LoRA on CPU.")

  if hasattr(processor, "tokenizer"):
    if processor.tokenizer.pad_token is None:
      processor.tokenizer.pad_token = processor.tokenizer.bos_token or "<pad>"
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
    processor.pad_token = processor.tokenizer.pad_token
    processor.pad_token_id = processor.tokenizer.pad_token_id
  elif processor.pad_token is None:
    processor.pad_token = processor.bos_token or "<pad>"
    processor.pad_token_id = processor.convert_tokens_to_ids(processor.pad_token)

  processor.model_name = model_id
  return model, processor


def apply_lora(model) -> object:
  lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
  return get_peft_model(model, lora_config)


def get_config() -> GRPOConfig:
  return GRPOConfig(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    num_generations=2,
    max_prompt_length=32,
    max_completion_length=32,
    temperature=0.7,
    beta=0.1,
    remove_unused_columns=False,
    log_completions=True,
  )


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
    "reward_funcs": lambda p, c, **kw: compute_reward(p, c, kw.get("ground_truths", [d["answer"] for d in train_data])),
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
