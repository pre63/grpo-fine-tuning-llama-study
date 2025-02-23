import os
import warnings
from typing import Dict, Tuple, Union

import torch
from peft import PeftModelForCausalLM, get_peft_model
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, MllamaForConditionalGeneration, MllamaProcessor

from grpo.config import get_bits_and_bytes_config, get_lora_config

# Set PyTorch memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Suppress warnings at the earliest point, before any imports
warnings.filterwarnings("ignore", category=UserWarning, module="(transformers|peft|trl).*")


def get_model(model_id: str, device_map, is_vision_model: bool) -> Union[LlamaForCausalLM, MllamaForConditionalGeneration]:
  print("Entering get_model")
  model_class = MllamaForConditionalGeneration if is_vision_model else AutoModelForCausalLM

  model = model_class.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if isinstance(device_map, dict) or device_map == "mps" else torch.float16,
    device_map=device_map,
    quantization_config=get_bits_and_bytes_config() if isinstance(device_map, dict) or device_map == "mps" else None,
  )

  model = apply_lora(model, device_map)  # Apply LoRA if needed

  print(f"Exiting get_model with model type: {type(model)}")

  return model


def get_processors(model_id: str, is_vision_model: bool) -> Dict[str, Union[AutoTokenizer, MllamaProcessor]]:
  print("Entering get_processors")

  vision = None
  if is_vision_model:
    vision = ensure_padding_token(MllamaProcessor.from_pretrained(model_id), model_id)
  text = ensure_padding_token(AutoTokenizer.from_pretrained(model_id), model_id)
  processors = {"text": text, "vision": vision}

  print(f"Exiting get_processors with processors type: {type(processors)}")
  return processors


def ensure_padding_token(processor, model_id: str) -> None:
  """Ensure the processor has a valid padding token and left padding."""
  print(f"Ensuring padding token for {model_id}")
  if hasattr(processor, "tokenizer"):  # MllamaProcessor case
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token or "<pad>"
      print(f"Set tokenizer.pad_token to {tokenizer.pad_token}")
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"  # Required by GRPOTrainer
    processor.pad_token = tokenizer.pad_token  # Sync top-level
    processor.pad_token_id = tokenizer.pad_token_id
    padding_side = tokenizer.padding_side  # Use tokenizer's attribute
  else:  # AutoTokenizer case
    if processor.pad_token is None:
      processor.pad_token = processor.eos_token or "<pad>"
      print(f"Set processor.pad_token to {processor.pad_token}")
    processor.pad_token_id = processor.convert_tokens_to_ids(processor.pad_token)
    processor.padding_side = "left"  # Required by GRPOTrainer
    padding_side = processor.padding_side  # Direct attribute

  processor.model_name = model_id
  print(f"Padding token: {processor.pad_token}, ID: {processor.pad_token_id}, Side: {padding_side}")
  return processor


def apply_lora(model, device_map) -> object:
  print("Entering apply_lora")
  lora_config = get_lora_config(device_map)
  model_with_lora = get_peft_model(model, lora_config)

  print("Exiting apply_lora")
  return model_with_lora


if __name__ == "__main__":
  print("Running test for model.py")

  import sys
  import unittest

  import torch

  from grpo.hardware import get_parameters

  class TestModelIntegration(unittest.TestCase):
    def test_model_loading_and_lora(self):
      print("Starting model integration test")
      model_id, _, _, device_map, is_vision_model = get_parameters()

      model = get_model(model_id, device_map, is_vision_model)
      processors = get_processors(model_id, is_vision_model)
      self.assertIsInstance(model, PeftModelForCausalLM)
      self.assertTrue(hasattr(model, "peft_config"))  # Check LoRA applied

      # Test tokenization with the processors, adapting for vision vs non-vision
      if is_vision_model:
        # Create a dummy image and include image token for vision processors
        dummy_image = Image.new("RGB", (10, 10), "blue")
        text_with_image_token = "<|image|> Hello"
        vision_processor = processors["vision"]
        inputs = vision_processor(text=text_with_image_token, images=[dummy_image], return_tensors="pt")
      else:
        text_processor = processors["text"]
        inputs = text_processor("Hello", return_tensors="pt")
      self.assertIn("input_ids", inputs)
      print("Model integration test completed")

  # Run tests and exit cleanly
  print("Running test suite")
  runner = unittest.TextTestRunner()
  suite = unittest.TestLoader().loadTestsFromTestCase(TestModelIntegration)
  result = runner.run(suite)
  torch.cuda.empty_cache()  # Clear any torch state
  print("Test suite completed")
  sys.exit(0 if result.wasSuccessful() else 1)
