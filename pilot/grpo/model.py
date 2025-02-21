import logging
import os
import warnings
from typing import Dict, Tuple, Union

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  LlamaForCausalLM,
  MllamaForConditionalGeneration,
  MllamaProcessor,
  PreTrainedTokenizerBase,
)

# Set PyTorch memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Suppress warnings at the earliest point, before any imports
warnings.filterwarnings("ignore", category=UserWarning, module="(transformers|peft|trl).*")

# Set up logging to trace execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_processor(model_id: str, device_map: Union[str, Dict[str, int]], is_vision_model) -> Tuple:
  logger.info("Entering load_model_and_processor")
  quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
  model_class = MllamaForConditionalGeneration if is_vision_model else AutoModelForCausalLM

  model = model_class.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if isinstance(device_map, dict) or device_map == "mps" else torch.float16,
    device_map=device_map,
    quantization_config=quantization_config if isinstance(device_map, dict) or device_map == "mps" else None,
  )

  if device_map == "cpu":
    print("Warning: QLoRA with bitsandbytes requires GPU/MPS. Using float16 + LoRA on CPU.")

  vision = None
  if is_vision_model:
    vision = ensure_padding_token(MllamaProcessor.from_pretrained(model_id), model_id)
  text = ensure_padding_token(AutoTokenizer.from_pretrained(model_id), model_id)
  processors = {"text": text, "vision": vision}

  logger.info(f"Exiting load_model_and_processor with model type: {type(model).__name__}, processors type: {type(processors)}")
  return model, processors


def ensure_padding_token(processor, model_id: str) -> None:
  """Ensure the processor has a valid padding token."""

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

  return processor


def apply_lora(model) -> object:
  logger.info("Entering apply_lora")
  lora_config = LoraConfig(r=2, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
  model_with_lora = get_peft_model(model, lora_config)

  logger.info("Exiting apply_lora")
  return model_with_lora


if __name__ == "__main__":
  print("Running test for model.py")

  import sys
  import unittest

  import torch

  from grpo.hardware import get_parameters

  class TestModelIntegration(unittest.TestCase):
    def test_model_loading_and_lora(self):
      logger.info("Starting model integration test")
      model_id, _, _, device_map, is_vision_model = get_parameters()

      model, processors = load_model_and_processor(model_id, device_map, is_vision_model)
      expected_model_class = MllamaForConditionalGeneration if is_vision_model else LlamaForCausalLM
      self.assertIsInstance(model, expected_model_class)  # Dynamic model class check
      model_with_lora = apply_lora(model)
      self.assertTrue(hasattr(model_with_lora, "peft_config"))  # Check LoRA applied

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
      logger.info("Model integration test completed")

  # Run tests and exit cleanly
  logger.info("Running test suite")
  runner = unittest.TextTestRunner()
  suite = unittest.TestLoader().loadTestsFromTestCase(TestModelIntegration)
  result = runner.run(suite)
  torch.cuda.empty_cache()  # Clear any torch state
  logger.info("Test suite completed")
  sys.exit(0 if result.wasSuccessful() else 1)
