import logging
import warnings
from typing import Dict, Tuple, Union

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  LlamaForCausalLM,
  MllamaForConditionalGeneration,
  MllamaProcessor,
  PreTrainedTokenizerBase,
)

# Set up logging to debug warning source
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_processor(model_id: str, device_map: Union[str, Dict[str, int]]) -> Tuple:
  is_vision_model = "vision" in model_id.lower()
  quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
  model_class = MllamaForConditionalGeneration if is_vision_model else AutoModelForCausalLM
  processor_class = MllamaProcessor if is_vision_model else AutoTokenizer

  model = model_class.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if isinstance(device_map, dict) or device_map == "mps" else torch.float16,
    device_map=device_map,
    quantization_config=quantization_config if isinstance(device_map, dict) or device_map == "mps" else None,
    trust_remote_code=True,
  )
  processor = processor_class.from_pretrained(model_id)

  if device_map == "cpu":
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


if __name__ == "__main__":
  import sys
  import unittest

  import torch

  from grpo.hardware import get_device_map

  # Suppress PEFT-related warnings more broadly
  warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")

  class TestModelIntegration(unittest.TestCase):
    def test_model_loading_and_lora(self):
      logger.info("Starting model integration test")
      # Use Llama-3.2-1B-Instruct for integration testing
      model_id = "meta-llama/Llama-3.2-1B-Instruct"
      device_map = get_device_map(cpu=True)  # Force CPU for simplicity
      model, processor = load_model_and_processor(model_id, device_map)
      self.assertIsInstance(model, LlamaForCausalLM)  # Check specific model class
      self.assertIsInstance(processor, PreTrainedTokenizerBase)  # Check base tokenizer class
      model_with_lora = apply_lora(model)
      self.assertTrue(hasattr(model_with_lora, "peft_config"))  # Check LoRA applied
      # Test tokenization with the processor
      inputs = processor("Hello", return_tensors="pt")
      self.assertIn("input_ids", inputs)
      logger.info("Model integration test completed")

  # Run tests and exit cleanly
  runner = unittest.TextTestRunner()
  suite = unittest.TestLoader().loadTestsFromTestCase(TestModelIntegration)
  result = runner.run(suite)
  torch.cuda.empty_cache()  # Clear any torch state to avoid bitsandbytes cleanup noise
  sys.exit(0 if result.wasSuccessful() else 1)
