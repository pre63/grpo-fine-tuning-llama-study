import base64
import logging
import os
import sys
from typing import Dict, Tuple, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from PIL import Image as PIL_Image
from transformers import LlamaForCausalLM, MllamaForConditionalGeneration, MllamaProcessor, PreTrainedTokenizerBase
from trl import GRPOTrainer

from grpo.config import get_config
from grpo.conversation import tokenize_example
from grpo.hardware import get_parameters, is_vision_model
from grpo.model import apply_lora, load_model_and_processor
from grpo.reward import compute_reward

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_trainer(model_id, model, processors, train_data, test_data):
  """Helper function to create a GRPOTrainer instance."""
  config = get_config(model_id, 8192)
  max_prompt_length = max(len(example["input_ids"]) for example in train_data)
  logger.info(f"Computed max_prompt_length: {max_prompt_length}")
  trainer_args = {
    "model": model,
    "reward_funcs": lambda prompts, completions, **kw: compute_reward(prompts, completions, kw.get("ground_truths", [d["answer"] for d in train_data])),
    "train_dataset": train_data,
    "eval_dataset": test_data,
    "args": config,
  }
  trainer = GRPOTrainer(**trainer_args)
  return trainer


if __name__ == "__main__":
  import unittest

  class TestGRPOIntegration(unittest.TestCase):
    def test_get_trainer_and_training_integration(self):
      logger.info("Starting GRPO training integration test with get_trainer")

      model_id, cpu, resume, device_map, is_vision_model = get_parameters()
      logger.info(f"Using MODEL: {model_id}, CPU: {cpu}, RESUME: {resume}, DEVICE_MAP: {device_map}")

      model, processors = load_model_and_processor(model_id, device_map, is_vision_model)
      self.assertTrue(isinstance(model, (LlamaForCausalLM, MllamaForConditionalGeneration)), "Model should be Llama or Mllama")
      self.assertIsInstance(processors, dict, "Processors should be a dictionary")

      model = apply_lora(model)
      self.assertTrue(hasattr(model, "peft_config"), "LoRA should be applied")

      dummy_image = base64.b64encode(Image.new("RGB", (224, 224), "blue")._repr_png_()).decode("utf-8")
      data = [
        {"question": "What is this?", "answer": "A test", "image": dummy_image},
        {"question": "What is that?", "answer": "Another test", "image": dummy_image},
        {"question": "What’s here?", "answer": "Yet another", "image": dummy_image},
        {"question": "What now?", "answer": "More tests", "image": ""},  # Empty string
      ]
      dataset = Dataset.from_list(data)
      train_size = int(0.5 * len(dataset))  # 2/2 split
      train_data = dataset.select(range(train_size)).map(lambda ex: tokenize_example(ex, processors, is_vision_model))
      test_data = dataset.select(range(train_size, len(dataset))).map(lambda ex: tokenize_example(ex, processors, is_vision_model))
      train_data = train_data.take(2)
      test_data = test_data.take(2)

      self.assertIsInstance(train_data, Dataset, "Train data should be a Dataset")
      self.assertIsInstance(test_data, Dataset, "Test data should be a Dataset")
      self.assertEqual(len(train_data), 2, "Train data should have 2 examples")
      self.assertEqual(len(test_data), 2, "Test data should have 2 examples")
      self.assertIn("labels", train_data[0], "Labels should be in dataset")

      trainer = get_trainer(model_id, model, processors, train_data, test_data)

      self.assertIsInstance(trainer, GRPOTrainer, "Trainer should be a GRPOTrainer")
      self.assertEqual(trainer.model, model, "Trainer model mismatch")
      self.assertEqual(trainer.train_dataset, train_data, "Trainer train dataset mismatch")
      self.assertEqual(trainer.eval_dataset, test_data, "Trainer eval dataset mismatch")

      trainer.train()
      logger.info("Training completed")

      torch.cuda.empty_cache() if torch.cuda.is_available() else None
      logger.info("GRPO training integration test with get_trainer completed")

  logger.info("Running GRPO integration test suite")
  runner = unittest.TextTestRunner()
  suite = unittest.TestLoader().loadTestsFromTestCase(TestGRPOIntegration)
  result = runner.run(suite)
  torch.cuda.empty_cache()
  logger.info("Test suite completed")
  sys.exit(0 if result.wasSuccessful() else 1)
