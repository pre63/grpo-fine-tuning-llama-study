import base64
import os
import sys
from typing import Dict, Tuple, Union

import torch
from datasets import Dataset
from peft import PeftModelForCausalLM
from PIL import Image
from transformers import LlamaForCausalLM, MllamaForConditionalGeneration
from trl import GRPOTrainer

from grpo.config import get_config, get_lora_config
from grpo.conversation import tokenize_example
from grpo.hardware import get_parameters
from grpo.model import get_model, get_processors
from grpo.reward import compute_reward
from hle.dataset import load_and_split_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() if torch.cuda.is_available() else None


def get_trainer(model_id: str, processors, train_data, test_data, device_map: str):
  print("Setting up GRPOTrainer")

  # Get config with device
  config = get_config(model_id, device_map)
  peft_config = get_lora_config(device_map)

  trainer_args = {
    "model": model_id,
    "reward_funcs": lambda prompts, completions, **kw: compute_reward(prompts, completions, kw.get("ground_truths", [d["answer"] for d in train_data])),
    "train_dataset": train_data,
    "eval_dataset": test_data,
    "args": config,
    "processing_class": processors["text"],
    "peft_config": peft_config,
  }

  trainer = GRPOTrainer(**trainer_args)

  return trainer


if __name__ == "__main__":
  import unittest

  class TestGRPOIntegration(unittest.TestCase):
    def test_get_trainer_and_training_integration(self):
      print("Starting GRPO training integration test with get_trainer")

      model_id, cpu, resume, device_map, is_vision_model = get_parameters()
      print(f"Using MODEL: {model_id}, CPU: {cpu}, RESUME: {resume}, DEVICE_MAP: {device_map}")

      processors = get_processors(model_id, is_vision_model)
      model = get_model(model_id, device_map, is_vision_model)

      self.assertTrue(
        isinstance(model, (PeftModelForCausalLM, LlamaForCausalLM, MllamaForConditionalGeneration)), "Model should be Llama or Mllama " + str(type(model))
      )
      self.assertIsInstance(processors, dict, "Processors should be a dictionary")

      self.assertTrue(hasattr(model, "peft_config"), "LoRA should be applied")

      train_data, test_data = load_and_split_dataset(test_size=0.3, tokenize_example=tokenize_example, processors=processors, is_vision_model=is_vision_model)
      train_data = train_data.take(2)
      test_data = test_data.take(2)

      self.assertIsInstance(train_data, Dataset, "Train data should be a Dataset " + str(type(train_data)))
      self.assertIsInstance(test_data, Dataset, "Test data should be a Dataset " + str(type(test_data)))
      self.assertEqual(len(train_data), 2, "Train data should have 2 examples " + str(len(train_data)))
      self.assertEqual(len(test_data), 2, "Test data should have 2 examples " + str(len(test_data)))
      self.assertIn("labels", train_data[0], "Labels should be in dataset " + str(train_data[0].keys()))

      trainer = get_trainer(model_id, processors, train_data, test_data, device_map)

      self.assertIsInstance(trainer, GRPOTrainer, "Trainer should be a GRPOTrainer")

      trainer.train()

      print("Training completed")

      torch.cuda.empty_cache() if torch.cuda.is_available() else None
      print("GRPO training integration test with get_trainer completed")

  print("Running GRPO integration test suite")
  runner = unittest.TextTestRunner()
  suite = unittest.TestLoader().loadTestsFromTestCase(TestGRPOIntegration)
  result = runner.run(suite)
  torch.cuda.empty_cache()
  print("Test suite completed")
  sys.exit(0 if result.wasSuccessful() else 1)
