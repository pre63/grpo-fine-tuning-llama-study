from datetime import datetime
from typing import Dict, Union

import torch
from trl import GRPOConfig


def get_config(model_id: str, device_map: Union[Dict, str] = "cpu", max_prompt_length: int = None, max_completion_length: int = None):
  is_cuda = isinstance(device_map, dict)
  date_path = datetime.now().strftime("%Y-%m-%d-%H-%M")

  return GRPOConfig(
    output_dir=f".grpo_output/{model_id}_{date_path}",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    num_generations=2,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    temperature=0.7,
    beta=0.1,
    remove_unused_columns=False,
    log_completions=True,
    # fp16=is_cuda,  # Enable mixed precision for CUDA
    # no_cuda=not is_cuda,  # Explicitly set based on device
    # report_to="none",  # Minimize logging overhead
  )


if __name__ == "__main__":
  print("Running test for config.py")

  import unittest

  class TestConfigIntegration(unittest.TestCase):
    def test_config_usage(self):
      config = get_config("itsatest")
      self.assertIsInstance(config, GRPOConfig)
      self.assertGreater(config.learning_rate, 0)
      self.assertTrue(isinstance(config.output_dir, str))

  unittest.main()
