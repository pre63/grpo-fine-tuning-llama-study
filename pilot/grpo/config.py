from datetime import datetime
from typing import Dict, Union

from trl import GRPOConfig


def get_config(model_id: str, device_map: Union[Dict, str] = "cpu", max_prompt_length: int = None, max_completion_length: int = None):
  is_cuda = isinstance(device_map, dict)
  date_path = datetime.now().strftime("%Y-%m-%d-%H-%M")

  return GRPOConfig(
    output_dir=f".grpo_output/{model_id}_{date_path}",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    num_generations=5,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    temperature=0.5,
    beta=0.1,
    remove_unused_columns=False,
    log_completions=True,
    fp16=is_cuda,  # Enable mixed precision for CUDA
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
