from datetime import datetime

from trl import GRPOConfig


def get_config(model_id, max_prompt_length=None) -> GRPOConfig:
  date_path = datetime.now().strftime("%Y-%m-%d-%H-%M")
  return GRPOConfig(
    output_dir=f".grpo_output/{model_id}_{date_path}",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    num_generations=2,
    max_prompt_length=max_prompt_length,
    max_completion_length=None,
    temperature=0.7,
    beta=0.1,
    remove_unused_columns=False,
    log_completions=True,
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
