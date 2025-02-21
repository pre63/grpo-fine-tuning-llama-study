from trl import GRPOConfig


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
  import unittest

  class TestConfigIntegration(unittest.TestCase):
    def test_config_usage(self):
      config = get_config()
      self.assertIsInstance(config, GRPOConfig)
      # Simulate usage in a trainer-like context
      self.assertEqual(config.max_prompt_length, 32)
      # Check if config attributes are accessible and valid
      self.assertGreater(config.learning_rate, 0)
      self.assertTrue(isinstance(config.output_dir, str))

  unittest.main()
