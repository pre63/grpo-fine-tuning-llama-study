from datetime import datetime
from typing import Dict, Union

from peft import LoraConfig, LoraRuntimeConfig
from trl import GRPOConfig


def get_config(model_id: str, device_map: Union[Dict, str] = "cpu", max_prompt_length=2048, max_completion_length=1024):
  is_cuda = isinstance(device_map, dict)
  date_path = datetime.now().strftime("%Y-%m-%d-%H-%M")

  # Base model_init_kwargs without use_cache
  model_init_kwargs = {"quantization_config": get_bits_and_bytes_config()} if is_cuda else {}

  return GRPOConfig(
    output_dir=f".grpo_output/{model_id}_{date_path}",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Reduced to lower memory usage per step
    per_device_eval_batch_size=4,  # Reduced for evaluation memory savings
    num_train_epochs=5,
    num_generations=4,
    gradient_accumulation_steps=16,  # Increased to maintain effective batch size
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    temperature=0.5,
    beta=0.1,
    remove_unused_columns=True,  # Remove unused dataset columns to save memory
    log_completions=True,  # Optional: set to False if logs aren't needed
    # fp16=is_cuda, # this parameter blocks the code from running
    # gradient_checkpointing=True, # use_cache issue
    # use_vllm=True if is_cuda else False, # this breaks memory allocation because it moves the tensors and it crashes
    model_init_kwargs=model_init_kwargs,
  )


def get_bits_and_bytes_config():
  return {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
  }


def get_lora_config(device_map: Union[Dict, str] = "cpu"):

  base_config = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "use_dora": True,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
  }
  # GPU Config
  if isinstance(device_map, dict):
    conf = LoraConfig(
      **base_config,
      runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True),
    )

  # CPU Config
  else:
    conf = LoraConfig(**base_config)

  return conf


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
