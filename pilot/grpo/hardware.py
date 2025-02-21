import os
from typing import Dict, Union

import torch


def is_vision_model(model_id: str) -> bool:
  LLAMA_MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct": False,
    "meta-llama/Llama-3.2-3B-Instruct": False,
    "meta-llama/Llama-3.2-11B-Vision-Instruct": True,
    "meta-llama/Llama-3.2-90B-Vision-Instruct": True,
  }

  return LLAMA_MODELS.get(model_id, False)


def get_device_map(cpu: bool) -> Union[str, Dict[str, int]]:
  use_cuda = torch.cuda.is_available() and not cpu
  use_mps = torch.backends.mps.is_available() and not cpu and not use_cuda
  if use_cuda:
    return {"": 0}
  elif use_mps:
    return "mps"
  else:
    return "cpu"


def get_parameters():
  model_id = os.getenv("MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
  if model_id.lower() == "none":
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

  cpu = os.getenv("CPU", "true").lower() == "true"
  resume = os.getenv("RESUME", "false").lower() == "true"
  device_map = get_device_map(cpu)

  is_vision = is_vision_model(model_id)

  print(f"Using MODEL: {model_id}, CPU: {cpu}, RESUME: {resume}, DEVICE_MAP: {device_map}")

  return model_id, cpu, resume, device_map, is_vision


if __name__ == "__main__":
  print("Running test for hardware.py")

  import unittest

  class TestHardwareIntegration(unittest.TestCase):
    def test_device_map_selection(self):
      # Test device map with actual hardware checks
      device_map_cpu = get_device_map(cpu=True)
      self.assertEqual(device_map_cpu, "cpu")
      # Test non-CPU case (depends on system; assume CPU if no GPU/MPS)
      device_map_auto = get_device_map(cpu=False)
      self.assertIn(device_map_auto, [{"": 0}, "mps", "cpu"])
      # Verify torch can use the device
      device = torch.device(device_map_auto if isinstance(device_map_auto, str) else "cuda")
      tensor = torch.tensor([1.0], device=device)
      self.assertEqual(tensor.device.type, device.type)

      model_id, cpu, resume, device_map, is_vision_model = get_parameters()
      self.assertIsInstance(model_id, str)
      self.assertIsInstance(cpu, bool)
      self.assertIsInstance(resume, bool)
      self.assertIsInstance(device_map, (str, dict))

  unittest.main()
