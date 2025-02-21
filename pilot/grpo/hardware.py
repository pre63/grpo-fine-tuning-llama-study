from typing import Dict, Union

import torch


def get_device_map(cpu: bool) -> Union[str, Dict[str, int]]:
  use_cuda = torch.cuda.is_available() and not cpu
  use_mps = torch.backends.mps.is_available() and not cpu and not use_cuda
  if use_cuda:
    return {"": 0}
  elif use_mps:
    return "mps"
  else:
    return "cpu"


if __name__ == "__main__":
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

  unittest.main()
