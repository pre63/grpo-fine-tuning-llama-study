import torch

from grpo.eval import evaluate, load_hle_dataset
from grpo.hardware import get_parameters
from grpo.model import get_model, get_processors

if __name__ == "__main__":
  model_id, cpu, resume, device_map, is_vision_model = get_parameters()

  model = get_model(model_id, device_map, is_vision_model)
  processors = get_processors(model_id, is_vision_model)

  device = torch.device("cuda" if isinstance(device_map, dict) else device_map)
  dataset = load_hle_dataset()

  evaluate(model, processors, dataset, device, model_id, resume, is_vision_model)
