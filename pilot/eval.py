import os

import torch

from grpo.eval import evaluate, get_output_filename, judge_predictions, load_hle_dataset, read_predictions_json
from grpo.hardware import get_parameters
from grpo.model import get_model, get_processors

if __name__ == "__main__":
  model_id, cpu, resume, device_map, is_vision_model = get_parameters()

  model = get_model(model_id, device_map, is_vision_model)
  processors = get_processors(model_id, is_vision_model)

  device = torch.device("cuda" if isinstance(device_map, dict) else device_map)
  dataset = load_hle_dataset()

  # Check for JUDGE environment variable
  if os.environ.get("JUDGE"):
    # Skip evaluation and directly call judge_predictions when JUDGE is present

    predictions = read_predictions_json(get_output_filename(model_id))
    judge_predictions(dataset, predictions, model, processors, device, model_id)
  else:
    # Normal evaluation path when JUDGE is not present
    evaluate(model, processors, dataset, device, model_id, resume, is_vision_model)
