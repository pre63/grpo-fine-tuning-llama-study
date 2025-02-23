import torch
from datasets import load_dataset

from grpo.audit import audit_predictions_and_judgements
from grpo.eval import get_judged_filename, get_output_filename
from grpo.hardware import get_parameters
from grpo.model import get_model, get_processors

if __name__ == "__main__":
  model_id, cpu, resume, device_map, is_vision_model = get_parameters()

  # Setup model and processors
  print(f"Loading model: {model_id}")
  model = get_model(model_id, device_map, is_vision_model)
  processors = get_processors(model_id, is_vision_model)

  # Setup device
  device = torch.device("cuda" if isinstance(device_map, dict) else device_map)
  print(f"Using device: {device}")

  # Load dataset
  print("Loading HLE test dataset")
  dataset = load_dataset("cais/hle", split="test")
  print(f"Loaded dataset with {len(dataset)} items")

  # Specify prediction and judgement files
  predictions_filepath = get_output_filename(model_id)
  judgements_filepath = get_judged_filename(model_id)

  # Run audit
  print("Starting audit process")
  audit_results = audit_predictions_and_judgements(
    dataset=dataset,
    predictions_filepath=predictions_filepath,
    judgements_filepath=judgements_filepath,
    model=model,
    processors=processors,
    device=device,
    model_id=model_id,
  )

  if audit_results is not None:
    print(f"Audit completed successfully. Results saved for {len(audit_results)} questions")
  else:
    print("Audit failed")

  # Clean up
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cleared CUDA cache")
