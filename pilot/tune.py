import os

import wandb
from eval import evaluate
from grpo.conversation import tokenize_example
from grpo.hardware import get_parameters
from grpo.model import apply_lora, load_model_and_processor
from grpo.trainer import get_trainer
from hle.dataset import load_and_split_dataset

if __name__ == "__main__":
  wandb.init(project="grpo")

  model_id, cpu, resume, device_map, is_vision_model = get_parameters()

  print("Loading model and processors...")
  model, processors = load_model_and_processor(model_id, device_map, is_vision_model)

  assert processors["text"].pad_token is not None, "Processor does not have eos_token."

  print("Applying LoRA...")
  model = apply_lora(model)

  print("Loading and splitting dataset...")
  train_data, test_data = load_and_split_dataset(test_size=0.3, tokenize_example=tokenize_example, processors=processors, is_vision_model=is_vision_model)

  print("Creating trainer...")
  trainer = get_trainer(model_id, model, processors, train_data, test_data, device_map)

  print("Training model...")
  trainer.train()

  print("Training complete.")

  # Save the model and processors with date and time
  path = f"./fine-tuned-model/{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
  os.makedirs(path, exist_ok=True)
  model.save_pretrained(path)
  processors["text"].save_pretrained(path)
  if is_vision_model:
    path_vision = f"./fine-tuned-model/{datetime.now().strftime('%Y-%m-%d-%H-%M')}/vision"
    processors["vision"].save_pretrained(path_vision)
  print("Model and processors saved.")

  if not device_map == "cpu":
    # immediately evaluate on test set with eval.py's evaluate function
    print("Evaluating on test set...")
    evaluate(model, processors, test_data, device, model_id, resume)
    print("Done!")

    print("Evaluating on test set with GRPOTrainer.evaluate()...")
    results = trainer.evaluate(test_data)
    print(results)
    print("Done!")
