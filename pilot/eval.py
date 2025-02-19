import argparse
import json
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(text):
  return text.strip().lower()


def local_judge(question, prediction):
  gt = normalize_text(question["answer"])
  pred = normalize_text(prediction)
  is_correct = "yes" if gt == pred else "no"
  confidence = 100 if is_correct == "yes" else 0
  return {
    "extracted_final_answer": prediction,
    "reasoning": "Local judgement: exact text match comparison.",
    "correct": is_correct,
    "confidence": confidence,
    "strict": True,
  }


def calib_err(confidences, corrects, beta=100):
  confidences = np.array(confidences)
  corrects = np.array(corrects, dtype=float)
  idxs = np.argsort(confidences)
  confidences = confidences[idxs]
  corrects = corrects[idxs]
  n = len(confidences)
  num_bins = n // beta if n // beta > 0 else 1
  bins = [(i * beta, (i + 1) * beta) for i in range(num_bins)]
  bins[-1] = (bins[-1][0], n)
  cerr = 0.0
  for start, end in bins:
    if end - start == 0:
      continue
    bin_conf = confidences[start:end]
    bin_corr = corrects[start:end]
    diff = abs(np.mean(bin_conf) - np.mean(bin_corr) * 100)
    cerr += (end - start) / n * (diff**2)
  return math.sqrt(cerr)


def get_output_filename(model_name):
  os.makedirs(".prediction", exist_ok=True)
  safe_name = model_name.replace("/", "_").replace(" ", "_")
  return os.path.join(".prediction", f"{safe_name}.json")


def get_judged_filename(model_name):
  os.makedirs(".prediction", exist_ok=True)
  safe_name = model_name.replace("/", "_").replace(" ", "_")
  return os.path.join(".prediction", f"judged_{safe_name}.json")


def load_model_and_tokenizer(model_id, device):
  print(f"Loading model '{model_id}' on device {device}...")
  try:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
  except Exception as e:
    print(f"Error loading model '{model_id}': {e}")
    exit(1)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  print("Model and tokenizer loaded successfully.")
  return model, tokenizer


def load_hle_dataset():
  print("Loading HLE test dataset...")
  dataset = load_dataset("cais/hle", split="test")
  print(f"Loaded {len(dataset)} questions from HLE test dataset.")
  return dataset


def generate_prediction(model, tokenizer, prompt, device):
  inputs = tokenizer(prompt, return_tensors="pt").to(device)
  outputs = model.generate(**inputs, max_length=512)
  return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_predictions(model, tokenizer, questions, device, model_id):
  output_filepath = get_output_filename(model_id)
  if os.path.exists(output_filepath):
    print(f"Predictions file '{output_filepath}' exists, skipping prediction generation.")
    with open(output_filepath, "r") as f:
      predictions = json.load(f)
  else:
    print("Generating predictions on test set...")
    predictions = {}
    total = len(questions)
    for idx, question in enumerate(questions):
      print(f"Generating prediction for question {idx+1}/{total} (id: {question['id']})...")
      prompt = question["question"]
      pred = generate_prediction(model, tokenizer, prompt, device)
      predictions[question["id"]] = {"model": model_id, "response": pred}
    with open(output_filepath, "w") as f:
      json.dump(predictions, f, indent=4)
    print(f"Predictions saved to '{output_filepath}'.")
  return predictions


def run_judgement(questions, predictions, model_id):
  judged_filepath = get_judged_filename(model_id)
  if os.path.exists(judged_filepath):
    print(f"Judged file '{judged_filepath}' exists, skipping judgement.")
    with open(judged_filepath, "r") as f:
      judged = json.load(f)
  else:
    print("Running local judgement on predictions...")
    judged = {}
    total = len(questions)
    corrects = []
    confidences = []
    for idx, question in enumerate(questions):
      qid = question["id"]
      if qid not in predictions:
        continue
      response = predictions[qid]["response"]
      judge_out = local_judge(question, response)
      judged[qid] = {"model": model_id, "response": response, "judge_response": judge_out}
      corrects.append(1 if judge_out["correct"] == "yes" else 0)
      confidences.append(judge_out["confidence"])
      print(f"Judged question {idx+1}/{total} (id: {qid}) - Correct: {judge_out['correct']}")
    with open(judged_filepath, "w") as f:
      json.dump(judged, f, indent=4)
    print(f"Judgement saved to '{judged_filepath}'.")
    compute_and_print_metrics(total, corrects, confidences)
  return judged


def compute_and_print_metrics(total, corrects, confidences):
  accuracy = round(100 * sum(corrects) / total, 2)
  conf_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / total), 2)
  calibration_error = round(calib_err(confidences, corrects), 2)
  print("\n*** Metrics ***")
  print(f"Accuracy: {accuracy}% +/- {conf_half_width}% | n = {total}")
  print(f"Calibration Error: {calibration_error}")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model",
    type=str,
    default="meta-llama/Llama-3.2-1B-Instruct",
    help="Path to a fine-tuned model or model id from HF. Use 'None' to default to meta-llama/Llama-3.2-1B-Instruct.",
  )
  parser.add_argument("--cpu", type=lambda x: x.lower() == "true", default=False, help="Force using CPU (True/False).")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  default_model = "meta-llama/Llama-3.2-1B-Instruct"
  model_id = args.model if args.model and args.model.lower() != "none" else default_model
  device = torch.device("cpu") if args.cpu else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

  model, tokenizer = load_model_and_tokenizer(model_id, device)
  questions = load_hle_dataset()
  predictions = generate_predictions(model, tokenizer, questions, device, model_id)
  run_judgement(questions, predictions, model_id)
