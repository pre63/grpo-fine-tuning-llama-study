import argparse
import json
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import HLE components for consistent prompt formatting and metrics
from hle.judge import JUDGE_PROMPT, calib_err, dump_metrics
from hle.prediction import SYSTEM_EXACT_ANSWER, SYSTEM_MC, format_message


def normalize_text(text):
  return text.strip().lower()


def local_judge(question, prediction):
  # Local judgement using exact string match on the answer.
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


def get_output_filename(model_name):
  os.makedirs(".predictions", exist_ok=True)
  safe_name = model_name.replace("/", "_").replace(" ", "_")
  return os.path.join(".predictions", f"{safe_name}.json")


def get_judged_filename(model_name):
  os.makedirs(".predictions", exist_ok=True)
  safe_name = model_name.replace("/", "_").replace(" ", "_")
  return os.path.join(".predictions", f"judged_{safe_name}.json")


def load_model_and_tokenizer(model_id, device):
  print(f"Loading model '{model_id}' on device {device}...")
  try:
    if device.type == "cuda":
      model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    else:
      model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).to(device)
  except Exception as e:
    print(f"Error loading model '{model_id}': {e}")
    exit(1)

  # Print the actual device(s) for the model parameters.
  device_set = {str(p.device) for p in model.parameters()}
  print("Model loaded on device(s):", device_set)

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


def get_prompt_as_str(question):
  # Use the imported format_message to get the prompt.
  prompt = format_message(question)
  if not isinstance(prompt, str):
    if isinstance(prompt, list):
      prompt = "\n".join(str(item) for item in prompt)
    else:
      prompt = str(prompt)
  return prompt


def generate_prediction(model, tokenizer, prompt, device):
  inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
  outputs = model.generate(**inputs, max_new_tokens=512)

  for i, output in enumerate(outputs):
    print(f"Output {i}: {tokenizer.decode(output, skip_special_tokens=True)}")

  return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_predictions(model, tokenizer, questions, device, model_id, resume):
  output_filepath = get_output_filename(model_id)
  predictions = {}
  if resume and os.path.exists(output_filepath):
    print(f"Resuming from existing predictions file '{output_filepath}'.")
    with open(output_filepath, "r") as f:
      predictions = json.load(f)
  elif os.path.exists(output_filepath):
    print(f"Found existing predictions file '{output_filepath}', but not resuming. Overwriting predictions.")
  else:
    print("No existing predictions file found. Starting fresh predictions.")

  total = len(questions)
  for idx, question in enumerate(questions):
    qid = question["id"]
    if resume and qid in predictions:
      print(f"Skipping question {idx+1}/{total} (id: {qid}) as prediction exists.")
      continue
    prompt = get_prompt_as_str(question)
    print(f"Generating prediction for question {idx+1}/{total} (id: {qid})...")
    pred = generate_prediction(model, tokenizer, prompt, device)
    predictions[qid] = {"model": model_id, "response": pred}
    # Save incrementally to allow resuming if interrupted.
    with open(output_filepath, "w") as f:
      json.dump(predictions, f, indent=4)
  print(f"All predictions saved to '{output_filepath}'.")
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
    dump_metrics(judged, total)
  return judged


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model",
    type=str,
    default="HF-Quantization/Llama-3.2-1B-GPTQ-INT4",
    help="Path to a fine-tuned model or model id from HF. Use 'None' to default to HF-Quantization/Llama-3.2-1B-GPTQ-INT4.",
  )
  parser.add_argument("--cpu", type=lambda x: x.lower() == "true", default=False, help="Force using CPU (True/False).")
  parser.add_argument("--resume", type=lambda x: x.lower() == "true", default=False, help="Resume predictions if file exists (True/False).")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  default_model = "HF-Quantization/Llama-3.2-1B-GPTQ-INT4"
  model_id = args.model if args.model and args.model.lower() != "none" else default_model
  device = torch.device("cpu") if args.cpu else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

  model, tokenizer = load_model_and_tokenizer(model_id, device)
  questions = load_hle_dataset()
  predictions = generate_predictions(model, tokenizer, questions, device, model_id, args.resume)
  run_judgement(questions, predictions, model_id)
