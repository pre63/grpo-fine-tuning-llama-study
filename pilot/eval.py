import argparse
import ast
import json
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import HLE components for consistent prompt formatting and metrics
from hle.judge import dump_metrics, format_judge_prompt
from hle.prediction import format_message


def normalize_text(text):
  return text.strip().lower()


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

  # Print the actual device(s) where parameters reside.
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
  prompt = format_message(question)
  if not isinstance(prompt, str):
    if isinstance(prompt, list):
      prompt = "\n".join(str(item) for item in prompt)
    else:
      prompt = str(prompt)
  return prompt


def generate_prediction(model, tokenizer, prompt, device, max_new_tokens=8192):
  # Do not truncate the prompt; send it in full
  inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
  outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
  for i, output in enumerate(outputs):
    print(f"Output {i}: {tokenizer.decode(output, skip_special_tokens=True)}")
  return tokenizer.decode(outputs[0], skip_special_tokens=True)


def process_prediction_output(prediction_str):
  """
    Given the model's raw prediction string (which contains multiple messages in a string format),
    parse it into individual messages, skip those with role 'system', and return the concatenated content.
    """
  messages = []
  for line in prediction_str.strip().splitlines():
    try:
      msg = ast.literal_eval(line.strip())
      if isinstance(msg, dict) and "role" in msg and "content" in msg:
        if msg["role"].lower() != "system":
          messages.append(msg["content"])
    except Exception as e:
      print("Failed to parse line:", line, "Error:", e)
  return "\n".join(messages).strip()


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
    raw_pred = generate_prediction(model, tokenizer, prompt, device, max_new_tokens=8192)

    # Process the raw prediction: skip messages we sent, extract the content.
    pred = process_prediction_output(raw_pred)
    predictions[qid] = {"model": model_id, "response": pred}
    with open(output_filepath, "w") as f:
      json.dump(predictions, f, indent=4)
  print(f"All predictions saved to '{output_filepath}'.")
  return predictions


def judge(question, prediction, model, tokenizer, device):
  judge_prompt = format_judge_prompt(question=question["question"], answer=question["answer"], response=prediction)
  judge_output = generate_prediction(model, tokenizer, judge_prompt, device, max_new_tokens=4096)
  return judge_output


def judge_predictions(dataset, predictions, model, tokenizer, device, model_id):
  judged_filepath = get_judged_filename(model_id)
  judged = {}
  if os.path.exists(judged_filepath):
    print(f"Found existing judged file '{judged_filepath}'. Resuming judgement.")
    with open(judged_filepath, "r") as f:
      judged = json.load(f)
  else:
    print("No judged file found. Starting fresh judgement.")

  total = len(dataset)
  for idx, question in enumerate(dataset):
    qid = question["id"]
    if qid not in predictions:
      continue
    if qid in judged:
      print(f"Skipping question {idx+1}/{total} (id: {qid}) as already judged.")
      continue
    response = predictions[qid]["response"]
    print(f"Judging question {idx+1}/{total} (id: {qid})...")
    judge_out = judge(question, response, model, tokenizer, device)

    # For simplicity, we assume if judge_out contains "yes" (case-insensitive) it's correct.
    judged_correct = "yes" if "yes" in judge_out.lower() else "no"
    judged[qid] = {"model": model_id, "response": response, "judge_response": judge_out, "judged_correct": judged_correct}

    # Save judgements incrementally.

    with open(judged_filepath, "w") as f:
      json.dump(judged, f, indent=4)
  print(f"All judgement results saved to '{judged_filepath}'.")
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


def main():
  args = parse_args()
  default_model = "HF-Quantization/Llama-3.2-1B-GPTQ-INT4"
  model_id = args.model if args.model and args.model.lower() != "none" else default_model
  device = torch.device("cpu") if args.cpu else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
  print(f"Using device: {device}")

  model, tokenizer = load_model_and_tokenizer(model_id, device)
  dataset = load_hle_dataset()
  predictions = generate_predictions(model, tokenizer, dataset, device, model_id, args.resume)
  judge_predictions(dataset, predictions, model, tokenizer, device, model_id)


if __name__ == "__main__":
  main()
