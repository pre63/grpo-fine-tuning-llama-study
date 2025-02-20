import argparse
import json
import os
from typing import Literal

import jsonlines
import torch
from datasets import load_dataset
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import HLE components for consistent prompt formatting and metrics
from hle.judge import JUDGE_PROMPT, dump_metrics, format_judge_prompt
from hle.prediction import format_message

# --- Pydantic Models for Validation ---


class Message(BaseModel):
  role: str
  content: str


class Prediction(BaseModel):
  question_id: str
  model: str
  content: str  # The processed assistant's response (without system messages)
  raw_response: str  # The full raw LLM output


class ExtractedAnswer(BaseModel):
  extracted_final_answer: str
  reasoning: str
  correct: Literal["yes", "no"]
  confidence: int
  strict: Literal[True]


# --- Basic Helpers ---


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
  dataset = dataset.filter(lambda x: x["image"] == "")
  print(f"Loaded {len(dataset)} questions from HLE test dataset.")
  return dataset


def prompt_as_str(question):
  msg = format_message(question)
  return msg if isinstance(msg, str) else "\n".join(map(str, msg))


# --- Composable Prediction Functions ---


def encode_prompt(prompt, tokenizer, device):
  return tokenizer(prompt, return_tensors="pt", truncation=False).to(device)


def generate_raw_output(model, inputs, max_new_tokens):
  return model.generate(**inputs, max_new_tokens=max_new_tokens)


def decode_outputs(outputs, tokenizer):
  return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def process_prediction_output(prediction_str):
  """
    Given the raw prediction string in NDJSON format (one JSON object per line),
    use jsonlines to iterate over each JSON object, validate with the Message model,
    and return the concatenated content for messages whose role is not 'system'.
    """
  with open("output.txt", "w") as f:
    f.write(prediction_str)

  messages = []
  for obj in jsonlines.Reader(prediction_str.splitlines()):
    try:
      msg = Message.model_validate(obj)

      if msg.role.lower() != "system":
        messages.append(msg.content)

    except Exception as e:
      print("Validation error for message:", e)

  return "\n".join(messages).strip()


def compose_prediction(model, tokenizer, prompt, device, max_new_tokens=8192):
  inputs = encode_prompt(prompt, tokenizer, device)
  raw_outputs = generate_raw_output(model, inputs, max_new_tokens)
  decoded = decode_outputs(raw_outputs, tokenizer)
  return decoded[0]


# --- Prediction & Judge Pipeline ---


def generate_predictions(model, tokenizer, questions, device, model_id, resume):
  output_filepath = get_output_filename(model_id)
  predictions = {}

  if resume and os.path.exists(output_filepath):
    print(f"Resuming from existing predictions file '{output_filepath}'.")

    with open(output_filepath, "r") as f:
      data = json.load(f)
      for k, v in data.items():
        try:
          predictions[k] = Prediction.model_validate(v).model_dump()
        except Exception as e:
          print(f"Prediction validation error for id {k}:", e)

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

    prompt = prompt_as_str(question)
    print(f"Generating prediction for question {idx+1}/{total} (id: {qid})...")
    raw_pred = compose_prediction(model, tokenizer, prompt, device, max_new_tokens=8192)
    content = process_prediction_output(raw_pred)
    predictions[qid] = Prediction(question_id=qid, model=model_id, content=content, raw_response=raw_pred).model_dump()

    with open(output_filepath, "w") as f:
      json.dump(predictions, f, indent=4)

  print(f"All predictions saved to '{output_filepath}'.")
  return predictions


def parse_judge_text(text: str) -> dict:
  """
    Parse a text block containing lines in the form "key: value" and return a dict.
    """
  result = {}
  for line in text.splitlines():

    if ":" in line:
      key, value = line.split(":", 1)
      result[key.strip()] = value.strip()
  return result


def extract_judge_answer(question_text, correct_answer, response, model, tokenizer, device):
  judge_prompt = JUDGE_PROMPT.format(question=question_text, correct_answer=correct_answer, response=response)
  raw_judge = compose_prediction(model, tokenizer, judge_prompt, device, max_new_tokens=4096)

  # If the model merely echoes the prompt, return a default null answer.

  if raw_judge.strip() == judge_prompt.strip():
    print("Judge output identical to prompt; returning default null answer.")

    return {
      "extracted_final_answer": "None",
      "reasoning": "",
      "correct": "no",
      "confidence": 0,
      "strict": True,
    }

  # Use composition: first parse the raw text into key/value pairs.
  parsed = parse_judge_text(raw_judge)
  try:
    extracted = ExtractedAnswer.model_validate(parsed)

    return {
      "correct_answer": correct_answer,
      "model_answer": extracted.extracted_final_answer,
      "reasoning": extracted.reasoning,
      "correct": extracted.correct,
      "confidence": extracted.confidence,
    }

  except Exception as e:
    print("Failed to validate judge output:", e)
    judged_correct = "yes" if "yes" in raw_judge.lower() else "no"

    return {
      "correct_answer": correct_answer,
      "model_answer": raw_judge,
      "reasoning": "Fallback: could not parse judge output.",
      "correct": judged_correct,
      "confidence": 100,
    }


def judge_predictions(dataset, predictions, model, tokenizer, device, model_id):
  judged_filepath = get_judged_filename(model_id)
  judged = {}

  if os.path.exists(judged_filepath):
    print(f"Found existing judged file '{judged_filepath}'. Resuming judgement.")

    with open(judged_filepath, "r") as f:
      data = json.load(f)
      for k, v in data.items():
        try:
          judged[k] = JudgedPrediction.model_validate(v).model_dump()
        except Exception as e:
          print(f"Judged prediction validation error for id {k}:", e)

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

    response = predictions[qid]["content"]
    print(f"Judging question {idx+1}/{total} (id: {qid})...")
    judge_result = extract_judge_answer(
      question_text=question["question"], correct_answer=question["answer"], response=response, model=model, tokenizer=tokenizer, device=device
    )

    judged[qid] = JudgedPrediction(
      extracted_final_answer=judge_result["extracted_final_answer"],
      reasoning=judge_result["reasoning"],
      correct=judge_result["correct"],
      confidence=judge_result["confidence"],
      strict=True,
    ).model_dump()

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
    default="meta-llama/Llama-3.2-11B-Vision-Instruct",
    help="Model id or path; use 'None' to default to meta-llama/Llama-3.2-11B-Vision-Instruct.",
  )
  parser.add_argument("--cpu", type=lambda x: x.lower() == "true", default=False, help="Force using CPU (True/False).")
  parser.add_argument("--resume", type=lambda x: x.lower() == "true", default=False, help="Resume predictions if file exists (True/False).")
  return parser.parse_args()


def main():
  args = parse_args()
  default_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
  model_id = args.model if args.model and args.model.lower() != "none" else default_model
  device = torch.device("cpu") if args.cpu else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
  print(f"Using device: {device}")

  model, tokenizer = load_model_and_tokenizer(model_id, device)
  dataset = load_hle_dataset()
  predictions = generate_predictions(model, tokenizer, dataset, device, model_id, args.resume)
  judge_predictions(dataset, predictions, model, tokenizer, device, model_id)


if __name__ == "__main__":
  main()
