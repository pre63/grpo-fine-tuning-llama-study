import json
import logging
import os
from typing import Dict, Literal, Union

import jsonlines
import torch
from datasets import load_dataset
from pydantic import BaseModel

from grpo.conversation import build_conversation
from grpo.hardware import get_parameters
from grpo.image_utils import decode_base64_image
from grpo.model import apply_lora, load_model_and_processor
from hle.judge import dump_metrics, format_judge_prompt

# Set PyTorch memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Pydantic Models
# --------------------------------------------------


class Message(BaseModel):
  role: str
  content: str


class Prediction(BaseModel):
  question_id: str
  model: str
  content: str
  raw_response: str


class ExtractedAnswer(BaseModel):
  extracted_final_answer: str
  reasoning: str
  correct: Literal["yes", "no"]
  confidence: int
  strict: Literal[True]


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------


def load_hle_dataset():
  logger.info("Loading HLE test dataset...")
  dataset = load_dataset("cais/hle", split="test")
  logger.info(f"Loaded {len(dataset)} questions from HLE test dataset.")
  return dataset


def normalize_text(text):
  return text.strip().lower()


def get_output_filename(model_name):
  from datetime import datetime

  date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  os.makedirs("./predictions", exist_ok=True)
  safe_name = model_name.replace("/", "_").replace(" ", "_")
  filename = f"{safe_name}_{date_time}.json"
  return os.path.join("./predictions", filename)


def get_judged_filename(model_name):
  from datetime import datetime

  date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  filename = f"{model_name.replace('/', '_')}_{date_time}_judged.json"
  os.makedirs("./judged", exist_ok=True)
  return os.path.join("./judged", filename)


# --------------------------------------------------
# Prediction and Judging Functions
# --------------------------------------------------


def process_prediction_output(prediction_str):
  with open("output.txt", "w") as f:
    f.write(prediction_str)
  try:
    messages = []
    for obj in jsonlines.Reader(prediction_str.splitlines()):
      try:
        msg = Message.model_validate(obj)
        if msg.role.lower() != "system":
          messages.append(msg.content)
      except Exception as e:
        logger.error(f"Validation error for message: {e}")
    return "\n".join(messages).strip()
  except Exception as e:
    logger.error(f"Error processing prediction output: {e}")
    return "Failed to process prediction output."


def decode(processor, input_ids, raw_output):
  input_length = input_ids.shape[1]
  generated_ids = raw_output[0, input_length:]
  decoded = processor.decode(generated_ids, skip_special_tokens=True)
  return decoded


def compose_prediction(model, processors, question, device, max_new_tokens=2048, is_vision_model=False):
  torch.cuda.empty_cache()
  has_images = is_vision_model and question.get("image", "").strip()
  conversation = build_conversation(question, has_images, include_system_prompt=True)

  text_processor = processors["text"]
  vision_processor = processors["vision"] if is_vision_model else None

  prompt = text_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
  images = [decode_base64_image(question["image"])] if has_images else []

  if is_vision_model:
    inputs = vision_processor(text=prompt, images=images, return_tensors="pt").to(device)
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    with torch.no_grad():
      raw_outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
      decoded = decode(vision_processor, inputs["input_ids"], raw_outputs)
  else:
    inputs = text_processor(prompt, return_tensors="pt").to(device)
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    with torch.no_grad():
      raw_outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
      decoded = decode(text_processor, inputs["input_ids"], raw_outputs)

  divider = "=" * 80
  line = f"\n{divider}\n{question['question']}\n---\n{decoded}\n{divider}\n"
  logger.info(line)

  return decoded


def generate_predictions(model, processors, questions, device, model_id, resume, is_vision_model):
  output_filepath = get_output_filename(model_id)
  predictions = {}

  if resume and os.path.exists(output_filepath):
    logger.info(f"Resuming from existing predictions file '{output_filepath}'.")
    with open(output_filepath, "r") as f:
      data = json.load(f)
      for k, v in data.items():
        try:
          predictions[k] = Prediction.model_validate(v).model_dump()
        except Exception as e:
          logger.error(f"Prediction validation error for id {k}: {e}")
  elif os.path.exists(output_filepath):
    logger.info(f"Found existing predictions file '{output_filepath}', but not resuming. Overwriting predictions.")
  else:
    logger.info("No existing predictions file found. Starting fresh predictions.")

  total = len(questions)
  for idx, question in enumerate(questions):
    qid = question["id"]
    if resume and qid in predictions:
      logger.info(f"Skipping question {idx+1}/{total} (id: {qid}) as prediction exists.")
      continue

    logger.info(f"Generating prediction for question {idx+1}/{total} (id: {qid})...")
    raw_pred = compose_prediction(model, processors, question, device, is_vision_model)
    content = process_prediction_output(raw_pred)
    predictions[qid] = Prediction(question_id=qid, model=model_id, content=content, raw_response=raw_pred).model_dump()

    with open(output_filepath, "w") as f:
      json.dump(predictions, f, indent=4)

  logger.info(f"All predictions saved to '{output_filepath}'.")
  return predictions


def parse_judge_text(text: str) -> dict:
  result = {}
  for line in text.splitlines():
    if ":" in line:
      key, value = line.split(":", 1)
      result[key.strip()] = value.strip()
  return result


def extract_judge_answer(question_text, correct_answer, response, model, processors, device):
  judge_prompt = format_judge_prompt(question_text, correct_answer, response)
  conversation = [{"role": "user", "content": [{"type": "text", "text": judge_prompt}]}]
  text_processor = processors["text"]
  prompt = text_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

  with torch.no_grad():
    inputs = text_processor(prompt, return_tensors="pt").to(device)
    raw_judge = model.generate(inputs["input_ids"], max_new_tokens=4096)
    decoded_judge = decode(text_processor, inputs["input_ids"], raw_judge)

  if decoded_judge.strip() == judge_prompt.strip():
    logger.info("Judge output identical to prompt; returning default null answer.")
    return {
      "extracted_final_answer": "None",
      "reasoning": "",
      "correct": "no",
      "confidence": 0,
      "strict": True,
    }

  parsed = parse_judge_text(decoded_judge)
  try:
    # Fix confidence parsing (remove % and convert to int)
    if "confidence" in parsed:
      parsed["confidence"] = int(parsed["confidence"].replace("%", ""))
    # Add strict if missing
    if "strict" not in parsed:
      parsed["strict"] = True
    extracted = ExtractedAnswer.model_validate(parsed)
    return {
      "correct_answer": correct_answer,
      "model_answer": extracted.extracted_final_answer,
      "reasoning": extracted.reasoning,
      "correct": extracted.correct,
      "confidence": extracted.confidence,
    }
  except Exception as e:
    logger.error(f"Failed to validate judge output: {e}")
    judged_correct = "yes" if "yes" in decoded_judge.lower() else "no"
    # Fallback with cleaned confidence
    confidence = 100
    for line in decoded_judge.splitlines():
      if "confidence" in line.lower():
        try:
          confidence = int(re.search(r"\d+", line).group())
        except:
          pass
    return {
      "correct_answer": correct_answer,
      "model_answer": decoded_judge.strip(),
      "reasoning": "Fallback: could not parse judge output.",
      "correct": judged_correct,
      "confidence": confidence,
    }


def judge_predictions(dataset, predictions, model, processors, device, model_id, is_vision_model):
  judged_filepath = get_judged_filename(model_id)
  judged = {}

  if os.path.exists(judged_filepath):
    logger.info(f"Found existing judged file '{judged_filepath}'. Resuming judgement.")
    with open(judged_filepath, "r") as f:
      data = json.load(f)
      for k, v in data.items():
        try:
          judged[k] = ExtractedAnswer.model_validate(v).model_dump()
        except Exception as e:
          logger.error(f"Judged prediction validation error for id {k}: {e}")
  else:
    logger.info("No judged file found. Starting fresh judgement.")

  total = len(dataset)
  for idx, question in enumerate(dataset):
    qid = question["id"]
    if qid not in predictions:
      continue
    if qid in judged:
      logger.info(f"Skipping question {idx+1}/{total} (id: {qid}) as already judged.")
      continue

    response = predictions[qid]["content"]
    logger.info(f"Judging question {idx+1}/{total} (id: {qid})...")
    judge_result = extract_judge_answer(
      question_text=question["question"],
      correct_answer=question["answer"],
      response=response,
      model=model,
      processors=processors,
      device=device,
    )
    judged[qid] = {
      "extracted_final_answer": judge_result["model_answer"],
      "reasoning": judge_result["reasoning"],
      "correct": judge_result["correct"],
      "confidence": judge_result["confidence"],
      "strict": True,
    }

    with open(judged_filepath, "w") as f:
      json.dump(judged, f, indent=4)

  logger.info(f"All judgement results saved to '{judged_filepath}'.")
  dump_metrics(judged, total)
  return judged


def evaluate(model, processors, test_data, device, model_id, resume, is_vision_model):
  predictions = generate_predictions(model, processors, test_data, device, model_id, resume, is_vision_model)
  judge_predictions(test_data, predictions, model, processors, device, model_id, is_vision_model)


if __name__ == "__main__":
  # Main evaluation script with QLoRA and apply_lora
  model_id, cpu, resume, device_map, is_vision_model = get_parameters()

  # Load model and processors with QLoRA
  model, processors = load_model_and_processor(model_id, device_map, is_vision_model)

  # Apply LoRA adapters
  model = apply_lora(model)

  device = torch.device("cuda" if isinstance(device_map, dict) else device_map)

  test_data = load_hle_dataset()
  test_data = test_data.take(2)

  evaluate(model, processors, test_data, device, model_id, resume, is_vision_model)
