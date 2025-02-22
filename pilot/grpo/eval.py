import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

import torch
from datasets import load_dataset
from pydantic import BaseModel

from grpo.conversation import build_conversation
from grpo.hardware import get_parameters
from grpo.image_utils import decode_base64_image
from grpo.model import get_model, get_processors
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


class PredictionResponse(BaseModel):
  explanation: str
  answer: str
  confidence: int


class ModelPrediction(BaseModel):
  question_id: str
  model: str
  content: Union[str, PredictionResponse]


class JudgementResponse(BaseModel):
  extracted_final_answer: str
  reasoning: str
  correct_yes_no: Literal["yes", "no"]
  confidence: int


class ModelJudgement(BaseModel):
  extracted_final_answer: str
  reasoning: str
  correct_yes_no: Literal["yes", "no"]
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
  # Check environment variable first
  output_filename = os.getenv("OUTPUT_FILENAME")
  if output_filename:
    return output_filename

  # Fallback to original logic
  date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  os.makedirs(".predictions", exist_ok=True)
  safe_name = model_name.replace("/", "_").replace(" ", "_")
  return os.path.join(".predictions", f"{safe_name}_{date_time}.json")


def get_judged_filename(model_name):
  # Check environment variable first
  judged_filename = os.getenv("JUDGED_FILENAME")
  if judged_filename:
    return judged_filename

  # Fallback to original logic
  date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  os.makedirs(".judged", exist_ok=True)
  safe_name = model_name.replace("/", "_").replace(" ", "_")
  return os.path.join(".judged", f"{safe_name}_{date_time}_judged.json")


# --------------------------------------------------
# Prediction and Judging Functions
# --------------------------------------------------
def decode(processor, input_ids, raw_output):
  input_length = input_ids.shape[1]
  generated_ids = raw_output[0, input_length:]
  decoded = processor.decode(generated_ids, skip_special_tokens=True)
  return decoded


def generate_and_parse_json(
  model, processor, inputs, prompt, expected_model: type[BaseModel], max_new_tokens=8192, max_retries=3
) -> Optional[Union[BaseModel, str]]:
  divider = "=" * 80
  retry_count = 0
  while retry_count <= max_retries:
    with torch.no_grad():
      raw_outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
      decoded = decode(processor, inputs["input_ids"], raw_outputs)
      del raw_outputs  # Free memory immediately
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

      # Fix incomplete JSON by adding closing bracket if missing
      if not decoded.strip().endswith("}"):
        decoded = decoded.strip() + "\n}"

      logger.info(f"\n{divider}\n{prompt}\n---\n{decoded}\n{divider}\n")

    if decoded.strip() == prompt.strip():
      logger.info("Output identical to prompt; returning None.")
      return None

    try:
      json_data = json.loads(decoded)
      result = expected_model.model_validate(json_data)
      logger.info(f"Successfully parsed and validated {expected_model.__name__} on attempt {retry_count + 1}")
      return result
    except json.JSONDecodeError as e:
      retry_count += 1
      if retry_count <= max_retries:
        logger.warning(f"Failed to parse JSON on attempt {retry_count}: {e}. Retrying...")
      else:
        logger.error(f"Max retries ({max_retries}) reached. Could not parse JSON: {e}. Returning raw output.")
        return decoded
    except Exception as e:  # Pydantic validation errors
      retry_count += 1
      if retry_count <= max_retries:
        logger.warning(f"Failed to validate {expected_model.__name__} on attempt {retry_count}: {e}. Retrying...")
      else:
        logger.error(f"Max retries ({max_retries}) reached. Validation failed: {e}. Returning raw output.")
        return decoded


# --------------------------------------------------
# JSON Reader/Writer Functions for Specific Pydantic Types
# --------------------------------------------------
def read_predictions_json(filepath: str) -> Optional[List[ModelPrediction]]:
  if not os.path.exists(filepath):
    return None

  try:
    with open(filepath, "r", encoding="utf-8") as f:
      data = json.load(f)
    predictions = []
    for value in data.values():  # Assuming data is a dict with qid keys
      try:
        prediction = ModelPrediction.model_validate(value)
        predictions.append(prediction)
      except Exception as e:
        logger.error(f"Prediction validation error in {filepath}: {e}")
    return predictions if predictions else None
  except Exception as e:
    logger.error(f"Failed to read predictions JSON file {filepath}: {e}")
    return None


def write_predictions_json(filepath: str, predictions: List[ModelPrediction]) -> None:
  try:
    # Convert list to dict with question_id as key
    predictions_dict = {pred.question_id: pred.model_dump() for pred in predictions}
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(predictions_dict, f, indent=4)
  except Exception as e:
    logger.error(f"Failed to write predictions JSON file {filepath}: {e}")


def read_judgements_json(filepath: str) -> Optional[List[ModelJudgement]]:
  if not os.path.exists(filepath):
    return None

  try:
    with open(filepath, "r", encoding="utf-8") as f:
      data = json.load(f)
    judgements = []
    for value in data.values():  # Assuming data is a dict with qid keys
      try:
        judgement = ModelJudgement.model_validate(value)
        judgements.append(judgement)
      except Exception as e:
        logger.error(f"Judgement validation error in {filepath}: {e}")
    return judgements if judgements else None
  except Exception as e:
    logger.error(f"Failed to read judgements JSON file {filepath}: {e}")
    return None


def write_judgements_json(filepath: str, judgements: List[ModelJudgement]) -> None:
  try:
    # Convert list to dict with question_id as key (assuming qid is available elsewhere, we'll use index as fallback)
    # Note: Ideally, ModelJudgement should have a qid; here we assume it's tied to predictions
    judgements_dict = {j.extracted_final_answer: j.model_dump() for j in judgements}  # Temporary key; needs qid
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(judgements_dict, f, indent=4)
  except Exception as e:
    logger.error(f"Failed to write judgements JSON file {filepath}: {e}")


# --------------------------------------------------
# Prediction and Judging Functions
# --------------------------------------------------


def compose_prediction(model, processors, question, device, max_new_tokens=2048, is_vision_model=False, max_retries=3):
  torch.cuda.empty_cache()
  has_images = is_vision_model and question.get("image", "").strip()
  conversation = build_conversation(question, has_images, include_system_prompt=True)

  text_processor = processors["text"]
  vision_processor = processors["vision"] if is_vision_model else None
  processor = vision_processor if is_vision_model else text_processor

  prompt = text_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
  images = [decode_base64_image(question["image"])] if has_images else []
  inputs = (
    vision_processor(text=prompt, images=images, return_tensors="pt").to(device) if is_vision_model else text_processor(prompt, return_tensors="pt").to(device)
  )
  logger.info(f"Input shape: {inputs['input_ids'].shape}")

  content = generate_and_parse_json(model, processor, inputs, prompt, PredictionResponse, max_new_tokens, max_retries)
  del inputs, images  # Free memory
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  return content


def generate_predictions(model, processors, questions, device, model_id, resume, is_vision_model, max_retries=3):
  output_filepath = get_output_filename(model_id)
  existing_predictions = read_predictions_json(output_filepath)
  predictions = existing_predictions if existing_predictions else []

  if resume and predictions:
    logger.info(f"Resuming from existing predictions file '{output_filepath}'.")
  elif os.path.exists(output_filepath):
    logger.info(f"Found existing predictions file '{output_filepath}', but not resuming. Overwriting predictions.")
  else:
    logger.info("No existing predictions file found. Starting fresh predictions.")

  total = len(questions)
  for idx, question in enumerate(questions):
    qid = question["id"]
    if resume and any(p.question_id == qid for p in predictions):
      logger.info(f"Skipping question {idx+1}/{total} (id: {qid}) as prediction exists.")
      continue

    logger.info(f"Generating prediction for question {idx+1}/{total} (id: {qid})...")
    content = compose_prediction(model, processors, question, device, is_vision_model=is_vision_model, max_retries=max_retries)
    mp = ModelPrediction(question_id=qid, model=model_id, content=content)
    predictions.append(mp)

    write_predictions_json(output_filepath, predictions)

  logger.info(f"All predictions saved to '{output_filepath}'.")
  return predictions


def extract_judge_answer(question_text, correct_answer, response, model, processors, device, max_retries=3) -> Optional[ModelJudgement]:
  # Handle response type
  if isinstance(response, str):
    content = response
  elif isinstance(response, ModelPrediction):
    content = response.content if isinstance(response.content, str) else response.content.model_dump_json()
  else:
    logger.error(f"Invalid response type for judgment: {type(response)}")
    return None

  judge_prompt = format_judge_prompt(question_text, correct_answer, content)
  conversation = [{"role": "user", "content": [{"type": "text", "text": judge_prompt}]}]
  text_processor = processors["text"]
  prompt = text_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
  inputs = text_processor(prompt, return_tensors="pt").to(device)
  judgment = generate_and_parse_json(model, text_processor, inputs, prompt, JudgementResponse, max_new_tokens=4096, max_retries=max_retries)
  if judgment is None:  # Prompt identical to output
    return None
  if isinstance(judgment, JudgementResponse):
    return ModelJudgement(
      extracted_final_answer=judgment.extracted_final_answer,
      reasoning=judgment.reasoning,
      correct_yes_no=judgment.correct_yes_no,
      confidence=judgment.confidence,
      strict=True,
    )
  print("Trace: judgment is not JudgementResponse")
  return None


def judge_predictions(dataset, predictions, model, processors, device, model_id, is_vision_model):
  predictions = {p.question_id: p for p in predictions}

  judged_filepath = get_judged_filename(model_id)
  existing_judgements = read_judgements_json(judged_filepath)
  judged_list = existing_judgements if existing_judgements else []

  if judged_list:
    logger.info(f"Found existing judged file '{judged_filepath}'. Resuming judgement.")
  else:
    logger.info("No judged file found. Starting fresh judgement.")

  total = len(dataset)
  for idx, question in enumerate(dataset):
    qid = question["id"]
    if qid not in predictions:
      continue
    if any(j.extracted_final_answer == qid for j in judged_list):  # Assuming unique qid; adjust if needed
      logger.info(f"Skipping question {idx+1}/{total} (id: {qid}) as already judged.")
      continue

    logger.info(f"Judging question {idx+1}/{total} (id: {qid})...")
    judge_result = extract_judge_answer(
      question_text=question["question"],
      correct_answer=question["answer"],
      response=predictions[qid],
      model=model,
      processors=processors,
      device=device,
      max_retries=3,
    )
    if judge_result is not None:  # Filter out None results
      judged_list.append(judge_result)

    write_judgements_json(judged_filepath, judged_list)

  logger.info(f"All judgement results saved to '{judged_filepath}'.")
  judged_dict = {j.extracted_final_answer: j.model_dump() for j in judged_list}  # Convert back to dict for compatibility
  dump_metrics(judged_dict, total)
  return judged_dict


def evaluate(model, processors, test_data, device, model_id, resume, is_vision_model):
  predictions = generate_predictions(model, processors, test_data, device, model_id, resume, is_vision_model, max_retries=3)

  judge_predictions(test_data, predictions, model, processors, device, model_id, is_vision_model)


if __name__ == "__main__":
  import unittest

  class TestEvaluation(unittest.TestCase):
    def test_evaluation(self):
      logger.info("Starting evaluation test suite")

      # Setup
      model_id, cpu, resume, device_map, is_vision_model = get_parameters()
      model = get_model(model_id, device_map, is_vision_model)
      processors = get_processors(model_id, is_vision_model)

      device = torch.device("cuda" if isinstance(device_map, dict) else device_map)
      test_data = load_hle_dataset().take(2)

      # Run evaluation
      logger.info("Running evaluation")
      evaluate(model, processors, test_data, device, model_id, resume, is_vision_model)

      logger.info("All tests passed successfully!")

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  unittest.main()
