import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type, Union

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
  content: PredictionResponse


class JudgementResponse(BaseModel):
  extracted_final_answer: str
  reasoning: str
  correct_yes_no: Literal["yes", "no"]


class ModelJudgement(BaseModel):
  question_id: str
  extracted_final_answer: str
  question_text: str
  correct_answer: str
  reasoning: str
  correct_yes_no: Literal["yes", "no"]
  confidence: int
  answer_type: str


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------


def load_hle_dataset():
  print("Loading HLE test dataset...")
  dataset = load_dataset("cais/hle", split="test")
  print(f"Loaded {len(dataset)} questions from HLE test dataset.")
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
  model: Any, processor: Any, inputs: Dict[str, torch.Tensor], prompt: str, expected_model: Type[BaseModel], max_new_tokens: int = 8192, max_retries: int = 3
) -> Optional[BaseModel]:
  divider: str = "=" * 80

  for attempt in range(max_retries + 1):
    with torch.no_grad():
      raw_outputs: torch.Tensor = model.generate(**inputs, max_new_tokens=max_new_tokens)
      decoded: str = decode(processor, inputs["input_ids"], raw_outputs).strip()
      del raw_outputs

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if decoded.strip() == prompt.strip():
      print("Output identical to prompt; returning None.")
      return None

    # Attempt to fix incomplete JSON
    fixed_decoded: str = _fix_incomplete_json(decoded)

    print(f"\n{divider}\n{prompt}\n---\n{fixed_decoded}\n{divider}\n")

    if fixed_decoded is None:
      print(f"Failed to fix JSON for attempt {attempt + 1}. Returning None. {decoded}")
      return None

    try:
      json_data: Dict[str, Any] = json.loads(fixed_decoded)
      result: BaseModel = expected_model.model_validate(json_data)
      return result
    except (json.JSONDecodeError, Exception) as e:
      if attempt < max_retries:
        print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
      else:
        print(f"Max retries ({max_retries}) reached. Failed with: {e}. Returning None. {decoded}")

      return None


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
        print(f"Prediction validation error in {filepath}: {e}")
    return predictions if predictions else None
  except Exception as e:
    print(f"Failed to read predictions JSON file {filepath}: {e}")
    return None


def write_predictions_json(filepath: str, predictions: List[ModelPrediction]) -> None:
  try:
    # Convert list to dict with question_id as key
    predictions_dict = {pred.question_id: pred.model_dump() for pred in predictions}
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(predictions_dict, f, indent=4)
  except Exception as e:
    print(f"Failed to write predictions JSON file {filepath}: {e}")


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
        print(f"Judgement validation error in {filepath}: {e}")
    return judgements if judgements else None
  except Exception as e:
    print(f"Failed to read judgements JSON file {filepath}: {e}")
    return None


def write_judgements_json(filepath: str, judgements: List[ModelJudgement]) -> None:
  try:
    # Convert list to dict with question_id as key (assuming qid is available elsewhere, we'll use index as fallback)
    judgements_dict = {j.question_id: j.model_dump() for j in judgements}
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(judgements_dict, f, indent=4)
  except Exception as e:
    print(f"Failed to write judgements JSON file {filepath}: {e}")


# --------------------------------------------------
# Prediction and Judging Functions
# --------------------------------------------------


def compose_prediction(model, processors, question, device, max_new_tokens=2048, is_vision_model=False, max_retries=3):
  torch.cuda.empty_cache()
  has_images = is_vision_model and question.get("image", "").strip()
  conversation = build_conversation(question, has_images, include_system_prompt=True, eval=True)

  text_processor = processors["text"]
  vision_processor = processors["vision"] if is_vision_model else None
  processor = vision_processor if is_vision_model else text_processor

  prompt = text_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
  images = [decode_base64_image(question["image"])] if has_images else []
  inputs = (
    vision_processor(text=prompt, images=images, return_tensors="pt").to(device) if is_vision_model else text_processor(prompt, return_tensors="pt").to(device)
  )
  print(f"Input shape: {inputs['input_ids'].shape}")

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
    print(f"Resuming from existing predictions file '{output_filepath}'.")
  elif os.path.exists(output_filepath):
    print(f"Found existing predictions file '{output_filepath}', but not resuming. Overwriting predictions.")
  else:
    print("No existing predictions file found. Starting fresh predictions.")

  total = len(questions)
  for idx, question in enumerate(questions):
    qid = question["id"]
    if resume and any(p.question_id == qid for p in predictions):
      print(f"Skipping question {idx+1}/{total} (id: {qid}) as prediction exists.")
      continue

    print(f"Generating prediction for question {idx+1}/{total} (id: {qid})...")
    content = compose_prediction(model, processors, question, device, is_vision_model=is_vision_model, max_retries=max_retries)

    if not isinstance(content, PredictionResponse):
      print(f"Invalid prediction type for question {qid}, skipping.")
      continue

    mp = ModelPrediction(question_id=qid, model=model_id, content=content)
    predictions.append(mp)

    write_predictions_json(output_filepath, predictions)

  print(f"All predictions saved to '{output_filepath}'.")
  return predictions


def extract_judge_answer(question, response: Union[str, ModelPrediction], model, processors, device, max_retries=3) -> Optional[ModelJudgement]:
  question_id = question["id"]
  question_text = question["question"]
  correct_answer = question["answer"]
  answer_type = question["answer_type"]

  # Handle response type
  if not isinstance(response, ModelPrediction):
    print(
      f"Invalid response type for judgment: {type(response)}, skipping question {question_id}, {response if isinstance(response, str) else response.content}"
    )
    return None

  judge_prompt = format_judge_prompt(question_text, correct_answer, response.content)
  conversation = [{"role": "user", "content": [{"type": "text", "text": judge_prompt}]}]
  text_processor = processors["text"]
  prompt = text_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
  inputs = text_processor(prompt, return_tensors="pt").to(device)
  judgment = generate_and_parse_json(model, text_processor, inputs, prompt, JudgementResponse, max_new_tokens=4096, max_retries=max_retries)

  if judgment is None:  # Prompt identical to output
    return None

  if isinstance(judgment, JudgementResponse):
    return ModelJudgement(
      question_id=question_id,
      extracted_final_answer=judgment.extracted_final_answer,
      question_text=question_text,
      correct_answer=correct_answer,
      reasoning=judgment.reasoning,
      correct_yes_no=judgment.correct_yes_no,
      confidence=response.content.confidence,
      answer_type=answer_type,
    )

  print("Trace: judgment is not JudgementResponse")
  return None


def judge_predictions(dataset, predictions, model, processors, device, model_id):
  predictions = {p.question_id: p for p in predictions}

  judged_filepath = get_judged_filename(model_id)
  existing_judgements = read_judgements_json(judged_filepath)
  judged_list = existing_judgements if existing_judgements else []

  if judged_list:
    print(f"Found existing judged file '{judged_filepath}'. Resuming judgement.")
  else:
    print("No judged file found. Starting fresh judgement.")

  total = len(dataset)
  for idx, question in enumerate(dataset):
    qid = question["id"]
    if qid not in predictions:
      continue
    if any(j.question_id == qid for j in judged_list):  # Skip if already judged
      print(f"Skipping question {idx+1}/{total} (id: {qid}) as already judged.")
      continue

    print(f"Judging question {idx+1}/{total} (id: {qid})...")
    judge_result = extract_judge_answer(
      question=question,
      response=predictions[qid],
      model=model,
      processors=processors,
      device=device,
      max_retries=3,
    )

    if judge_result is not None:  # Filter out None results
      judged_list.append(judge_result)

    write_judgements_json(judged_filepath, judged_list)

  print(f"All judgement results saved to '{judged_filepath}'.")
  judged_dict = {j.question_id: j.model_dump() for j in judged_list}
  dump_metrics(judged_dict, total)
  return judged_dict


def _fix_incomplete_json(raw: str) -> Optional[str]:
  """Fix incomplete or malformed JSON, escaping control characters and appending "}."""
  cleaned = raw.strip()
  start_idx = cleaned.find("{")

  if not cleaned or start_idx == -1:
    return None

  # Take everything from the first { onward
  cleaned = cleaned[start_idx:]

  # Escape control characters only within quoted strings
  def escape_control_chars(match):
    value = match.group(0)
    # Double invalid single backslashes not followed by valid escapes
    value = re.sub(r'\\(?![ntrfb"\\])', r"\\\\", value)
    return value

  # Apply escaping only to content within double quotes
  cleaned = re.sub(r".*?", escape_control_chars, cleaned, flags=re.DOTALL)

  # Take up to the last } if it exists and try parsing
  last_brace_idx = cleaned.rfind("}")
  if last_brace_idx != -1:
    candidate = cleaned[: last_brace_idx + 1]
    try:
      json.loads(candidate)
      return candidate  # If it parses, we’re done
    except json.JSONDecodeError:
      pass

  # Trim trailing junk and append "}"
  cleaned = cleaned.rstrip("\"}:, '`\t\n") + '"\n}'
  return cleaned


def evaluate(model, processors, test_data, device, model_id, resume, is_vision_model):
  predictions = generate_predictions(model, processors, test_data, device, model_id, resume, is_vision_model, max_retries=3)

  judge_predictions(test_data, predictions, model, processors, device, model_id)


if __name__ == "__main__":
  import unittest

  class TestEvaluation(unittest.TestCase):

    def _test_json_parsing(self, raw: str, expect_none: bool = False):
      """Helper method to test JSON fixing and parsing."""
      fixed = _fix_incomplete_json(raw)
      if expect_none:
        self.assertIsNone(fixed)
      else:
        try:
          parsed = json.loads(fixed)
          self.assertIsInstance(parsed, dict)
        except json.JSONDecodeError as e:
          self.fail(f"Failed to parse: {e} - {fixed}")

    def test_extra_closing_brace(self):
      raw = """
          {
              "explanation": "The average adult height of the population is 3 feet and 6 inches.",
              "answer": "3.06",
              "confidence": "100"
          }}
          """
      self._test_json_parsing(raw)

    def test_extra_closing_brace_with_code_block(self):
      raw = """
          ```
          {
              "explanation": "The average adult height of the population is 3 feet and 6 inches.",
              "answer": "3.06",
              "confidence": "100"
          }
          ```
          """
      self._test_json_parsing(raw)

    def test_extra_text_after_valid_json(self):
      raw = """
          Here is the response to your query in valid json:
          ```
          {
              "explanation": "The average adult height of the population is 3 feet and 6 inches.",
              "answer": "3.06",
              "confidence": "100"
          }
          ```
          Enjoy the solution.
          """
      self._test_json_parsing(raw)

    def test_truncated_string_value(self):
      raw = """
          {
              "explanation": "The third homotopy group is T^3.",
              "answer": "3",
              "confidence": "100
          }
          """
      self._test_json_parsing(raw)

    def test_completely_valid_json(self):
      raw = """
          {
              "explanation": "All good here.",
              "answer": "yes",
              "confidence": "95"
          }
          """
      self._test_json_parsing(raw)

    def test_missing_closing_brace(self):
      raw = """
          {
              "explanation": "Incomplete JSON",
              "answer": "no"
          """
      self._test_json_parsing(raw)

    def test_empty_string(self):
      raw = ""
      self._test_json_parsing(raw, expect_none=True)

    def test_trailing_comma(self):
      raw = """
          {
              "explanation": "Extra comma issue",
              "answer": "yes",
              "confidence": "90",
          }
          """
      self._test_json_parsing(raw)

    def test_malformed_number_with_extra_brace(self):
      raw = """
          {
              "explanation": "Height example",
              "answer": "3.06",
              "confidence": "100}
          }
          """
      self._test_json_parsing(raw)

    def test_complex_latex_content(self):
      raw = """
          {"explanation": "The first nontrivial group of symmetries of the unit square is the dihedral group $D_4$, which consists of 8 elements: $4$ rotations and 4 reflections. The moduli space $X$ of nondegenerate lattices in $\\mathbb{R}^2$ with unit area is the quotient space $D_4/X$, where $X$ is the quotient space of the identity in $D_4$ by the conjugacy relation. This quotient space can be identified with the projective plane $P^1$, whose points are the equivalence classes of lines through the origin. The group $D_4$ acts on $P^1$ by translations and rotations, and this action is transitive. Therefore, the moduli space $X$ is homeomorphic to the real line $\\mathbb{R}$, and its fundamental group is isomorphic to $\\mathbb{Z}$. The first nontrivial element of the fundamental group is the element of order 2, which corresponds to the rotation by $90^\\circ$ about the origin. Therefore, the first nontrivial homology group $H_1(X, \\mathbb{Z})$ is the trivial group, which is $\\boxed{0}$."}
          """
      self._test_json_parsing(raw)

    def test_evaluation(self):
      print("Starting evaluation test suite")

      # Setup
      model_id, cpu, resume, device_map, is_vision_model = get_parameters()
      model = get_model(model_id, device_map, is_vision_model)
      processors = get_processors(model_id, is_vision_model)

      device = torch.device("cuda" if isinstance(device_map, dict) else device_map)
      test_data = load_hle_dataset().take(2)

      # Run evaluation
      print("Running evaluation")
      evaluate(model, processors, test_data, device, model_id, resume, is_vision_model)

      print("All tests passed successfully!")

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  unittest.main()
