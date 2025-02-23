import json
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import torch
from datasets import load_dataset
from pydantic import BaseModel

from grpo.eval import decode, read_judgements_json, read_predictions_json
from grpo.hardware import get_parameters
from grpo.model import get_model, get_processors


def audit_predictions_and_judgements(dataset, predictions_filepath: str, judgements_filepath: str, model, processors, device, model_id):
  print(f"Reading predictions from {predictions_filepath}")
  predictions = read_predictions_json(predictions_filepath)
  print(f"Reading judgements from {judgements_filepath}")
  judgements = read_judgements_json(judgements_filepath)

  if not predictions or not judgements:
    print(f"Missing predictions or judgements: {predictions_filepath}, {judgements_filepath}")
    print(f"Predictions: {predictions}, Judgements: {judgements}")
    return None

  predictions_dict = {p.question_id: p for p in predictions}
  judgements_dict = {j.question_id: j for j in judgements}

  audit_filepath = get_audit_filename(model_id)
  existing_audit = read_audit_json(audit_filepath)
  audit_dict = {entry.question_id: entry.model_dump() for entry in existing_audit} if existing_audit else {}

  if audit_dict:
    print(f"Found existing audit file '{audit_filepath}'. Resuming audit with {len(audit_dict)} entries.")
  else:
    print("No audit file found. Starting fresh audit.")

  total = len(dataset)
  text_processor = processors["text"]

  for idx, question in enumerate(dataset):
    qid = question["id"]

    print(f"Processing question {idx+1}/{total} with ID: {qid}")

    if qid not in predictions_dict or qid not in judgements_dict:
      print(f"Question {qid} not found in predictions or judgements. Skipping.")
      continue

    if qid in audit_dict:
      print(f"Skipping question {idx+1}/{total} (id: {qid}) as already audited.")
      continue

    print(f"Auditing question {idx+1}/{total} (id: {qid})...")

    model_answer = predictions_dict[qid].content.answer
    correct_answer = question["answer"]

    audit_prompt = f"""Compare these two statements. Respond with exactly one word: 'same' if they are identical, 'similar' if they are close but not identical, 'different' if they are distinct. Use only 'same', 'different', or 'similar'.\n- {model_answer}\n- {correct_answer}\n\nAnswer: """

    conversation = [{"role": "user", "content": [{"type": "text", "text": audit_prompt}]}]
    prompt = text_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = text_processor(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
      raw_outputs = model.generate(**inputs, max_new_tokens=16)
      decoded = decode(text_processor, inputs["input_ids"], raw_outputs).strip()
      print(f"Model output for {qid}: '{decoded}'")
      del raw_outputs

    # remove all non-alpha characters
    audit_result = "".join([c for c in decoded if c.isalpha()])
    audit_result = audit_result.lower().strip()
    if audit_result not in ["same", "different", "similar"]:
      print(f"Invalid audit response for {qid}: '{audit_result}'. Defaulting to 'different'")
      audit_result = "different"

    audit_entry = AuditEntry(
      question_id=qid,
      question_text=question["question"],
      correct_answer=correct_answer,
      predicted_answer=model_answer,
      judgement_correct_yes_no=judgements_dict[qid].correct_yes_no,
      audit=audit_result,
    )

    audit_dict[qid] = audit_entry.model_dump()
    try:
      with open(audit_filepath, "w", encoding="utf-8") as f:
        json.dump(audit_dict, f, indent=4)
      print(f"Updated audit JSON with entry for {qid}. Total entries: {len(audit_dict)}")
    except Exception as e:
      print(f"Failed to write audit JSON for {qid}: {e}")

    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  # Summary and Statistics
  print("\n=== Audit Summary ===")
  success_count = 0
  failure_count = 0
  false_positive_count = 0
  false_negative_count = 0
  total_processed = len(audit_dict)

  print("Audit Successes (audit aligns with judgement):")
  for qid, entry in audit_dict.items():
    judgement = entry["judgement_correct_yes_no"]
    audit = entry["audit"]
    is_success = (judgement == "yes" and audit in ["same", "similar"]) or (judgement == "no" and audit == "different")
    if is_success:
      success_count += 1
      if idx < 5:
        print(f"- QID: {qid}, Predicted: '{entry['predicted_answer']}', Correct: '{entry['correct_answer']}', Judgement: '{judgement}', Audit: '{audit}'")

  print("\nAudit Failures (audit contradicts judgement):")
  for idx, (qid, entry) in enumerate(audit_dict.items()):
    judgement = entry["judgement_correct_yes_no"]
    audit = entry["audit"]
    is_failure = (judgement == "yes" and audit == "different") or (judgement == "no" and audit in ["same", "similar"])
    if is_failure:
      failure_count += 1
      if judgement == "no" and audit in ["same", "similar"]:
        false_positive_count += 1
        if idx < 5:
          print(
            f"- False Positive: QID: {qid}, Predicted: '{entry['predicted_answer']}', Correct: '{entry['correct_answer']}', Judgement: 'no', Audit: '{audit}'"
          )
      elif judgement == "yes" and audit == "different":
        false_negative_count += 1
        if idx < 5:

          print(
            f"- False Negative: QID: {qid}, Predicted: '{entry['predicted_answer']}', Correct: '{entry['correct_answer']}', Judgement: 'yes', Audit: 'different'"
          )

  print(f"\nStatistics:")
  print(f"Total questions processed: {total_processed}")
  print(f"Audit Successes: {success_count} ({success_count/total_processed*100:.2f}%)")
  print(f"Audit Failures: {failure_count} ({failure_count/total_processed*100:.2f}%)")
  print(f"False Positives (judged 'no', audited 'same'/'similar'): {false_positive_count} ({false_positive_count/total_processed*100:.2f}%)")
  print(f"False Negatives (judged 'yes', audited 'different'): {false_negative_count} ({false_negative_count/total_processed*100:.2f}%)")

  print(f"All audit results saved to '{audit_filepath}'. Processed {len(audit_dict)} entries.")
  return audit_dict


class AuditEntry(BaseModel):
  question_id: str
  question_text: str
  correct_answer: str
  predicted_answer: str
  judgement_correct_yes_no: Literal["yes", "no"]
  audit: Literal["same", "different", "similar"]


def get_audit_filename(model_name):
  audit_filename = os.getenv("AUDIT_FILENAME")
  if audit_filename:
    return audit_filename

  date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  os.makedirs(".audit", exist_ok=True)
  safe_name = model_name.replace("/", "_").replace(" ", "_")
  return os.path.join(".audit", f"{safe_name}_{date_time}_audit.json")


def read_audit_json(filepath: str) -> Optional[List[AuditEntry]]:
  if not os.path.exists(filepath):
    return None

  try:
    with open(filepath, "r", encoding="utf-8") as f:
      data = json.load(f)
    audit_entries = []
    for value in data.values():
      try:
        entry = AuditEntry.model_validate(value)
        audit_entries.append(entry)
      except Exception as e:
        print(f"Audit entry validation error in {filepath}: {e}")
    return audit_entries if audit_entries else None
  except Exception as e:
    print(f"Failed to read audit JSON file {filepath}: {e}")
    return None


def write_audit_json(filepath: str, audit_entries: List[AuditEntry]) -> None:
  try:
    audit_dict = {entry.question_id: entry.model_dump() for entry in audit_entries}
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(audit_dict, f, indent=4)
  except Exception as e:
    print(f"Failed to write audit JSON file {filepath}: {e}")


if __name__ == "__main__":
  import unittest

  class TestAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
      cls.model_id, cls.cpu, cls.resume, cls.device_map, cls.is_vision_model = get_parameters()
      cls.model = get_model(cls.model_id, cls.device_map, cls.is_vision_model)
      cls.processors = get_processors(cls.model_id, cls.is_vision_model)
      cls.device = torch.device("cuda" if isinstance(cls.device_map, dict) else cls.device_map)

      cls.test_dataset = load_dataset("cais/hle", split="test").select(range(2))

      cls.test_predictions = {
        cls.test_dataset[0]["id"]: {
          "question_id": cls.test_dataset[0]["id"],
          "model": cls.model_id,
          "content": {"explanation": "Test prediction 1", "answer": cls.test_dataset[0]["answer"], "confidence": 95},
        },
        cls.test_dataset[1]["id"]: {
          "question_id": cls.test_dataset[1]["id"],
          "model": cls.model_id,
          "content": {"explanation": "Test prediction 2", "answer": "hello", "confidence": 90},
        },
      }
      cls.test_judgements = {
        cls.test_dataset[0]["id"]: {
          "question_id": cls.test_dataset[0]["id"],
          "extracted_final_answer": cls.test_dataset[0]["answer"],
          "question_text": cls.test_dataset[0]["question"],
          "correct_answer": cls.test_dataset[0]["answer"],
          "reasoning": "Correct",
          "correct_yes_no": "yes",
          "confidence": 95,
          "answer_type": cls.test_dataset[0]["answer_type"],
        },
        cls.test_dataset[1]["id"]: {
          "question_id": cls.test_dataset[1]["id"],
          "extracted_final_answer": "wrong",
          "question_text": cls.test_dataset[1]["question"],
          "correct_answer": cls.test_dataset[1]["answer"],
          "reasoning": "Incorrect",
          "correct_yes_no": "no",
          "confidence": 90,
          "answer_type": cls.test_dataset[1]["answer_type"],
        },
      }

      cls.predictions_filepath = "test_predictions.json"
      cls.judgements_filepath = "test_judgements.json"

      print(f"Writing test predictions to {cls.predictions_filepath}")
      with open(cls.predictions_filepath, "w") as f:
        json.dump(cls.test_predictions, f)
      print(f"Writing test judgements to {cls.judgements_filepath}")
      with open(cls.judgements_filepath, "w") as f:
        json.dump(cls.test_judgements, f)

    @classmethod
    def tearDownClass(cls):
      if os.path.exists(cls.predictions_filepath):
        os.remove(cls.predictions_filepath)
      if os.path.exists(cls.judgements_filepath):
        os.remove(cls.judgements_filepath)
      audit_file = get_audit_filename(cls.model_id)
      if os.path.exists(audit_file):
        os.remove(audit_file)
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def test_audit_predictions_and_judgements(self):
      print("Starting audit test")

      result = audit_predictions_and_judgements(
        self.test_dataset, self.predictions_filepath, self.judgements_filepath, self.model, self.processors, self.device, self.model_id
      )

      self.assertIsNotNone(result, "Audit result should not be None")
      self.assertIsInstance(result, dict, "Result should be a dictionary")
      self.assertEqual(len(result), 2, f"Expected 2 entries, got {len(result)}")

      audit_file = get_audit_filename(self.model_id)
      self.assertTrue(os.path.exists(audit_file), f"Audit file {audit_file} should exist")

      with open(audit_file, "r") as f:
        audit_data = json.load(f)

      self.assertEqual(len(audit_data), 2, "Audit file should contain 2 entries")

      for i, qid in enumerate([self.test_dataset[0]["id"], self.test_dataset[1]["id"]]):
        self.assertIn(qid, audit_data, f"Question {qid} should be in audit results")
        entry = audit_data[qid]
        self.assertEqual(entry["question_id"], qid)
        self.assertEqual(entry["question_text"], self.test_dataset[i]["question"])
        self.assertEqual(entry["correct_answer"], self.test_dataset[i]["answer"])
        self.assertIn(entry["audit"], ["same", "different", "similar"])

    def test_missing_files(self):
      result = audit_predictions_and_judgements(
        self.test_dataset, "nonexistent_predictions.json", "nonexistent_judgements.json", self.model, self.processors, self.device, self.model_id
      )
      self.assertIsNone(result)

  unittest.main()
