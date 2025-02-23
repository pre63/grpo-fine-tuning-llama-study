"""
MIT License

Copyright (c) 2025 centerforaisafety

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from typing import Dict

import numpy as np

JUDGE_PROMPT = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]:
 - Explanation: {explanation}
 - Answer: {answer}

Your judgement must be in the format and criteria specified below:

[correct_answer]: {correct_answer}

Please provide your judgment in the following JSON format:
{
  "extracted_final_answer": "{The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response}",
  "reasoning": "{Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on background, do not solve the problem, do not argue for a different answer, focus only on whether the answers match}",
  "correct_yes_no": "{Answer 'yes' if extracted_final_answer matches [correct_answer] or is within a small margin of error for numerical problems, 'no' if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect}",
}

Ensure the output is properly formatted JSON with double-quoted keys and values.
"""


def calib_err(confidence, correct, p="2", beta=100):
  """
    Calculate calibration error for confidence scores against correctness.

    Args:
        confidence (np.ndarray): Array of confidence scores.
        correct (np.ndarray): Array of boolean correctness values.
        p (str): Norm type ('1', '2', or 'infty'). Defaults to '2'.
        beta (int): Target bin size. Defaults to 100.

    Returns:
        float: Calibration error, or 0.0 if no data or invalid input.
    """
  if not isinstance(confidence, np.ndarray) or not isinstance(correct, np.ndarray) or len(confidence) == 0 or len(confidence) != len(correct):
    return 0.0  # Return 0 for invalid or empty input

  # Sort confidence and correct arrays
  idxs = np.argsort(confidence)
  confidence = confidence[idxs]
  correct = correct[idxs]

  # Calculate number of bins, ensuring at least one if data exists
  num_bins = max(1, len(confidence) // beta)
  bins = [[i * beta, min((i + 1) * beta, len(confidence))] for i in range(num_bins)]

  cerr = 0
  total_examples = len(confidence)
  for bin_start, bin_end in bins:
    bin_confidence = confidence[bin_start:bin_end]
    bin_correct = correct[bin_start:bin_end]
    num_examples_in_bin = len(bin_confidence)

    if num_examples_in_bin > 0:
      avg_confidence = np.nanmean(bin_confidence)
      avg_correct = np.nanmean(bin_correct)
      difference = np.abs(avg_confidence - avg_correct)

      if p == "2":
        cerr += num_examples_in_bin / total_examples * np.square(difference)
      elif p == "1":
        cerr += num_examples_in_bin / total_examples * difference
      elif p in ["infty", "infinity", "max"]:
        cerr = np.maximum(cerr, difference)
      else:
        raise ValueError("p must be '1', '2', or 'infty'")

  return np.sqrt(cerr) if p == "2" else cerr


def dump_metrics(predictions: Dict[str, Dict], n: int) -> None:
  """
    Compute and print evaluation metrics for predictions.

    Args:
        predictions (Dict[str, Dict]): Dictionary of prediction results keyed by question ID.
        n (int): Total number of questions.

    Returns:
        None: Prints metrics to console.
    """
  correct = []
  confidence = []

  for k, v in predictions.items():
    if "correct_yes_no" in v and "confidence" in v:  # Check for direct fields instead of nested judge_response
      correct.append(v["correct_yes_no"] == "yes")
      confidence.append(int(v["confidence"]))
    else:
      print(f"Missing judge response for {k}, you should rerun the judge")

  correct = np.array(correct, dtype=bool)
  confidence = np.array(confidence, dtype=float)

  if len(correct) != n:
    print(f"Available predictions: {len(correct)} | Total questions: {n}")

  if len(correct) == 0:
    print("No valid judgments to evaluate")
    print("*** Metrics ***")
    print(f"Accuracy: N/A | n = {n}")
    print("Calibration Error: N/A")
    return

  accuracy = round(100 * sum(correct) / n, 2)
  confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2) if n > 0 else 0
  calibration_error = round(calib_err(confidence, correct, p="2", beta=100), 2)

  print("*** Metrics ***")
  print(f"Accuracy: {accuracy}% +/- {confidence_half_width}% | n = {n}")
  print(f"Calibration Error: {calibration_error}")


def format_judge_prompt(question, correct_answer, content):
  explanation = content.explanation
  answer = content.answer

  replacements = {
    "{question}": question,
    "{correct_answer}": correct_answer,
    "{explanation}": explanation,
    "{answer}": answer,
  }
  judge_prompt = JUDGE_PROMPT
  for key, value in replacements.items():
    judge_prompt = judge_prompt.replace(key, value)

  return judge_prompt
