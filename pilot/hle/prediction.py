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

SYSTEM_EXACT_ANSWER = """
Your response must be a valid JSON object containing the following fields:
{
  "explanation": "{your explanation for your final answer}",
  "answer": "{your succinct, exact, final answer}",
  "confidence": "{your confidence score as an integer between 0 and 100}"
}

Example:
{
  "explanation": "The problem involves calculating 2 + 2, which is a simple arithmetic operation.",
  "answer": "4",
  "confidence": "100"
}

Ensure the output is only the properly formatted JSON with double-quoted keys and values, and Confidence as an integer (no % symbol).
"""


SYSTEM_MC = """
Your response must be a valid JSON object containing the following fields:
{
  "explanation": "{your explanation for your answer choice}",
  "answer": "{your chosen answer}",
  "confidence": "{your confidence score as an integer between 0 and 100}"
}

Example:
{
  "explanation": "The question asks for the capital of France, and among the options, Paris is the correct city.",
  "answer": "Paris",
  "confidence": "95"
}

Ensure the output is only the properly formatted JSON with double-quoted keys and values, and Confidence as an integer (no % symbol).
"""


def get_system_prompt(question):
  answer_type = question["answer_type"]
  return SYSTEM_EXACT_ANSWER if answer_type == "exactMatch" else SYSTEM_MC
