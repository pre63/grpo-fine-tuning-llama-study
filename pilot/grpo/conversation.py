from typing import Dict, List, Optional

from PIL import Image as PIL_Image

from grpo.image_utils import decode_base64_image


def build_conversation(example: Dict, is_vision_model: bool) -> List[Dict]:
  user_content = [{"type": "text", "text": example["question"]}]
  if is_vision_model and example.get("image", "").strip():
    image = decode_base64_image(example["image"])
    user_content = [{"type": "image"}] + user_content if image else user_content
  return [
    {"role": "user", "content": user_content},
    {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
  ]


def tokenize_prompt(processor, conversation: List[Dict], images: List[PIL_Image.Image]) -> Dict:
  prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
  is_vision_model = "vision" in processor.model_name.lower()
  tokens = (
    processor(text=prompt, images=images if is_vision_model and images else None, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    if is_vision_model
    else processor(prompt, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
  )
  input_ids = tokens["input_ids"][0].tolist()
  labels = [-100 if token == processor.pad_token_id else token for token in input_ids]
  return {"input_ids": input_ids, "labels": labels, "answer": conversation[1]["content"][0]["text"]}


def tokenize_example(example: Dict, processor) -> Dict:
  is_vision_model = "vision" in processor.model_name.lower()
  images = [decode_base64_image(example["image"]) or None] if is_vision_model and example.get("image", "").strip() else []
  conversation = build_conversation(example, is_vision_model)
  tokenized = tokenize_prompt(processor, conversation, [img for img in images if img])
  tokenized["prompt"] = example["question"]
  return tokenized


if __name__ == "__main__":
  import base64
  import io
  import unittest

  from PIL import Image
  from transformers import AutoTokenizer

  class TestConversationIntegration(unittest.TestCase):
    def test_conversation_to_tokenization(self):
      # Use Llama-3.2-1B-Instruct tokenizer
      model_id = "meta-llama/Llama-3.2-1B-Instruct"
      tokenizer = AutoTokenizer.from_pretrained(model_id)
      if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
      # Manually set model_name to mimic load_model_and_processor
      tokenizer.model_name = model_id
      example = {"question": "What is this?", "answer": "A test", "image": base64.b64encode(Image.new("RGB", (10, 10), "blue")._repr_png_()).decode("utf-8")}
      result = tokenize_example(example, tokenizer)
      self.assertIn("input_ids", result)
      self.assertIn("labels", result)
      self.assertEqual(result["prompt"], "What is this?")
      self.assertEqual(result["answer"], "A test")
      self.assertEqual(len(result["input_ids"]), 32)  # Matches max_length

  unittest.main()
