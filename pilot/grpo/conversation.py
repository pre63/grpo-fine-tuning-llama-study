import logging
from typing import Dict, List, Optional

from PIL import Image as PIL_Image

from grpo.image_utils import decode_base64_image
from grpo.model import ensure_padding_token
from hle.prediction import get_system_prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_conversation(example: Dict, has_images: bool, include_system_prompt: bool = False) -> List[Dict]:
  user_text = example["question"]
  if has_images:
    user_text = "<|image|>\n" + user_text
  user_content = [{"type": "text", "text": user_text}]

  conversation = []
  if include_system_prompt:
    system_prompt = get_system_prompt(example)
    conversation.append({"role": "system", "content": system_prompt})

  conversation.extend(
    [
      {"role": "user", "content": user_content},
      {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
    ]
  )

  return conversation


def tokenize_example(example: Dict, processors, is_vision_model: Optional[bool] = None) -> Dict:
  image_value = example.get("image")
  image = decode_base64_image(image_value) if isinstance(image_value, str) and image_value.strip() else None
  has_images = is_vision_model and image is not None
  images = [image] if has_images else None

  conversation = build_conversation(example, has_images)
  tokenized = tokenize_prompt(processors, conversation, images, is_vision_model)
  tokenized["prompt"] = example["question"]
  tokenized["images"] = images  # Always include images key, None if no image
  return tokenized


def tokenize_prompt(processors, conversation: List[Dict], images: List[PIL_Image.Image], is_vision_model: Optional[bool] = None):
  text = processors["text"]

  prompt = text.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
  images_input = images if images and any(img is not None for img in images) else None

  if is_vision_model and images_input:
    vision = processors["vision"]
    assert len(images_input) > 0, "At least one image must be provided for vision model"
    assert type(vision).__name__ == "MllamaProcessor", f"Processor must be MllamaProcessor for vision model, got {type(vision).__name__}"

    tokens = vision(text=prompt, images=images_input, return_tensors="pt")
  else:
    assert type(text).__name__ == "PreTrainedTokenizerFast", f"Processor must be AutoTokenizer for text model, got {type(text).__name__}"

    tokens = text(prompt, return_tensors="pt")

  result = {
    "input_ids": tokens["input_ids"][0].tolist(),
    "attention_mask": tokens["attention_mask"][0].tolist() if "attention_mask" in tokens else None,
  }

  pad_token_id = text.pad_token_id if hasattr(text, "pad_token_id") else text.tokenizer.pad_token_id
  result["labels"] = [-100 if token == pad_token_id else token for token in result["input_ids"]]

  return result


if __name__ == "__main__":
  print("Running test for conversation.py")

  import base64
  import unittest

  from PIL import Image
  from transformers import AutoTokenizer, MllamaProcessor

  processors = {
    "text": ensure_padding_token(AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"), "meta-llama/Llama-3.2-1B-Instruct"),
    "vision": ensure_padding_token(MllamaProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct"), "meta-llama/Llama-3.2-11B-Vision-Instruct"),
  }

  class TestConversationIntegration(unittest.TestCase):
    def test_conversation_with_image_vision_model(self):
      """Test tokenization with an image for a vision model."""
      model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

      dummy_image = base64.b64encode(Image.new("RGB", (10, 10), "blue")._repr_png_()).decode("utf-8")
      example = {"question": "What is this?", "answer": "A test", "image": dummy_image}
      result = tokenize_example(example, processors, is_vision_model=True)

      self.assertIn("input_ids", result)
      self.assertIn("labels", result)
      self.assertIn("attention_mask", result)
      self.assertEqual(result["prompt"], "What is this?")
      self.assertIn("images", result)
      self.assertIsInstance(result["images"], list)
      self.assertIsInstance(result["images"][0], PIL_Image.Image)

    def test_conversation_no_image_vision_model(self):
      """Test tokenization without an image for a vision model."""
      model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
      example = {"question": "What is this?", "answer": "A test", "image": ""}  # No image
      result = tokenize_example(example, processors, is_vision_model=True)

      self.assertIn("input_ids", result)
      self.assertIn("labels", result)
      self.assertIn("attention_mask", result)
      self.assertEqual(result["prompt"], "What is this?")
      self.assertIn("images", result)
      self.assertIsNone(result["images"], "images should be None when no image")

    def test_conversation_empty_image_vision_model(self):
      """Test tokenization with an empty image string for a vision model."""
      model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

      example = {"question": "What is this?", "answer": "A test", "image": ""}  # Empty string
      result = tokenize_example(example, processors, is_vision_model=True)

      self.assertIn("input_ids", result)
      self.assertIn("labels", result)
      self.assertIn("attention_mask", result)
      self.assertEqual(result["prompt"], "What is this?")
      self.assertIn("images", result)
      self.assertIsNone(result["images"], "images should be None with empty image")

    def test_conversation_with_image_text_model(self):
      """Test tokenization with an image for a text-only model."""
      model_id = "meta-llama/Llama-3.2-1B-Instruct"

      dummy_image = base64.b64encode(Image.new("RGB", (10, 10), "blue")._repr_png_()).decode("utf-8")
      example = {"question": "What is this?", "answer": "A test", "image": dummy_image}
      result = tokenize_example(example, processors, is_vision_model=False)

      self.assertIn("input_ids", result)
      self.assertIn("labels", result)
      self.assertIn("attention_mask", result)
      self.assertEqual(result["prompt"], "What is this?")
      self.assertIn("images", result)
      self.assertIsNone(result["images"], "images should be None for text model with image")

    def test_conversation_no_image_text_model(self):
      """Test tokenization without an image for a text-only model."""
      model_id = "meta-llama/Llama-3.2-1B-Instruct"

      example = {"question": "What is this?", "answer": "A test", "image": ""}  # No image
      result = tokenize_example(example, processors, is_vision_model=False)

      self.assertIn("input_ids", result)
      self.assertIn("labels", result)
      self.assertIn("attention_mask", result)
      self.assertEqual(result["prompt"], "What is this?")
      self.assertIn("images", result)
      self.assertIsNone(result["images"], "images should be None for text model without image")

  unittest.main()
