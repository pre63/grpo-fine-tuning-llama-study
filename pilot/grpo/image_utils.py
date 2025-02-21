import base64
import io
from typing import Optional

from PIL import Image as PIL_Image


def decode_base64_image(image_str: str) -> Optional[PIL_Image.Image]:
  try:
    if "," in image_str:
      image_str = image_str.split(",", 1)[1]
    image_data = base64.b64decode(image_str)
    return PIL_Image.open(io.BytesIO(image_data)).convert("RGB")
  except Exception as e:
    print(f"Error decoding image: {e}")
    return None


if __name__ == "__main__":
  print("Running test for image_utils.py")

  import unittest

  from PIL import Image

  class TestImageUtilsIntegration(unittest.TestCase):
    def test_image_decode_and_reencode(self):
      # Create a test image, encode it, and decode it back
      img = Image.new("RGB", (10, 10), color="red")
      img_byte_arr = io.BytesIO()
      img.save(img_byte_arr, format="PNG")
      base64_str = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
      decoded_img = decode_base64_image(base64_str)
      self.assertIsInstance(decoded_img, Image.Image)
      self.assertEqual(decoded_img.size, (10, 10))
      # Reencode and check if it’s still valid
      reencoded_bytes = io.BytesIO()
      decoded_img.save(reencoded_bytes, format="PNG")
      self.assertGreater(reencoded_bytes.tell(), 0)

  unittest.main()
