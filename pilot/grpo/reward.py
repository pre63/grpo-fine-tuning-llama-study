from typing import List, Optional

from sentence_transformers import SentenceTransformer, util

try:
  embedder = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError as e:
  print(f"Failed to load SentenceTransformer: {e}")
  embedder = None


def compute_reward(prompts: List[str], completions: List[str], ground_truths: List[Optional[str]]) -> List[float]:
  if embedder is None:
    raise RuntimeError("SentenceTransformer not available")
  # Ensure ground_truths matches completions length, filling with None if shorter
  ground_truths = (ground_truths or []) + [None] * (len(completions) - len(ground_truths or []))
  rewards = []
  for prompt, completion, truth in zip(prompts, completions, ground_truths):
    if not truth:
      reward = 0.5
    else:
      # Compare completion directly to ground truth for higher similarity
      reward = (util.cos_sim(embedder.encode(completion, convert_to_tensor=True), embedder.encode(truth, convert_to_tensor=True)).item() + 1) / 2
    rewards.append(reward)
  return rewards


if __name__ == "__main__":
  import unittest

  class TestRewardIntegration(unittest.TestCase):
    def test_reward_with_full_pipeline(self):
      # Simulate a pipeline with prompts and completions
      prompts = ["What is 2+2?"] * 2
      completions = ["4", "5"]
      ground_truths = ["4"]
      rewards = compute_reward(prompts, completions, ground_truths)
      self.assertEqual(len(rewards), 2)
      self.assertGreater(rewards[0], rewards[1])  # "4" should score higher
      self.assertGreaterEqual(rewards[0], 0.95)  # Expect high similarity, not exactly 1.0
      # Second completion gets a default or lower score
      self.assertTrue(0.5 <= rewards[1] <= 1.0)

  unittest.main()
