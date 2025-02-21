from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


def load_and_split_dataset(test_size=0.3, tokenize_example=None, processors=None, is_vision_model=False):
  """
    This function loads the dataset and splits it into training, validation, and testing sets.
    """
  data = load_dataset("cais/hle")

  data = data["test"].train_test_split(test_size=test_size)

  train_data = data["train"]
  test_data = data["test"]

  if processors is None or tokenize_example is None:
    return train_data, test_data

  train_data = train_data.map(lambda x: tokenize_example(x, processors, is_vision_model)).remove_columns(["question", "image"])
  test_data = test_data.map(lambda x: tokenize_example(x, processors, is_vision_model)).remove_columns(["question", "image"])

  return train_data, test_data
