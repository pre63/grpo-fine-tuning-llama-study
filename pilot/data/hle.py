import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtune.data import Message

SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


def get_train_loader(train_data, tokenizer, batch_size=8, shuffle=True, max_seq_len=512):
  """
    This function creates a DataLoader for the training data.
    """
  return DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: custom_collate_fn(batch, max_seq_len, tokenizer))


def load_and_split_dataset(test_size=0.3, val_size=0.1):
  """
    This function loads the dataset and splits it into training, validation, and testing sets.
    """
  data = load_dataset("cais/hle")

  # Split the "test" split into training and testing
  data = data["test"].train_test_split(test_size=test_size)

  # Further split the training portion into train and validation
  train_val = data["test"].train_test_split(test_size=val_size)

  val_data = train_val["test"]
  train_data = train_val["train"]
  test_data = data["test"]

  return train_data, val_data, test_data


def tokenize_function(example, tokenizer):
  """
    This function tokenizes a single example into a format that the model can understand.
    """
  msg = Message(role="user", text=example["question"])
  tokens, mask = tokenizer.tokenize_messages([msg])
  return {"input_ids": tokens, "attention_mask": mask}


def custom_collate_fn(batch, max_seq_len=2, tokenizer=None):
  input_ids_list = [item["input_ids"].clone().detach() if isinstance(item["input_ids"], torch.Tensor) else torch.tensor(item["input_ids"]) for item in batch]
  attention_mask_list = [
    item["attention_mask"].clone().detach() if isinstance(item["attention_mask"], torch.Tensor) else torch.tensor(item["attention_mask"]) for item in batch
  ]
  pad_token_id = tokenizer.pad_token_id if tokenizer is not None and hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
  input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
  attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
  if input_ids_padded.size(1) > max_seq_len:
    input_ids_padded = input_ids_padded[:, :max_seq_len]
    attention_mask_padded = attention_mask_padded[:, :max_seq_len]
  return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded}

  return inner_collate_fn


def format_message(question):
  """
    This function formats a question into a list of messages that the model can understand.
    """
  answer_type = question["answer_type"]

  system_prompt = SYSTEM_EXACT_ANSWER if answer_type == "exact_match" else SYSTEM_MC

  question_text = question["question"]

  system_role = "system"

  messages = [Message(role=system_role, content=system_prompt), Message(role="user", content=question_text)]

  if question["image"]:
    messages.append(Message(role="user", content=question["image"]))

  return messages


def tokenize_batch(tokenizer):
  """
    This function tokenizes a batch of examples into a format that the model can understand.
    """

  def inner_func(examples):
    batch_input_ids = []
    batch_attention_mask = []

    for q, answer_type, image in zip(examples["question"], examples["answer_type"], examples["image"]):
      example = {"question": q, "answer_type": answer_type, "image": image}
      messages = format_message(example)
      tokens, mask = tokenizer.tokenize_messages(messages)
      batch_input_ids.append(tokens)
      batch_attention_mask.append(mask)

    return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

  return inner_func
