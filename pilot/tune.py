import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data.hle import load_and_split_dataset


# Define a function to apply the chat template
def apply_chat_template(example):
  messages = [{"role": "user", "content": example["question"]}, {"role": "assistant", "content": example["answer"]}]
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  return {"prompt": prompt}


# Tokenize the data
def tokenize_function(example):
  tokens = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128)
  # Set padding token labels to -100 to ignore them in loss calculation
  tokens["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in tokens["input_ids"]]
  return tokens


if __name__ == "__main__":

  # Load the base model and tokenizer
  model_id = "meta-llama/Llama-3.2-1B-Instruct"
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto")  # Must be float32 for MacBooks!
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token

  train_data, val_data, test_data = load_and_split_dataset()

  # Apply the chat template function to the dataset
  train_data = train_data.map(apply_chat_template)
  train_data = train_data.map(tokenize_function)

  test_data = test_data.map(apply_chat_template)
  test_data = test_data.map(tokenize_function)

  # Apply tokenize_function to each row
  tokenized_dataset = test_data.map(tokenize_function)
  tokenized_dataset = tokenized_dataset.remove_columns(["question", "answer", "prompt"])

  # Define training arguments
  model.train()
  training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",  # To evaluate during training
    eval_steps=40,
    logging_steps=40,
    save_steps=150,
    per_device_train_batch_size=2,  # Adjust based on your hardware
    per_device_eval_batch_size=2,
    num_train_epochs=2,  # How many times to loop through the dataset
    fp16=False,  # Must be False for MacBooks
    report_to="none",  # Here we can use something like tensorboard to see the training metrics
    log_level="info",
    learning_rate=1e-5,  # Would avoid larger values here
    max_grad_norm=2,  # Clipping the gradients is always a good idea
    use_cpu=True,  # Must be True for MacBooks
  )

  # Initialize Trainer
  trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=test_data, tokenizer=tokenizer)

  # Train the model
  trainer.train()

  # Save the model and tokenizer
  trainer.save_model("./fine-tuned-model")
  tokenizer.save_pretrained("./fine-tuned-model")
