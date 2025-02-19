import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer

from data.hle import load_and_split_dataset


def apply_chat_template(example):
  messages = [{"role": "user", "content": example["question"]}, {"role": "assistant", "content": example["answer"]}]
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  return {"prompt": prompt}


def tokenize_function(example):
  tokens = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128)
  tokens["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in tokens["input_ids"]]
  return tokens


def compute_rewards(responses):
  return torch.tensor([1.0] * len(responses))


if __name__ == "__main__":
  cpu = True
  device_map = "cpu" if cpu else "auto"

  model_id = "meta-llama/Llama-3.2-1B-Instruct"
  print("Loading model and tokenizer...")
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map=device_map)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token

  print("Loading and splitting dataset...")
  train_data, _, test_data = load_and_split_dataset(test_size=0.05, val_size=0.1)

  print("Applying chat template and tokenizing training data...")
  train_data = train_data.map(apply_chat_template)
  train_data = train_data.map(tokenize_function)
  train_data = train_data.remove_columns(["question", "answer", "prompt"])

  print("Creating reference model...")
  ref_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map=device_map)

  ppo_config = PPOConfig(
    model_name=model_id,
    learning_rate=1e-5,
    log_with=None,
    batch_size=2,
    forward_batch_size=1,
  )

  print("Initializing PPOTrainer...")
  ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

  train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)

  num_epochs = 2
  print("Starting PPO fine-tuning...")
  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs} started.")
    for step, batch in enumerate(train_dataloader):
      input_ids = batch["input_ids"]
      attention_mask = batch["attention_mask"]

      responses = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
      rewards = compute_rewards(responses)

      stats = ppo_trainer.step(input_ids, responses, rewards)

      if step % 40 == 0:
        print(f"Epoch {epoch+1}, Step {step}: {stats}")
    print(f"Epoch {epoch+1} completed.")
  print("PPO fine-tuning finished. Saving model and tokenizer...")
  model.save_pretrained("./fine-tuned-model")
  tokenizer.save_pretrained("./fine-tuned-model")
  print("Model and tokenizer saved.")
