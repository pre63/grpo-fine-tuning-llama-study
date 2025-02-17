import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.models.llama3_2 import llama3_2_1b
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from torchtune.training.quantization import Int4WeightOnlyQATQuantizer

from data.hle import get_train_loader, load_and_split_dataset, tokenize_batch
from network import GRPONetwork, compute_loss

# Hyperparameters

batch_size = 2
num_epochs = 3
learning_rate = 1e-4
max_seq_len = 131072
shuffle_data = True

lora_ran = 8
lora_alpha = 16
lora_dropout = 0.0
target_modules = ["q_proj", "v_proj"]

tokenizer_path = ".model/Llama-3.2-1B/tokenizer.model"
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
  print("Loading dataset...")
  train_data, val_data, test_data = load_and_split_dataset()
  tokenizer = llama3_tokenizer(path=tokenizer_path)
  train_data = train_data.map(tokenize_batch(tokenizer), batched=True)
  train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
  train_loader = get_train_loader(train_data, tokenizer, batch_size=batch_size, shuffle=shuffle_data, max_seq_len=max_seq_len)
  print("Dataset loaded.")

  print("Preparing base model...")
  base_model = llama3_2_1b().to(device)

  quantizer = Int4WeightOnlyQATQuantizer(groupsize=256, inner_k_tiles=8, precision=torch.bfloat16, scales_precision=torch.bfloat16)
  base_model = quantizer.prepare(base_model)

  if hasattr(base_model, "gradient_checkpointing_enable"):
    base_model.gradient_checkpointing_enable()
  print("Base model ready.")

  print("Injecting LoRA adapters...")
  lora_config = LoraConfig(r=lora_ran, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none")
  qlora_model = get_peft_model(base_model, lora_config)
  set_trainable_params(qlora_model, get_adapter_params(qlora_model))
  print("LoRA adapters injected.")

  print("Wrapping model for training...")
  model = GRPONetwork(qlora_model, device=device)

  adapter_params = [p for n, p in model.named_parameters() if p.requires_grad]
  if not adapter_params:
    for name, p in model.named_parameters():
      if "lora" in name:
        p.requires_grad = True
        adapter_params.append(p)
  if not adapter_params:
    raise ValueError("No trainable parameters found! Check target_modules in LoraConfig.")
  print(f"Number of trainable parameters: {sum(p.numel() for p in adapter_params)}")

  optimizer = optim.Adam(adapter_params, lr=learning_rate)
  scaler = torch.amp.GradScaler(device=device)

  print("Starting training...")
  model.train()

  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0.0
    batch_count = 0

    for batch in train_loader:

      input_ids = batch["input_ids"].to(device)
      optimizer.zero_grad()

      with torch.autocast(device_type=device):
        logits = model(input_ids)
        targets = input_ids[:, 1:]
        pred_logits = logits[:, :-1, :]
        loss = compute_loss(pred_logits, targets)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      epoch_loss += loss.item()
      batch_count += 1
    avg_loss = epoch_loss / batch_count if batch_count > 0 else float("inf")
    print(f"Epoch {epoch+1}/{num_epochs}: Average Loss: {avg_loss:.4f}")
  print("Training complete.")

  print("Converting model to quantized version...")
  quantized_base = quantizer.convert(base_model)
  print("Quantization complete.")
