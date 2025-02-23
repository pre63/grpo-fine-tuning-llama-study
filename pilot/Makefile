.PHONY: install test fix tune eval

# Use CPU instead of GPU
export CPU=True

# Model to use
export MODEL=meta-llama/Llama-3.2-1B-Instruct

# Resume from a previous run
export RESUME=True

# WandB 
export WANDB_SILENT=true
export WANDB_API_KEY=4b28b7410bc92bce660b446e56bd56f33dca3e44

# Hugging Face
export HF_TOKEN=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
export TOKENIZERS_PARALLELISM=false



test:
	@python -m grpo.eval
	@python -m grpo.trainer
	@python -m grpo.conversation
	@python -m grpo.model
	@python -m grpo.reward
	@python -m grpo.hardware
	@python -m grpo.config
	@python -m grpo.image_utils

env:
	@python3 -m venv ../.venv

colab:
	@pip install triton

	# Uninstall conflicting packages
	@pip uninstall -y tensorflow torch torch-xla accelerate trl transformers

	# Install compatible versions
	@pip install torch==2.4.0 torch-xla==2.4 accelerate==0.34.0 trl==0.15.1 transformers==4.46.0 tensorflow-cpu

	# Restart runtime (manual step in Colab), then test
	@python -c "from trl import GRPOTrainer; print('Success')"


install:
	@pip install -U -r requirements.txt

fix:
	@isort . && black .

rl: tune

tune:
	@$(MAKE) fix
	@python tune.py

eval:
	@$(MAKE) fix
	@python eval.py

local:
	$(MAKE) test CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) tune CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"

cuda:
	$(MAKE) test CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) test CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) test CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

	$(MAKE) tune CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) tune CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

	$(MAKE) eval CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) eval CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
