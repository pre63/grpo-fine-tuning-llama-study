.PHONY: install test fix tune eval audit judge local cuda env py pym

# Environment Variables
CPU ?= True
MODEL ?= meta-llama/Llama-3.2-1B-Instruct
RESUME ?= True
WANDB_SILENT ?= true
WANDB_API_KEY ?= 4b28b7410bc92bce660b446e56bd56f33dca3e44
HF_TOKEN ?= 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
TOKENIZERS_PARALLELISM ?= false
OUTPUT_FILENAME ?= new_merged_predictions.json
JUDGED_FILENAME ?= new_judged_questions.json

# Create virtual environment
env:
	@python3 -m venv ../.venv

# Install dependencies
install:
	@. ../.venv/bin/activate && pip install -U -r requirements.txt

# Format code
fix:
	@. ../.venv/bin/activate && isort . && black .

# Run a Python script (e.g., make py F=todo.py)
py:
	@. ../.venv/bin/activate && \
	CPU=$(CPU) MODEL=$(MODEL) RESUME=$(RESUME) WANDB_SILENT=$(WANDB_SILENT) WANDB_API_KEY=$(WANDB_API_KEY) HF_TOKEN=$(HF_TOKEN) TOKENIZERS_PARALLELISM=$(TOKENIZERS_PARALLELISM) OUTPUT_FILENAME=$(OUTPUT_FILENAME) JUDGED_FILENAME=$(JUDGED_FILENAME) \
	python $(F)

# Run a Python module (e.g., make pym M=grpo.eval)
pym:
	@. ../.venv/bin/activate && \
	CPU=$(CPU) MODEL=$(MODEL) RESUME=$(RESUME) WANDB_SILENT=$(WANDB_SILENT) WANDB_API_KEY=$(WANDB_API_KEY) HF_TOKEN=$(HF_TOKEN) TOKENIZERS_PARALLELISM=$(TOKENIZERS_PARALLELISM) OUTPUT_FILENAME=$(OUTPUT_FILENAME) JUDGED_FILENAME=$(JUDGED_FILENAME) \
	python -m $(M)

# Test various modules
test:
	@$(MAKE) pym M="grpo.audit"
	@$(MAKE) pym M="grpo.eval"
	@$(MAKE) pym M="grpo.conversation"
	@$(MAKE) pym M="grpo.model"
	@$(MAKE) pym M="grpo.reward"
	@$(MAKE) pym M="grpo.hardware"
	@$(MAKE) pym M="grpo.config"
	@$(MAKE) pym M="grpo.image_utils"
	@$(MAKE) pym M="grpo.trainer"

# Specific tasks
audit:
	@$(MAKE) fix
	@$(MAKE) py F="audit.py"

tune:
	@$(MAKE) fix
	@$(MAKE) py F="tune.py"

eval:
	@$(MAKE) fix
	@$(MAKE) py F="eval.py"

judge:
	@$(MAKE) fix
	@$(MAKE) py F="eval.py JUDGE=1"

# Local and CUDA configurations
local:
	@$(MAKE) test CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"
	@$(MAKE) tune CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"

cuda:
	@$(MAKE) test CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"
	@$(MAKE) test CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	@$(MAKE) test CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
	@$(MAKE) tune CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	@$(MAKE) tune CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
	@$(MAKE) eval CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	@$(MAKE) eval CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"