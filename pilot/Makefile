.PHONY: install test fix rl eval

# Use CPU instead of GPU
CPU = True

# Model to use
MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

# Resume from a previous run
RESUME=True

test:
	@$(MAKE) fix
	@python -m grpo.conversation
	@python -m grpo.model
	@python -m grpo.trainer
	@if [ "$(CPU)" = "False" ]; then \
		python -m grpo.eval; \
	fi
	@python -m grpo.reward
	@python -m grpo.hardware
	@python -m grpo.config
	@python -m grpo.image_utils

install:
	@python3 -m venv .venv
	@pip install -U -r requirements.txt

fix:
	@isort . && black .

rl:
	@$(MAKE) fix
	@python rl.py

eval:
	@$(MAKE) fix
	@python eval.py

local:
	$(MAKE) test CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) rl CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"

cuda:
	$(MAKE) test CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) test CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) test CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

	$(MAKE) rl CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) rl CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

	$(MAKE) eval CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) eval CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
