.PHONY: install test fix rl eval

# Use CPU instead of GPU
CPU = True

# Model to use
MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

# Resume from a previous run
RESUME=True

test:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python -m grpo.conversation
	@. ../.venv/bin/activate && python -m grpo.model
	@. ../.venv/bin/activate && python -m grpo.trainer
	@if [ "$(CPU)" = "False" ]; then \
		. ../.venv/bin/activate && python -m grpo.eval; \
	fi
	@. ../.venv/bin/activate && python -m grpo.reward
	@. ../.venv/bin/activate && python -m grpo.hardware
	@. ../.venv/bin/activate && python -m grpo.config
	@. ../.venv/bin/activate && python -m grpo.image_utils

install:
	@python3 -m venv ../.venv
	@. ../.venv/bin/activate && pip install -U -r requirements.txt

fix:
	@. ../.venv/bin/activate && isort . && black .

rl:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python rl.py

eval:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python eval.py

local:
	@$(MAKE) fix
	$(MAKE) test CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) rl CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"

cuda:
	@$(MAKE) fix

	$(MAKE) test CPU=True MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) test CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) test CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

	$(MAKE) rl CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) rl CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

	$(MAKE) eval CPU=False MODEL="meta-llama/Llama-3.2-1B-Instruct"
	$(MAKE) eval CPU=False MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
