.PHONY: tune install test fix rl eval

CPU = True

MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
RESUME=True

tune:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python tune.py

install:
	@python3 -m venv ../.venv
	@. ../.venv/bin/activate && pip install -U -r requirements.txt

test:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python network.py
	@. ../.venv/bin/activate && python -m grpo.image_utils
	@. ../.venv/bin/activate && python -m grpo.conversation
	@. ../.venv/bin/activate && python -m grpo.reward
	@. ../.venv/bin/activate && python -m grpo.hardware
	@. ../.venv/bin/activate && python -m grpo.model
	@. ../.venv/bin/activate && python -m grpo.config

fix:
	@. ../.venv/bin/activate && isort . && black .

rl:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python rl.py --cpu $(CPU) --model $(MODEL) 

eval:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python eval.py --model $(MODEL) --cpu $(CPU) --resume $(RESUME)