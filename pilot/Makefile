.PHONY: tune install test fix rl eval

CPU = True
MODEL = None

tune:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python tune.py

install:
	@python3 -m venv ../.venv
	@. ../.venv/bin/activate && pip install -U -r requirements.txt

test:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python network.py

fix:
	@. ../.venv/bin/activate && isort . && black .

rl:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python rl.py --cpu $(CPU)

eval:
	@$(MAKE) fix
	@. ../.venv/bin/activate && python eval.py --model $(MODEL) --cpu $(CPU)