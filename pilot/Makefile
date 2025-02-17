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

