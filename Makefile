.PHONY: ingest features train serve test

ingest:
	@echo "not implemented"

features:
	@echo "not implemented"

train:
	@echo "not implemented"

serve:
	@echo "not implemented"

test:
	uv run pytest tests/ -v
