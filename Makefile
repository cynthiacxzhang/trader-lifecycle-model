.PHONY: ingest features train serve test

ingest:
	uv run python -m src.ingestion.portfolio_snapshots
	uv run python -m src.ingestion.market_data

features:
	uv run python -m src.features.pipeline

train:
	@echo "not implemented"

serve:
	@echo "not implemented"

test:
	uv run pytest tests/ -v
