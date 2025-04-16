.PHONY: init index up

init:
	@echo "Initializing environment..."
	@bash setup.sh
	@echo "Note: Environment variables might not be available in this shell session."
	@echo "To ensure environment variables are set, run 'source setup.sh' directly."

index:
	@echo "Running data loader..."
	python3 src/scripts/load_data.py --upsert

up:
	@echo "Starting server..."
	uvicorn src.app:app --host "0.0.0.0" --port 5000