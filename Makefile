PYTHON = python
PIP = pip

.PHONY: setup data train clean test help

help:
	@echo "Available commands:"
	@echo "  make setup   - Install dependencies"
	@echo "  make data    - Download and process data"
	@echo "  make train   - Train both models"
	@echo "  make test    - Run tests"
	@echo "  make clean   - Remove data and models"

setup:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) -m src.data.process

train:
	$(PYTHON) -m src.models.train_cf
	$(PYTHON) -m src.models.train_bandit

test:
	$(PYTHON) -m unittest discover tests

clean:
	rm -rf data/raw/* data/interim/* data/processed/* models/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
