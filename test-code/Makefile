.PHONY: install setup clean

install:
	pip install -r requirements.txt

setup:
	mkdir -p data/raw data/processed data/embeddings
	mkdir -p features models notebooks api

lint:
	flake8 . --count --exit-zero

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build dist *.egg-info
