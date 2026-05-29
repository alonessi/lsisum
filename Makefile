.PHONY: install install-parsers install-pipeline fetch pipeline reports test dashboard all clean

PY ?= python3

install: install-parsers install-pipeline

install-parsers:
	cd parsers && $(PY) -m pip install -r requirements.txt

install-pipeline:
	$(PY) -m pip install -e ".[models,dashboard,dev]"

# --- Data layer (parsers) -------------------------------------------------

fetch:
	cd parsers && $(PY) scripts/fetch_all.py
	mkdir -p data/raw
	cp parsers/data/processed/*.csv data/raw/

# --- Analytics layer (main pipeline) --------------------------------------

pipeline:
	$(PY) scripts/run_pipeline.py

reports:
	$(PY) scripts/eda.py
	$(PY) scripts/eval_metrics.py

test:
	$(PY) -m pytest -q

dashboard:
	$(PY) -m streamlit run dashboard/app.py

# --- Full end-to-end ------------------------------------------------------

all:
	$(PY) scripts/fetch_and_run.py

clean:
	rm -rf artifacts/*.parquet artifacts/*.csv artifacts/*.json
	rm -rf reports/figures/*.png reports/*.md
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
