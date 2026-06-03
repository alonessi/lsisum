.PHONY: install fetch init update dashboard test clean

PY ?= py

install:
	$(PY) -m pip install -r requirements.txt

fetch:
	cd parsers && $(PY) scripts/fetch_all.py
	if not exist data\raw mkdir data\raw
	copy parsers\data\processed\*.csv data\raw\

init:
	$(PY) scripts/incremental_nsvm.py --init

update:
	$(PY) scripts/incremental_nsvm.py

dashboard:
	$(PY) -m streamlit run dashboard/app.py

test:
	$(PY) -m pytest -q

clean:
	del /Q artifacts\*.parquet artifacts\*.csv artifacts\*.json artifacts\*.pkl artifacts\*.pt 2>NUL
	for /R . %%d in (__pycache__) do @if exist "%%d" rmdir /S /Q "%%d"
	for /R . %%f in (*.pyc) do @del /Q "%%f"
