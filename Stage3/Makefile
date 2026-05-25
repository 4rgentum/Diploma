PY ?= python

.PHONY: install install-dev test smoke fmt lint clean download preprocess train evaluate attacker drift demo experiments reproduce

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[service,ui,dev]"

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/02_build_dataset.py --config configs/data/unsw_nb15.yaml --smoke
	$(PY) scripts/03_train.py --model configs/models/cnn_lstm.yaml --train configs/train/smoke.yaml --seed 42

download:
	$(PY) scripts/01_download_unsw.py

preprocess:
	$(PY) scripts/02_build_dataset.py --config configs/data/unsw_nb15.yaml

train:
	$(PY) scripts/03_train.py --model configs/models/cnn_lstm.yaml --train configs/train/full.yaml --seed 42

evaluate:
	$(PY) scripts/04_evaluate.py --model configs/models/cnn_lstm.yaml --checkpoint models/cnn_lstm_seed42.pt

attacker:
	$(PY) scripts/05_run_attacker.py --config configs/attacker/policy.yaml --ticks 600

drift:
	$(PY) scripts/06_drift_test.py --config configs/attacker/drift.yaml

demo:
	$(PY) scripts/07_demo_realtime.py

experiments: experiment-E1 experiment-E2 experiment-E3 experiment-E4 experiment-E5 experiment-E6

experiment-E1:
	$(PY) scripts/10_run_experiment_E1.py

experiment-E2:
	$(PY) scripts/11_run_experiment_E2_error_analysis.py

experiment-E3:
	$(PY) scripts/12_run_experiment_E3_calibration.py

experiment-E4:
	$(PY) scripts/13_run_experiment_E4_attacker.py

experiment-E5:
	$(PY) scripts/14_run_experiment_E5_drift.py

experiment-E6:
	$(PY) scripts/15_run_experiment_E6_perf.py

reproduce: preprocess train evaluate experiments

fmt:
	$(PY) -m ruff format src tests scripts

lint:
	$(PY) -m ruff check src tests scripts

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache __pycache__ src/**/__pycache__ tests/__pycache__
