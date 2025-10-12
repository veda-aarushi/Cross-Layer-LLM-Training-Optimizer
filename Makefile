PY=python

bench:
	$(PY) -u bench/bench_tokens.py --mode baseline
	$(PY) -u bench/bench_tokens.py --mode pinned
	$(PY) -u bench/bench_tokens.py --mode compile

plot:
	$(PY) -u utils/plot_results.py

test:
	$(PY) -m pytest -q

all: bench plot
