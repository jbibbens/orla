# SWE-bench Lite experiments.
# From repo root: make -C examples/swe_bench_lite baseline
# From this dir:  make baseline
# In Docker, entrypoint runs "make <target>" (e.g. make baseline).
ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || echo "../..")

baseline:
	go build -o baseline ./examples/swe_bench_lite/cmd/baseline
	exec ./baseline