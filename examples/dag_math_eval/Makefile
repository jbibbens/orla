# DAG-Math memory evaluation experiments. In Docker the binary is at /usr/local/bin (on PATH); entrypoint runs "make <target>".
# Locally: EXPERIMENT_MODE=flush_per_workflow go run ./examples/dag_math_eval/cmd/run

dag_math_flush_per_request:
	EXPERIMENT_MODE=flush_per_request exec dag_math_eval

dag_math_flush_per_workflow:
	EXPERIMENT_MODE=flush_per_workflow exec dag_math_eval
