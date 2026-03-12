# DAG-Math memory evaluation image. Build from repo root:
#   docker build -f examples/dag_math_eval/Dockerfile .
# Run with EXPERIMENT_MODE env var to select mode (flush_per_request, flush_per_workflow).
# Dataset: dataset.zip; unzipped at build time to /dataset.

FROM golang:1.25-alpine AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o /build/dag_math_eval ./examples/dag_math_eval/cmd/run

FROM alpine:3.21
RUN apk add --no-cache ca-certificates bash curl make
COPY --from=builder /build/dag_math_eval /usr/local/bin/dag_math_eval

COPY examples/dag_math_eval/dataset.zip /tmp/dataset.zip
RUN mkdir -p /dataset && \
    unzip -o /tmp/dataset.zip -d /dataset && \
    rm -f /tmp/dataset.zip

COPY examples/dag_math_eval/Makefile /Makefile
COPY examples/dag_math_eval/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/usr/local/bin:$PATH"
ENV ORLA_URL=http://orla:8081
ENV OUTPUT_PATH=/output/metrics.json
ENV METRICS_PATH=/output/metrics.json
WORKDIR /

ENTRYPOINT ["/entrypoint.sh"]
