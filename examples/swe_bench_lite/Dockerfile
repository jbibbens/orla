# SWE-bench experiments image. Build from repo root:
#   docker build -f examples/swe_bench_lite/Dockerfile .                    # Lite (default)
#   docker build -f examples/swe_bench_lite/Dockerfile --build-arg FULL_SWE_BENCH=1 .   # Full
# Run with EXPERIMENT_MODE env var to select mode (baseline, stage_mapping, sjf).
# Dataset: dataset.zip (Lite) or dataset_full_*.zip (Full); unzipped at build time to /dataset.

ARG FULL_SWE_BENCH=0

FROM golang:1.25-alpine AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o /build/single_shot ./examples/swe_bench_lite/cmd/single_shot

FROM alpine:3.21
ARG FULL_SWE_BENCH=0
RUN apk add --no-cache ca-certificates bash curl git jq make patch py3-pip python3 unzip
COPY --from=builder /build/single_shot /usr/local/bin/single_shot

# Copy dataset zips (Lite: dataset.zip; Full: dataset_full_001.zip, dataset_full_002.zip, ...)
COPY examples/swe_bench_lite/dataset*.zip /tmp/

# Unzip the selected dataset based on FULL_SWE_BENCH
RUN mkdir -p /dataset && \
    if [ "$FULL_SWE_BENCH" = "1" ]; then \
        sh -c 'for z in /tmp/dataset_full_*.zip; do unzip -o "$z" -d /dataset; done'; \
    else \
        unzip -o /tmp/dataset.zip -d /dataset; \
    fi && \
    rm -f /tmp/dataset*.zip

COPY examples/swe_bench_lite/Makefile /Makefile
COPY examples/swe_bench_lite/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/usr/local/bin:$PATH"
ENV ORLA_URL=http://orla:8081
ENV WORKDIR_ROOT=/workdir
ENV OUTPUT_PATH=/output/predictions.jsonl
WORKDIR /

ENTRYPOINT ["/entrypoint.sh"]
