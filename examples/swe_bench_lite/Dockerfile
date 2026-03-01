# SWE-bench Lite experiments image. Build from repo root:
#   docker build -f examples/swe_bench_lite/Dockerfile .
# Run with make target as command, e.g.: docker run ... baseline
# Populate examples/swe_bench_lite/dataset first (fetch_dataset.py) so the image has instances.

FROM golang:1.25-alpine AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
# Build each experiment binary; add more lines as experiments are added.
RUN CGO_ENABLED=0 go build -o /build/baseline ./examples/swe_bench_lite/cmd/baseline

FROM alpine:3.21
RUN apk add --no-cache ca-certificates bash git make
COPY --from=builder /build/baseline /usr/local/bin/baseline
COPY examples/swe_bench_lite/dataset /dataset
COPY examples/swe_bench_lite/Makefile /Makefile
COPY examples/swe_bench_lite/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ORLA_URL=http://orla:8081
ENV WORKDIR_ROOT=/workdir
ENV OUTPUT_PATH=/output/predictions.jsonl
WORKDIR /

ENTRYPOINT ["/entrypoint.sh"]
CMD ["baseline"]
