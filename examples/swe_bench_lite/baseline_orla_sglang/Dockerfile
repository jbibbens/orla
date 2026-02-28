# Build from repo root so we have go.mod and pkg/api:
#   docker build -f examples/swe_bench_lite/baseline_orla_sglang/Dockerfile .

FROM golang:1.25-alpine AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o /baseline ./examples/swe_bench_lite/baseline_orla_sglang

FROM alpine:3.21
RUN apk add --no-cache ca-certificates bash git jq
COPY --from=builder /baseline /usr/local/bin/baseline
# When building with -f examples/swe_bench_lite/baseline_orla_sglang/Dockerfile from repo root:
COPY examples/swe_bench_lite/baseline_orla_sglang/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ORLA_URL=http://orla:8081
ENV WORKDIR_ROOT=/workdir
ENV OUTPUT_PATH=/output/predictions.jsonl

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/instances/instance.json"]
