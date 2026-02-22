# Build stage
FROM golang:1.25-alpine AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
ARG VERSION=dev
ARG BUILD_DATE
RUN CGO_ENABLED=0 go build -ldflags "-X main.version=${VERSION} -X main.buildDate=${BUILD_DATE:-unknown}" \
	-o /orla ./cmd/orla

# Runtime stage
FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=builder /orla /usr/local/bin/orla

# Default: run the Orla server. Mount a config at /config/orla.yaml (see deploy/*.yaml) or override CMD with --config /path/to/orla.yaml.
ENTRYPOINT ["orla"]
CMD ["serve", "--config", "/config/orla.yaml"]
