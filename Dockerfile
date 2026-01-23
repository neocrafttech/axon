FROM rust:1.92-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libssl-dev \
    pkg-config \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/nyas

COPY . .

RUN chmod +x bolt.sh && ./bolt.sh build

FROM debian:bookworm-slim

# Install runtime dependencies if needed
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/bin

COPY --from=builder /usr/src/nyas/target/release/vecd .

EXPOSE 50051

CMD ["./vecd"]
