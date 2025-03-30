# Használj egy megfelelő toolchain-t és statikusan linkelj
FROM --platform=linux/arm/v7 rust AS builder
#FROM rust AS builder
WORKDIR /app

# Telepítsd a musl és a glibc kompatibilitást
RUN apt-get update && apt-get install -y musl-dev musl-tools

# Másold a Cargo fájlokat
COPY Cargo.toml Cargo.lock ./

# Cache build: először csak a függőségeket fordítjuk
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo install --path .

# Másold be a teljes forráskódot és fordítsd le statikusan
COPY src ./src/
COPY model.onnx ./model.onnx

RUN cargo install --path .

# Minimális futtatókörnyezet
FROM --platform=linux/arm/v7 debian:bookworm-slim
#FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libc6 && rm -rf /var/lib/apt/lists/*

# Másoljuk át a statikusan linkelt binárist
COPY --from=builder /usr/local/cargo/bin/tract_minimal /usr/local/bin/
COPY --from=builder /app/model.onnx /usr/local/bin/
COPY --from=builder /app/model.onnx /app/model.onnx

#start the program
CMD ["tract_minimal", "model.onnx"]