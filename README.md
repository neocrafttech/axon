# axon : A simple Vector Database
Axon is a lightweight and efficient vector database engine designed for similarity search of embeddings.
It supports multiple distance metrics (L2, Cosine, Dot), hybrid CPU/GPU execution, and flexible storage backends.

---

## References

- *“DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node”* (NeurIPS 2019)
  https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf

- *“FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search”* (arXiv 2021)
  https://arxiv.org/abs/2105.09613

These papers influenced Axon’s design, particularly in indexing structures, graph search, and efficient query traversal.

---

## Quick Start

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (latest stable)
- [Protobuf Compiler](https://grpc.io/docs/protoc-installation/) (for gRPC)
- [Docker](https://docs.docker.com/engine/install/) (optional, for containerized run)

### Running the gRPC Service (vecd)

#### Using Cargo
```bash
# Start the VectorDB service
cargo run -p vecd
```

#### Using Docker
```bash
# Build the image
docker build -t vecd .

# Run the container
docker run -p 50051:50051 vecd
```

### Running Examples

#### SIFT 10k gRPC Example
This example demonstrates connecting to the running `vecd` service, inserting the SIFT 10k dataset, and calculating recall.

1. Ensure `vecd` is running (see above).
2. Download the SIFT 10k dataset into `examples/data/siftsmall`:(ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz)
   *(Note: The example expects `siftsmall_base.fvecs`, `siftsmall_query.fvecs`, and `siftsmall_groundtruth.ivecs`)*
3. Run the example:
   ```bash
   cargo run --bin sift10k_index_grpc --release
   ```
