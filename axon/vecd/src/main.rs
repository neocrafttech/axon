use diskann::index_view::IndexView;
use mimalloc::MiMalloc;
use service::{VectorService, vector};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Initialize IndexView with index name "vector_db"
    let index_view = IndexView::new("vector_db").await?;
    let svc = VectorService::new(index_view);

    let addr = "0.0.0.0:50051".parse()?;
    println!("VectorDB gRPC server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(vector::vector_db_server::VectorDbServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
