use std::time::{Duration, SystemTime};

use cpu_time::ProcessTime;
use service::vector::vector_db_client::VectorDbClient;
use service::vector::{InsertVectorRequest, SearchVectorRequest};
use tokio::io;
use tonic::Request;

use crate::index_utils::SiftDataset;
mod index_utils;

#[tokio::main]
async fn main() -> io::Result<()> {
    let base_folder = "examples/data/siftsmall";
    let start_cpu = ProcessTime::now();
    let start_wall = SystemTime::now();

    let (base, query, ground_truth) = tokio::try_join!(
        SiftDataset::from_fvecs(format!("{}/siftsmall_base.fvecs", base_folder)),
        SiftDataset::from_fvecs(format!("{}/siftsmall_query.fvecs", base_folder)),
        SiftDataset::from_ivecs(format!("{}/siftsmall_groundtruth.ivecs", base_folder))
    )?;

    println!("Base dataset: {} vectors of dimension {}", base.vectors.len(), base.dimension);
    println!("Query dataset: {} vectors of dimension {}", query.vectors.len(), query.dimension);
    println!("Ground truth: {} queries", ground_truth.len());

    let mut client = VectorDbClient::connect("http://0.0.0.0:50051")
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))?;

    // Since we are connecting via gRPC, we don't check for local file existence here
    // as easily. For simplicity in this example, we'll try to insert if not found
    // or just always try to insert (server can handle duplicates if implemented).
    // However, the original code had:
    // let path = Path::new(index_name);
    // if !path.exists() { ... }

    // We'll skip the check for now as the server manages the index.
    // If we want to avoid re-inserting, we'd need a way to check if index is built on server.

    println!("Inserting vectors...");
    for (index, vector) in base.vectors.iter().enumerate() {
        let request = Request::new(InsertVectorRequest {
            id: index.to_string(),
            vector: vector.to_f32_vec(),
        });

        let _response = client.insert_vector(request).await.map_err(io::Error::other)?;

        if index % 1000 == 0 && index > 0 {
            println!("Inserted {} vectors", index);
        }
    }

    let index_cpu_time = start_cpu.elapsed();
    let index_wall_time = start_wall.elapsed().unwrap();
    println!("Indexing time: CPU {:?}, Wall {:?}", index_cpu_time, index_wall_time);

    for k in [1, 10, 100] {
        let mut results = Vec::new();
        for q in query.vectors.iter() {
            let request =
                Request::new(SearchVectorRequest { vector: q.to_f32_vec(), top_k: k as u32 });
            let response = client.search_vector(request).await.map_err(io::Error::other)?;
            let search_res = response.into_inner();

            // Map string IDs back to u32
            let u32_ids: Vec<u32> =
                search_res.ids.iter().filter_map(|id: &String| id.parse::<u32>().ok()).collect();
            results.push(u32_ids);
        }
        let recall = index_utils::compute_recall(&results, &ground_truth, k);
        println!("Recall for k={}: {}", k, recall);
    }

    let cpu_time: Duration = start_cpu.elapsed();
    let wall_time = start_wall.elapsed().unwrap();

    println!("Total CPU time: {:?}", cpu_time);
    println!("Total Wall time: {:?}", wall_time);

    Ok(())
}
