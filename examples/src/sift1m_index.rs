use std::path::Path;
use std::time::{Duration, SystemTime};

use cpu_time::ProcessTime;
use diskann::index_view::IndexView;
use futures::StreamExt;
use system::vector_point::VectorPoint;
use tokio::io;

use crate::index_utils::SiftDataset;
mod index_utils;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse::<usize>().ok().filter(|v| *v > 0)
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let base_folder = "examples/data/sift";
    let start_cpu = ProcessTime::now();
    let start_wall = SystemTime::now();

    let (base, query, ground_truth) = tokio::try_join!(
        SiftDataset::from_fvecs(format!("{}/sift_base.fvecs", base_folder)),
        SiftDataset::from_fvecs(format!("{}/sift_query.fvecs", base_folder)),
        SiftDataset::from_ivecs(format!("{}/sift_groundtruth.ivecs", base_folder))
    )?;

    println!("Base dataset: {} vectors of dimension {}", base.vectors.len(), base.dimension);
    println!("Query dataset: {} vectors of dimension {}", query.vectors.len(), query.dimension);
    println!("Ground truth: {} queries", ground_truth.len());

    let index_name = "sift1m";
    let index_view = IndexView::new(index_name).await.expect("Failed to create IndexView");

    let path = Path::new(index_name);

    if !path.exists() {
        println!("Starting insertion of {} points...", base.vectors.len());

        let insert_parallelism = env_usize("NYAS_INSERT_CONCURRENCY").unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(4)
        });
        println!("Insert concurrency: {}", insert_parallelism);
        let index_view_ref = &index_view;

        futures::stream::iter(base.vectors.iter().enumerate())
            .for_each_concurrent(insert_parallelism, |(index, vector)| async move {
                let point = VectorPoint::new(index as u32, vector.clone());
                index_view_ref.insert(&point).await.unwrap();
            })
            .await;

        println!("Finished insertion.");

        println!("Starting final streaming merge to disk...");
        let merge_start = SystemTime::now();
        index_view.streaming_merge().await.expect("Failed to perform final streaming merge");
        println!("Final streaming merge took: {:?}", merge_start.elapsed().unwrap());
    }

    let index_cpu_time = start_cpu.elapsed();
    let index_wall_time = start_wall.elapsed().unwrap();
    println!("Indexing time: CPU {:?}, Wall {:?}", index_cpu_time, index_wall_time);

    for k in [1, 10, 100] {
        index_utils::compute_recall_at_k(&index_view, &query, &ground_truth, k).await;
    }

    let cpu_time: Duration = start_cpu.elapsed();
    let wall_time = start_wall.elapsed().unwrap();

    println!("CPU time: {:?}", cpu_time);
    println!("Wall time: {:?}", wall_time);

    Ok(())
}
