use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use diskann::index_view::IndexView;
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;

fn dataset_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/data/siftsmall")
}

fn read_fvecs(path: &Path) -> io::Result<Vec<VectorData>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let mut dim_bytes = [0u8; 4];
        match reader.read_exact(&mut dim_bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        let dim = u32::from_le_bytes(dim_bytes) as usize;
        let mut vec = vec![0f32; dim];
        let mut buf = [0u8; 4];
        for item in &mut vec {
            reader.read_exact(&mut buf)?;
            *item = f32::from_le_bytes(buf);
        }
        vectors.push(VectorData::from_f32(vec));
    }

    Ok(vectors)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let data_root = dataset_root();
    let base = read_fvecs(&data_root.join("siftsmall_base.fvecs"))?;
    let query = read_fvecs(&data_root.join("siftsmall_query.fvecs"))?;

    let run_id =
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0)).as_millis();
    let index_name = format!("sift10k_search_only_{}", run_id);
    let index_view = IndexView::new(&index_name).await?;

    for (id, vector) in base.iter().enumerate() {
        index_view
            .insert(&VectorPoint::new(id as u32, vector.clone()))
            .await
            .map_err(io::Error::other)?;
    }

    let search_count = 10_000usize;
    let mut latencies = Vec::with_capacity(search_count);
    let start = Instant::now();
    for i in 0..search_count {
        let q = &query[i % query.len()];
        let t0 = Instant::now();
        let _ = index_view.search(q, 10, 128).await;
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort_unstable();
    let qps =
        if total.as_secs_f64() > 0.0 { search_count as f64 / total.as_secs_f64() } else { 0.0 };

    println!("DiskANN SIFT10K Search Benchmark");
    println!("base vectors: {}", base.len());
    println!("query vectors: {}", query.len());
    println!("search count: {}", search_count);
    println!("search total wall time: {:?}", total);
    println!("search QPS: {:.2}", qps);
    println!("search p50 latency: {:?}", percentile(&latencies, 0.50));
    println!("search p95 latency: {:?}", percentile(&latencies, 0.95));
    println!("search p99 latency: {:?}", percentile(&latencies, 0.99));

    let _ = std::fs::remove_dir_all(&index_name);
    Ok(())
}
