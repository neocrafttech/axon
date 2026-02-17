use std::collections::HashSet;
use std::path::Path;

use diskann::index_view::IndexView;
use system::vector_data::VectorData;
use tokio::io;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse::<usize>().ok().filter(|v| *v > 0)
}

#[derive(Debug)]
pub struct SiftDataset {
    pub dimension: u32,
    pub vectors: Vec<VectorData>,
}

impl SiftDataset {
    pub async fn from_fvecs<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let bytes = tokio::fs::read(path).await?;
        if bytes.len() < 4 {
            return Ok(SiftDataset { dimension: 0, vectors: Vec::new() });
        }

        let dimension = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let record_size = 4usize + 4usize * dimension as usize;
        if record_size == 0 || bytes.len() % record_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid fvecs layout or truncated file",
            ));
        }

        let mut vectors = Vec::with_capacity(bytes.len() / record_size);
        let mut offset = 0usize;
        while offset + record_size <= bytes.len() {
            let dim = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            if dim != dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Inconsistent vector dimensions",
                ));
            }

            let mut vec = vec![0f32; dimension as usize];
            let data_start = offset + 4;
            for (i, item) in vec.iter_mut().enumerate() {
                let s = data_start + i * 4;
                *item = f32::from_le_bytes(bytes[s..s + 4].try_into().unwrap());
            }
            vectors.push(VectorData::from_f32(vec));
            offset += record_size;
        }

        Ok(SiftDataset { dimension, vectors })
    }

    #[allow(dead_code)]
    pub async fn from_bvecs<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let bytes = tokio::fs::read(path).await?;
        if bytes.len() < 4 {
            return Ok(SiftDataset { dimension: 0, vectors: Vec::new() });
        }

        let dimension = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let record_size = 4usize + dimension as usize;
        if record_size == 0 || bytes.len() % record_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid bvecs layout or truncated file",
            ));
        }

        let mut vectors = Vec::with_capacity(bytes.len() / record_size);
        let mut offset = 0usize;
        while offset + record_size <= bytes.len() {
            let dim = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            if dim != dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Inconsistent vector dimensions",
                ));
            }
            let data_start = offset + 4;
            let vec = bytes[data_start..data_start + dimension as usize]
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>();
            vectors.push(VectorData::from_f32(vec));
            offset += record_size;
        }

        Ok(SiftDataset { dimension, vectors })
    }

    pub async fn from_ivecs<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<u32>>> {
        let bytes = tokio::fs::read(path).await?;
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        let mut offset = 0usize;
        while offset + 4 <= bytes.len() {
            let k = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let record_bytes = k * 4;
            if offset + record_bytes > bytes.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid ivecs layout or truncated file",
                ));
            }

            let mut neighbors = Vec::with_capacity(k);
            for i in 0..k {
                let s = offset + i * 4;
                neighbors.push(u32::from_le_bytes(bytes[s..s + 4].try_into().unwrap()));
            }
            results.push(neighbors);
            offset += record_bytes;
        }

        Ok(results)
    }
}

pub fn compute_recall(results: &[Vec<u32>], ground_truth: &[Vec<u32>], k: usize) -> f64 {
    assert_eq!(results.len(), ground_truth.len(), "Results and groundtruth must have same length");

    let num_queries = results.len();
    let mut total_matches = 0;

    for (result, truth) in results.iter().zip(ground_truth.iter()) {
        let truth_set: HashSet<u32> = truth.iter().take(k).copied().collect();

        let matches = result.iter().take(k).filter(|&&idx| truth_set.contains(&idx)).count();
        total_matches += matches;
    }

    total_matches as f64 / (num_queries * k) as f64
}

#[allow(dead_code)]
pub async fn compute_recall_at_k(
    index_view: &IndexView, query: &SiftDataset, ground_truth: &[Vec<u32>], k: usize,
) {
    use futures::StreamExt;

    let query_len = query.vectors.len();
    let parallelism = env_usize("NYAS_SEARCH_CONCURRENCY").unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(4)
    });

    let mut indexed_results: Vec<(usize, Vec<u32>)> =
        futures::stream::iter(query.vectors.iter().enumerate())
            .map(|(i, q)| async move { (i, index_view.search(q, k, 128).await) })
            .buffer_unordered(parallelism)
            .collect()
            .await;

    indexed_results.sort_unstable_by_key(|(i, _)| *i);
    let results: Vec<Vec<u32>> = indexed_results.into_iter().map(|(_, r)| r).collect();
    debug_assert_eq!(results.len(), query_len);

    let recall = compute_recall(&results, ground_truth, k);
    println!("Recall  for k: {}: {:?}", k, recall);
}
