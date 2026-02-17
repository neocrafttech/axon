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

#[tokio::main]
async fn main() -> io::Result<()> {
    let data_root = dataset_root();
    let base = read_fvecs(&data_root.join("siftsmall_base.fvecs"))?;

    let run_id =
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0)).as_millis();
    let index_name = format!("sift10k_index_only_{}", run_id);
    let index_view = IndexView::new(&index_name).await?;

    let start = Instant::now();
    for (id, vector) in base.iter().enumerate() {
        index_view
            .insert(&VectorPoint::new(id as u32, vector.clone()))
            .await
            .map_err(io::Error::other)?;
    }
    let elapsed = start.elapsed();

    println!("DiskANN SIFT10K Index Benchmark");
    println!("base vectors: {}", base.len());
    println!("indexing wall time: {:?}", elapsed);

    let _ = std::fs::remove_dir_all(&index_name);
    Ok(())
}
