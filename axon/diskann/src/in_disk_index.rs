use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use dashmap::{DashMap, DashSet};
use system::metric::MetricType;
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;
use tokio::io;
use tokio::sync::RwLock;

use crate::disk_index_storage::DiskIndexStorage;
use crate::in_mem_index::InMemIndex;

pub struct InDiskIndex {
    /// Long-term index on SSD (simplified as in-memory for this demo)
    lti: Arc<RwLock<DiskIndexStorage>>,
    /// Read-write temporary index
    rw_temp_index: Arc<RwLock<InMemIndex>>,
    /// Read-only temporary indices
    ro_temp_indices: Arc<RwLock<Vec<InMemIndex>>>,
    /// Delete list
    delete_list: Arc<DashSet<u32>>,
    /// Maximum size of temp index before merge
    max_temp_size: usize,
    /// Parameters
    r: usize,
    alpha: f64,
    l_build: usize,
    metric: MetricType,
}

impl InDiskIndex {
    pub async fn new(
        index_name: &str, r: usize, alpha: f64, l_build: usize, max_temp_size: usize,
        metric: MetricType,
    ) -> Result<Self, std::io::Error> {
        let lti = Arc::new(RwLock::new(DiskIndexStorage::new(index_name).await?));
        let rw_temp_index = Arc::new(RwLock::new(InMemIndex::new(r, alpha, l_build, metric)));
        let ro_temp_indices = Arc::new(RwLock::new(Vec::new()));
        let delete_list = Arc::new(DashSet::new());

        Ok(InDiskIndex {
            lti,
            rw_temp_index,
            ro_temp_indices,
            delete_list,
            max_temp_size,
            r,
            alpha,
            l_build,
            metric,
        })
    }

    pub async fn insert(&self, point: &VectorPoint) -> Result<(), String> {
        {
            let rw_temp = self.rw_temp_index.read().await;
            rw_temp.insert(point);

            if rw_temp.size() < self.max_temp_size {
                return Ok(());
            }
        }

        // Only reach here if we need to snapshot
        self.snapshot_temp_index().await?;

        Ok(())
    }

    pub fn delete(&self, point_id: u32) {
        self.delete_list.insert(point_id);
    }

    pub async fn search(&self, query: &VectorData, k: usize, l: usize) -> Vec<u32> {
        let mut visited_map: HashMap<u32, f64> = HashMap::new();

        // Search LTI
        let lti = self.lti.read().await;
        lti.search(query, l, &mut visited_map).await;

        // Search RW-TempIndex
        let rw_temp = self.rw_temp_index.read().await;
        rw_temp.greedy_search_for_lti(query, l, &mut visited_map);
        // Search RO-TempIndices
        let ro_temps = self.ro_temp_indices.read().await;
        for ro_temp in ro_temps.iter() {
            ro_temp.greedy_search_for_lti(query, l, &mut visited_map);
        }

        let mut result_with_dist: Vec<_> = visited_map.into_iter().collect();

        if result_with_dist.len() > k {
            let _ = result_with_dist.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)
            });
            result_with_dist.truncate(k);
        }

        result_with_dist
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));

        // Filter out deleted points
        result_with_dist
            .into_iter()
            .filter_map(|(id, _)| (!self.delete_list.contains(&id)).then_some(id))
            .take(k)
            .collect()
    }

    #[allow(dead_code)]
    async fn get_point_distance(
        &self, point_id: u32, query: &VectorData, lti: &InMemIndex, rw: &InMemIndex,
    ) -> f64 {
        if let Some(p) = lti.points.get(&point_id) {
            return p.distance_to_vector(query, self.metric);
        }
        if let Some(p) = rw.points.get(&point_id) {
            return p.distance_to_vector(query, self.metric);
        }

        let ro_temps = self.ro_temp_indices.read().await;
        for ro in ro_temps.iter() {
            if let Some(p) = ro.points.get(&point_id) {
                return p.distance_to_vector(query, self.metric);
            }
        }
        f64::MAX
    }

    /// Snapshot RW-TempIndex to RO-TempIndex
    async fn snapshot_temp_index(&self) -> Result<(), String> {
        let mut rw_temp = self.rw_temp_index.write().await;
        let mut ro_temps = self.ro_temp_indices.write().await;

        // Clone the current RW index to RO
        let snapshot = InMemIndex {
            graph: rw_temp.graph.clone(),
            points: rw_temp.points.clone(),
            start_node: std::sync::RwLock::new(*rw_temp.start_node.read().unwrap()),
            r: rw_temp.r,
            alpha: rw_temp.alpha,
            l_build: rw_temp.l_build,
            locks: DashMap::new(), // RO doesn't need locks
            metric: self.metric,
        };

        ro_temps.push(snapshot);

        // Create new empty RW-TempIndex
        *rw_temp = InMemIndex::new(self.r, self.alpha, self.l_build, self.metric);

        Ok(())
    }

    #[allow(dead_code)]
    pub async fn streaming_merge(&self) -> io::Result<()> {
        let deletes: Vec<_> = self.delete_list.iter().map(|e| *e).collect();
        let delete_set: std::collections::HashSet<u32> = deletes.iter().copied().collect();

        let mut rw_temp = self.rw_temp_index.write().await;
        let mut ro_temps = self.ro_temp_indices.write().await;

        if ro_temps.is_empty() && rw_temp.points.is_empty() {
            return Ok(());
        }

        // Fast path: no deletes and only one staged index to flush.
        if delete_set.is_empty() {
            if ro_temps.is_empty() {
                let mut lti = self.lti.write().await;
                lti.insert(&rw_temp).await?;
                drop(lti);

                *rw_temp = InMemIndex::new(self.r, self.alpha, self.l_build, self.metric);
                self.delete_list.clear();
                return Ok(());
            }

            if ro_temps.len() == 1 && rw_temp.points.is_empty() {
                let mut lti = self.lti.write().await;
                lti.insert(&ro_temps[0]).await?;
                drop(lti);

                ro_temps.clear();
                self.delete_list.clear();
                return Ok(());
            }
        }

        println!("Starting streaming merge of {} RO indices and RW index...", ro_temps.len());

        // Deduplicate by ID while collecting with "newest wins" semantics:
        // RW (newest) first, then RO snapshots from newest to oldest.
        let mut seen = std::collections::HashSet::new();
        let mut dedup_points: Vec<VectorPoint> = Vec::new();

        // Collect points from RW index and filter deleted IDs.
        for point_ref in rw_temp.points.iter() {
            let point = point_ref.value().clone();
            if !delete_set.contains(&point.id) && seen.insert(point.id) {
                dedup_points.push(point);
            }
        }

        // Collect points from RO indices from newest to oldest.
        for ro_temp in ro_temps.iter().rev() {
            for point_ref in ro_temp.points.iter() {
                let point = point_ref.value().clone();
                if !delete_set.contains(&point.id) && seen.insert(point.id) {
                    dedup_points.push(point);
                }
            }
        }

        if dedup_points.is_empty() {
            rw_temp.points.clear();
            ro_temps.clear();
            self.delete_list.clear();
            return Ok(());
        }

        println!("Merging {} unique points...", dedup_points.len());

        let combined = InMemIndex::new(self.r, self.alpha, self.l_build, self.metric);

        // Rebuild in parallel from deduplicated points.
        use rayon::prelude::*;
        dedup_points.into_par_iter().for_each(|point| {
            combined.insert(&point);
        });

        println!("Persisting combined index to disk...");
        let mut lti = self.lti.write().await;
        lti.insert(&combined).await?;
        drop(lti);

        println!("Streaming merge completed. Clearing temporary indices.");
        *rw_temp = InMemIndex::new(self.r, self.alpha, self.l_build, self.metric);
        ro_temps.clear();
        self.delete_list.clear();
        Ok(())
    }
}

#[cfg(test)]
mod in_disk_index_test {
    use std::time::SystemTime;

    use system::metric::MetricType;
    use system::vector_data::VectorData;
    use system::vector_point::VectorPoint;

    use super::*;
    use crate::in_mem_index::InMemIndex;

    #[tokio::test]
    async fn test_in_disk_index() {
        let system = InDiskIndex::new("test_index_1", 32, 1.2, 50, 20, MetricType::L2)
            .await
            .expect("Expects In Disk Index creation");

        for i in 0..50 {
            let vector = VectorData::from_f32(vec![i as f32, (i * 2) as f32]);
            system.insert(&VectorPoint::new(i, vector)).await.unwrap();
        }

        system.delete(5);
        system.delete(10);

        let query = VectorData::from_f32(vec![25.0, 50.0]);
        let results = system.search(&query, 5, 20).await;

        assert!(!results.contains(&5));
        assert!(!results.contains(&10));

        println!("Search results: {:?}", results);
    }

    #[tokio::test]
    async fn test_streaming_merge() {
        let system = InDiskIndex::new("test_index_2", 32, 1.2, 50, 20, MetricType::L2)
            .await
            .expect("Expects In Disk Index creation");
        let in_mem_index = InMemIndex::new(32, 1.2, 50, MetricType::L2);

        for i in 0..50 {
            let vector = VectorData::from_f32(vec![i as f32, (i * 2) as f32, i as f32 / 2.0]);
            let point = VectorPoint::new(i, vector);
            system.insert(&point).await.unwrap();
            in_mem_index.insert(&point);
        }

        system.streaming_merge().await.unwrap();

        let query = VectorData::from_f32(vec![25.0, 50.0, 12.5]);
        let results = system.search(&query, 5, 20).await;
        println!("Disk Search results: {:?}", results);
        let in_mem_results = in_mem_index.search(&query, 5, 20);
        println!("In Memory Search results: {:?}", in_mem_results);
        for result in [23, 24, 25, 26, 27] {
            assert!(results.contains(&result));
            assert!(in_mem_results.contains(&result));
        }

        let query = VectorData::from_f32(vec![20.0, 41.0, 12.5]);
        let results = system.search(&query, 5, 20).await;
        println!("Boundary Disk Search results: {:?}", results);
        // Expect most of the closest IDs to be present
        let mut count = 0;
        for result in [19, 20, 21, 22, 23] {
            if results.contains(&result) {
                count += 1;
            }
        }
        assert!(count >= 4);
    }

    #[tokio::test]
    async fn test_streaming_merge_large() {
        let system = InDiskIndex::new("test_index_large", 32, 1.2, 50, 100, MetricType::L2)
            .await
            .expect("Expects In Disk Index creation");

        println!("Inserting 10000 points...");
        for i in 0..10000 {
            let vector = VectorData::from_f32(vec![i as f32, (i * 2) as f32, i as f32 / 2.0]);
            let point = VectorPoint::new(i, vector);
            system.insert(&point).await.unwrap();
        }

        println!("Starting streaming_merge...");
        let start = SystemTime::now();
        system.streaming_merge().await.unwrap();
        println!("Streaming merge took: {:?}", start.elapsed().unwrap());

        let query = VectorData::from_f32(vec![500.0, 1000.0, 250.0]);
        let results = system.search(&query, 5, 20).await;
        println!("Disk Search results: {:?}", results);
    }

    #[tokio::test]
    async fn test_pq_optimized_search() {
        let dim = 64;
        let num_points = 200;
        let system = InDiskIndex::new("test_index_pq", 32, 1.2, 100, 500, MetricType::L2)
            .await
            .expect("Expects In Disk Index creation");

        for i in 0..num_points {
            let mut data = vec![0.0; dim];
            data[0] = i as f32;
            let vector = VectorData::from_f32(data);
            system.insert(&VectorPoint::new(i as u32, vector)).await.unwrap();
        }

        // Search for a point that was added
        let mut query_data = vec![0.0; dim];
        query_data[0] = 50.5;
        let query = VectorData::from_f32(query_data);

        let results = system.search(&query, 5, 20).await;
        println!("PQ Search results for query 50.5: {:?}", results);

        // Expect results near 50 and 51
        assert!(results.contains(&50) || results.contains(&51));
    }
}
