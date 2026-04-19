use std::collections::HashMap;
use std::sync::Arc;

use diskann::index_view::IndexView;
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use crate::vector::vector_db_server::VectorDb;
use crate::vector::{
    BuildIndexRequest, BuildIndexResponse, DeleteVectorRequest, DeleteVectorResponse, IndexType,
    InsertVectorRequest, InsertVectorResponse, SearchVectorRequest, SearchVectorResponse,
};

pub mod vector {
    tonic::include_proto!("vector");
}

pub struct VectorService {
    pub index: Arc<IndexView>,
    pub id_map: Arc<RwLock<HashMap<String, u32>>>,
    pub reverse_id_map: Arc<RwLock<HashMap<u32, String>>>,
}

impl VectorService {
    pub fn new(index: IndexView) -> Self {
        Self {
            index: Arc::new(index),
            id_map: Arc::new(RwLock::new(HashMap::new())),
            reverse_id_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn get_or_create_u32_id(&self, string_id: String) -> u32 {
        let mut id_map = self.id_map.write().await;
        if let Some(&u32_id) = id_map.get(&string_id) {
            u32_id
        } else {
            let u32_id = id_map.len() as u32;
            id_map.insert(string_id.clone(), u32_id);
            self.reverse_id_map.write().await.insert(u32_id, string_id);
            u32_id
        }
    }

    async fn get_string_id(&self, u32_id: u32) -> Option<String> {
        self.reverse_id_map.read().await.get(&u32_id).cloned()
    }
}

impl BuildIndexRequest {
    pub fn index_type_enum(&self) -> IndexType {
        IndexType::try_from(self.index_type).unwrap_or(IndexType::Unspecified)
    }
}

#[tonic::async_trait]
impl VectorDb for VectorService {
    async fn insert_vector(
        &self, request: Request<InsertVectorRequest>,
    ) -> Result<Response<InsertVectorResponse>, Status> {
        let req = request.into_inner();
        let u32_id = self.get_or_create_u32_id(req.id).await;
        let point = VectorPoint::new(u32_id, VectorData::from_f32(req.vector));

        match self.index.insert(&point).await {
            Ok(_) => Ok(Response::new(InsertVectorResponse { success: true })),
            Err(e) => Err(Status::internal(e)),
        }
    }

    async fn search_vector(
        &self, request: Request<SearchVectorRequest>,
    ) -> Result<Response<SearchVectorResponse>, Status> {
        let req = request.into_inner();
        let query = VectorData::from_f32(req.vector);
        // Using default L value of 100 for now, k from request
        let k = req.top_k as usize;
        let l = std::cmp::max(k, 100);

        let u32_ids: Vec<u32> = self.index.search(&query, k, l).await;

        let mut ids = Vec::with_capacity(u32_ids.len());
        for u32_id in u32_ids {
            if let Some(string_id) = self.get_string_id(u32_id).await {
                ids.push(string_id);
            }
        }

        // Diskann currently doesn't return distances in IndexView::search
        // We might need to update IndexView if distances are required.
        let distances = vec![0.0; ids.len()];

        Ok(Response::new(SearchVectorResponse { ids, distances }))
    }

    async fn build_index(
        &self, _request: Request<BuildIndexRequest>,
    ) -> Result<Response<BuildIndexResponse>, Status> {
        // Build index logic might be different now with InDiskIndex integration
        // For now, return success if it's already using diskann
        Ok(Response::new(BuildIndexResponse { success: true }))
    }

    async fn delete_vector(
        &self, request: Request<DeleteVectorRequest>,
    ) -> Result<Response<DeleteVectorResponse>, Status> {
        let req = request.into_inner();
        let u32_id_opt = { self.id_map.read().await.get(&req.id).copied() };

        if let Some(u32_id) = u32_id_opt {
            self.index.delete(u32_id);
            Ok(Response::new(DeleteVectorResponse { success: true }))
        } else {
            Err(Status::not_found(format!("Vector with ID {} not found", req.id)))
        }
    }
}
