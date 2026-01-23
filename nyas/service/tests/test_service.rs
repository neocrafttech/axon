use service::VectorService;
use service::vector::vector_db_server::VectorDb;
use service::vector::{InsertVectorRequest, SearchVectorRequest};
use tonic::Request;

#[tokio::test]
async fn test_insert_vector() {
    let index = diskann::index_view::IndexView::new("test_insert").await.unwrap();
    let service = VectorService::new(index);
    let dim = 128;
    let input: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
    let req = InsertVectorRequest { id: "vec1".to_string(), vector: input };

    let response = service.insert_vector(Request::new(req)).await.unwrap();
    let res_inner = response.into_inner();

    assert!(res_inner.success, "Insert should succeed");
}

#[tokio::test]
async fn test_search_vector() {
    let index = diskann::index_view::IndexView::new("test_search").await.unwrap();
    let service = VectorService::new(index);
    let dim = 128;
    let input: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
    let req = InsertVectorRequest { id: "vec1".to_string(), vector: input };
    service.insert_vector(Request::new(req)).await.expect("Insert failed");

    let search_vec: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 0.01).collect();
    let req = SearchVectorRequest { vector: search_vec, top_k: 1 };

    let response = service.search_vector(Request::new(req)).await.expect("Search failed");
    let res_inner = response.into_inner();

    assert_eq!(res_inner.ids.len(), 1, "Should return one id");
    assert_eq!(res_inner.distances.len(), 1, "Should return one distance");
    assert_eq!(res_inner.ids[0], "vec1");
}
