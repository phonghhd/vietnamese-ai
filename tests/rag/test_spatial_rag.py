import numpy as np

from vietnamese_ai.rag.spatial_store import SpatialVectorStore


def test_spatial_vector_store():
    # Khởi tạo Vector Store với chiều dữ liệu = 4
    store = SpatialVectorStore(kich_thuoc=4, khoang_cach="cosine")

    # Giả lập 2 vector ngữ nghĩa hoàn toàn giống nhau (đều là "Cốc nước")
    vector_coc_nuoc = np.array([0.1, 0.2, 0.3, 0.4])

    # Nhưng tọa độ 3D khác nhau
    store.chen_khong_gian("coc_bep", vector_coc_nuoc, toa_do=(0.0, 0.0, 0.0), metadata={"color": "red"})
    store.chen_khong_gian("coc_ban_khach", vector_coc_nuoc, toa_do=(100.0, 100.0, 100.0), metadata={"color": "blue"})

    # Truy vấn tại tọa độ (1.0, 1.0, 1.0), ưu tiên không gian (alpha = 0.1)
    results = store.tim_kiem_khong_gian(
        query_vector=vector_coc_nuoc,
        toa_do_truy_van=(1.0, 1.0, 1.0),
        top_k=1,
        alpha=0.1
    )

    assert len(results) == 1
    # Kết quả phải trả về "coc_bep" vì tọa độ (0,0,0) gần (1,1,1) hơn rất nhiều so với (100,100,100)
    assert results[0]["ma"] == "coc_bep"
    assert results[0]["diem_khong_gian"] > 0.0
