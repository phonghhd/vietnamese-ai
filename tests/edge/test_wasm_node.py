from vietnamese_ai.edge.intelligent_router import EdgeRouter
from vietnamese_ai.edge.p2p_network import P2PTracker
from vietnamese_ai.edge.wasm_node import WebBrowserNode


def test_wasm_node_khoi_tao():
    node = WebBrowserNode(session_id="user_123")
    assert node.session_id == "user_123"
    assert node.trang_thai == "dang_cho"


def test_wasm_node_sinh_van_ban():
    node = WebBrowserNode(session_id="user_123")
    ket_qua = node.sinh_van_ban("Xin chào")
    assert "[WASM-WebGPU]" in ket_qua
    assert "Xin chào" in ket_qua


def test_wasm_node_tich_hop_router():
    tracker = P2PTracker()
    wasm_node = WebBrowserNode(session_id="client_xyz")

    # Đăng ký WASM Node vào mạng P2P với tốc độ ưu tiên
    tracker.dang_ky_node("web_client_xyz", wasm_node, toc_do_du_kien=20.0)

    # Cấu hình Router sử dụng mạng P2P
    router = EdgeRouter(p2p_tracker=tracker, cloud_api_key="dummy")

    # Chạy speculative routing
    ket_qua = router.sinh_van_ban_song_song("Chào trình duyệt")

    # Do WASM Node có tốc độ 20.0 (cao), nó sẽ chiến thắng cuộc đua
    assert "WASM-WebGPU" in ket_qua
