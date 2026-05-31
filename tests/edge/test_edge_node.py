from unittest.mock import MagicMock, patch

from vietnamese_ai.edge.intelligent_router import EdgeRouter
from vietnamese_ai.edge.node_llama import NodeLlamaEngine


@patch("subprocess.Popen")
@patch("shutil.which", return_value="/usr/bin/npx")
@patch("os.path.exists", return_value=True)
@patch("vietnamese_ai.edge.node_llama.NodeLlamaEngine._wait_for_server")
def test_node_llama_engine_start(mock_wait, mock_exists, mock_which, mock_popen):
    """Test khởi tạo và chạy process node-llama-cpp."""
    mock_process = MagicMock()
    mock_popen.return_value = mock_process

    engine = NodeLlamaEngine(model_path="~/.vietnamese_ai/models/model.gguf", port=8080)

    # Kiểm tra lệnh gọi subprocess
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert "npx" in args[0]
    assert "node-llama-cpp" in args[0]

    # Kiểm tra cleanup
    engine.stop_server()
    mock_process.terminate.assert_called_once()


@patch(
    "vietnamese_ai.edge.intelligent_router.EdgeRouter._call_cloud", return_value="Trả lời từ Cloud"
)
def test_edge_router(mock_call_cloud):
    """Test logic định tuyến của EdgeRouter."""
    mock_edge = MagicMock()
    mock_edge.sinh_van_ban.return_value = "Trả lời từ Edge"
    del mock_edge.chay_suy_luan_an_toan

    router = EdgeRouter(edge_engine=mock_edge, cloud_api_key="fake_key")

    # Test câu hỏi đơn giản (vào Edge)
    res_edge = router.sinh_van_ban("Xin chào")
    assert res_edge == "Trả lời từ Edge"

    # Test câu hỏi phức tạp (vào Cloud)
    res_cloud = router.sinh_van_ban("Giải thích chi tiết thuật toán Transformer")
    assert res_cloud == "Trả lời từ Cloud"

    # Test ép buộc Edge
    res_force = router.sinh_van_ban("Giải thích chi tiết", force_edge=True)
    assert res_force == "Trả lời từ Edge"
