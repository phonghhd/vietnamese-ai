import json
import os
from unittest.mock import MagicMock, patch

from vietnamese_ai.agents.polyglot_tools import CongCuTuXa
from vietnamese_ai.api.grpc_server import MockEvoNetServiceServicer


def test_grpc_proto_structure():
    """Kiểm tra file Protobuf tồn tại và chứa cấu trúc chuẩn."""
    proto_file = os.path.join(
        os.path.dirname(__file__), "../../vietnamese_ai/api/proto/evonet.proto"
    )
    assert os.path.exists(proto_file)

    with open(proto_file, "r") as f:
        content = f.read()

    assert 'syntax = "proto3";' in content
    assert "service EvoNetService" in content
    assert "rpc ChatCompletion" in content


def test_grpc_mock_servicer():
    """Kiểm tra Servicer gRPC xử lý Request đúng chuẩn."""

    class DummyRequest:
        def __init__(self, model, prompt):
            self.model = model
            self.prompt = prompt

    servicer = MockEvoNetServiceServicer(bo_xu_ly=None)  # Fallback native
    req = DummyRequest("test-model", "Xin chao")
    res = servicer.ChatCompletion(req, None)

    assert res.model == "test-model"
    assert "[gRPC Native]" in res.text
    assert "grpc-" in res.id


@patch("urllib.request.urlopen")
def test_polyglot_remote_tool(mock_urlopen):
    """Kiểm tra công cụ MCP Remote Tool Fetch đúng chuẩn."""
    # Giả lập Response từ Node.js Webhook
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"status": "success", "message": "Da xoa DB"}'
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    remote_tool = CongCuTuXa(
        ten="clear_db",
        mo_ta="Xoá Database trên Node.js",
        webhook_url="http://localhost:3000/webhook/clear_db",
    )

    # Thực thi tool
    ket_qua = remote_tool.chay(table="users")

    assert "Da xoa DB" in ket_qua
    mock_urlopen.assert_called_once()

    # Kiểm tra Request Object
    req_obj = mock_urlopen.call_args[0][0]
    assert req_obj.full_url == "http://localhost:3000/webhook/clear_db"

    payload = json.loads(req_obj.data.decode("utf-8"))
    assert payload["tool"] == "clear_db"
    assert payload["parameters"]["table"] == "users"


def test_wasm_webgpu_integration():
    """Kiểm tra WebGPU có trong WASM Loader không."""
    wasm_file = os.path.join(os.path.dirname(__file__), "../../sdks/wasm/evonet_wasm.js")
    with open(wasm_file, "r") as f:
        content = f.read()

    assert "initWebGPU()" in content
    assert "navigator.gpu.requestAdapter" in content
