import os

from vietnamese_ai.mobile.power_manager import PowerManager
from vietnamese_ai.mobile.sdk_generator import SDKGenerator


def test_sdk_generator(tmp_path):
    flutter_file = SDKGenerator.generate_flutter(str(tmp_path))
    rn_file = SDKGenerator.generate_react_native(str(tmp_path))

    assert os.path.exists(flutter_file)
    assert os.path.exists(rn_file)

    with open(flutter_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "EvoNetConnector" in content
        assert "hardware_sync" in content
        assert "offlineQueue" in content
        assert "encryptPayload" in content

    with open(rn_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "export class EvoNetConnector" in content
        assert "offlineQueue" in content
        assert "encryptPayload" in content


def test_power_manager_hardware_sync():
    # Xóa trạng thái nếu có từ các test khác
    PowerManager._trang_thai_tu_app = {}

    # Lúc đầu dùng psutil hoặc fallback
    pin_cu, _ = PowerManager.get_battery_status()

    # App gọi SDK Bridge gửi trạng thái Pin thật
    PowerManager.dong_bo_phan_cung(pin=12.5, cam_sac=False, nhiet_do="critical")

    # get_battery_status phải ưu tiên trả về dữ liệu từ SDK App (12.5)
    pin_moi, sạc = PowerManager.get_battery_status()

    assert pin_moi == 12.5
    assert sạc is False

import base64
import json
from unittest.mock import MagicMock

import pytest

from vietnamese_ai.mobile.bridge import MobileAppBridge


@pytest.mark.asyncio
async def test_bridge_e2e_encryption():
    agent = MagicMock()
    agent.chay.return_value = "Đã phân tích xong."

    bridge = MobileAppBridge(mobile_agent=agent)

    # Mã hóa chuỗi gửi lên (giả lập Client)
    payload_client = {"message": "Báo cáo tài chính"}
    payload_encoded = base64.b64encode(json.dumps(payload_client).encode("utf-8")).decode("utf-8")

    # Xử lý request qua Bridge (với Background=True để test Push)
    ket_qua_encoded = await bridge.handle_request("chat", payload_encoded, is_background=True)

    # Giải mã chuỗi nhận về
    ket_qua_decoded = json.loads(base64.b64decode(ket_qua_encoded).decode("utf-8"))

    assert ket_qua_decoded["text"] == "Đã phân tích xong."
    assert ket_qua_decoded["is_done"] is True

