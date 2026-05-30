import pytest
from unittest.mock import patch
from vietnamese_ai.mobile.power_manager import PowerManager

@patch("vietnamese_ai.mobile.power_manager.PowerManager.get_battery_status")
def test_power_manager_high_battery(mock_status):
    # Pin 80%, không sạc
    mock_status.return_value = (80.0, False)
    config = PowerManager.dieu_tiet_do_chinh_xac()
    assert config["precision"] == "4-bit"
    assert config["load_in_4bit"] == True
    
@patch("vietnamese_ai.mobile.power_manager.PowerManager.get_battery_status")
def test_power_manager_low_battery(mock_status):
    # Pin 15%, không sạc -> Kích hoạt chế độ sinh tồn 1.58-bit
    mock_status.return_value = (15.0, False)
    config = PowerManager.dieu_tiet_do_chinh_xac()
    assert config["precision"] == "1.58-bit"
    assert config["use_bitnet"] == True
    assert config["load_in_8bit"] == False
