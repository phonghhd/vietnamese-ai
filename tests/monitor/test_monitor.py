import os
import time
from unittest.mock import patch, mock_open, MagicMock

import pytest

from vietnamese_ai.monitor import TheoDoiGPU, TheoDoiHeThong, MayTinhTien


@patch("ctypes.CDLL")
def test_theo_doi_gpu_khong_loi_khi_khong_co_driver(mock_cdll):
    # Cố tình ném lỗi khi tải CDLL
    mock_cdll.side_effect = OSError("image not found")
    
    # Không được ném lỗi ra ngoài, chỉ in log warning
    gpu_monitor = TheoDoiGPU()
    assert gpu_monitor._da_khoi_tao is False
    
    # Hàm lấy thông tin phải trả về rỗng an toàn
    thong_tin = gpu_monitor.lay_thong_tin_tat_ca_gpu()
    assert isinstance(thong_tin, list)
    assert len(thong_tin) == 0

@patch("ctypes.CDLL")
def test_theo_doi_gpu_hoat_dong(mock_cdll):
    # Mock thư viện C
    mock_lib = MagicMock()
    # nvmlInit_v2 thành công
    mock_lib.nvmlInit_v2.return_value = 0 
    # Có 1 GPU
    mock_lib.nvmlDeviceGetCount_v2.side_effect = lambda c: setattr(c._obj, 'value', 1) or 0
    # nvmlDeviceGetHandleByIndex_v2 thành công
    mock_lib.nvmlDeviceGetHandleByIndex_v2.return_value = 0
    # Tên GPU
    mock_lib.nvmlDeviceGetName.side_effect = lambda h, name, l: setattr(name, 'value', b"RTX 3090 Ti") or 0
    # Nhiệt độ 75C
    mock_lib.nvmlDeviceGetTemperature.side_effect = lambda h, st, t: setattr(t._obj, 'value', 75) or 0
    # Công suất 250000 mW (250W)
    mock_lib.nvmlDeviceGetPowerUsage.side_effect = lambda h, p: setattr(p._obj, 'value', 250000) or 0
    
    mock_cdll.return_value = mock_lib
    
    gpu_monitor = TheoDoiGPU()
    assert gpu_monitor._da_khoi_tao is True
    
    thong_tin = gpu_monitor.lay_thong_tin_tat_ca_gpu()
    assert len(thong_tin) == 1
    assert thong_tin[0]["ten"] == "RTX 3090 Ti"
    assert thong_tin[0]["nhiet_do_c"] == 75
    assert thong_tin[0]["cong_suat_w"] == 250.0

@patch("os.path.exists", return_value=True)
def test_theo_doi_he_thong_linux_proc(mock_exists):
    # Giả lập /proc/stat
    # Format: cpu  user nice system idle iowait irq softirq steal
    mock_stat = "cpu  1000 0 500 8000 500 0 0 0\n"
    # Giả lập /proc/meminfo
    mock_mem = "MemTotal: 16000000 kB\nMemAvailable: 4000000 kB\n"
    
    # Dùng side_effect cho mock_open để trả về nội dung khác nhau cho các file
    def mock_file(filename, *args, **kwargs):
        if "stat" in filename:
            return mock_open(read_data=mock_stat)()
        elif "meminfo" in filename:
            return mock_open(read_data=mock_mem)()
        return mock_open()()

    with patch("builtins.open", side_effect=mock_file):
        monitor = TheoDoiHeThong()
        assert monitor._ho_tro_proc is True
        
        # Test RAM
        thong_tin = monitor.lay_thong_tin()
        assert thong_tin["ram_tong_mb"] == 16000000 // 1024
        assert thong_tin["ram_su_dung_mb"] == (16000000 - 4000000) // 1024
        
@patch("time.time")
def test_may_tinh_tien(mock_time):
    # Thời điểm t=0
    mock_time.return_value = 0.0
    may_tinh = MayTinhTien(gia_dien_kwh=3000, gia_thue_gpu_gio=10000)
    
    # 1 giờ sau (t=3600), chạy với công suất 250W
    mock_time.return_value = 3600.0
    may_tinh.cap_nhat_muc_tieu_thu(cong_suat_w=250.0)
    
    # Lúc này năng lượng tiêu thụ là 250W * 3600s = 900,000 Joules = 0.25 kWh
    bao_cao = may_tinh.lay_bao_cao_chi_phi()
    
    assert bao_cao["thoi_gian_chay_giay"] == 3600.0
    assert bao_cao["tong_kwh"] == 0.25
    assert bao_cao["chi_phi_dien_vnd"] == 0.25 * 3000  # 750 VNĐ
    assert bao_cao["chi_phi_thue_vnd"] == 1.0 * 10000  # 10000 VNĐ
    assert bao_cao["tong_chi_phi_vnd"] == 10750.0

    # Test reset
    may_tinh.reset()
    bao_cao2 = may_tinh.lay_bao_cao_chi_phi()
    assert bao_cao2["thoi_gian_chay_giay"] == 0.0
    assert bao_cao2["tong_chi_phi_vnd"] == 0.0
