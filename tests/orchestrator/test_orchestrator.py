import json
import threading
import time
import urllib.request
from unittest.mock import patch, MagicMock

import pytest

from vietnamese_ai.orchestrator import CanBangTai, NutChinh, NutPhu


def test_can_bang_tai_round_robin():
    balancer = CanBangTai(chien_luoc="round_robin")
    workers = ["w1", "w2", "w3"]
    
    # Khởi tạo chi_so_hien_tai = 0 -> Lần chọn đầu tiên sẽ tăng lên 1 -> "w2"
    assert balancer.chon_worker(workers) == "w2"
    assert balancer.chon_worker(workers) == "w3"
    assert balancer.chon_worker(workers) == "w1"
    
def test_can_bang_tai_empty():
    balancer = CanBangTai()
    assert balancer.chon_worker([]) is None

def test_can_bang_tai_canary():
    balancer = CanBangTai(chien_luoc="canary")
    balancer.cap_nhat_trong_so({"w1": 0.9, "w2": 0.1})
    workers = ["w1", "w2", "w3"]
    
    with patch("random.choices", return_value=["w2"]):
        # Chỉ truyền w1, w2 vì w3 không có trọng số
        chon = balancer.chon_worker(workers)
        assert chon == "w2"

def test_can_bang_tai_random():
    balancer = CanBangTai(chien_luoc="random")
    workers = ["w1"]
    assert balancer.chon_worker(workers) == "w1"

def ham_xu_ly_mock(data):
    return {"nhan": "test", "diem": 0.99}

def test_nut_chinh_khoi_tao():
    master = NutChinh(port=18080)
    assert master.host == "0.0.0.0"
    assert master.port == 18080
    assert master.can_bang_tai.chien_luoc == "round_robin"

@patch("multiprocessing.Process")
def test_nut_chinh_them_worker_va_khoi_dong(mock_process):
    master = NutChinh()
    master.them_worker(port=18081, bo_xu_ly=ham_xu_ly_mock)
    
    # Kiểm tra cấu hình đã được thêm
    worker_id = "127.0.0.1:18081"
    assert worker_id in master.workers_config
    
    # Mock Process instance
    mock_p_instance = MagicMock()
    mock_process.return_value = mock_p_instance
    
    master._khoi_dong_worker(worker_id)
    
    mock_process.assert_called_once()
    mock_p_instance.start.assert_called_once()
    assert master.workers_config[worker_id]["process"] == mock_p_instance

@patch("urllib.request.urlopen")
def test_nut_chinh_kiem_tra_suc_khoe(mock_urlopen):
    # Mock response cho health check
    mock_response = MagicMock()
    mock_response.status = 200
    mock_urlopen.return_value.__enter__.return_value = mock_response

    master = NutChinh()
    master.them_worker(port=18081, bo_xu_ly=ham_xu_ly_mock)
    worker_id = "127.0.0.1:18081"
    
    # Giả lập process đang chạy
    mock_p = MagicMock()
    mock_p.is_alive.return_value = True
    master.workers_config[worker_id]["process"] = mock_p
    
    # Bật health check tạm thời (chạy 1 lần trong luồng chính để test)
    master._chay_health_check = True
    
    # Tránh vòng lặp vô tận, mock sleep để ném exception thoát vòng lặp
    with patch("time.sleep", side_effect=InterruptedError):
        try:
            master._kiem_tra_suc_khoe()
        except InterruptedError:
            pass
            
    # Kết quả là worker_id phải có mặt trong danh_sach_active
    assert worker_id in master.danh_sach_active

@patch("urllib.request.urlopen")
def test_nut_chinh_auto_healing(mock_urlopen):
    master = NutChinh()
    master.them_worker(port=18081, bo_xu_ly=ham_xu_ly_mock)
    worker_id = "127.0.0.1:18081"
    
    # Giả lập process ĐÃ CHẾT
    mock_p = MagicMock()
    mock_p.is_alive.return_value = False
    master.workers_config[worker_id]["process"] = mock_p
    
    # Chúng ta mock hàm _khoi_dong_worker để xem nó có được gọi không
    with patch.object(master, "_khoi_dong_worker") as mock_khoi_dong:
        master._chay_health_check = True
        with patch("time.sleep", side_effect=InterruptedError):
            try:
                master._kiem_tra_suc_khoe()
            except InterruptedError:
                pass
                
        # Phải gọi _khoi_dong_worker để cứu sống (Auto-healing)
        mock_khoi_dong.assert_called_once_with(worker_id)
