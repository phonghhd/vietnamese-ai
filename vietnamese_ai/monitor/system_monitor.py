"""
TheoDoiHeThong (System Monitor) - Đọc thông số CPU/RAM trực tiếp từ Linux /proc
nhằm đảm bảo Zero-Dependency (không cần cài psutil).
"""

import os
import time
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("V-Monitor")

class TheoDoiHeThong:
    """Theo dõi thông số CPU và RAM (Chỉ hoạt động trên Linux)."""

    def __init__(self):
        self._ho_tro_proc = os.path.exists("/proc/stat") and os.path.exists("/proc/meminfo")
        if not self._ho_tro_proc:
            logger.warning("[TheoDoiHeThong] Tính năng chỉ hỗ trợ trên Linux. Sẽ trả về 0 trên Windows/macOS.")
        
        self._cpu_truoc: Tuple[int, int] = (0, 0) # (idle, total)
        self._cap_nhat_cpu_tick_truoc()

    def _cap_nhat_cpu_tick_truoc(self):
        """Đọc và lưu lại tick của CPU để tính phần trăm."""
        if not self._ho_tro_proc:
            return
            
        try:
            with open("/proc/stat", "r") as f:
                dong_dau = f.readline()
                
            phan_tu = dong_dau.split()[1:] # Bỏ chữ "cpu"
            gia_tri = [int(x) for x in phan_tu]
            
            # idle = idle + iowait
            idle = gia_tri[3] + gia_tri[4]
            # total = user + nice + system + idle + iowait + irq + softirq + steal
            total = sum(gia_tri)
            
            self._cpu_truoc = (idle, total)
        except Exception:
            pass

    def lay_thong_tin(self) -> Dict[str, Any]:
        """Lấy phần trăm CPU và RAM (MB)."""
        if not self._ho_tro_proc:
            return {
                "cpu_phan_tram": 0.0,
                "ram_tong_mb": 0,
                "ram_su_dung_mb": 0
            }

        # --- 1. Lấy CPU ---
        cpu_phan_tram = 0.0
        try:
            with open("/proc/stat", "r") as f:
                dong_dau = f.readline()
                
            phan_tu = dong_dau.split()[1:]
            gia_tri = [int(x) for x in phan_tu]
            
            idle_hien_tai = gia_tri[3] + gia_tri[4]
            total_hien_tai = sum(gia_tri)
            
            idle_delta = idle_hien_tai - self._cpu_truoc[0]
            total_delta = total_hien_tai - self._cpu_truoc[1]
            
            if total_delta > 0:
                # Usage = (Total - Idle) / Total
                cpu_phan_tram = 100.0 * (total_delta - idle_delta) / total_delta
                
            self._cpu_truoc = (idle_hien_tai, total_hien_tai)
        except Exception as e:
            logger.debug(f"Lỗi đọc CPU: {e}")

        # --- 2. Lấy RAM ---
        ram_tong_mb = 0
        ram_su_dung_mb = 0
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
                
            mem_dict = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    # Value format "123456 kB"
                    val = int(parts[1].strip().split()[0]) 
                    mem_dict[key] = val
                    
            if "MemTotal" in mem_dict and "MemAvailable" in mem_dict:
                ram_tong_mb = mem_dict["MemTotal"] // 1024
                # Used = Total - Available
                ram_su_dung_mb = (mem_dict["MemTotal"] - mem_dict["MemAvailable"]) // 1024
        except Exception as e:
            logger.debug(f"Lỗi đọc RAM: {e}")

        return {
            "cpu_phan_tram": round(cpu_phan_tram, 1),
            "ram_tong_mb": ram_tong_mb,
            "ram_su_dung_mb": ram_su_dung_mb
        }
