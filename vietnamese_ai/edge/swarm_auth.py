"""Mô-đun Xác Thực & Kết Nối An Toàn Cụm Swarm CPU (Swarm Zero-Trust Auth & Tailscale Integration).

Cung cấp các cơ chế:
1. Tự động nhận diện dải IP Tailscale WireGuard Mesh (100.64.0.0/10).
2. Xác thực bắt tay Thách thức - Phản hồi (HMAC-SHA256 Challenge-Response) bằng Mã PIN 6 số.
3. Chống tấn công vét cạn (Anti-Brute-Force Rate Limiting).
4. Cơ chế Phê duyệt 1 chạm (One-Click Approval Queue) cho Web UI EvoNet OS và CLI.
5. Quản lý phiên suy luận được ký số (Session Token Management).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import secrets
import shutil
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional


def phat_hien_ip_tailscale() -> Optional[str]:
    """
    Tự động phát hiện địa chỉ IP của thiết bị trong mạng Tailscale.

    Tailscale sử dụng dải địa chỉ Carrier-Grade NAT (CGNAT) chính thức:
    100.64.0.0/10 (từ 100.64.0.0 đến 100.127.255.255).

    Trả về:
        Địa chỉ IPv4 Tailscale nếu tìm thấy (vd: '100.85.12.1'), ngược lại trả về None.
    """
    # Cách 1: Thử gọi lệnh CLI của Tailscale nếu được cài đặt trên máy
    if shutil.which("tailscale"):
        try:
            res = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            ip_ts = res.stdout.strip()
            if ip_ts and _la_ip_tailscale_hop_le(ip_ts):
                return ip_ts
        except Exception:
            pass

    # Cách 2: Quét qua các card mạng nội bộ qua 'ip -4 addr' trên Linux/Unix
    if shutil.which("ip"):
        try:
            res = subprocess.run(
                ["ip", "-4", "-o", "addr", "show"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            for line in res.stdout.splitlines():
                parts = line.split()
                for p in parts:
                    if "/" in p:
                        raw_ip = p.split("/")[0]
                        if _la_ip_tailscale_hop_le(raw_ip):
                            return raw_ip
        except Exception:
            pass

    # Cách 3: Quét qua socket getaddrinfo và hostname
    try:
        ten_host = socket.gethostname()
        thong_tin_ip = socket.getaddrinfo(ten_host, None)
        for muc in thong_tin_ip:
            ip = muc[4][0]
            if isinstance(ip, str) and _la_ip_tailscale_hop_le(ip):
                return ip
    except Exception:
        pass

    # Kiểm tra biến môi trường dự phòng (dành cho container/môi trường kiểm thử)
    env_ip = os.environ.get("TAILSCALE_IP")
    if env_ip and _la_ip_tailscale_hop_le(env_ip):
        return env_ip

    return None


def _la_ip_tailscale_hop_le(ip: str) -> bool:
    """Kiểm tra một chuỗi IP có thuộc dải Tailscale CGNAT (100.64.0.0/10) hay không."""
    mau_ip = r"^100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.\d{1,3}\.\d{1,3}$"
    return bool(re.match(mau_ip, ip.strip()))


def tao_ma_pin_ngau_nhien(do_dai: int = 6) -> str:
    """Sinh mã PIN ngẫu nhiên an toàn mật mã học (mặc định 6 chữ số)."""
    gioi_han = 10**do_dai
    so_ngau_nhien = secrets.randbelow(gioi_han)
    return f"{so_ngau_nhien:0{do_dai}d}"


class XacThucCumSwarm:
    """
    Bộ Quản Lý Xác Thực Zero-Trust Cho Cụm Swarm CPU:
    - Bắt tay Thách thức - Phản hồi (HMAC-SHA256 Challenge-Response).
    - Bảo vệ chống tấn công vét cạn (Brute-Force Rate Limiting).
    - Quản lý phiên suy luận và hàng đợi phê duyệt 1 chạm.
    """

    def __init__(
        self,
        ma_pin: Optional[str] = None,
        yeu_cau_phe_duyet: bool = False,
        so_lan_sai_toi_da: int = 5,
        thoi_gian_khoa_giay: float = 900.0,
    ):
        self.ma_pin = str(ma_pin) if ma_pin is not None else tao_ma_pin_ngau_nhien(6)
        self.yeu_cau_phe_duyet = yeu_cau_phe_duyet
        self.so_lan_sai_toi_da = so_lan_sai_toi_da
        self.thoi_gian_khoa_giay = thoi_gian_khoa_giay

        # IP -> timestamp hết hạn khóa
        self._ip_bi_khoa: Dict[str, float] = {}
        # IP -> Danh sách timestamp các lần thử sai trong 60 giây gần nhất
        self._lich_su_that_bai: Dict[str, List[float]] = {}
        # Session token -> Thông tin nút được ủy quyền
        self._phien_hoat_dong: Dict[str, Dict[str, Any]] = {}
        # nut_id -> Thông tin chờ duyệt kèm asyncio.Future
        self._hang_doi_phe_duyet: Dict[str, Dict[str, Any]] = {}

    def kiem_tra_ip_bi_khoa(self, ip: str) -> bool:
        """Kiểm tra xem địa chỉ IP này có đang nằm trong danh sách đen tạm thời hay không."""
        bay_gio = time.time()
        het_han = self._ip_bi_khoa.get(ip)
        if het_han is not None:
            if bay_gio < het_han:
                return True
            # Đã hết thời gian phạt
            del self._ip_bi_khoa[ip]
            self._lich_su_that_bai.pop(ip, None)
        return False

    def ghi_nhan_that_bai(self, ip: str) -> bool:
        """
        Ghi nhận một lần xác thực thất bại từ IP.
        Trả về True nếu IP này đã bị khóa do vượt ngưỡng cho phép.
        """
        bay_gio = time.time()
        ds = self._lich_su_that_bai.setdefault(ip, [])
        # Chỉ giữ lại các lần thử trong 60 giây qua
        ds = [t for t in ds if bay_gio - t <= 60.0]
        ds.append(bay_gio)
        self._lich_su_that_bai[ip] = ds

        if len(ds) >= self.so_lan_sai_toi_da:
            self._ip_bi_khoa[ip] = bay_gio + self.thoi_gian_khoa_giay
            return True
        return False

    def xoa_lich_su_that_bai(self, ip: str) -> None:
        """Xóa lịch sử thất bại của IP khi xác thực thành công."""
        self._lich_su_that_bai.pop(ip, None)
        self._ip_bi_khoa.pop(ip, None)

    def tao_thach_thuc(self) -> str:
        """Tạo một chuỗi số ngẫu nhiên ngẫu nhiên (Nonce) dùng cho một lượt bắt tay duy nhất."""
        return secrets.token_hex(16)

    def ky_thach_thuc(self, nonce: str, pin: Optional[str] = None) -> str:
        """
        Tạo chữ ký số phản hồi thách thức bằng mã băm HMAC-SHA256 với mã PIN.
        """
        pin_dung = pin if pin is not None else self.ma_pin
        key = pin_dung.encode("utf-8")
        msg = nonce.encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).hexdigest()

    def xac_minh_thach_thuc(self, nonce: str, chu_ky_nhan_duoc: str, pin: Optional[str] = None) -> bool:
        """
        Xác minh chữ ký thách thức từ máy đối tác qua hmac.compare_digest chống timing attack.
        """
        chu_ky_mong_doi = self.ky_thach_thuc(nonce, pin=pin)
        return hmac.compare_digest(chu_ky_mong_doi, chu_ky_nhan_duoc)

    def tao_session_token(self, nut_id: str, ip: str = "") -> str:
        """Cấp Session Token ủy quyền phiên suy luận có giá trị bảo mật cao."""
        token = secrets.token_hex(24)
        self._phien_hoat_dong[token] = {
            "nut_id": nut_id,
            "ip": ip,
            "tao_luc": time.time(),
        }
        return token

    def xac_minh_session_token(self, token: str) -> bool:
        """Kiểm tra tính hợp lệ của token phiên suy luận."""
        return token in self._phien_hoat_dong

    def thu_hoi_session_token(self, token: str) -> None:
        """Thu hồi session token khi ngắt kết nối."""
        self._phien_hoat_dong.pop(token, None)

    def dang_ky_cho_duyet(
        self,
        nut_id: str,
        ip: str,
        thong_tin: Dict[str, Any],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> asyncio.Future:
        """Đưa thiết bị mới vào hàng đợi phê duyệt 1 chạm của chủ máy."""
        vong_lap = loop or asyncio.get_event_loop()
        future: asyncio.Future = vong_lap.create_future()
        self._hang_doi_phe_duyet[nut_id] = {
            "nut_id": nut_id,
            "ip": ip,
            "thong_tin": thong_tin,
            "thoi_gian": time.time(),
            "future": future,
        }
        return future

    def phe_duyet_nut(self, nut_id: str, cho_phep: bool) -> bool:
        """Chủ máy phê duyệt (True) hoặc từ chối (False) yêu cầu gia nhập của thiết bị."""
        muc = self._hang_doi_phe_duyet.pop(nut_id, None)
        if not muc:
            return False
        fut: asyncio.Future = muc["future"]
        if not fut.done():
            fut.set_result(cho_phep)
        return True

    def lay_danh_sach_cho_duyet(self) -> List[Dict[str, Any]]:
        """Lấy danh sách các thiết bị đang đợi phê duyệt trên giao diện Web."""
        return [
            {
                "nut_id": k,
                "ip": v["ip"],
                "thong_tin": v["thong_tin"],
                "thoi_gian": v["thoi_gian"],
            }
            for k, v in self._hang_doi_phe_duyet.items()
        ]
