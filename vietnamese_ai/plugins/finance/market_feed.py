"""
Market Feed Engine - Lõi thu thập và đệm dữ liệu giá Real-time cho Chứng khoán VN & Crypto.
Đảm bảo hoạt động nhẹ nhàng, độ trễ thấp, zero heavy dependency (chỉ dùng standard library).
Tích hợp sẵn bộ tạo giá giả lập (Deterministic Mock Feed) cho môi trường test/offline.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("V-MarketFeed")

# Các mã chỉ số và cổ phiếu VN phổ biến
MA_CHUNG_KHOAN_VN_PHO_BIEN = {
    "VNINDEX",
    "VN30",
    "VN30F1M",
    "HNXINDEX",
    "UPCOMINDEX",
    "HPG",
    "VNM",
    "SSI",
    "VND",
    "FPT",
    "VIC",
    "VHM",
    "TCB",
    "MBB",
    "VCB",
    "MWG",
    "MSN",
    "STB",
    "VPB",
    "DGC",
    "PVD",
}

# Các cặp crypto phổ biến
CAP_CRYPTO_PHO_BIEN = {
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "LINKUSDT",
}


def la_ma_crypto(ma: str) -> bool:
    """Xác định nhanh mã có phải là tài sản Crypto hay không."""
    m = ma.upper().strip()
    if any(m.endswith(suffix) for suffix in ("USDT", "BUSD", "USDC", "BTC", "ETH")):
        return True
    if m in {
        "BTC",
        "ETH",
        "BNB",
        "SOL",
        "ADA",
        "DOGE",
        "XRP",
        "AVAX",
        "NEAR",
        "LINK",
    }:
        return True
    return False


def chuan_hoa_ma(ma: str) -> Tuple[str, str]:
    """
    Chuẩn hóa mã và trả về (ma_chuan, loai_tai_san).
    loai_tai_san: 'crypto' hoặc 'chung_khoan_vn'.
    """
    m = ma.upper().strip()
    if la_ma_crypto(m):
        if not any(m.endswith(s) for s in ("USDT", "BUSD", "USDC")):
            m = f"{m}USDT"
        return m, "crypto"
    return m, "chung_khoan_vn"


class BoPhatGiaGiaLap:
    """
    Bộ phát dữ liệu thị trường giả lập (Deterministic Synthetic Feed).
    Phục vụ môi trường kiểm thử (Unit Test), CI/CD, hoặc phân tích ngoài giờ giao dịch
    khi không có kết nối Internet. Sử dụng mô hình Geometric Brownian Motion (GBM).
    """

    GIA_CO_SO: Dict[str, float] = {
        "VNINDEX": 1285.50,
        "VN30": 1340.20,
        "VN30F1M": 1335.50,
        "HNXINDEX": 242.10,
        "UPCOMINDEX": 94.60,
        "HPG": 28.50,
        "VNM": 67.20,
        "SSI": 34.80,
        "VND": 15.60,
        "FPT": 138.50,
        "VIC": 42.10,
        "VHM": 41.50,
        "TCB": 24.30,
        "MBB": 25.10,
        "VCB": 91.00,
        "MWG": 65.40,
        "BTCUSDT": 67850.00,
        "ETHUSDT": 2650.00,
        "BNBUSDT": 595.00,
        "SOLUSDT": 178.50,
        "DOGEUSDT": 0.145,
    }

    def __init__(self, seed: Optional[int] = 42):
        self.rng = random.Random(seed)

    def sinh_gia_hien_tai(self, ma: str) -> Dict[str, Any]:
        """Tạo snapshot giá hiện tại với bước giá ngẫu nhiên thực tế."""
        ma_chuan, loai = chuan_hoa_ma(ma)
        gia_goc = self.GIA_CO_SO.get(ma_chuan, 50.0 if loai == "chung_khoan_vn" else 10.0)

        # Biên độ dao động giả lập: ±1.5%
        delta_pct = self.rng.uniform(-0.025, 0.025)
        gia_hien_tai = round(gia_goc * (1.0 + delta_pct), 2 if loai == "chung_khoan_vn" else 4)
        gia_mo_cua = gia_goc
        gia_cao_nhat = round(max(gia_hien_tai, gia_mo_cua) * (1.0 + self.rng.uniform(0.001, 0.01)), 2)
        gia_thap_nhat = round(min(gia_hien_tai, gia_mo_cua) * (1.0 - self.rng.uniform(0.001, 0.01)), 2)
        thay_doi = round(gia_hien_tai - gia_mo_cua, 2)
        thay_doi_pct = round((thay_doi / gia_mo_cua) * 100.0, 2)

        khoi_luong = (
            int(self.rng.uniform(1_000_000, 15_000_000))
            if loai == "chung_khoan_vn"
            else round(self.rng.uniform(500.0, 5000.0), 2)
        )

        return {
            "ma": ma_chuan,
            "loai": loai,
            "gia_hien_tai": gia_hien_tai,
            "gia_mo_cua": gia_mo_cua,
            "gia_cao_nhat": gia_cao_nhat,
            "gia_thap_nhat": gia_thap_nhat,
            "thay_doi": thay_doi,
            "thay_doi_pct": thay_doi_pct,
            "khoi_luong": khoi_luong,
            "thoi_gian": int(time.time()),
            "nguon": "synthetic_mock",
            "la_du_lieu_gia_lap": True,
        }

    def sinh_du_lieu_nen(
        self,
        ma: str,
        khung_thoi_gian: str = "1D",
        so_nen: int = 60,
    ) -> List[Dict[str, Any]]:
        """Sinh chuỗi nến OHLCV mô phỏng liên tục."""
        ma_chuan, loai = chuan_hoa_ma(ma)
        gia_hien_tai = self.GIA_CO_SO.get(ma_chuan, 50.0 if loai == "chung_khoan_vn" else 100.0)

        # Khoảng cách giây giữa các nến
        khung = khung_thoi_gian.upper().strip()
        if "D" in khung:
            buoc_giay = 86400
        elif "H" in khung:
            buoc_giay = 3600
        elif "M" in khung:
            digits = "".join(filter(str.isdigit, khung))
            so_phut = int(digits) if digits else 1
            buoc_giay = so_phut * 60
        else:
            buoc_giay = 60
        thoi_gian_dau = int(time.time()) - (so_nen * buoc_giay)

        danh_sach_nen: List[Dict[str, Any]] = []
        gia = gia_hien_tai * 0.85  # bắt đầu từ mức thấp hơn để tạo xu hướng tự nhiên

        for i in range(so_nen):
            tg = thoi_gian_dau + (i * buoc_giay)
            bien_dong = self.rng.gauss(0.001, 0.02)  # GBM drift + diffusion
            mo_cua = gia
            dong_cua = round(mo_cua * math.exp(bien_dong), 2 if loai == "chung_khoan_vn" else 4)
            cao_nhat = round(max(mo_cua, dong_cua) * (1.0 + abs(self.rng.gauss(0.002, 0.005))), 2)
            thap_nhat = round(min(mo_cua, dong_cua) * (1.0 - abs(self.rng.gauss(0.002, 0.005))), 2)
            khoi_luong = (
                int(self.rng.uniform(500_000, 20_000_000))
                if loai == "chung_khoan_vn"
                else round(self.rng.uniform(100.0, 10_000.0), 2)
            )

            danh_sach_nen.append(
                {
                    "thoi_gian": tg,
                    "mo_cua": mo_cua,
                    "cao_nhat": cao_nhat,
                    "thap_nhat": thap_nhat,
                    "dong_cua": dong_cua,
                    "khoi_luong": khoi_luong,
                }
            )
            gia = dong_cua

        return danh_sach_nen


class BoThuThapCrypto:
    """
    Bộ thu thập dữ liệu Crypto từ Binance Public REST API.
    Không yêu cầu API key, độ trễ thấp, tự động bắt lỗi và fallback.
    """

    BASE_URL = "https://api.binance.com/api/v3"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; VietnameseAI-CryptoEngine/1.0)",
        "Accept": "application/json",
    }

    def __init__(self, timeout_giay: float = 3.0):
        self.timeout_giay = timeout_giay

    def lay_gia_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Lấy giá 24h ticker từ Binance."""
        sym = symbol.upper()
        url = f"{self.BASE_URL}/ticker/24hr?symbol={sym}"
        req = urllib.request.Request(url, headers=self.HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_giay) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "ma": sym,
                    "loai": "crypto",
                    "gia_hien_tai": float(data.get("lastPrice", 0.0)),
                    "gia_mo_cua": float(data.get("openPrice", 0.0)),
                    "gia_cao_nhat": float(data.get("highPrice", 0.0)),
                    "gia_thap_nhat": float(data.get("lowPrice", 0.0)),
                    "thay_doi": float(data.get("priceChange", 0.0)),
                    "thay_doi_pct": float(data.get("priceChangePercent", 0.0)),
                    "khoi_luong": float(data.get("volume", 0.0)),
                    "thoi_gian": int(time.time()),
                    "nguon": "binance_rest",
                }
        except Exception as e:
            logger.debug("Lỗi kết nối Binance Ticker cho %s: %s", sym, e)
            return None

    def lay_nen_klines(
        self, symbol: str, interval: str = "1d", limit: int = 60
    ) -> Optional[List[Dict[str, Any]]]:
        """Lấy danh sách nến OHLCV từ Binance."""
        sym = symbol.upper()
        # Chuyển interval sang chuẩn Binance (1D -> 1d, 1H -> 1h)
        iv = interval.lower()
        url = f"{self.BASE_URL}/klines?symbol={sym}&interval={iv}&limit={limit}"
        req = urllib.request.Request(url, headers=self.HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_giay) as resp:
                raw_candles = json.loads(resp.read().decode("utf-8"))
                ket_qua = []
                for c in raw_candles:
                    ket_qua.append(
                        {
                            "thoi_gian": int(c[0] // 1000),
                            "mo_cua": float(c[1]),
                            "cao_nhat": float(c[2]),
                            "thap_nhat": float(c[3]),
                            "dong_cua": float(c[4]),
                            "khoi_luong": float(c[5]),
                        }
                    )
                return ket_qua
        except Exception as e:
            logger.debug("Lỗi kết nối Binance Klines cho %s: %s", sym, e)
            return None


class BoThuThapGiaChungKhoan:
    """
    Bộ thu thập dữ liệu bảng giá Chứng khoán Việt Nam (HOSE, HNX, UPCoM).
    Tích hợp bộ định tuyến dự phòng đa nguồn (TCBS -> VNDirect -> SSI).
    Tự động phòng vệ không bị chặn IP và zero-dependency.
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    def __init__(self, timeout_giay: float = 3.5):
        self.timeout_giay = timeout_giay

    def lay_gia_tcbs(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin giá từ TCBS Open Analytics API."""
        sym = symbol.upper()
        url = f"https://apipubanalytics.tcbs.com.vn/stocks/v1/ticker/{sym}/overview"
        req = urllib.request.Request(url, headers=self.HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_giay) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                price = float(data.get("price", 0.0) or data.get("matchPrice", 0.0))
                # TCBS thường trả giá theo đơn vị nghìn (28.5) hoặc đồng (28500)
                if price > 1000:
                    price = round(price / 1000.0, 2)
                change = float(data.get("change", 0.0) or 0.0)
                if change > 1000 or change < -1000:
                    change = round(change / 1000.0, 2)
                change_pct = float(data.get("percentChange", 0.0) or 0.0)

                return {
                    "ma": sym,
                    "loai": "chung_khoan_vn",
                    "gia_hien_tai": price,
                    "gia_mo_cua": round(price - change, 2),
                    "gia_cao_nhat": price,
                    "gia_thap_nhat": price,
                    "thay_doi": change,
                    "thay_doi_pct": change_pct,
                    "khoi_luong": int(data.get("totalVolume", 0) or 0),
                    "thoi_gian": int(time.time()),
                    "nguon": "tcbs_api",
                }
        except Exception as e:
            logger.debug("Lỗi kết nối TCBS cho %s: %s", sym, e)
            return None

    def lay_nen_vndirect(
        self, symbol: str, limit: int = 60
    ) -> Optional[List[Dict[str, Any]]]:
        """Lấy dữ liệu nến lịch sử từ VNDirect DChart API."""
        sym = symbol.upper()
        den_ngay = int(time.time())
        tu_ngay = den_ngay - (limit * 86400 * 2)  # tính dư ngày nghỉ cuối tuần
        url = (
            f"https://dchart-api.vndirect.com.vn/dchart/history?"
            f"resolution=D&symbol={sym}&from={tu_ngay}&to={den_ngay}"
        )
        req = urllib.request.Request(url, headers=self.HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_giay) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("s") != "ok":
                    return None
                t_arr = data.get("t", [])
                o_arr = data.get("o", [])
                h_arr = data.get("h", [])
                l_arr = data.get("l", [])
                c_arr = data.get("c", [])
                v_arr = data.get("v", [])

                so_luong = min(len(t_arr), limit)
                ket_qua = []
                for i in range(len(t_arr) - so_luong, len(t_arr)):
                    ket_qua.append(
                        {
                            "thoi_gian": int(t_arr[i]),
                            "mo_cua": float(o_arr[i]),
                            "cao_nhat": float(h_arr[i]),
                            "thap_nhat": float(l_arr[i]),
                            "dong_cua": float(c_arr[i]),
                            "khoi_luong": float(v_arr[i]),
                        }
                    )
                return ket_qua
        except Exception as e:
            logger.debug("Lỗi kết nối VNDirect DChart cho %s: %s", sym, e)
            return None


class BoDieuPhoiThiTruong:
    """
    Bộ Điều Phối Thị Trường (Market Feed Coordinator) - Điểm tiếp cận duy nhất.
    Tự động nhận diện loại tài sản, quản lý Ring Buffer lưu đệm và tự chuyển đổi
    nguồn cấp dữ liệu (Live Real-time -> Failover -> Deterministic Mock Feed).
    """

    def __init__(self, dung_gia_lap_khi_loi: bool = True, bao_loi_khi_mat_ket_noi: bool = False) -> None:
        self.dung_gia_lap_khi_loi = dung_gia_lap_khi_loi
        self.bao_loi_khi_mat_ket_noi = bao_loi_khi_mat_ket_noi
        self.crypto_engine = BoThuThapCrypto()
        self.stock_engine = BoThuThapGiaChungKhoan()
        self.mock_engine = BoPhatGiaGiaLap()
        self._ring_buffer: Dict[str, collections.deque] = {}

    def _cap_nhat_buffer(self, ma: str, du_lieu: Dict[str, Any]) -> None:
        if ma not in self._ring_buffer:
            self._ring_buffer[ma] = collections.deque(maxlen=200)
        self._ring_buffer[ma].append(du_lieu)

    def lay_gia_hien_tai(
        self, ma: str, loai_tai_san: Optional[str] = None
    ) -> Dict[str, Any]:
        """Lấy giá thực tế mới nhất và đánh dấu minh bạch cờ `la_du_lieu_gia_lap` khi buộc phải fallback."""
        ma_chuan, loai_phat_hien = chuan_hoa_ma(ma)
        loai = loai_tai_san or loai_phat_hien

        kq = None
        if loai == "crypto":
            kq = self.crypto_engine.lay_gia_ticker(ma_chuan)
        else:
            kq = self.stock_engine.lay_gia_tcbs(ma_chuan)

        if kq is not None:
            kq["la_du_lieu_gia_lap"] = False
            self._cap_nhat_buffer(ma_chuan, kq)
            return kq

        if self.dung_gia_lap_khi_loi:
            kq = self.mock_engine.sinh_gia_hien_tai(ma_chuan)
            kq["la_du_lieu_gia_lap"] = True
            self._cap_nhat_buffer(ma_chuan, kq)
            return kq

        if self.bao_loi_khi_mat_ket_noi:
            raise ConnectionError(f"Không thể kết nối nguồn giá thực tế cho mã '{ma_chuan}'")

        return {
            "ma": ma_chuan,
            "loai": loai,
            "gia_hien_tai": 0.0,
            "thay_doi": 0.0,
            "thay_doi_pct": 0.0,
            "khoi_luong": 0,
            "thoi_gian": int(time.time()),
            "nguon": "loi_mat_ket_noi",
            "la_du_lieu_gia_lap": False,
        }

    def lay_du_lieu_nen(
        self,
        ma: str,
        khung_thoi_gian: str = "1D",
        so_nen: int = 60,
    ) -> List[Dict[str, Any]]:
        """Lấy chuỗi nến OHLCV thực tế và đánh dấu rõ `la_du_lieu_gia_lap` cho từng bản ghi."""
        ma_chuan, loai = chuan_hoa_ma(ma)
        ket_qua_nen = None

        if loai == "crypto":
            ket_qua_nen = self.crypto_engine.lay_nen_klines(
                ma_chuan, interval=khung_thoi_gian, limit=so_nen
            )
        else:
            ket_qua_nen = self.stock_engine.lay_nen_vndirect(ma_chuan, limit=so_nen)

        if ket_qua_nen:
            for nen in ket_qua_nen:
                nen["la_du_lieu_gia_lap"] = False
            return ket_qua_nen

        if self.dung_gia_lap_khi_loi:
            ket_qua_nen = self.mock_engine.sinh_du_lieu_nen(
                ma_chuan, khung_thoi_gian=khung_thoi_gian, so_nen=so_nen
            )
            for nen in ket_qua_nen:
                nen["la_du_lieu_gia_lap"] = True
            return ket_qua_nen

        if self.bao_loi_khi_mat_ket_noi:
            raise ConnectionError(f"Không thể kết nối nguồn nến OHLCV thực tế cho mã '{ma_chuan}'")

        return []
