"""Sổ Cái Giao Dịch Chuỗi Băm SHA-256 (`TokenLedger`) & Bộ Điều Phối P2P Có Kiểm Tra TCP (`P2PTracker`)."""

from __future__ import annotations

import hashlib
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TokenLedger:
    """
    Sổ cái Token có nhật ký giao dịch nối chuỗi băm SHA-256 (Tamper-Evident Hash-Chained Ledger):
    - Lưu số dư thực tế, hỗ trợ ghi bền vững ra tệp JSONL khi cung cấp `duong_dan_so_cai`.
    - Cho phép kiểm toán toàn vẹn toàn bộ lịch sử giao dịch bằng `xac_minh_toan_ven_so_cai()`.
    """

    def __init__(self, duong_dan_so_cai: Optional[Union[str, Path]] = None) -> None:
        self.so_du: Dict[str, float] = {}
        self.lich_su_giao_dich: List[Dict[str, Any]] = []
        self._duong_dan = Path(duong_dan_so_cai) if duong_dan_so_cai else None
        if self._duong_dan and self._duong_dan.exists():
            self._tai_tu_tep()

    @staticmethod
    def _tinh_hash_giao_dich(ban_ghi: Dict[str, Any]) -> str:
        """Tính băm SHA-256 xác định liên kết bản ghi hiện tại với `prev_hash`."""
        payload = json.dumps(
            {
                "index": ban_ghi["index"],
                "prev_hash": ban_ghi["prev_hash"],
                "node_id": ban_ghi["node_id"],
                "so_luong": ban_ghi["so_luong"],
                "ly_do": ban_ghi["ly_do"],
                "thoi_gian": ban_ghi["thoi_gian"],
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _tai_tu_tep(self) -> None:
        """Nạp lại lịch sử giao dịch và tính lại số dư từ tệp JSONL bền vững."""
        if not self._duong_dan:
            return
        for dong in self._duong_dan.read_text(encoding="utf-8").splitlines():
            dong_s = dong.strip()
            if not dong_s:
                continue
            tx = json.loads(dong_s)
            self.lich_su_giao_dich.append(tx)
            nid = str(tx["node_id"])
            self.so_du[nid] = self.so_du.get(nid, 0.0) + float(tx["so_luong"])

    def _ghi_giao_dich(self, node_id: str, so_luong: float, ly_do: str) -> Dict[str, Any]:
        """Ghi nhận giao dịch nối chuỗi băm SHA-256 và đồng bộ đĩa nếu bật lưu bền vững."""
        prev_hash = self.lich_su_giao_dich[-1]["tx_hash"] if self.lich_su_giao_dich else "0" * 64
        tx: Dict[str, Any] = {
            "index": len(self.lich_su_giao_dich),
            "prev_hash": prev_hash,
            "node_id": node_id,
            "so_luong": float(so_luong),
            "ly_do": ly_do,
            "thoi_gian": round(time.time(), 6),
        }
        tx["tx_hash"] = self._tinh_hash_giao_dich(tx)
        self.lich_su_giao_dich.append(tx)
        self.so_du[node_id] = self.so_du.get(node_id, 0.0) + float(so_luong)

        if self._duong_dan is not None:
            self._duong_dan.parent.mkdir(parents=True, exist_ok=True)
            with open(self._duong_dan, "a", encoding="utf-8") as f:
                f.write(json.dumps(tx, ensure_ascii=False) + "\n")
        return tx

    def xac_minh_toan_ven_so_cai(self) -> bool:
        """Xác minh toàn bộ chuỗi băm SHA-256 của lịch sử giao dịch không bị chỉnh sửa trái phép."""
        expected_prev = "0" * 64
        for idx, tx in enumerate(self.lich_su_giao_dich):
            if tx.get("index") != idx or tx.get("prev_hash") != expected_prev:
                return False
            if tx.get("tx_hash") != self._tinh_hash_giao_dich(tx):
                return False
            expected_prev = str(tx["tx_hash"])
        return True

    def thuong_token(self, node_id: str, so_luong: float = 10.0) -> Dict[str, Any]:
        """Thưởng Token và ghi nhận giao dịch băm SHA-256 khi Node hoàn thành tốt tác vụ."""
        return self._ghi_giao_dich(node_id, so_luong, "hoan_thanh_suy_luan")

    def thuong_token_danh_gia(self, user_id: str, chat_luong_danh_gia: float = 1.0) -> Dict[str, Any]:
        """Thưởng Token cho đóng góp gán nhãn RLHF/DPO có kiểm chứng chuỗi băm."""
        phan_thuong = 5.0 * float(chat_luong_danh_gia)
        return self._ghi_giao_dich(user_id, phan_thuong, "danh_gia_rlhf")

    def lay_so_du(self, node_id: str) -> float:
        """Trả về số dư hiện tại của `node_id`."""
        return self.so_du.get(node_id, 0.0)


class P2PTracker:
    """
    Sổ đăng ký & điều phối Edge Node hỗ trợ kiểm tra sức khỏe TCP thực tế và lưu bền vững danh bạ nút.
    """

    def __init__(self, duong_dan_danh_ba: Optional[Union[str, Path]] = None) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self._duong_dan = Path(duong_dan_danh_ba) if duong_dan_danh_ba else None

    def _luu_danh_ba(self) -> None:
        """Lưu metadata các nút đã đăng ký ra tệp JSON nếu cấu hình `duong_dan_danh_ba`."""
        if self._duong_dan is None:
            return
        self._duong_dan.parent.mkdir(parents=True, exist_ok=True)
        du_lieu = {
            nid: {
                "toc_do": info["toc_do"],
                "trang_thai": info["trang_thai"],
                "endpoint": info.get("endpoint"),
                "cap_nhat_cuoi": info.get("cap_nhat_cuoi"),
            }
            for nid, info in self.nodes.items()
        }
        self._duong_dan.write_text(json.dumps(du_lieu, indent=2, ensure_ascii=False), encoding="utf-8")

    def dang_ky_node(
        self,
        node_id: str,
        engine: Any,
        toc_do_du_kien: float = 1.0,
        endpoint: Optional[str] = None,
    ) -> None:
        """Đăng ký một Edge Node kèm thông tin tốc độ và địa chỉ mạng TCP (nếu có)."""
        ep = endpoint or getattr(engine, "endpoint_url", None)
        self.nodes[node_id] = {
            "engine": engine,
            "toc_do": float(toc_do_du_kien),
            "trang_thai": "san_sang",
            "endpoint": ep,
            "cap_nhat_cuoi": time.time(),
        }
        self._luu_danh_ba()

    def huy_dang_ky(self, node_id: str) -> None:
        """Hủy đăng ký nút khỏi bộ điều phối."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._luu_danh_ba()

    def kiem_tra_suc_khoe_node(self, node_id: str, timeout: float = 1.0) -> bool:
        """Kiểm tra kết nối TCP thực tế tới endpoint của nút (hoặc tính khả dụng cục bộ nếu không có endpoint)."""
        info = self.nodes.get(node_id)
        if info is None:
            return False
        endpoint = info.get("endpoint")
        if not endpoint:
            return info.get("engine") is not None
        try:
            clean = str(endpoint).replace("tcp://", "").replace("http://", "").replace("https://", "")
            host_port = clean.split("/")[0]
            host, port_s = host_port.rsplit(":", 1)
            with socket.create_connection((host, int(port_s)), timeout=timeout):
                info["trang_thai"] = "san_sang"
                info["cap_nhat_cuoi"] = time.time()
                return True
        except Exception:
            info["trang_thai"] = "mat_ket_noi"
            return False

    def tim_node_tot_nhat(self) -> Optional[Any]:
        """Tìm Edge Node đang sẵn sàng có điểm tốc độ cao nhất."""
        node_tot_nhat = None
        max_toc_do = -1.0

        for thong_tin in self.nodes.values():
            if thong_tin["trang_thai"] == "san_sang" and thong_tin["toc_do"] > max_toc_do:
                max_toc_do = thong_tin["toc_do"]
                node_tot_nhat = thong_tin["engine"]

        return node_tot_nhat
