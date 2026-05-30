"""StudioKeoTha - No-code Studio kéo thả xây dựng ML pipeline."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger

LOAI_NODE = {
    "du_lieu": {
        "ten": "Nguồn dữ liệu",
        "mo_ta": "Tạo hoặc tải dữ liệu",
        "mau": "#4CAF50",
        "dau_ra": ["du_lieu"],
    },
    "tien_xu_ly": {
        "ten": "Tiền xử lý",
        "mo_ta": "Chuẩn hóa, xử lý missing values",
        "mau": "#2196F3",
        "dau_vao": ["du_lieu"],
        "dau_ra": ["du_lieu"],
    },
    "mo_hinh": {
        "ten": "Mô hình",
        "mo_ta": "Huấn luyện mô hình ML",
        "mau": "#FF9800",
        "dau_vao": ["du_lieu"],
        "dau_ra": ["mo_hinh", "du_lieu"],
    },
    "danh_gia": {
        "ten": "Đánh giá",
        "mo_ta": "Đánh giá hiệu suất mô hình",
        "mau": "#9C27B0",
        "dau_vao": ["mo_hinh", "du_lieu"],
        "dau_ra": ["ket_qua"],
    },
    "truc_quan_hoa": {
        "ten": "Trực quan hóa",
        "mo_ta": "Biểu đồ hóa kết quả",
        "mau": "#E91E63",
        "dau_vao": ["ket_qua"],
        "dau_ra": ["bieu_do"],
    },
    "xuat": {
        "ten": "Xuất kết quả",
        "mo_ta": "Lưu mô hình hoặc kết quả",
        "mau": "#607D8B",
        "dau_vao": ["mo_hinh", "ket_qua"],
        "dau_ra": [],
    },
}

TAT_CA_THUAT_TOAN = {
    "phan_loai": [
        "logistic", "knn", "svm", "rung_ngau_nhien",
        "gradient_boosting", "naive_bayes",
    ],
    "hoi_quy": [
        "tuyen_tinh", "ridge", "lasso",
        "rung_ngau_nhien", "gradient_boosting",
    ],
    "phan_cum": ["kmeans", "dbscan", "hierarchical"],
}


class Node:
    """Một node trong pipeline kéo thả."""

    def __init__(
        self,
        ma: str,
        loai: str,
        ten: str,
        vi_tri: Optional[Dict[str, float]] = None,
        tham_so: Optional[Dict[str, Any]] = None,
    ):
        if loai not in LOAI_NODE:
            raise ValueError(
                f"Loại node '{loai}' không hợp lệ. "
                f"Chọn: {', '.join(LOAI_NODE.keys())}"
            )

        self.ma = ma
        self.loai = loai
        self.ten = ten
        self.vi_tri = vi_tri or {"x": 0, "y": 0}
        self.tham_so = tham_so or {}
        self.ket_qua: Optional[Any] = None
        self.da_chay = False
        self.loi: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ma": self.ma,
            "loai": self.loai,
            "ten": self.ten,
            "vi_tri": self.vi_tri,
            "tham_so": self.tham_so,
            "da_chay": self.da_chay,
            "loi": self.loi,
        }


class KetNoi:
    """Kết nối giữa 2 nodes."""

    def __init__(self, ma: str, tu_node: str, den_node: str, cua: str = "du_lieu"):
        self.ma = ma
        self.tu_node = tu_node
        self.den_node = den_node
        self.cua = cua

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ma": self.ma,
            "tu_node": self.tu_node,
            "den_node": self.den_node,
            "cua": self.cua,
        }


TEMPLATE_PIPELINES = {
    "phan_loai_co_ban": {
        "ten": "Phân loại cơ bản",
        "mo_ta": "Pipeline phân loại với tiền xử lý",
        "nodes": [
            {"loai": "du_lieu", "ten": "Dữ liệu mẫu", "tham_so": {"nguon": "mau"}},
            {"loai": "tien_xu_ly", "ten": "Chuẩn hóa Z-Score", "tham_so": {"phuong_phap": "zscore"}},
            {"loai": "mo_hinh", "ten": "Random Forest", "tham_so": {"thuat_toan": "rung_ngau_nhien", "nhiem_vu": "phan_loai"}},
            {"loai": "danh_gia", "ten": "Đánh giá", "tham_so": {"chi_so": "do_chinh_xac"}},
        ],
        "ket_noi": [
            {"tu": 0, "den": 1},
            {"tu": 1, "den": 2},
            {"tu": 2, "den": 3},
        ],
    },
    "hoi_quy_nang_cao": {
        "ten": "Hồi quy nâng cao",
        "mo_ta": "Pipeline hồi quy với feature engineering",
        "nodes": [
            {"loai": "du_lieu", "ten": "Dữ liệu mẫu", "tham_so": {"nguon": "mau", "nhiem_vu": "hoi_quy"}},
            {"loai": "tien_xu_ly", "ten": "Xử lý missing", "tham_so": {"phuong_phap": "trung_vi"}},
            {"loai": "tien_xu_ly", "ten": "Chuẩn hóa Min-Max", "tham_so": {"phuong_phap": "minmax"}},
            {"loai": "mo_hinh", "ten": "Gradient Boosting", "tham_so": {"thuat_toan": "gradient_boosting", "nhiem_vu": "hoi_quy"}},
            {"loai": "danh_gia", "ten": "Đánh giá MSE", "tham_so": {"chi_so": "mse"}},
        ],
        "ket_noi": [
            {"tu": 0, "den": 1},
            {"tu": 1, "den": 2},
            {"tu": 2, "den": 3},
            {"tu": 3, "den": 4},
        ],
    },
    "so_sanh_nhieu_mo_hinh": {
        "ten": "So sánh nhiều mô hình",
        "mo_ta": "So sánh hiệu suất nhiều thuật toán",
        "nodes": [
            {"loai": "du_lieu", "ten": "Dữ liệu", "tham_so": {"nguon": "mau"}},
            {"loai": "tien_xu_ly", "ten": "Chuẩn hóa", "tham_so": {"phuong_phap": "zscore"}},
            {"loai": "mo_hinh", "ten": "Logistic", "tham_so": {"thuat_toan": "logistic", "nhiem_vu": "phan_loai"}},
            {"loai": "mo_hinh", "ten": "Random Forest", "tham_so": {"thuat_toan": "rung_ngau_nhien", "nhiem_vu": "phan_loai"}},
            {"loai": "danh_gia", "ten": "So sánh", "tham_so": {"chi_so": "do_chinh_xac"}},
        ],
        "ket_noi": [
            {"tu": 0, "den": 1},
            {"tu": 1, "den": 2},
            {"tu": 1, "den": 3},
            {"tu": 2, "den": 4},
            {"tu": 3, "den": 4},
        ],
    },
}


class StudioKeoTha:
    """
    No-code Studio kéo thả để xây dựng ML pipeline.

    Tính năng:
    - Kéo thả nodes (data, preprocessing, model, evaluate, export)
    - Kết nối nodes thành pipeline
    - Chạy pipeline
    - Templates có sẵn
    - Lưu/tải pipeline configs

    Sử dụng:
        >>> studio = StudioKeoTha()
        >>> studio.them_node("du_lieu", "Dữ liệu mẫu", tham_so={"nguon": "mau"})
        >>> studio.them_node("mo_hinh", "Logistic", tham_so={"thuat_toan": "logistic"})
        >>> studio.ket_noi("node_0", "node_1")
        >>> ket_qua = studio.chay()
    """

    def __init__(self, ten: str = "Vietnamese AI Studio"):
        self.ten = ten
        self.logger = Logger("StudioKeoTha")
        self._nodes: Dict[str, Node] = {}
        self._ket_noi: List[KetNoi] = []
        self._counter = 0
        self._ket_qua_chay: Dict[str, Any] = {}
        self._da_chay = False

    def _tao_ma_node(self) -> str:
        ma = f"node_{self._counter}"
        self._counter += 1
        return ma

    def _tao_ma_ket_noi(self) -> str:
        return f"conn_{len(self._ket_noi)}"

    def them_node(
        self,
        loai: str,
        ten: str,
        vi_tri: Optional[Dict[str, float]] = None,
        tham_so: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Thêm node vào canvas.

        Args:
            loai: Loại node (du_lieu, tien_xu_ly, mo_hinh, danh_gia, truc_quan_hoa, xuat)
            ten: Tên hiển thị
            vi_tri: Vị trí trên canvas {"x": float, "y": float}
            tham_so: Tham số cho node

        Returns:
            Dict chứa thông tin node
        """
        ma = self._tao_ma_node()
        node = Node(ma, loai, ten, vi_tri, tham_so)
        self._nodes[ma] = node
        self._da_chay = False

        self.logger.info(f"Thêm node: {ten} ({ma}, {loai})")
        return node.to_dict()

    def xoa_node(self, ma_node: str) -> None:
        """Xóa node và các kết nối liên quan."""
        if ma_node not in self._nodes:
            raise KeyError(f"Node '{ma_node}' không tồn tại")

        del self._nodes[ma_node]
        self._ket_noi = [
            kn for kn in self._ket_noi
            if kn.tu_node != ma_node and kn.den_node != ma_node
        ]
        self._da_chay = False
        self.logger.info(f"Đã xóa node: {ma_node}")

    def sua_node(self, ma_node: str, tham_so: Dict[str, Any]) -> Dict[str, Any]:
        """Cập nhật tham số node."""
        if ma_node not in self._nodes:
            raise KeyError(f"Node '{ma_node}' không tồn tại")

        node = self._nodes[ma_node]
        node.tham_so.update(tham_so)
        node.da_chay = False
        self._da_chay = False
        return node.to_dict()

    def ket_noi(self, tu_node: str, den_node: str, cua: str = "du_lieu") -> Dict[str, Any]:
        """
        Kết nối 2 nodes.

        Args:
            tu_node: Mã node nguồn
            den_node: Mã node đích
            cua: Loại kết nối

        Returns:
            Dict chứa thông tin kết nối
        """
        if tu_node not in self._nodes:
            raise KeyError(f"Node nguồn '{tu_node}' không tồn tại")
        if den_node not in self._nodes:
            raise KeyError(f"Node đích '{den_node}' không tồn tại")
        if tu_node == den_node:
            raise ValueError("Không thể kết nối node với chính nó")

        for kn in self._ket_noi:
            if kn.tu_node == tu_node and kn.den_node == den_node:
                raise ValueError(f"Kết nối {tu_node} -> {den_node} đã tồn tại")

        ma = self._tao_ma_ket_noi()
        kn = KetNoi(ma, tu_node, den_node, cua)
        self._ket_noi.append(kn)
        self._da_chay = False

        ten_tu = self._nodes[tu_node].ten
        ten_den = self._nodes[den_node].ten
        self.logger.info(f"Kết nối: {ten_tu} -> {ten_den}")
        return kn.to_dict()

    def huy_ket_noi(self, tu_node: str, den_node: str) -> None:
        """Hủy kết nối giữa 2 nodes."""
        ban_dau = len(self._ket_noi)
        self._ket_noi = [
            kn for kn in self._ket_noi
            if not (kn.tu_node == tu_node and kn.den_node == den_node)
        ]
        if len(self._ket_noi) == ban_dau:
            raise KeyError(f"Không tìm thấy kết nối {tu_node} -> {den_node}")
        self._da_chay = False

    def _sap_xep_topo(self) -> List[str]:
        """Sắp xếp topological để chạy đúng thứ tự."""
        ban_deg: Dict[str, int] = {ma: 0 for ma in self._nodes}
        adjacency: Dict[str, List[str]] = {ma: [] for ma in self._nodes}

        for kn in self._ket_noi:
            if kn.den_node in ban_deg:
                ban_deg[kn.den_node] += 1
            if kn.tu_node in adjacency:
                adjacency[kn.tu_node].append(kn.den_node)

        hang_doi = [ma for ma, deg in ban_deg.items() if deg == 0]
        thu_tu = []

        while hang_doi:
            ma = hang_doi.pop(0)
            thu_tu.append(ma)
            for neighbor in adjacency.get(ma, []):
                ban_deg[neighbor] -= 1
                if ban_deg[neighbor] == 0:
                    hang_doi.append(neighbor)

        if len(thu_tu) != len(self._nodes):
            raise RuntimeError("Pipeline có vòng lặp (cycle). Kiểm tra lại kết nối.")

        return thu_tu

    def _chay_node_du_lieu(self, node: Node) -> np.ndarray:
        """Chạy node dữ liệu."""
        from vietnamese_ai.datalake.sample_data import DuLieuMau

        nhiem_vu = node.tham_so.get("nhiem_vu", "phan_loai")
        so_mau = node.tham_so.get("so_mau", 200)
        so_dac_trung = node.tham_so.get("so_dac_trung", 5)

        if nhiem_vu == "hoi_quy":
            X, y = DuLieuMau.hoi_quy_don_gian(so_mau=so_mau)
        else:
            X, y = DuLieuMau.phan_loai_don_gian(so_mau=so_mau, so_dac_trung=so_dac_trung)

        return {"X": X, "y": y, "nhiem_vu": nhiem_vu}

    def _chay_node_tien_xu_ly(self, node: Node, du_lieu_vao: Any) -> Any:
        """Chạy node tiền xử lý."""
        from vietnamese_ai.preprocessing.numerical import XuLySo

        phuong_phap = node.tham_so.get("phuong_phap", "zscore")
        X = du_lieu_vao["X"]

        xl = XuLySo()
        if phuong_phap == "zscore":
            X = xl.chuan_hoa_zscore(X)
        elif phuong_phap == "minmax":
            X = xl.chuan_hoa_minmax(X)
        elif phuong_phap == "trung_vi":
            X = XuLySo.xu_ly_gia_tri_thieu(X, "trung_vi")

        du_lieu_vao["X"] = X
        return du_lieu_vao

    def _chay_node_mo_hinh(self, node: Node, du_lieu_vao: Any) -> Any:
        """Chạy node mô hình."""
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.models.clustering import PhanCum
        from vietnamese_ai.models.regression import HoiQuy
        from vietnamese_ai.preprocessing.numerical import XuLySo

        thuat_toan = node.tham_so.get("thuat_toan", "logistic")
        nhiem_vu = node.tham_so.get("nhiem_vu", du_lieu_vao.get("nhiem_vu", "phan_loai"))

        X = du_lieu_vao["X"]
        y = du_lieu_vao.get("y")

        if nhiem_vu == "phan_cum":
            mo_hinh = PhanCum(so_cum=node.tham_so.get("so_cum", 3), thuat_toan=thuat_toan)
            mo_hinh.huan_luyen(X)
        elif nhiem_vu == "hoi_quy":
            X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)
            mo_hinh = HoiQuy(thuat_toan=thuat_toan)
            mo_hinh.huan_luyen(X_train, y_train)
            du_lieu_vao["X_test"] = X_test
            du_lieu_vao["y_test"] = y_test
        else:
            X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)
            mo_hinh = PhanLoai(thuat_toan=thuat_toan)
            mo_hinh.huan_luyen(X_train, y_train)
            du_lieu_vao["X_test"] = X_test
            du_lieu_vao["y_test"] = y_test

        du_lieu_vao["mo_hinh"] = mo_hinh
        return du_lieu_vao

    def _chay_node_danh_gia(self, node: Node, du_lieu_vao: Any) -> Any:
        """Chạy node đánh giá."""
        mo_hinh = du_lieu_vao.get("mo_hinh")
        if mo_hinh is None:
            raise RuntimeError("Chưa có mô hình để đánh giá")

        ket_qua = {}
        X_test = du_lieu_vao.get("X_test", du_lieu_vao["X"])
        y_test = du_lieu_vao.get("y_test", du_lieu_vao.get("y"))

        if y_test is not None:
            diem = mo_hinh.danh_gia(X_test, y_test)
            ket_qua["diem"] = float(diem)

            if hasattr(mo_hinh, "bao_cao"):
                ket_qua["bao_cao"] = mo_hinh.bao_cao(X_test, y_test)
        else:
            du_doan = mo_hinh.du_doan(X_test)
            ket_qua["du_doan_sample"] = du_doan[:5].tolist()

        du_lieu_vao["ket_qua_danh_gia"] = ket_qua
        return du_lieu_vao

    def chay(self) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline.

        Returns:
            Dict chứa kết quả của tất cả nodes
        """
        if not self._nodes:
            raise RuntimeError("Pipeline trống. Thêm nodes trước.")

        self.logger.info("=" * 50)
        self.logger.info(f"CHẠY PIPELINE: {self.ten}")
        self.logger.info(f"Nodes: {len(self._nodes)}, Kết nối: {len(self._ket_noi)}")
        self.logger.info("=" * 50)

        thu_tu = self._sap_xep_topo()
        du_lieu_node: Dict[str, Any] = {}
        bat_dau = time.time()
        loi_count = 0

        for ma_node in thu_tu:
            node = self._nodes[ma_node]
            self.logger.info(f"Chạy node: {node.ten} ({node.loai})")

            try:
                dau_vao = None
                for kn in self._ket_noi:
                    if kn.den_node == ma_node:
                        dau_vao = du_lieu_node.get(kn.tu_node)
                        break

                if node.loai == "du_lieu":
                    ket_qua = self._chay_node_du_lieu(node)
                elif node.loai == "tien_xu_ly":
                    ket_qua = self._chay_node_tien_xu_ly(node, dau_vao)
                elif node.loai == "mo_hinh":
                    ket_qua = self._chay_node_mo_hinh(node, dau_vao)
                elif node.loai == "danh_gia":
                    ket_qua = self._chay_node_danh_gia(node, dau_vao)
                else:
                    ket_qua = dau_vao

                du_lieu_node[ma_node] = ket_qua
                node.da_chay = True
                node.ket_qua = ket_qua
                node.loi = None

            except Exception as e:
                node.loi = str(e)
                node.da_chay = False
                loi_count += 1
                self.logger.error(f"Lỗi node {node.ten}: {e}")

        tong_thoi_gian = time.time() - bat_dau
        self._da_chay = True

        self._ket_qua_chay = {
            "trang_thai": "thanh_cong" if loi_count == 0 else "co_loi",
            "so_node": len(self._nodes),
            "so_node_thanh_cong": len(self._nodes) - loi_count,
            "so_loi": loi_count,
            "tong_thoi_gian": round(tong_thoi_gian, 3),
            "chi_tiet": {
                ma: {
                    "ten": self._nodes[ma].ten,
                    "loai": self._nodes[ma].loai,
                    "da_chay": self._nodes[ma].da_chay,
                    "loi": self._nodes[ma].loi,
                }
                for ma in thu_tu
            },
        }

        for ma in thu_tu:
            node = self._nodes[ma]
            if node.loai == "danh_gia" and node.ket_qua:
                self._ket_qua_chay["ket_qua_danh_gia"] = node.ket_qua.get(
                    "ket_qua_danh_gia", {}
                )

        self.logger.info(
            f"Pipeline hoàn tất: {self._ket_qua_chay['so_node_thanh_cong']}"
            f"/{self._ket_qua_chay['so_node']} nodes OK "
            f"({tong_thoi_gian:.3f}s)"
        )

        return self._ket_qua_chay

    def lay_canvas(self) -> Dict[str, Any]:
        """Lấy trạng thái canvas hiện tại."""
        return {
            "ten": self.ten,
            "nodes": {ma: n.to_dict() for ma, n in self._nodes.items()},
            "ket_noi": [kn.to_dict() for kn in self._ket_noi],
            "da_chay": self._da_chay,
        }

    def danh_sach_loai_node(self) -> Dict[str, Dict]:
        """Liệt kê tất cả loại node có sẵn."""
        return LOAI_NODE.copy()

    def danh_sach_thuat_toan(self) -> Dict[str, List[str]]:
        """Liệt kê tất cả thuật toán có sẵn."""
        return TAT_CA_THUAT_TOAN.copy()

    def lay_templates(self) -> Dict[str, Dict]:
        """Lấy danh sách templates có sẵn."""
        return {
            ten: {
                "ten": tpl["ten"],
                "mo_ta": tpl["mo_ta"],
                "so_nodes": len(tpl["nodes"]),
            }
            for ten, tpl in TEMPLATE_PIPELINES.items()
        }

    def tai_template(self, ten_template: str) -> Dict[str, Any]:
        """
        Tải template vào canvas.

        Args:
            ten_template: Tên template

        Returns:
            Dict chứa thông tin canvas
        """
        if ten_template not in TEMPLATE_PIPELINES:
            raise KeyError(
                f"Template '{ten_template}' không tồn tại. "
                f"Chọn: {', '.join(TEMPLATE_PIPELINES.keys())}"
            )

        tpl = TEMPLATE_PIPELINES[ten_template]
        self._nodes.clear()
        self._ket_noi.clear()
        self._counter = 0
        self.ten = tpl["ten"]

        node_map = {}
        for i, node_def in enumerate(tpl["nodes"]):
            node = self.them_node(
                loai=node_def["loai"],
                ten=node_def["ten"],
                vi_tri={"x": i * 200, "y": 100},
                tham_so=node_def.get("tham_so", {}),
            )
            node_map[i] = node["ma"]

        for conn_def in tpl["ket_noi"]:
            self.ket_noi(
                node_map[conn_def["tu"]],
                node_map[conn_def["den"]],
            )

        self.logger.info(f"Đã tải template: {tpl['ten']}")
        return self.lay_canvas()

    def luu(self, duong_dan: str) -> str:
        """Lưu pipeline config ra file JSON."""
        config = {
            "ten": self.ten,
            "version": "1.0",
            "thoi_gian_luu": time.time(),
            "nodes": [
                {
                    "ma": n.ma,
                    "loai": n.loai,
                    "ten": n.ten,
                    "vi_tri": n.vi_tri,
                    "tham_so": n.tham_so,
                }
                for n in self._nodes.values()
            ],
            "ket_noi": [
                {"tu": kn.tu_node, "den": kn.den_node, "cua": kn.cua}
                for kn in self._ket_noi
            ],
        }

        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Đã lưu pipeline: {duong_dan}")
        return str(duong_dan_path)

    @classmethod
    def tai(cls, duong_dan: str) -> "StudioKeoTha":
        """Tải pipeline từ file JSON."""
        with open(duong_dan, "r", encoding="utf-8") as f:
            config = json.load(f)

        studio = cls(ten=config.get("ten", "Pipeline"))
        node_map = {}

        for node_def in config.get("nodes", []):
            node = studio.them_node(
                loai=node_def["loai"],
                ten=node_def["ten"],
                vi_tri=node_def.get("vi_tri"),
                tham_so=node_def.get("tham_so"),
            )
            node_map[node_def["ma"]] = node["ma"]

        for conn_def in config.get("ket_noi", []):
            tu = node_map.get(conn_def["tu"], conn_def["tu"])
            den = node_map.get(conn_def["den"], conn_def["den"])
            try:
                studio.ket_noi(tu, den, conn_def.get("cua", "du_lieu"))
            except (KeyError, ValueError):
                pass

        Logger("StudioKeoTha").info(f"Đã tải pipeline từ: {duong_dan}")
        return studio

    def chay_giao_dien_web(self, port: int = 5000):
        """Khởi động Visual UI trên nền trình duyệt."""
        try:
            from vietnamese_ai.ui.core import UIApp, KieuDang
            from vietnamese_ai.ui.components import KhungKeoThaDAG
        except ImportError:
            raise ImportError("Không tìm thấy V-UI. Hãy đảm bảo thư mục vietnamese_ai/ui tồn tại.")
        
        self.logger.info(f"Đang chuẩn bị giao diện Web Visual Builder...")
        app = UIApp(tieu_de=self.ten, theme=KieuDang.DARK)
        
        # Đưa lõi DAG vào Component UI
        ui_dag = KhungKeoThaDAG(
            ham_lay_dag=self.lay_canvas,
            ham_chay_dag=self.chay
        )
        app.them_cot(ui_dag)
        
        # Khởi động Web Server (Zero-Dependency)
        app.chay(port=port)

