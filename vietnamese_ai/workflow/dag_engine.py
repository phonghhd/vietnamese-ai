"""
DongChayDAG (DAG Engine) - Phân tích, sắp xếp và thực thi song song các Nút
trong Đồ thị có hướng (Directed Acyclic Graph).
"""

import concurrent.futures
import logging
from collections import defaultdict, deque
from typing import Any, Dict, List

from .node import NutCongViec

logger = logging.getLogger("V-Workflow")


class LoiVongLap(Exception):
    """Lỗi khi phát hiện vòng lặp (Cycle) trong DAG."""

    pass


class DongChayDAG:
    """Lõi định tuyến Luồng công việc."""

    def __init__(self):
        self.danh_sach_nut: Dict[str, NutCongViec] = {}
        # Ánh xạ biến (biến A được tạo ra bởi Nút X)
        self.nguon_cung_cap_bien: Dict[str, str] = {}

    def them_nut(self, nut: NutCongViec):
        """Đăng ký một nút vào Đồ thị."""
        if nut.id_nut in self.danh_sach_nut:
            raise ValueError(f"Nút có ID '{nut.id_nut}' đã tồn tại.")
        self.danh_sach_nut[nut.id_nut] = nut

        # Cập nhật nguồn cung cấp biến
        for dau_ra in nut.dau_ra:
            if dau_ra in self.nguon_cung_cap_bien:
                logger.warning(
                    f"[DAG] Cảnh báo: Nút '{nut.id_nut}' ghi đè biến '{dau_ra}' của nút '{self.nguon_cung_cap_bien[dau_ra]}'"
                )
            self.nguon_cung_cap_bien[dau_ra] = nut.id_nut

    def xac_dinh_lien_ket(self) -> Dict[str, List[str]]:
        """
        Xây dựng danh sách kề (Adjacency List) mô tả đồ thị.
        Dựa trên việc Nút B cần đầu vào được sinh ra bởi Nút A.
        Returns:
            Dict: id_nut -> [danh_sach_id_nut_phu_thuoc_no]
        """
        do_thi: Dict[str, List[str]] = defaultdict(list)

        # Đảm bảo tất cả nút đều có key (kể cả nút không ai phụ thuộc)
        for id_nut in self.danh_sach_nut:
            do_thi[id_nut] = []

        for id_nut, nut in self.danh_sach_nut.items():
            for dau_vao in nut.dau_vao:
                if dau_vao in self.nguon_cung_cap_bien:
                    nut_nguon = self.nguon_cung_cap_bien[dau_vao]
                    # Nút Nguồn -> Nút Hiện Tại
                    if id_nut not in do_thi[nut_nguon]:
                        do_thi[nut_nguon].append(id_nut)

        return do_thi

    def sap_xep_topo(self) -> List[str]:
        """
        Thuật toán Topological Sort bằng Kahn's Algorithm.
        Giúp phát hiện Vòng lặp và trả về danh sách thứ tự ưu tiên chạy.
        """
        do_thi = self.xac_dinh_lien_ket()
        bac_vao = {id_nut: 0 for id_nut in self.danh_sach_nut}

        # Tính in-degree (Bậc vào)
        for u in do_thi:
            for v in do_thi[u]:
                bac_vao[v] += 1

        # Hàng đợi chứa các nút không có phụ thuộc (in-degree == 0)
        hang_doi = deque([nut for nut, bac in bac_vao.items() if bac == 0])
        thu_tu_chay = []

        while hang_doi:
            u = hang_doi.popleft()
            thu_tu_chay.append(u)

            for v in do_thi[u]:
                bac_vao[v] -= 1
                if bac_vao[v] == 0:
                    hang_doi.append(v)

        # Kiểm tra vòng lặp
        if len(thu_tu_chay) != len(self.danh_sach_nut):
            cac_nut_bi_lap = [nut for nut, bac in bac_vao.items() if bac > 0]
            raise LoiVongLap(f"Phát hiện Vòng lặp (Cycle) ở các nút: {cac_nut_bi_lap}")

        return thu_tu_chay

    def thuc_thi(self, du_lieu_khoi_tao: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Thực thi đồ thị song song bằng Multi-threading.

        Args:
            du_lieu_khoi_tao: Biến toàn cục ban đầu có sẵn (Nếu có).
        """
        # Kiểm tra vòng lặp trước khi chạy
        self.sap_xep_topo()

        kho_du_lieu = du_lieu_khoi_tao or {}

        do_thi = self.xac_dinh_lien_ket()
        bac_vao = {id_nut: 0 for id_nut in self.danh_sach_nut}
        for u in do_thi:
            for v in do_thi[u]:
                bac_vao[v] += 1

        nut_san_sang = set([nut for nut, bac in bac_vao.items() if bac == 0])
        nut_dang_chay = set()
        nut_hoan_thanh = set()

        logger.info(f"[DAG] Bắt đầu chạy Workflow ({len(self.danh_sach_nut)} nút).")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tuong_lai_den_nut = {}  # Future -> id_nut

            while nut_san_sang or nut_dang_chay:
                # 1. Kích hoạt tất cả nút sẵn sàng
                for id_nut in nut_san_sang.copy():
                    nut = self.danh_sach_nut[id_nut]

                    # Submit task vào ThreadPool
                    logger.debug(f"[DAG] Bắn tác vụ '{id_nut}' vào ThreadPool")
                    tuong_lai = executor.submit(nut.chay, kho_du_lieu)
                    tuong_lai_den_nut[tuong_lai] = id_nut

                    nut_san_sang.remove(id_nut)
                    nut_dang_chay.add(id_nut)

                # 2. Đợi ít nhất 1 nút hoàn thành
                hoan_thanh, chua_hoan_thanh = concurrent.futures.wait(
                    tuong_lai_den_nut.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )

                # 3. Xử lý các nút vừa xong
                for tuong_lai in hoan_thanh:
                    id_nut = tuong_lai_den_nut.pop(tuong_lai)
                    nut_dang_chay.remove(id_nut)
                    nut_hoan_thanh.add(id_nut)

                    # Cập nhật Kho Dữ Liệu
                    try:
                        ket_qua = tuong_lai.result()
                        kho_du_lieu.update(ket_qua)
                        logger.info(f"[DAG] Hoàn thành nút '{id_nut}'")
                    except Exception as e:
                        logger.error(f"[DAG] Nút '{id_nut}' ném ra lỗi: {e}")
                        # Dừng toàn bộ Workflow nếu 1 nút chết
                        raise RuntimeError(f"Workflow dừng đột ngột do Nút '{id_nut}' bị lỗi: {e}")

                    # 4. Mở khóa cho các nút phụ thuộc
                    for v in do_thi[id_nut]:
                        bac_vao[v] -= 1
                        if bac_vao[v] == 0:
                            nut_san_sang.add(v)

        logger.info("[DAG] Workflow thực thi xong toàn bộ.")
        return kho_du_lieu
