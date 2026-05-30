"""
HoDuLieu (DataLake Manager) - Quản lý, lưu trữ và Streaming dữ liệu lớn cho quá trình huấn luyện.
"""

import os
import shutil
import logging
from typing import Iterator, List, Dict, Any

from .stream_reader import DocDuLieuStream
from .dvc_core import QuanLyPhienBan

logger = logging.getLogger("V-DataLake")


class HoDuLieu:
    """
    Quản lý Hồ Dữ Liệu (DataLake). 
    Giao tiếp chính cho EvoNet-Studio để Import, Phiên bản hóa và Đọc dữ liệu.
    """

    def __init__(self, duong_dan_luu_tru: str = "."):
        """
        Khởi tạo Hồ Dữ Liệu.
        
        Args:
            duong_dan_luu_tru: Nơi chứa thư mục .v-dvc và các file dataset (Ví dụ ổ đĩa ngoài).
        """
        self.duong_dan_luu_tru = os.path.abspath(duong_dan_luu_tru)
        self.dvc = QuanLyPhienBan(self.duong_dan_luu_tru)
        self.thu_muc_datasets = os.path.join(self.duong_dan_luu_tru, "datasets")
        
        os.makedirs(self.thu_muc_datasets, exist_ok=True)

    def nhap_du_lieu(self, duong_dan_nguon: str, ten_dataset: str, tin_nhan_commit: str = "Import mới") -> str:
        """
        Sao chép file dữ liệu vào DataLake và đánh phiên bản (Commit).
        
        Args:
            duong_dan_nguon: Đường dẫn file gốc (JSONL/CSV).
            ten_dataset: Tên file mới sẽ được lưu trong DataLake (ví dụ: 'train_data.jsonl').
            tin_nhan_commit: Mô tả cho phiên bản này.
            
        Returns:
            Mã băm SHA256 của phiên bản vừa tạo.
        """
        if not os.path.exists(duong_dan_nguon):
            raise FileNotFoundError(f"Không tìm thấy file nguồn: {duong_dan_nguon}")
            
        duong_dan_dich = os.path.join(self.thu_muc_datasets, ten_dataset)
        
        # Sao chép file vào DataLake (nếu nó chưa nằm trong DataLake)
        if os.path.abspath(duong_dan_nguon) != os.path.abspath(duong_dan_dich):
            logger.info(f"[HoDuLieu] Đang nạp dữ liệu vào hồ: {ten_dataset}...")
            shutil.copy2(duong_dan_nguon, duong_dan_dich)
            
        # Gọi DVC để đóng gói và đánh mã băm
        ma_bam = self.dvc.commit(duong_dan_dich, tin_nhan_commit)
        return ma_bam

    def phuc_hoi_phien_ban(self, ten_dataset: str, ma_bam: str):
        """
        Khôi phục một phiên bản dataset cũ đã bị ghi đè/xóa.
        """
        duong_dan_dich = os.path.join(self.thu_muc_datasets, ten_dataset)
        self.dvc.checkout(ma_bam, duong_dan_dich)

    def doc_luong_du_lieu(self, ten_dataset: str, kich_thuoc_lo: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """
        Trả về một Generator để duyệt qua dữ liệu khổng lồ theo từng lô (Batch).
        Giải quyết hoàn toàn vấn đề OOM (Tràn RAM).
        """
        duong_dan_file = os.path.join(self.thu_muc_datasets, ten_dataset)
        if not os.path.exists(duong_dan_file):
            raise FileNotFoundError(f"Dataset {ten_dataset} không tồn tại trong hồ.")
            
        if ten_dataset.endswith(".csv"):
            return DocDuLieuStream.doc_csv(duong_dan_file, kich_thuoc_lo)
        else:
            return DocDuLieuStream.doc_jsonl(duong_dan_file, kich_thuoc_lo)

    def xem_lich_su(self) -> list:
        """Xem lịch sử các phiên bản dataset."""
        return self.dvc.lay_lich_su()
