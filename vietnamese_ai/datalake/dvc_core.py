"""
QuanLyPhienBan (Data Version Control Core) - Quản lý phiên bản dữ liệu lớn
bằng thuật toán băm SHA256, hoạt động tương tự Git LFS.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("V-DataLake")


class QuanLyPhienBan:
    """Lõi quản lý phiên bản dữ liệu (DVC) thuần Python."""

    def __init__(self, thu_muc_goc: str = "."):
        """
        Khởi tạo DVC.
        
        Args:
            thu_muc_goc: Thư mục gốc chứa dự án và dữ liệu.
        """
        self.thu_muc_goc = os.path.abspath(thu_muc_goc)
        self.thu_muc_dvc = os.path.join(self.thu_muc_goc, ".v-dvc")
        self.thu_muc_objects = os.path.join(self.thu_muc_dvc, "objects")
        self.file_index = os.path.join(self.thu_muc_dvc, "index.json")
        
        self._khoi_tao_thu_muc()

    def _khoi_tao_thu_muc(self):
        """Tạo cấu trúc thư mục .v-dvc nếu chưa có."""
        os.makedirs(self.thu_muc_objects, exist_ok=True)
        if not os.path.exists(self.file_index):
            with open(self.file_index, 'w', encoding='utf-8') as f:
                json.dump({"commits": []}, f)

    def _tinh_ma_bam(self, duong_dan: str, kich_thuoc_chunk: int = 4096 * 1024) -> str:
        """
        Tính mã băm SHA256 của một file siêu lớn bằng cách đọc từng chunk 4MB.
        Tránh OOM cho file chục GB.
        """
        sha256_hash = hashlib.sha256()
        with open(duong_dan, "rb") as f:
            for byte_block in iter(lambda: f.read(kich_thuoc_chunk), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _doc_index(self) -> Dict[str, Any]:
        """Đọc file index.json."""
        try:
            with open(self.file_index, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"commits": []}

    def _ghi_index(self, data: Dict[str, Any]):
        """Ghi dữ liệu vào index.json."""
        with open(self.file_index, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def commit(self, duong_dan_file: str, tin_nhan: str = "Cập nhật dữ liệu") -> str:
        """
        Lưu một phiên bản mới của file dữ liệu.
        
        Args:
            duong_dan_file: Đường dẫn tới file cần theo dõi (dataset).
            tin_nhan: Mô tả thay đổi.
            
        Returns:
            Mã băm SHA256 của commit.
        """
        if not os.path.exists(duong_dan_file):
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan_file}")
            
        logger.info(f"[DVC] Đang tính toán mã băm cho {duong_dan_file}...")
        ma_bam = self._tinh_ma_bam(duong_dan_file)
        
        # Đường dẫn lưu trong kho objects
        file_dich = os.path.join(self.thu_muc_objects, ma_bam)
        
        # Nếu chưa tồn tại trong kho thì copy vào
        if not os.path.exists(file_dich):
            logger.info(f"[DVC] Đang lưu trữ phiên bản mới ({ma_bam[:8]})...")
            # Tối ưu hóa: Dùng copy2 để giữ nguyên metadata
            shutil.copy2(duong_dan_file, file_dich)
            logger.debug("Sử dụng Copy tiêu chuẩn.")
        else:
            logger.info(f"[DVC] Phiên bản ({ma_bam[:8]}) đã tồn tại, bỏ qua lưu trữ vật lý.")

        # Ghi nhận vào index
        index_data = self._doc_index()
        commit_moi = {
            "hash": ma_bam,
            "file": os.path.relpath(duong_dan_file, self.thu_muc_goc),
            "tin_nhan": tin_nhan,
            "thoi_gian": time.time()
        }
        index_data["commits"].append(commit_moi)
        self._ghi_index(index_data)
        
        logger.info(f"[DVC] Commit thành công: {ma_bam[:8]} - {tin_nhan}")
        return ma_bam

    def checkout(self, ma_bam: str, duong_dan_dich: str):
        """
        Khôi phục (checkout) file dữ liệu từ kho vật lý dựa vào mã băm.
        
        Args:
            ma_bam: Mã băm đầy đủ hoặc 8 ký tự đầu của commit.
            duong_dan_dich: Nơi sẽ đặt file được khôi phục.
        """
        # Tìm mã băm đầy đủ nếu người dùng chỉ truyền 8 ký tự đầu
        hash_day_du = None
        for f in os.listdir(self.thu_muc_objects):
            if f.startswith(ma_bam):
                hash_day_du = f
                break
                
        if not hash_day_du:
            raise ValueError(f"Không tìm thấy phiên bản nào có mã băm: {ma_bam}")
            
        file_nguon = os.path.join(self.thu_muc_objects, hash_day_du)
        
        logger.info(f"[DVC] Đang khôi phục phiên bản {hash_day_du[:8]}...")
        # Xóa file cũ nếu có để tránh lỗi
        if os.path.exists(duong_dan_dich):
            os.remove(duong_dan_dich)
            
        shutil.copy2(file_nguon, duong_dan_dich)
            
        logger.info(f"[DVC] Khôi phục thành công tới {duong_dan_dich}")
        
    def lay_lich_su(self) -> list:
        """Lấy danh sách các commit đã lưu."""
        return self._doc_index().get("commits", [])
