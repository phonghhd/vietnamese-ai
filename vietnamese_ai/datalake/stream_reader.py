"""
DocDuLieuStream (Streaming Data Reader) - Đọc file dữ liệu khổng lồ theo lô (batch)
bằng Generator để tránh tràn RAM (OOM).
"""

import csv
import json
import logging
from typing import Any, Dict, Iterator, List

logger = logging.getLogger("V-DataLake")


class DocDuLieuStream:
    """Đọc dữ liệu lớn (JSONL, CSV) dưới dạng Streaming."""

    @staticmethod
    def doc_jsonl(duong_dan: str, kich_thuoc_lo: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """
        Đọc file JSON Lines (.jsonl) và trả về từng lô (batch) dữ liệu.

        Args:
            duong_dan: Đường dẫn tới file JSONL.
            kich_thuoc_lo: Số lượng dòng trong một lô (mặc định 1000).

        Yields:
            Một danh sách chứa các JSON object (dict).
        """
        lo_hien_tai = []
        try:
            with open(duong_dan, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        lo_hien_tai.append(obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"[DocDuLieuStream] Bỏ qua dòng bị lỗi JSON: {e}")

                    if len(lo_hien_tai) >= kich_thuoc_lo:
                        yield lo_hien_tai
                        lo_hien_tai = []

            # Yield phần còn lại nếu file đã hết nhưng chưa đủ lô
            if lo_hien_tai:
                yield lo_hien_tai

        except FileNotFoundError:
            logger.error(f"[DocDuLieuStream] Không tìm thấy file: {duong_dan}")
            raise

    @staticmethod
    def doc_csv(duong_dan: str, kich_thuoc_lo: int = 1000) -> Iterator[List[Dict[str, str]]]:
        """
        Đọc file CSV và trả về từng lô dữ liệu dưới dạng Dict.
        Dòng đầu tiên của CSV bắt buộc phải là Header.
        """
        lo_hien_tai = []
        try:
            with open(duong_dan, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lo_hien_tai.append(row)

                    if len(lo_hien_tai) >= kich_thuoc_lo:
                        yield lo_hien_tai
                        lo_hien_tai = []

            if lo_hien_tai:
                yield lo_hien_tai

        except FileNotFoundError:
            logger.error(f"[DocDuLieuStream] Không tìm thấy file: {duong_dan}")
            raise
