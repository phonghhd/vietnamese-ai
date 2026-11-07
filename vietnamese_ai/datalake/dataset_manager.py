"""
HoDuLieu (DataLake Manager) - Quản lý, lưu trữ và Streaming dữ liệu lớn cho quá trình huấn luyện.
"""

import logging
import os
import shutil
from typing import Any, Dict, Iterator, List

from .dvc_core import QuanLyPhienBan
from .stream_reader import DocDuLieuStream

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

    def nhap_du_lieu(
        self, duong_dan_nguon: str, ten_dataset: str, tin_nhan_commit: str = "Import mới"
    ) -> str:
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

    def doc_luong_du_lieu(
        self, ten_dataset: str, kich_thuoc_lo: int = 1000
    ) -> Iterator[List[Dict[str, Any]]]:
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

    def nhap_danh_sach(
        self,
        du_lieu: List[Dict[str, Any]],
        ten_dataset: str,
        tin_nhan_commit: str = "Import từ danh sách bộ nhớ",
        quet_an_toan: bool = True,
    ) -> str:
        """
        Lưu trực tiếp danh sách bản ghi (List[Dict]) vào DataLake dưới dạng JSONL
        (kèm quét tự động chống Sleeper Backdoor, Split-Payload & PII) và đánh phiên bản SHA-256.
        """
        import json

        ban_ghi_luu = du_lieu
        if quet_an_toan and du_lieu:
            from vietnamese_ai.security.data_defense import PhongThuDuLieu

            ban_ghi_luu, _ = PhongThuDuLieu.quet_va_lam_sach_dataset(du_lieu)

        if not ten_dataset.endswith((".jsonl", ".json")):
            ten_dataset = f"{ten_dataset}.jsonl"
        duong_dan_dich = os.path.join(self.thu_muc_datasets, ten_dataset)
        with open(duong_dan_dich, "w", encoding="utf-8") as f:
            for item in ban_ghi_luu:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return self.dvc.commit(duong_dan_dich, tin_nhan_commit)

    def quet_phong_thu_dataset(
        self,
        ten_dataset: str,
        kich_thuoc_lo: int = 500,
        nguong_kich_hoat_mau: int = 20,
    ) -> Dict[str, Any]:
        """
        Quét kiểm toán toàn bộ dataset lớn trong DataLake theo luồng (không tràn RAM)
        nhằm phát hiện "Bom hẹn giờ" Sleeper Backdoor, Split-Payload và Prompt Injection.
        """
        from vietnamese_ai.security.data_defense import BoGiamSatTichTuDuLieu
        from vietnamese_ai.security.llm_firewall import TuongLuaAI

        giam_sat = BoGiamSatTichTuDuLieu(nguong_kich_hoat_mau=nguong_kich_hoat_mau)
        tuong_lua = TuongLuaAI(ngat_ket_noi_khi_phat_hien=False)
        so_dong_doc_hai = 0

        for lo in self.doc_luong_du_lieu(ten_dataset, kich_thuoc_lo=kich_thuoc_lo):
            giam_sat.cap_nhat_lo(lo)
            for mau in lo:
                if isinstance(mau, dict):
                    chuoi = " ".join(str(v) for v in mau.values() if isinstance(v, str))
                    ok, _ = tuong_lua.kiem_tra_prompt(chuoi)
                    if not ok:
                        so_dong_doc_hai += 1

        bc = giam_sat.kiem_tra_bom_tich_tu()
        bc["so_mau_prompt_injection_don_le"] = so_dong_doc_hai
        bc["an_toan"] = (not bc["phat_hien_bom_hen_gio"]) and so_dong_doc_hai == 0
        return bc

    def xuat_instruction_dataset(
        self, ten_dataset: str, che_do: str = "alpaca"
    ) -> Any:
        """
        Chuyển đổi một dataset đã phiên bản hóa trong DataLake thành `InstructionDataset`
        để truyền trực tiếp vào `BoTinhChinhMoHinh`, `SFTTrainer` hoặc `InstructionTuningTrainer`.
        """
        from vietnamese_ai.fine_tuning.dataset import InstructionDataset

        ds = InstructionDataset(che_do=che_do)
        items: List[Dict[str, Any]] = []
        for batch in self.doc_luong_du_lieu(ten_dataset, kich_thuoc_lo=500):
            items.extend(batch)
        ds.tai_tu_list(items)
        return ds

    def nap_vao_rag(
        self,
        ten_dataset: str,
        rag_pipeline: Any,
        truong_van_ban: str = "text",
        truong_id: str = "id",
    ) -> int:
        """
        Đọc luồng dữ liệu từ DataLake và nạp trực tiếp các tài liệu vào `RAGPipeline`.
        Trả về tổng số tài liệu đã lập chỉ mục.
        """
        dem = 0
        for batch in self.doc_luong_du_lieu(ten_dataset, kich_thuoc_lo=200):
            for idx, record in enumerate(batch):
                noi_dung = str(
                    record.get(
                        truong_van_ban,
                        record.get(
                            "van_ban",
                            record.get(
                                "output",
                                record.get("content", record.get("instruction", "")),
                            ),
                        ),
                    )
                ).strip()
                if not noi_dung:
                    continue
                doc_id = str(record.get(truong_id, f"{ten_dataset}_{dem + idx}"))
                rag_pipeline.them_tai_lieu(doc_id, noi_dung)
                dem += 1
        return dem

    def xuat_xy(
        self,
        ten_dataset: str,
        cot_nhan: str = "label",
        nhung_van_ban: bool = True,
    ) -> tuple:
        """
        Trích xuất cặp `(X, y)` từ dataset trong DataLake để huấn luyện mô hình học máy hoặc phân loại.
        Nếu dữ liệu chứa văn bản (`text`/`van_ban`) và `nhung_van_ban=True`, tự động nhúng bằng `BoNhungVanBan`.
        """
        import numpy as np

        records: List[Dict[str, Any]] = []
        for batch in self.doc_luong_du_lieu(ten_dataset, kich_thuoc_lo=1000):
            records.extend(batch)

        if not records:
            return np.empty((0, 1)), np.empty((0,))

        first = records[0]
        if ("text" in first or "van_ban" in first) and nhung_van_ban:
            from vietnamese_ai.rag.embedder import BoNhungVanBan

            texts = [str(r.get("text", r.get("van_ban", ""))) for r in records]
            labels = [r.get(cot_nhan, r.get("nhan", 0)) for r in records]
            X = BoNhungVanBan(kich_thuoc=128).nhung(texts)
            return X, np.array(labels)

        feature_keys = [k for k in first.keys() if k not in (cot_nhan, "nhan", "id")]
        X_rows = [[float(r.get(k, 0.0)) for k in feature_keys] for r in records]
        y_rows = [r.get(cot_nhan, r.get("nhan", 0)) for r in records]
        return np.array(X_rows, dtype=float), np.array(y_rows)

    BO_DU_LIEU_MAU_TIENG_VIET: Dict[str, List[Dict[str, Any]]] = {
        "uit_vsfc": [
            {
                "text": "Giảng viên dạy rất nhiệt tình dễ hiểu bài giảng sinh động",
                "instruction": "Phân loại cảm xúc phản hồi sinh viên",
                "input": "Giảng viên dạy rất nhiệt tình dễ hiểu bài giảng sinh động",
                "output": "tích cực",
                "label": 1,
            },
            {
                "text": "Tài liệu môn học đầy đủ phòng học thoáng mát hiện đại",
                "instruction": "Phân loại cảm xúc phản hồi sinh viên",
                "input": "Tài liệu môn học đầy đủ phòng học thoáng mát hiện đại",
                "output": "tích cực",
                "label": 1,
            },
            {
                "text": "Bài tập quá nặng thời gian kiểm tra gấp gáp khó theo kịp",
                "instruction": "Phân loại cảm xúc phản hồi sinh viên",
                "input": "Bài tập quá nặng thời gian kiểm tra gấp gáp khó theo kịp",
                "output": "tiêu cực",
                "label": 0,
            },
            {
                "text": "Thiết bị phòng thực hành cũ kỹ thường xuyên bị lỗi mạng",
                "instruction": "Phân loại cảm xúc phản hồi sinh viên",
                "input": "Thiết bị phòng thực hành cũ kỹ thường xuyên bị lỗi mạng",
                "output": "tiêu cực",
                "label": 0,
            },
        ],
        "viquad": [
            {
                "text": "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam, nằm ở vùng đồng bằng sông Hồng.",
                "instruction": "Thủ đô của Việt Nam nằm ở vùng nào?",
                "output": "Thủ đô Hà Nội của Việt Nam nằm ở vùng đồng bằng sông Hồng.",
                "label": 1,
            },
            {
                "text": "PhoBERT là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt do nhóm nghiên cứu VinAI phát triển.",
                "instruction": "Ai đã phát triển mô hình PhoBERT?",
                "output": "Nhóm nghiên cứu VinAI đã phát triển mô hình PhoBERT.",
                "label": 1,
            },
            {
                "text": "Vịnh Hạ Long thuộc tỉnh Quảng Ninh được UNESCO công nhận là Di sản thiên nhiên thế giới.",
                "instruction": "Vịnh Hạ Long thuộc tỉnh nào?",
                "output": "Vịnh Hạ Long thuộc tỉnh Quảng Ninh.",
                "label": 1,
            },
        ],
        "phoner": [
            {
                "text": "Nguyễn Nhật Ánh sinh tại tỉnh Quảng Nam và làm việc tại Thành phố Hồ Chí Minh.",
                "instruction": "Nhận diện thực thể tên riêng trong câu",
                "output": "PER: Nguyễn Nhật Ánh | LOC: Quảng Nam, Thành phố Hồ Chí Minh",
                "label": 1,
            },
            {
                "text": "Tập đoàn FPT hợp tác nghiên cứu AI tại Hà Nội.",
                "instruction": "Nhận diện thực thể tên riêng trong câu",
                "output": "ORG: Tập đoàn FPT | LOC: Hà Nội",
                "label": 1,
            },
        ],
        "vmlu": [
            {
                "text": "Quốc hoa truyền thống phổ biến của Việt Nam là hoa sen.",
                "instruction": "Loài hoa nào thường được tôn vinh là quốc hoa của Việt Nam?",
                "output": "Hoa sen",
                "label": 1,
            },
            {
                "text": "Đỉnh núi Fansipan thuộc dãy Hoàng Liên Sơn là đỉnh núi cao nhất Việt Nam với độ cao 3147.3 mét.",
                "instruction": "Đỉnh núi cao nhất Việt Nam tên là gì?",
                "output": "Đỉnh Fansipan thuộc dãy Hoàng Liên Sơn.",
                "label": 1,
            },
        ],
        "vi_alpaca": [
            {
                "instruction": "Giải thích sự khác biệt giữa Học có giám sát và Học không giám sát.",
                "input": "",
                "output": "Học có giám sát (Supervised Learning) sử dụng dữ liệu đã gán nhãn để học ánh xạ đầu vào - đầu ra, trong khi Học không giám sát (Unsupervised Learning) tự khám phá cấu trúc hoặc phân cụm trên dữ liệu chưa có nhãn.",
                "text": "Học có giám sát dùng dữ liệu có nhãn, học không giám sát tìm cấu trúc ẩn không cần nhãn.",
                "label": 1,
            },
            {
                "instruction": "Viết 3 mẹo tối ưu hóa truy vấn SQL trên bảng dữ liệu lớn.",
                "input": "",
                "output": "1. Đánh chỉ mục (Index) trên các cột lọc WHERE và JOIN. 2. Tránh dùng SELECT * mà chỉ lấy cột cần thiết. 3. Phân vùng bảng (Partitioning) theo thời gian.",
                "text": "Tối ưu SQL bằng cách tạo Index, tránh SELECT * và dùng Partitioning.",
                "label": 1,
            },
        ],
        "vi_dpo_preference": [
            {
                "prompt": "Làm thế nào để bảo mật mật khẩu người dùng trong cơ sở dữ liệu?",
                "chosen": "Sử dụng thuật toán băm mật khẩu chuyên dụng như Argon2id hoặc bcrypt kèm muối ngẫu nhiên (salt) riêng biệt cho từng người dùng.",
                "rejected": "Lưu trực tiếp mật khẩu dưới dạng văn bản thuần hoặc chỉ mã hóa bằng MD5 để dễ tra cứu lại.",
                "instruction": "Làm thế nào để bảo mật mật khẩu người dùng trong cơ sở dữ liệu?",
                "output": "Sử dụng thuật toán băm mật khẩu chuyên dụng như Argon2id hoặc bcrypt kèm muối ngẫu nhiên.",
                "label": 1,
            },
            {
                "prompt": "Giải thích kỹ thuật LoRA khi tinh chỉnh mô hình ngôn ngữ lớn.",
                "chosen": "LoRA đóng băng trọng số gốc W và chỉ huấn luyện hai ma trận hạng thấp A, B sao cho delta_W = A @ B, giúp giảm hơn 90% số tham số cần cập nhật.",
                "rejected": "LoRA là kỹ thuật xóa bớt một nửa số tầng của mạng nơ-ron để chạy nhanh hơn.",
                "instruction": "Giải thích kỹ thuật LoRA khi tinh chỉnh mô hình ngôn ngữ lớn.",
                "output": "LoRA đóng băng trọng số gốc W và chỉ huấn luyện hai ma trận hạng thấp A, B.",
                "label": 1,
            },
        ],
        "vi_reasoning_cot": [
            {
                "instruction": "Một cửa hàng giảm giá 20% cho chiếc áo giá 500.000đ, sau đó giảm thêm 10% trên giá đã giảm cho thành viên VIP. Hỏi khách VIP phải trả bao nhiêu?",
                "output": "<think>Bước 1: Giá sau khi giảm 20% đầu tiên là 500.000 * (1 - 0.20) = 400.000đ. Bước 2: Giảm thêm 10% trên mức 400.000đ là 400.000 * (1 - 0.10) = 360.000đ.</think>\nKhách hàng VIP phải trả 360.000 đồng.",
                "text": "Áo 500.000đ giảm 20% còn 400.000đ, giảm tiếp 10% còn 360.000đ.",
                "label": 1,
            },
            {
                "instruction": "Nếu 3 máy chủ huấn luyện xong 3 mô hình trong 3 giờ, thì 9 máy chủ huấn luyện 9 mô hình cùng loại mất bao lâu?",
                "output": "<think>Mỗi máy chủ cần 3 giờ để hoàn thành 1 mô hình độc lập. Khi 9 máy chủ chạy song song để huấn luyện 9 mô hình, mỗi máy chủ vẫn phụ trách 1 mô hình.</think>\nThời gian cần thiết vẫn là 3 giờ.",
                "text": "9 máy chủ chạy song song huấn luyện 9 mô hình trong 3 giờ.",
                "label": 1,
            },
            {
                "instruction": "Một bể nước có 2 vòi: vòi A chảy đầy bể trong 4 giờ, vòi B chảy đầy bể trong 6 giờ. Mở cả 2 vòi cùng lúc thì sau bao lâu đầy bể?",
                "output": "<think>Bước 1: Mỗi giờ vòi A chảy được 1/4 bể, vòi B chảy được 1/6 bể. Bước 2: Cả 2 vòi mỗi giờ chảy được 1/4 + 1/6 = 5/12 bể. Bước 3: Thời gian đầy bể là 1 / (5/12) = 12/5 = 2.4 giờ.</think>\nSau 2.4 giờ (2 giờ 24 phút) bể sẽ đầy.",
                "text": "Hai vòi cùng chảy đạt 5/12 bể mỗi giờ nên mất 2.4 giờ.",
                "label": 1,
            },
            {
                "instruction": "Tìm số tiếp theo trong dãy số: 2, 6, 12, 20, 30, ?",
                "output": "<think>Bước 1: Xét hiệu giữa các số liên tiếp: 6-2=4, 12-6=6, 20-12=8, 30-20=10. Bước 2: Hiệu tăng đều thêm 2 đơn vị, vậy hiệu tiếp theo là 12. Bước 3: 30 + 12 = 42.</think>\nSố tiếp theo trong dãy là 42.",
                "text": "Quy luật n*(n+1), số thứ 6 là 6*7 = 42.",
                "label": 1,
            },
            {
                "instruction": "Một đoàn tàu dài 150m chạy qua cây cầu dài 450m với vận tốc 20m/s. Hỏi tàu mất bao nhiêu giây để vượt qua hoàn toàn cây cầu?",
                "output": "<think>Bước 1: Tổng quãng đường từ lúc đầu tàu chạm cầu đến lúc đuôi tàu rời khỏi cầu là 150 + 450 = 600m. Bước 2: Thời gian = 600 / 20 = 30 giây.</think>\nĐoàn tàu mất 30 giây để vượt qua cầu.",
                "text": "Tổng quãng đường 600m chia vận tốc 20m/s bằng 30 giây.",
                "label": 1,
            },
        ],
        "vi_tool_calling": [
            {
                "instruction": "Tra cứu thời tiết tại Đà Nẵng hôm nay và đổi 25 độ C sang độ F.",
                "output": 'Hành động: lay_thoi_tiet\nTham số: {"thanh_pho": "Đà Nẵng", "don_vi": "celsius"}\nKết quả: 25 độ C (77 độ F).',
                "text": "Gọi công cụ lay_thoi_tiet với tham số thanh_pho Đà Nẵng.",
                "label": 1,
            },
            {
                "instruction": "Tính lãi kép cho khoản gửi 100 triệu đồng với lãi suất 7%/năm trong 3 năm.",
                "output": 'Hành động: tinh_lai_kep\nTham số: {"von_goc": 100000000, "lai_suat": 0.07, "so_nam": 3}\nKết quả: 122.504.300 VNĐ.',
                "text": "Gọi công cụ tinh_lai_kep tính lãi kép ngân hàng.",
                "label": 1,
            },
        ],
        "vi_phap_luat": [
            {
                "text": "Theo Bộ luật Lao động Việt Nam, thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.",
                "instruction": "Thời giờ làm việc bình thường tối đa theo Bộ luật Lao động là bao nhiêu?",
                "output": "Thời giờ làm việc bình thường không quá 08 giờ/ngày và không quá 48 giờ/tuần.",
                "label": 1,
            },
            {
                "text": "Hợp đồng dân sự có thể được giao kết bằng lời nói, bằng văn bản hoặc bằng hành vi cụ thể theo quy định của Bộ luật Dân sự.",
                "instruction": "Hợp đồng dân sự có thể được giao kết dưới những hình thức nào?",
                "output": "Hợp đồng dân sự có thể được giao kết bằng lời nói, bằng văn bản hoặc bằng hành vi cụ thể.",
                "label": 1,
            },
        ],
        "vi_y_te": [
            {
                "text": "Bệnh sốt xuất huyết Dengue lây truyền qua vết đốt của muỗi vằn Aedes aegypti. Người bệnh sốt cao cần bù nước điện giải Oresol và đi khám bác sĩ, tuyệt đối không tự ý dùng Ibuprofen hoặc Aspirin.",
                "instruction": "Người mắc sốt xuất huyết cần lưu ý gì khi hạ sốt?",
                "output": "Cần bù nước điện giải và thăm khám y tế, tuyệt đối không tự ý uống Ibuprofen hoặc Aspirin vì gây nguy cơ xuất huyết.",
                "label": 1,
            },
            {
                "text": "Chỉ số khối cơ thể BMI được tính bằng cân nặng (kg) chia cho bình phương chiều cao (m). Mức BMI bình thường ở người trưởng thành châu Á nằm trong khoảng 18.5 đến 22.9.",
                "instruction": "Cách tính chỉ số BMI là gì?",
                "output": "BMI bằng cân nặng (kg) chia cho bình phương chiều cao (m).",
                "label": 1,
            },
        ],
        "vi_tai_chinh": [
            {
                "text": "Chỉ số P/E (Price-to-Earnings) đo lường tỷ lệ giữa thị giá cổ phiếu và thu nhập trên mỗi cổ phiếu (EPS), phản ánh mức giá nhà đầu tư sẵn sàng trả cho 1 đồng lợi nhuận.",
                "instruction": "Chỉ số tài chính P/E có ý nghĩa gì?",
                "output": "P/E là tỷ số giữa giá thị trường của cổ phiếu và lợi nhuận trên mỗi cổ phiếu (EPS).",
                "label": 1,
            },
            {
                "text": "Trái phiếu doanh nghiệp xanh có kỳ hạn 5 năm hỗ trợ vốn cho các dự án năng lượng tái tạo tại Việt Nam.",
                "instruction": "Trái phiếu doanh nghiệp xanh dùng để tài trợ cho lĩnh vực nào?",
                "output": "Tài trợ vốn cho các dự án năng lượng tái tạo và bảo vệ môi trường.",
                "label": 1,
            },
        ],
        "vi_code_python": [
            {
                "instruction": "Viết hàm Python tìm kiếm nhị phân (Binary Search) trên danh sách đã sắp xếp.",
                "output": "def tim_nhi_phan(arr: list, muc_tieu: int) -> int:\n    trai, phai = 0, len(arr) - 1\n    while trai <= phai:\n        giua = (trai + phai) // 2\n        if arr[giua] == muc_tieu:\n            return giua\n        elif arr[giua] < muc_tieu:\n            trai = giua + 1\n        else:\n            phai = giua - 1\n    return -1",
                "text": "Hàm Python tìm kiếm nhị phân độ phức tạp O(log n).",
                "label": 1,
            },
            {
                "instruction": "Viết hàm NumPy tính chuẩn hóa Softmax ổn định số học.",
                "output": "def softmax(x: np.ndarray) -> np.ndarray:\n    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))\n    return e_x / np.sum(e_x, axis=-1, keepdims=True)",
                "text": "Hàm Softmax ổn định số học bằng NumPy.",
                "label": 1,
            },
        ],
        "vi_tom_tat_tin_tuc": [
            {
                "text": "Ngành công nghiệp bán dẫn và trí tuệ nhân tạo tại Việt Nam đang thu hút nhiều tập đoàn công nghệ lớn xây dựng trung tâm R&D tại Hà Nội, Đà Nẵng và TP.HCM, đồng thời đào tạo 50.000 kỹ sư thiết kế vi mạch đến năm 2030.",
                "instruction": "Tóm tắt nội dung chính của bản tin công nghệ trên.",
                "output": "Việt Nam đẩy mạnh thu hút R&D về AI, bán dẫn và đặt mục tiêu đào tạo 50.000 kỹ sư vi mạch vào năm 2030.",
                "label": 1,
            },
        ],
        "vi_dich_thuat": [
            {
                "instruction": "Dịch câu sau sang tiếng Anh chuyên ngành AI: 'Tinh chỉnh mô hình ngôn ngữ lớn với bộ điều hợp hạng thấp giúp tiết kiệm bộ nhớ GPU.'",
                "output": "Fine-tuning large language models with low-rank adapters significantly reduces GPU memory consumption.",
                "text": "Dịch thuật Việt - Anh chuyên ngành Trí tuệ nhân tạo.",
                "label": 1,
            },
        ],
        "vi_chinh_ta": [
            {
                "instruction": "Chuẩn hóa và sửa lỗi chính tả câu tiếng Việt sau: 'mik đag học lập trinh tri tuệ nhân tạo ở trường đh'",
                "output": "Mình đang học lập trình trí tuệ nhân tạo ở trường đại học.",
                "text": "Mình đang học lập trình trí tuệ nhân tạo ở trường đại học.",
                "label": 1,
            },
        ],
        "vi_cham_soc_khach_hang": [
            {
                "instruction": "Khách hàng hỏi: 'Đơn hàng #VN2026 của tôi đặt từ 3 ngày trước khi nào giao tới?'",
                "output": "Chào bạn! Mình đã kiểm tra đơn hàng #VN2026 của bạn hiện đã tới bưu cục phát tại quận của bạn và dự kiến sẽ được giao trong hôm nay. Cảm ơn bạn đã kiên nhẫn nhé!",
                "text": "Phản hồi chăm sóc khách hàng tra cứu trạng thái giao hàng #VN2026.",
                "label": 1,
            },
        ],
    }

    @classmethod
    def danh_sach_bo_mau(cls) -> Dict[str, int]:
        """Liệt kê tất cả các bộ dữ liệu mẫu tiếng Việt có sẵn cùng số bản ghi gốc."""
        return {ten: len(ds) for ten, ds in cls.BO_DU_LIEU_MAU_TIENG_VIET.items()}

    def sinh_them_mau(self, ten_bo: str = "uit_vsfc", so_luong: int = 20) -> Dict[str, Any]:
        """
        Tự động nhân bản và biến thể hóa (Data Augmentation) một bộ dữ liệu mẫu tiếng Việt
        lên `so_luong` bản ghi để phục vụ huấn luyện hoặc benchmark quy mô lớn.
        """
        key = ten_bo.lower().strip()
        if key not in self.BO_DU_LIEU_MAU_TIENG_VIET:
            hop_le = ", ".join(self.BO_DU_LIEU_MAU_TIENG_VIET.keys())
            raise ValueError(f"Bộ dữ liệu '{ten_bo}' không tồn tại. Chọn một trong: {hop_le}")

        goc = self.BO_DU_LIEU_MAU_TIENG_VIET[key]
        mo_rong: List[Dict[str, Any]] = []
        hau_to = ["(mẫu chuẩn)", "(thực tế)", "(kiểm chứng)", "(mở rộng)", "(nâng cao)"]

        for i in range(max(1, so_luong)):
            mau_goc = dict(goc[i % len(goc)])
            tag = hau_to[i % len(hau_to)]
            if i >= len(goc):
                if "text" in mau_goc:
                    mau_goc["text"] = f"{mau_goc['text']} {tag}"
                if "instruction" in mau_goc:
                    mau_goc["instruction"] = f"{mau_goc['instruction']} [{i + 1}]"
            mo_rong.append(mau_goc)

        ten_file = f"{key}_aug_{so_luong}.jsonl"
        commit_hash = self.nhap_danh_sach(mo_rong, ten_file)
        return {
            "ten_dataset": ten_file,
            "commit_hash": commit_hash,
            "so_ban_ghi": len(mo_rong),
            "du_lieu": mo_rong,
        }

    def tai_bo_mau(self, ten_bo: str = "uit_vsfc") -> Dict[str, Any]:
        """
        Tải một bộ dữ liệu chuẩn tiếng Việt dựng sẵn (`uit_vsfc`, `viquad`, `phoner`, `vmlu`)
        và lưu tự động vào DataLake kèm băm phiên bản SHA-256.
        """
        key = ten_bo.lower().strip()
        if key not in self.BO_DU_LIEU_MAU_TIENG_VIET:
            hop_le = ", ".join(self.BO_DU_LIEU_MAU_TIENG_VIET.keys())
            raise ValueError(f"Bộ dữ liệu '{ten_bo}' không tồn tại. Chọn một trong: {hop_le}")

        du_lieu = self.BO_DU_LIEU_MAU_TIENG_VIET[key]
        ten_file = f"{key}.jsonl"
        commit_hash = self.nhap_danh_sach(du_lieu, ten_file)
        return {
            "ten_dataset": ten_file,
            "commit_hash": commit_hash,
            "so_ban_ghi": len(du_lieu),
            "du_lieu": du_lieu,
        }

    def tai_bo_du_lieu_mau(self, ten_bo: str = "uit_vsfc") -> List[Dict[str, Any]]:
        """Trả về trực tiếp danh sách bản ghi của bộ dữ liệu mẫu tiếng Việt."""
        return self.tai_bo_mau(ten_bo)["du_lieu"]

    def tai_tu_huggingface(
        self,
        ten_hf: str,
        split: str = "train",
        gioi_han: int = 100,
        che_do_offline: bool = True,
    ) -> Dict[str, Any]:
        """
        Tải trực tiếp bộ dữ liệu từ HuggingFace Datasets (`datasets.load_dataset`) vào DataLake.
        Tự động chuyển sang bộ dữ liệu mẫu tiếng Việt nội bộ nếu chạy ở `che_do_offline=True` hoặc không có mạng.
        """
        if not che_do_offline:
            try:
                from datasets import load_dataset  # type: ignore

                ds = load_dataset(ten_hf, split=split)
                records = [dict(ds[i]) for i in range(min(len(ds), gioi_han))]
                safe_name = ten_hf.replace("/", "_") + ".jsonl"
                commit_hash = self.nhap_danh_sach(records, safe_name)
                return {
                    "ten_dataset": safe_name,
                    "commit_hash": commit_hash,
                    "so_ban_ghi": len(records),
                    "nguon": "huggingface",
                }
            except Exception:
                pass

        # Fallback về bộ dữ liệu mẫu tiếng Việt phù hợp nhất
        lower_name = ten_hf.lower()
        if "quad" in lower_name or "qa" in lower_name:
             fallback_key = "viquad"
        elif "ner" in lower_name:
            fallback_key = "phoner"
        elif "vmlu" in lower_name:
            fallback_key = "vmlu"
        else:
            fallback_key = "uit_vsfc"

        kq = self.tai_bo_mau(fallback_key)
        kq["nguon"] = f"offline_fallback:{fallback_key}"
        return kq
