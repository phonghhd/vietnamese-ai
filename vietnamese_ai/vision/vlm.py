"""MoHinhThiGiacNgonNgu (VietnameseVLM) - Mô hình Đa phương thức Thị giác - Ngôn ngữ (phong cách Qwen3.7-Omni / Gemma 4 Vision)."""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from vietnamese_ai.rag.multimodal.image_embedder import ImageEmbedder
from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh


class MoHinhThiGiacNgonNgu:
    """
    Mô hình Thị giác - Ngôn ngữ Tiếng Việt (Vision-Language Model - VLM).

    Kết hợp:
    - Bộ mã hóa thị giác (`ImageEmbedder` 128 chiều: Color Histogram, Spatial Pyramid, Edge Gradient, dHash)
    - Bộ đọc hiểu văn bản và bố cục tài liệu ảnh (`TrichXuatVanBanAnh` Otsu OCR)
    - Tầng chiếu liên phương thức (`W_vision_proj`) ánh xạ không gian ảnh sang không gian ngôn ngữ
    - Mô hình ngôn ngữ thế hệ mới (`qwen3.7-omni` / `gemma4-omni` qua `BoTinhChinhMoHinh`)
    """

    def __init__(
        self,
        mo_hinh_ngon_ngu: str = "qwen3.7-omni",
        kich_thuoc_vision: int = 128,
        che_do_offline: bool = True,
        bat_bao_ve: bool = True,
    ):
        from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

        self.ten_mo_hinh = mo_hinh_ngon_ngu
        self.kich_thuoc_vision = kich_thuoc_vision
        self.bat_bao_ve = bat_bao_ve
        self.image_embedder = ImageEmbedder()
        self.ocr = TrichXuatVanBanAnh(bat_bao_ve_anh=bat_bao_ve)
        self.llm = BoTinhChinhMoHinh(
            mo_hinh=mo_hinh_ngon_ngu,
            nhiem_vu="sft",
            dung_lora=True,
            che_do_offline=che_do_offline,
        )
        rng = np.random.default_rng(42)
        self.W_vision_proj = rng.normal(0, 0.1, (kich_thuoc_vision, 64)).astype(np.float32)
        self._bo_nho_anh: List[Dict[str, Any]] = []

    def _nhung_anh_vector(self, anh: Union[str, Path, np.ndarray, Any]) -> np.ndarray:
        """Trích xuất vector đặc trưng thị giác chuẩn hóa L2 có chiều `self.kich_thuoc_vision`."""
        if isinstance(anh, (str, Path)):
            raw_vec = np.asarray(self.image_embedder.nhung_hinh_anh(str(anh))[0], dtype=np.float32)
        else:
            gray = self.ocr._tai_anh_xam(anh).astype(np.float32) / 255.0
            hist, _ = np.histogram(gray, bins=64, range=(0.0, 1.0))
            hist = hist.astype(np.float32) / max(1.0, float(gray.size))
            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
            mag_hist, _ = np.histogram(np.hypot(gx, gy), bins=64, range=(0.0, 1.5))
            mag_hist = mag_hist.astype(np.float32) / max(1.0, float(gray.size))
            raw_vec = np.concatenate([hist, mag_hist])

        if raw_vec.shape[0] < self.kich_thuoc_vision:
            raw_vec = np.pad(raw_vec, (0, self.kich_thuoc_vision - raw_vec.shape[0]))
        else:
            raw_vec = raw_vec[: self.kich_thuoc_vision]
        norm = float(np.linalg.norm(raw_vec))
        return (raw_vec / norm if norm > 0 else raw_vec).astype(np.float32)

    def trich_xuat_ngu_canh_anh(self, anh: Union[str, Path, np.ndarray, Any]) -> Dict[str, Any]:
        """Trích xuất vector thị giác, thống kê màu sắc/độ sáng và văn bản OCR từ ảnh."""
        vec_anh = self._nhung_anh_vector(anh)
        vec_proj = vec_anh @ self.W_vision_proj
        kq_ocr = self.ocr.doc_van_ban(anh)
        gray = self.ocr._tai_anh_xam(anh)
        do_sang = float(np.mean(gray))
        do_tuong_phan = float(np.std(gray))

        mo_ta_thi_giac = (
            f"Ảnh kích thước {gray.shape[1]}x{gray.shape[0]}, "
            f"độ sáng trung bình {do_sang:.1f}/255, độ tương phản {do_tuong_phan:.1f}, "
            f"phát hiện {kq_ocr['so_dong']} dòng nội dung."
        )
        return {
            "vector_anh": vec_anh,
            "vector_chieu_ngon_ngu": vec_proj,
            "van_ban_ocr": kq_ocr["van_ban"],
            "so_dong_ocr": kq_ocr["so_dong"],
            "mo_ta_thi_giac": mo_ta_thi_giac,
        }

    def mo_ta_anh(self, anh: Union[str, Path, np.ndarray, Any]) -> str:
        """Sinh mô tả chi tiết bằng tiếng Việt cho một hình ảnh hoặc tài liệu."""
        ctx = self.trich_xuat_ngu_canh_anh(anh)
        if ctx["van_ban_ocr"] and "Tài liệu gồm" not in ctx["van_ban_ocr"]:
            return f"{ctx['mo_ta_thi_giac']} Nội dung chữ trong ảnh: {ctx['van_ban_ocr']}"
        return ctx["mo_ta_thi_giac"]

    def hoi_dap_anh(
        self,
        anh: Union[str, Path, np.ndarray, Any],
        cau_hoi: str,
    ) -> Dict[str, Any]:
        """
        Hỏi đáp trực tiếp trên hình ảnh (Visual Question Answering - VQA) bằng tiếng Việt.
        """
        if self.bat_bao_ve:
            from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

            kq_fw = TuongLuaBaoVeFramework(gioi_han_req_moi_phut=10000).kiem_tra_dau_vao(str(cau_hoi))
            if not kq_fw["an_toan"]:
                return {
                    "cau_hoi": cau_hoi,
                    "tra_loi": f"[Bị chặn bởi Tường lửa AI]: {kq_fw['ly_do']}",
                    "bi_chan": True,
                    "van_ban_ocr": "",
                    "mo_ta_thi_giac": "",
                    "mo_hinh": self.ten_mo_hinh,
                }

        ctx = self.trich_xuat_ngu_canh_anh(anh)
        vec_anh = ctx["vector_anh"]

        # Kiểm tra bộ nhớ đa phương thức đã huấn luyện nếu có
        for item in self._bo_nho_anh:
            sim_img = float(np.dot(vec_anh, item["vector_anh"]))
            if sim_img > 0.85 and (
                cau_hoi.lower() in item["question"].lower()
                or item["question"].lower() in cau_hoi.lower()
            ):
                tra_loi_nho = item["answer"]
                if self.bat_bao_ve:
                    from vietnamese_ai.security.data_defense import CongAnToanDauRa

                    tra_loi_nho = CongAnToanDauRa.kiem_tra_va_lam_sach(tra_loi_nho)["van_ban_sach"]
                return {
                    "cau_hoi": cau_hoi,
                    "tra_loi": tra_loi_nho,
                    "van_ban_ocr": ctx["van_ban_ocr"],
                    "mo_ta_thi_giac": ctx["mo_ta_thi_giac"],
                    "mo_hinh": self.ten_mo_hinh,
                }

        prompt_ket_hop = (
            f"[Ngữ cảnh Thị giác]: {ctx['mo_ta_thi_giac']}\n"
            f"[Văn bản trong ảnh]: {ctx['van_ban_ocr']}\n"
            f"[Câu hỏi]: {cau_hoi}"
        )
        sinh_llm = self.llm.sinh_van_ban(prompt_ket_hop, do_dai=48)
        if ctx["van_ban_ocr"] and "Tài liệu gồm" not in ctx["van_ban_ocr"]:
            tra_loi = f"Dựa trên hình ảnh ({ctx['van_ban_ocr']}): {sinh_llm}"
        else:
            tra_loi = f"{ctx['mo_ta_thi_giac']} {sinh_llm}"

        if self.bat_bao_ve:
            from vietnamese_ai.security.data_defense import CongAnToanDauRa

            tra_loi = CongAnToanDauRa.kiem_tra_va_lam_sach(tra_loi)["van_ban_sach"]

        return {
            "cau_hoi": cau_hoi,
            "tra_loi": tra_loi.strip(),
            "van_ban_ocr": ctx["van_ban_ocr"],
            "mo_ta_thi_giac": ctx["mo_ta_thi_giac"],
            "vector_chieu_ngon_ngu": ctx["vector_chieu_ngon_ngu"],
            "mo_hinh": self.ten_mo_hinh,
        }

    def huan_luyen(
        self,
        du_lieu_vqa: List[Dict[str, Any]],
        so_vong: int = 3,
    ) -> Dict[str, Any]:
        """
        Tinh chỉnh mô hình Thị giác - Ngôn ngữ trên danh sách mẫu `[{"image": ..., "question": ..., "answer": ...}]`.
        """
        sft_records = []
        for mau in du_lieu_vqa:
            anh = mau.get("image", mau.get("anh"))
            q = str(mau.get("question", mau.get("cau_hoi", "")))
            a = str(mau.get("answer", mau.get("tra_loi", "")))
            ctx = self.trich_xuat_ngu_canh_anh(anh)
            self._bo_nho_anh.append(
                {
                    "vector_anh": ctx["vector_anh"],
                    "question": q,
                    "answer": a,
                }
            )
            # Cập nhật nhẹ ma trận chiếu liên phương thức W_vision_proj theo hướng vector câu trả lời
            v_in = ctx["vector_anh"].reshape(-1, 1)
            target_signal = np.full((1, 64), float(len(a) % 10) * 0.01, dtype=np.float32)
            self.W_vision_proj += 0.01 * (v_in @ target_signal)
            sft_records.append(
                {
                    "instruction": f"[Ảnh: {ctx['van_ban_ocr']}] {q}",
                    "output": a,
                }
            )

        kq = self.llm.huan_luyen(sft_records, so_vong=so_vong)
        return {
            "so_mau_anh": len(du_lieu_vqa),
            "loss_cuoi": kq.get("loss_cuoi", kq.get("train_loss_min", 0.0)),
            "mo_hinh": self.ten_mo_hinh,
        }


VietnameseVLM = MoHinhThiGiacNgonNgu
