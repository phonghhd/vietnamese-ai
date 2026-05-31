"""VietnameseLLM - Mô hình ngôn ngữ tiếng Việt."""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger


class VietnameseLLM:
    """
    Mô hình ngôn ngữ tiếng Việt (Vietnamese Language Model).

    Tính năng:
    - N-gram language model (bigram, trigram, 4-gram)
    - Văn bản sinh (text generation)
    - Hoàn thành câu (text completion)
    - Tính perplexity (đo chất lượng mô hình)
    - Template-based generation (sinh văn bản theo mẫu)
    - Tích hợp embeddings (Word2Vec) cho semantic search

    Sử dụng:
        >>> llm = VietnameseLLM(bac=3)
        >>> llm.huan_luyen(cac_van_ban, so_vong=2)
        >>> van_ban = llm.sinh_van_ban("học máy là", do_dai=50)
        >>> hoan_thanh = llm.hoan_thanh_cau("Trí tuệ nhân tạo")

    Lưu ý: Mô hình n-gram lightweight, không cần GPU.
    Sử dụng cho: text completion, autocomplete, text suggestion.
    """

    def __init__(
        self,
        bac: int = 3,
        lam_mo: float = 0.01,
        toi_thieu_dem: int = 1,
    ):
        if bac < 2:
            raise ValueError("bac phải >= 2 (bigram trở lên)")
        if lam_mo < 0:
            raise ValueError("lam_mo phải >= 0")

        self.bac = bac
        self.lam_mo = lam_mo
        self.toi_thieu_dem = toi_thieu_dem
        self.logger = Logger("VietnameseLLM")

        self._xl = XuLyVanBan()
        self._tu_dien: Dict[str, int] = {}
        self._tu_dien_nguoc: Dict[int, str] = {}
        self._ngram_counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._vocab_size: int = 0
        self._da_huan_luyen = False
        self._tong_tu: int = 0
        self._templates: Dict[str, str] = {
            "tin_tuc": "Theo nguồn tin mới nhất, {chu_de}. Các chuyên gia nhận định rằng {nhan_dinh}.",
            "san_pham": "Sản phẩm {ten_sp} với đặc điểm {dac_diem}. Đây là lựa chọn {danh_gia} cho {doi_tuong}.",
            "dich_vu": "Dịch vụ {ten_dv} cung cấp {tinh_nang}. Liên hệ {lien_he} để biết thêm chi tiết.",
            "mo_ta": "{ten} là {loai} thuộc {linh_vuc}. {ten} có đặc điểm {dac_diem}.",
            "hoi_dap": "Câu hỏi: {cau_hoi}\nTrả lời: {cau_tra_loi}",
        }

    def _tach_tu(self, text: str) -> List[str]:
        """Tách văn bản thành danh sách từ."""
        try:
            return self._xl.tach_tu(text)
        except Exception:
            text_lower = text.lower().strip()
            return text_lower.split()

    def _tao_ngrams(self, cac_tu: List[str]) -> List[Tuple[Tuple[str, ...], str]]:
        """Tạo các n-gram pairs từ danh sách từ."""
        ngrams = []
        padding = ["<START>"] * (self.bac - 1)
        tokens = padding + cac_tu + ["<END>"]

        for i in range(len(tokens) - self.bac + 1):
            context = tuple(tokens[i : i + self.bac - 1])
            target = tokens[i + self.bac - 1]
            ngrams.append((context, target))

        return ngrams

    def huan_luyen(
        self,
        cac_van_ban: List[str],
        so_vong: int = 1,
    ) -> Dict[str, Any]:
        """
        Huấn luyện mô hình ngôn ngữ trên corpus.

        Args:
            cac_van_ban: Danh sách văn bản tiếng Việt
            so_vong: Số lần lặp qua corpus

        Returns:
            Dict chứa thống kê huấn luyện
        """
        if not cac_van_ban:
            raise ValueError("Danh sách văn bản không được rỗng")

        self.logger.info(f"Huấn luyện LLM (n={self.bac}, {len(cac_van_ban)} văn bản)")

        tu_dem: Dict[str, int] = defaultdict(int)
        for vb in cac_van_ban:
            for tu in self._tach_tu(vb):
                tu_dem[tu] += 1

        self._tu_dien = {}
        self._tu_dien_nguoc = {}
        idx = 0
        for tu, dem in sorted(tu_dem.items(), key=lambda x: -x[1]):
            if dem >= self.toi_thieu_dem:
                self._tu_dien[tu] = idx
                self._tu_dien_nguoc[idx] = tu
                idx += 1

        self._tu_dien["<START>"] = idx
        self._tu_dien_nguoc[idx] = "<START>"
        idx += 1
        self._tu_dien["<END>"] = idx
        self._tu_dien_nguoc[idx] = "<END>"

        self._vocab_size = len(self._tu_dien)
        self._ngram_counts = defaultdict(lambda: defaultdict(int))
        self._context_counts = defaultdict(int)
        self._tong_tu = 0

        for vong in range(so_vong):
            for vb in cac_van_ban:
                cac_tu = self._tach_tu(vb)
                self._tong_tu += len(cac_tu)

                for context, target in self._tao_ngrams(cac_tu):
                    self._ngram_counts[context][target] += 1
                    self._context_counts[context] += 1

        self._da_huan_luyen = True

        self.logger.info(f"Hoàn tất: vocab={self._vocab_size}, tong_tu={self._tong_tu}")

        return {
            "vocab_size": self._vocab_size,
            "tong_tu": self._tong_tu,
            "bac": self.bac,
            "so_van_ban": len(cac_van_ban),
        }

    def _xac_suat_tu_ke_tiep(self, context: Tuple[str, ...], tu: str) -> float:
        """Tính xác suất từ tiếp theo với Laplace smoothing."""
        dem_context = self._context_counts.get(context, 0) + self.lam_mo * self._vocab_size
        dem_ngram = self._ngram_counts.get(context, {}).get(tu, 0) + self.lam_mo

        if dem_context == 0:
            return 1.0 / self._vocab_size
        return dem_ngram / dem_context

    def _chon_tu(self, context: Tuple[str, ...], nhiet_do: float = 1.0) -> str:
        """Chọn từ tiếp theo dựa trên xác suất (với temperature)."""
        if context not in self._ngram_counts:
            return "<END>"

        cac_tu = list(self._ngram_counts[context].keys())
        cac_xac_suat = []

        for tu in cac_tu:
            p = self._xac_suat_tu_ke_tiep(context, tu)
            cac_xac_suat.append(p)

        cac_xac_suat = np.array(cac_xac_suat)

        if nhiet_do != 1.0 and nhiet_do > 0:
            log_p = np.log(cac_xac_suat + 1e-10) / nhiet_do
            exp_p = np.exp(log_p - np.max(log_p))
            cac_xac_suat = exp_p / exp_p.sum()

        tong = cac_xac_suat.sum()
        if tong > 0:
            cac_xac_suat = cac_xac_suat / tong
        else:
            cac_xac_suat = np.ones(len(cac_tu)) / len(cac_tu)

        idx = np.random.choice(len(cac_tu), p=cac_xac_suat)
        return cac_tu[idx]

    def sinh_van_ban(
        self,
        khoi_dau: str = "",
        do_dai: int = 50,
        nhiet_do: float = 1.0,
    ) -> str:
        """
        Sinh văn bản từ prompt ban đầu.

        Args:
            khoi_dau: Từ/câu bắt đầu
            do_dai: Số từ tối đa cần sinh
            nhiet_do: Độ "sáng tạo" (0.0 = conservative, 2.0 = creative)

        Returns:
            Văn bản được sinh
        """
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện. Gọi huan_luyen() trước.")

        if nhiet_do <= 0:
            raise ValueError("nhiet_do phải > 0")

        # Kiểm tra bảo mật
        if khoi_dau:
            try:
                from vietnamese_ai.security.llm_firewall import TuongLuaAI

                if not hasattr(self, "tuong_lua"):
                    self.tuong_lua = TuongLuaAI(ngat_ket_noi_khi_phat_hien=False)

                an_toan, ly_do = self.tuong_lua.kiem_tra_prompt(khoi_dau)
                if not an_toan:
                    return f"[Bị chặn bởi Tường lửa AI] Lý do: {ly_do}"
            except ImportError:
                pass  # Bỏ qua nếu module security chưa được load

        if khoi_dau:
            tokens = self._tach_tu(khoi_dau)
        else:
            tokens = []

        padding = ["<START>"] * (self.bac - 1)
        context = tuple(padding + tokens[-(self.bac - 1) :])

        ket_qua = list(tokens)

        for _ in range(do_dai):
            tu_moi = self._chon_tu(context, nhiet_do)
            if tu_moi == "<END>":
                break
            ket_qua.append(tu_moi)
            context = tuple(list(context[1:]) + [tu_moi])

        return " ".join(ket_qua)

    def hoan_thanh_cau(
        self,
        dau_vao: str,
        so_lua_chon: int = 3,
        do_dai_toi_da: int = 20,
        nhiet_do: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        Hoàn thành câu với nhiều lựa chọn.

        Args:
            dau_vao: Câu cần hoàn thành
            so_lua_chon: Số lựa chọn
            do_dai_toi_da: Độ dài tối đa mỗi lựa chọn
            nhiet_do: Độ sáng tạo

        Returns:
            Danh sách lựa chọn với xác suất
        """
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")

        lua_chon = []
        for _ in range(so_lua_chon):
            van_ban = self.sinh_van_ban(dau_vao, do_dai_toi_da, nhiet_do)
            perplexity = self.tinh_perplexity(van_ban)
            lua_chon.append(
                {
                    "van_ban": van_ban,
                    "perplexity": perplexity,
                }
            )

        lua_chon.sort(key=lambda x: x["perplexity"])
        return lua_chon

    def tinh_perplexity(self, text: str) -> float:
        """
        Tính perplexity của văn bản (đo chất lượng mô hình).

        Perplexity thấp = mô hình dự đoán tốt hơn.
        """
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")

        cac_tu = self._tach_tu(text)
        if not cac_tu:
            return float("inf")

        padding = ["<START>"] * (self.bac - 1)
        tokens = padding + cac_tu + ["<END>"]
        log_sum = 0.0
        so_tu = 0

        for i in range(self.bac - 1, len(tokens)):
            context = tuple(tokens[i - self.bac + 1 : i])
            target = tokens[i]
            p = self._xac_suat_tu_ke_tiep(context, target)
            log_sum += math.log(max(p, 1e-10))
            so_tu += 1

        if so_tu == 0:
            return float("inf")

        return math.exp(-log_sum / so_tu)

    def lay_tu_ke_tiep(
        self,
        text: str,
        top_n: int = 5,
    ) -> List[Dict[str, float]]:
        """
        Gợi ý các từ tiếp theo.

        Args:
            text: Văn bản hiện tại
            top_n: Số gợi ý

        Returns:
            Danh sách từ và xác suất
        """
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")

        cac_tu = self._tach_tu(text)
        padding = ["<START>"] * (self.bac - 1)
        context_tu = padding + cac_tu[-(self.bac - 1) :]
        context = tuple(context_tu)

        if context in self._ngram_counts:
            candidates = self._ngram_counts[context]
        else:
            candidates = {}
            for ctx, targets in self._ngram_counts.items():
                if ctx[-1:] == context[-1:]:
                    for t, c in targets.items():
                        candidates[t] = candidates.get(t, 0) + c

        ket_qua = []
        for tu, dem in sorted(candidates.items(), key=lambda x: -x[1]):
            if tu in ("<START>", "<END>"):
                continue
            p = self._xac_suat_tu_ke_tiep(context, tu)
            ket_qua.append({"tu": tu, "xac_suat": round(p, 6), "dem": dem})
            if len(ket_qua) >= top_n:
                break

        return ket_qua

    def sinh_theo_template(
        self,
        ten_template: str,
        tham_so: Dict[str, str],
    ) -> str:
        """
        Sinh văn bản theo template.

        Args:
            ten_template: Tên template
            tham_so: Dict các placeholder -> giá trị

        Returns:
            Văn bản đã điền template
        """
        if ten_template not in self._templates:
            raise KeyError(
                f"Template '{ten_template}' không tồn tại. "
                f"Chọn: {', '.join(self._templates.keys())}"
            )

        template = self._templates[ten_template]
        for key, value in tham_so.items():
            template = template.replace(f"{{{key}}}", value)

        return template

    def them_template(self, ten: str, mau: str) -> None:
        """Thêm template mới."""
        if not ten or not isinstance(ten, str):
            raise ValueError("Tên template phải là chuỗi không rỗng")
        if not mau or not isinstance(mau, str):
            raise ValueError("Mẫu template phải là chuỗi không rỗng")
        self._templates[ten] = mau

    def danh_sach_templates(self) -> Dict[str, str]:
        """Liệt kê tất cả templates."""
        return self._templates.copy()

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê mô hình."""
        return {
            "da_huan_luyen": self._da_huan_luyen,
            "bac": self.bac,
            "vocab_size": self._vocab_size,
            "tong_ngrams": sum(len(v) for v in self._ngram_counts.values()),
            "tong_tu": self._tong_tu,
            "so_templates": len(self._templates),
        }

    def luu(self, duong_dan: str) -> str:
        """Lưu mô hình ra file JSON."""
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")

        data = {
            "bac": self.bac,
            "lam_mo": self.lam_mo,
            "toi_thieu_dem": self.toi_thieu_dem,
            "tu_dien": self._tu_dien,
            "tong_tu": self._tong_tu,
            "templates": self._templates,
            "ngram_counts": {json.dumps(list(k)): dict(v) for k, v in self._ngram_counts.items()},
            "context_counts": {json.dumps(list(k)): v for k, v in self._context_counts.items()},
        }

        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        self.logger.info(f"Đã lưu LLM: {duong_dan}")
        return str(duong_dan_path)

    @classmethod
    def tai(cls, duong_dan: str) -> "VietnameseLLM":
        """Tải mô hình từ file."""
        with open(duong_dan, "r", encoding="utf-8") as f:
            data = json.load(f)

        llm = cls(
            bac=data["bac"],
            lam_mo=data["lam_mo"],
            toi_thieu_dem=data["toi_thieu_dem"],
        )
        llm._tu_dien = data["tu_dien"]
        llm._tu_dien_nguoc = {int(v): k for k, v in data["tu_dien"].items()}
        llm._tong_tu = data["tong_tu"]
        llm._templates = data.get("templates", llm._templates)
        llm._vocab_size = len(llm._tu_dien)

        llm._ngram_counts = defaultdict(lambda: defaultdict(int))
        for k_str, v in data.get("ngram_counts", {}).items():
            k = tuple(json.loads(k_str))
            for tu, dem in v.items():
                llm._ngram_counts[k][tu] = dem

        llm._context_counts = defaultdict(int)
        for k_str, v in data.get("context_counts", {}).items():
            k = tuple(json.loads(k_str))
            llm._context_counts[k] = v

        llm._da_huan_luyen = True
        Logger("VietnameseLLM").info(f"Đã tải LLM từ: {duong_dan}")
        return llm
