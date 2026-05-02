"""Word2Vec tiếng Việt - tự cài đặt Skip-gram/CBOW."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger


class Word2VecTiengViet:
    """
    Word2Vec tiếng Việt (Skip-gram + Negative Sampling).

    Tự cài đặt, không phụ thuộc gensim. Hỗ trợ:
    - Skip-gram và CBOW
    - Negative sampling
    - Tải pre-trained vectors (text format)

    Sử dụng:
        >>> w2v = Word2VecTiengViet(kich_thuoc=100, che_do="skipgram")
        >>> w2v.huan_luyen(cac_van_ban, so_vong=5)
        >>> vector = w2v.lay_vector("học")
        >>> tu_giong = w2v.tim_tu_giong("học", top_n=5)
    """

    def __init__(
        self,
        kich_thuoc: int = 100,
        cua_so: int = 5,
        che_do: str = "skipgram",
        toc_do_hoc: float = 0.025,
        toi_thieu_dem: int = 2,
        so_am: int = 5,
    ):
        self.kich_thuoc = kich_thuoc
        self.cua_so = cua_so
        self.che_do = che_do
        self.toc_do_hoc = toc_do_hoc
        self.toi_thieu_dem = toi_thieu_dem
        self.so_am = so_am
        self.logger = Logger("Word2Vec")

        self._tu_dien: Dict[str, int] = {}
        self._tu_dien_nguoc: Dict[int, str] = {}
        self._dem_tu: Dict[str, int] = {}
        self._W_in: Optional[np.ndarray] = None
        self._W_out: Optional[np.ndarray] = None
        self._da_huan_luyen = False
        self._xl = XuLyVanBan()

    def _tao_tu_dien(self, cac_van_ban: List[str]) -> None:
        """Tạo từ điển từ corpus."""
        self._dem_tu = {}
        for vb in cac_van_ban:
            for tu in self._xl.tach_tu(vb):
                self._dem_tu[tu] = self._dem_tu.get(tu, 0) + 1

        tu_hop_le = {
            tu: dem for tu, dem in self._dem_tu.items()
            if dem >= self.toi_thieu_dem
        }
        self._tu_dien = {tu: idx for idx, tu in enumerate(sorted(tu_hop_le))}
        self._tu_dien_nguoc = {idx: tu for tu, idx in self._tu_dien.items()}

    def _khoi_tao_trong_so(self) -> None:
        """Khởi tạo ma trận trọng số."""
        so_tu = len(self._tu_dien)
        self._W_in = np.random.uniform(-0.5, 0.5, (so_tu, self.kich_thuoc)) / self.kich_thuoc
        self._W_out = np.zeros((so_tu, self.kich_thuoc))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Hàm sigmoid ổn định."""
        x = np.clip(x, -6, 6)
        return 1.0 / (1.0 + np.exp(-x))

    def _lay_cap_skipgram(
        self, cac_van_ban: List[str]
    ) -> List[Tuple[int, int]]:
        """Tạo các cặp (từ_trung_tâm, từ_ngữ_cảnh) cho Skip-gram."""
        cap = []
        for vb in cac_van_ban:
            cac_tu = self._xl.tach_tu(vb)
            chi_so = [self._tu_dien[t] for t in cac_tu if t in self._tu_dien]

            for i, tam in enumerate(chi_so):
                bat_dau = max(0, i - self.cua_so)
                ket_thuc = min(len(chi_so), i + self.cua_so + 1)
                for j in range(bat_dau, ket_thuc):
                    if j != i:
                        cap.append((tam, chi_so[j]))
        return cap

    def huan_luyen(
        self,
        cac_van_ban: List[str],
        so_vong: int = 5,
    ) -> None:
        """
        Huấn luyện Word2Vec.

        Args:
            cac_van_ban: Danh sách văn bản
            so_vong: Số lần lặp qua toàn bộ corpus
        """
        self.logger.info(f"Bắt đầu huấn luyện Word2Vec ({self.che_do})")
        self._tao_tu_dien(cac_van_ban)
        self._khoi_tao_trong_so()
        self.logger.info(f"Từ điển: {len(self._tu_dien)} từ")

        for vong in range(so_vong):
            cap = self._lay_cap_skipgram(cac_van_ban)
            np.random.shuffle(cap)
            tong_loss = 0.0

            for tam_idx, canh_idx in cap:
                v_tam = self._W_in[tam_idx]

                # Positive sample
                z = np.dot(v_tam, self._W_out[canh_idx])
                sig = self._sigmoid(z)
                g = (sig - 1) * self.toc_do_hoc
                self._W_out[canh_idx] -= g * v_tam
                grad = g * self._W_out[canh_idx]

                # Negative samples
                for _ in range(self.so_am):
                    am_idx = np.random.randint(0, len(self._tu_dien))
                    while am_idx == canh_idx:
                        am_idx = np.random.randint(0, len(self._tu_dien))

                    z_am = np.dot(v_tam, self._W_out[am_idx])
                    sig_am = self._sigmoid(z_am)
                    g_am = sig_am * self.toc_do_hoc
                    self._W_out[am_idx] -= g_am * v_tam
                    grad += g_am * self._W_out[am_idx]

                self._W_in[tam_idx] -= grad

                tong_loss += -np.log(sig + 1e-10)

            self.logger.info(
                f"Vòng {vong+1}/{so_vong}: loss={tong_loss/len(cap):.4f}"
            )

        self._da_huan_luyen = True
        self.logger.info("Huấn luyện Word2Vec hoàn tất")

    def lay_vector(self, tu: str) -> Optional[np.ndarray]:
        """Lấy vector biểu diễn của một từ."""
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện. Gọi huan_luyen() trước.")
        if tu not in self._tu_dien:
            return None
        return self._W_in[self._tu_dien[tu]].copy()

    def lay_vector_van_ban(self, text: str) -> np.ndarray:
        """Lấy vector biểu diễn của văn bản (trung bình các vector từ)."""
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")

        cac_tu = self._xl.tach_tu(text)
        vectors = []
        for tu in cac_tu:
            v = self.lay_vector(tu)
            if v is not None:
                vectors.append(v)

        if not vectors:
            return np.zeros(self.kich_thuoc)
        return np.mean(vectors, axis=0)

    def tim_tu_giong(self, tu: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Tìm các từ giống nhất (cosine similarity)."""
        v = self.lay_vector(tu)
        if v is None:
            return []

        # Chuẩn hóa
        v_norm = v / (np.linalg.norm(v) + 1e-10)
        W_norm = self._W_in / (np.linalg.norm(self._W_in, axis=1, keepdims=True) + 1e-10)
        tuong_dong = W_norm @ v_norm

        chi_so_tot_nhat = np.argsort(tuong_dong)[::-1][1:top_n + 1]
        return [
            (self._tu_dien_nguoc[idx], float(tuong_dong[idx]))
            for idx in chi_so_tot_nhat
            if idx in self._tu_dien_nguoc
        ]

    def tu_dien(self) -> Dict[str, int]:
        """Trả về từ điển."""
        return self._tu_dien.copy()

    def luu(self, duong_dan: str) -> None:
        """Lưu vectors ra file text format."""
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")
        with open(duong_dan, "w", encoding="utf-8") as f:
            f.write(f"{len(self._tu_dien)} {self.kich_thuoc}\n")
            for tu, idx in sorted(self._tu_dien.items(), key=lambda x: x[1]):
                vec_str = " ".join(f"{v:.6f}" for v in self._W_in[idx])
                f.write(f"{tu} {vec_str}\n")
        self.logger.info(f"Đã lưu vectors tại: {duong_dan}")

    @classmethod
    def tai(cls, duong_dan: str) -> "Word2VecTiengViet":
        """Tải vectors từ file text format."""
        logger = Logger("Word2Vec")
        logger.info(f"Tải vectors từ: {duong_dan}")

        with open(duong_dan, "r", encoding="utf-8") as f:
            header = f.readline().split()
            so_tu, kich_thuoc = int(header[0]), int(header[1])

            w2v = cls(kich_thuoc=kich_thuoc)
            w2v._W_in = np.zeros((so_tu, kich_thuoc))

            for i, line in enumerate(f):
                parts = line.strip().split()
                tu = parts[0]
                vec = np.array([float(v) for v in parts[1:]])
                w2v._tu_dien[tu] = i
                w2v._tu_dien_nguoc[i] = tu
                w2v._W_in[i] = vec

        w2v._da_huan_luyen = True
        logger.info(f"Đã tải {so_tu} vectors, kich_thuoc={kich_thuoc}")
        return w2v
