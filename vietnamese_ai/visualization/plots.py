"""BieuDo - Các biểu đồ trực quan hóa."""

from typing import Dict, List, Optional

import numpy as np


class BieuDo:
    """
    Bộ công cụ trực quan hóa dữ liệu.

    Sử dụng:
        >>> bd = BieuDo()
        >>> bd.phan_bo_du_lieu(data, tieu_de="Phân bố dữ liệu")
        >>> bd.matran_nham_lan(y_thuc, y_du_doan)
        >>> bd.lich_su_huan_luyen(loss_history)
    """

    @staticmethod
    def _kiem_tra_matplotlib():
        try:
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            raise ImportError(
                "Cần cài đặt matplotlib: pip install matplotlib"
            )

    @staticmethod
    def phan_bo_du_lieu(
        data: np.ndarray,
        tieu_de: str = "Phân bố dữ liệu",
        ten_truc_x: str = "Giá trị",
        ten_truc_y: str = "Tần suất",
        luu_tai: Optional[str] = None,
    ) -> None:
        """Vẽ biểu đồ phân bố dữ liệu."""
        plt = BieuDo._kiem_tra_matplotlib()

        data = np.asarray(data).flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        plt.title(tieu_de, fontsize=14)
        plt.xlabel(ten_truc_x)
        plt.ylabel(ten_truc_y)
        plt.grid(True, alpha=0.3)

        if luu_tai:
            plt.savefig(luu_tai, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def matran_nham_lan(
        y_thuc: np.ndarray,
        y_du_doan: np.ndarray,
        ten_lop: Optional[List[str]] = None,
        tieu_de: str = "Ma trận nhầm lẫn",
        luu_tai: Optional[str] = None,
    ) -> None:
        """Vẽ ma trận nhầm lẫn."""
        plt = BieuDo._kiem_tra_matplotlib()

        y_thuc, y_du_doan = np.asarray(y_thuc), np.asarray(y_du_doan)
        cac_lop = np.unique(np.concatenate([y_thuc, y_du_doan]))
        so_lop = len(cac_lop)

        ma_tran = np.zeros((so_lop, so_lop), dtype=int)
        for thuc, du_doan in zip(y_thuc, y_du_doan):
            i = np.where(cac_lop == thuc)[0][0]
            j = np.where(cac_lop == du_doan)[0][0]
            ma_tran[i, j] += 1

        plt.figure(figsize=(8, 6))
        plt.imshow(ma_tran, interpolation="nearest", cmap="Blues")
        plt.title(tieu_de, fontsize=14)
        plt.colorbar()

        nhan = ten_lop or [str(lop) for lop in cac_lop]
        tick_marks = np.arange(so_lop)
        plt.xticks(tick_marks, nhan, rotation=45)
        plt.yticks(tick_marks, nhan)

        for i in range(so_lop):
            for j in range(so_lop):
                mau = "white" if ma_tran[i, j] > ma_tran.max() / 2 else "black"
                plt.text(j, i, str(ma_tran[i, j]), ha="center", va="center", color=mau)

        plt.ylabel("Nhãn thực tế")
        plt.xlabel("Nhãn dự đoán")
        plt.tight_layout()

        if luu_tai:
            plt.savefig(luu_tai, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def lich_su_huan_luyen(
        loss_history: List[float],
        tieu_de: str = "Lịch sử huấn luyện",
        ten_truc_x: str = "Vòng lặp",
        ten_truc_y: str = "Loss",
        luu_tai: Optional[str] = None,
    ) -> None:
        """Vẽ biểu đồ loss theo vòng lặp huấn luyện."""
        plt = BieuDo._kiem_tra_matplotlib()

        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, color="steelblue", linewidth=2)
        plt.title(tieu_de, fontsize=14)
        plt.xlabel(ten_truc_x)
        plt.ylabel(ten_truc_y)
        plt.grid(True, alpha=0.3)

        if luu_tai:
            plt.savefig(luu_tai, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def scatter_2d(
        X: np.ndarray,
        nhan: Optional[np.ndarray] = None,
        tam_cum: Optional[np.ndarray] = None,
        tieu_de: str = "Biểu đồ Scatter 2D",
        luu_tai: Optional[str] = None,
    ) -> None:
        """Vẽ biểu đồ scatter 2D (dữ liệu đã giảm chiều)."""
        plt = BieuDo._kiem_tra_matplotlib()

        X = np.asarray(X)
        if X.shape[1] > 2:
            from vietnamese_ai.preprocessing.feature_engineering import TaoDacTrung
            X = TaoDacTrung.giam_chieu_pca(X, so_chieu=2)

        plt.figure(figsize=(10, 8))

        if nhan is not None:
            nhan = np.asarray(nhan)
            cac_lop = np.unique(nhan)
            mau = plt.cm.tab10(np.linspace(0, 1, len(cac_lop)))
            for i, lop in enumerate(cac_lop):
                mask = nhan == lop
                plt.scatter(X[mask, 0], X[mask, 1], c=[mau[i]], label=f"Lớp {lop}", alpha=0.6)
            plt.legend()
        else:
            plt.scatter(X[:, 0], X[:, 1], alpha=0.6, color="steelblue")

        if tam_cum is not None:
            plt.scatter(
                tam_cum[:, 0], tam_cum[:, 1],
                c="red", marker="x", s=200, linewidths=3, label="Tâm cụm"
            )
            plt.legend()

        plt.title(tieu_de, fontsize=14)
        plt.xlabel("Thành phần 1")
        plt.ylabel("Thành phần 2")
        plt.grid(True, alpha=0.3)

        if luu_tai:
            plt.savefig(luu_tai, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def so_sanh_mo_hinh(
        ket_qua: Dict[str, float],
        tieu_de: str = "So sánh mô hình",
        ten_chi_so: str = "Điểm số",
        luu_tai: Optional[str] = None,
    ) -> None:
        """Vẽ biểu đồ so sánh hiệu suất các mô hình."""
        plt = BieuDo._kiem_tra_matplotlib()

        ten = list(ket_qua.keys())
        gia_tri = list(ket_qua.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(ten, gia_tri, color="steelblue", edgecolor="black", alpha=0.7)
        plt.title(tieu_de, fontsize=14)
        plt.ylabel(ten_chi_so)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, gia_tri):
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=10
            )

        plt.tight_layout()

        if luu_tai:
            plt.savefig(luu_tai, dpi=150, bbox_inches="tight")
        plt.show()
