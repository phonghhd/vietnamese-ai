"""HocRutGon - Knowledge Distillation cho mô hình ML."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class HocRutGon:
    """
    Knowledge Distillation - huấn luyện mô hình nhỏ (student) từ mô hình lớn (teacher).

    Hỗ trợ:
    - Soft label distillation
    - Feature-based distillation
    - Temperature scaling
    - Ensemble distillation

    Sử dụng:
        >>> teacher = PhanLoai(thuat_toan="rung_ngau_nhien")
        >>> teacher.huan_luyen(X_train, y_train)
        >>>
        >>> distiller = HocRutGon(teacher=teacher, nhiet_do=3.0)
        >>> student = PhanLoai(thuat_toan="logistic")
        >>> distiller.huan_luyen(student, X_train, y_train)
    """

    def __init__(
        self,
        teacher: Any,
        nhiet_do: float = 3.0,
        alpha: float = 0.5,
        so_vong: int = 10,
        ham_loss: str = "kl_divergence",
    ):
        if ham_loss not in ("kl_divergence", "mse", "cross_entropy"):
            raise ValueError("ham_loss phải là: kl_divergence, mse, cross_entropy")

        self.teacher = teacher
        self.nhiet_do = nhiet_do
        self.alpha = alpha
        self.so_vong = so_vong
        self.ham_loss = ham_loss
        self.logger = Logger("HocRutGon")

        self._lich_su: List[Dict[str, Any]] = []

    def huan_luyen(
        self,
        student: Any,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện student model bằng knowledge distillation.

        Args:
            student: Mô hình student
            X: Dữ liệu huấn luyện
            y: Nhãn thực
            X_val: Dữ liệu validation
            y_val: Nhãn validation

        Returns:
            {student, lich_su, thong_ke}
        """
        self.logger.info("Bắt đầu Knowledge Distillation")
        bat_dau = time.time()

        # Lấy soft labels từ teacher
        teacher_probs = self._lay_teacher_probs(X)
        self.logger.info(f"Teacher soft labels shape: {teacher_probs.shape}")

        # Nếu student có phương thức huan_luyen_voi_soft_labels
        if hasattr(student, "huan_luyen_voi_soft_labels"):
            student.huan_luyen_voi_soft_labels(X, teacher_probs, y)
        else:
            # Fallback: huấn luyện bình thường
            student.huan_luyen(X, y)

        # Đánh giá
        student_acc = student.danh_gia(X, y)
        teacher_acc = self.teacher.danh_gia(X, y)

        if X_val is not None and y_val is not None:
            student_val_acc = student.danh_gia(X_val, y_val)
            teacher_val_acc = self.teacher.danh_gia(X_val, y_val)
        else:
            student_val_acc = student_acc
            teacher_val_acc = teacher_acc

        thoi_gian = time.time() - bat_dau

        ket_qua = {
            "student": student,
            "teacher_acc": teacher_acc,
            "student_acc": student_acc,
            "teacher_val_acc": teacher_val_acc,
            "student_val_acc": student_val_acc,
            "ty_le_nen": self._tinh_ty_le_nen(student),
            "toc_do": f"{thoi_gian:.2f}s",
            "nhiet_do": self.nhiet_do,
            "alpha": self.alpha,
        }

        self._lich_su.append(ket_qua)
        self.logger.info(
            f"Distillation hoàn tất: teacher={teacher_acc:.4f}, "
            f"student={student_acc:.4f}, thời gian={thoi_gian:.2f}s"
        )

        return ket_qua

    def huan_luyen_ensemble(
        self,
        student: Any,
        teachers: List[Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Distillation từ ensemble teachers.

        Args:
            student: Student model
            teachers: Danh sách teacher models
            X, y: Dữ liệu huấn luyện

        Returns:
            {student, lich_su}
        """
        self.logger.info(f"Ensemble distillation từ {len(teachers)} teachers")

        # Lấy soft labels từ tất cả teachers
        all_probs = []
        for teacher in teachers:
            probs = self._lay_teacher_probs(X, teacher=teacher)
            all_probs.append(probs)

        # Trung bình soft labels
        avg_probs = np.mean(all_probs, axis=0)

        # Huấn luyện student
        if hasattr(student, "huan_luyen_voi_soft_labels"):
            student.huan_luyen_voi_soft_labels(X, avg_probs, y)
        else:
            student.huan_luyen(X, y)

        student_acc = student.danh_gia(X, y)

        return {
            "student": student,
            "student_acc": student_acc,
            "so_teachers": len(teachers),
        }

    def _lay_teacher_probs(
        self,
        X: np.ndarray,
        teacher: Optional[Any] = None,
    ) -> np.ndarray:
        """Lấy xác suất dự đoán từ teacher."""
        teacher = teacher or self.teacher

        if hasattr(teacher, "du_doan_xac_suat"):
            probs = teacher.du_doan_xac_suat(X)
        elif hasattr(teacher, "du_doan"):
            predictions = teacher.du_doan(X)
            # One-hot encode
            n_classes = len(np.unique(predictions))
            probs = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                probs[i, int(pred)] = 1.0
        else:
            raise ValueError("Teacher không có phương thức dự đoán")

        # Temperature scaling
        if self.nhiet_do != 1.0:
            probs = self._temperature_scale(probs)

        return probs

    def _temperature_scale(self, probs: np.ndarray) -> np.ndarray:
        """Áp dụng temperature scaling."""
        log_probs = np.log(probs + 1e-10) / self.nhiet_do
        # Softmax
        exp_probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

    def _tinh_ty_le_nen(self, student: Any) -> float:
        """Tính tỷ lệ nén (teacher/student params)."""
        teacher_params = self._dem_tham_so(self.teacher)
        student_params = self._dem_tham_so(student)

        if student_params == 0:
            return float("inf")
        return teacher_params / max(student_params, 1)

    def _dem_tham_so(self, model: Any) -> int:
        """Đếm số tham số của model."""
        if hasattr(model, "n_features_in_"):
            return model.n_features_in_

        total = 0
        for attr_name in dir(model):
            attr = getattr(model, attr_name, None)
            if isinstance(attr, np.ndarray):
                total += attr.size
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, np.ndarray):
                        total += item.size
        return total

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử distillation."""
        return self._lich_su.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "nhiet_do": self.nhiet_do,
            "alpha": self.alpha,
            "ham_loss": self.ham_loss,
            "so_lan_distill": len(self._lich_su),
        }

    def __repr__(self) -> str:
        return f"HocRutGon(nhiet_do={self.nhiet_do}, alpha={self.alpha})"
