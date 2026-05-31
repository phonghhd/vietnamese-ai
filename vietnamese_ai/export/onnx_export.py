"""XuatONNX - Xuất mô hình sang ONNX format."""

from pathlib import Path
from typing import Any

import numpy as np

from vietnamese_ai.utils.logger import Logger


class XuatONNX:
    """
    Xuất mô hình sang ONNX format để deploy cross-platform.

    ONNX (Open Neural Network Exchange) cho phép chạy mô hình trên:
    - ONNX Runtime (Python, C++, C#, Java)
    - TensorFlow Serving
    - Azure ML
    - AWS SageMaker

    Sử dụng:
        >>> xuat = XuatONNX()
        >>> xuat.xuat_sklearn(mo_hinh, "model.onnx", kich_thuoc_dau_vao=(5,))
        >>> du_doan = xuat.chay_onnx("model.onnx", X_test)
    """

    def __init__(self):
        self.logger = Logger("XuatONNX")

    def xuat_sklearn(
        self,
        mo_hinh: Any,
        duong_dan: str,
        kich_thuoc_dau_vao: tuple = (None,),
        ten_input: str = "input",
        ten_output: str = "output",
    ) -> str:
        """
        Xuất mô hình scikit-learn sang ONNX.

        Args:
            mo_hinh: Mô hình sklearn (hoặc wrapper có ._mo_hinh)
            duong_dan: Đường dẫn file .onnx
            kich_thuoc_dau_vao: Shape của input (VD: (None, 5))
            ten_input: Tên input
            ten_output: Tên output

        Returns:
            Đường dẫn file đã lưu
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError("Cần cài đặt skl2onnx: pip install skl2onnx onnxruntime")

        model = mo_hinh
        if hasattr(mo_hinh, "_mo_hinh"):
            model = mo_hinh._mo_hinh

        initial_type = [(ten_input, FloatTensorType(kich_thuoc_dau_vao))]

        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)

        with open(duong_dan, "wb") as f:
            f.write(onnx_model.SerializeToString())

        self.logger.info(f"Đã xuất ONNX: {duong_dan} ({duong_dan.stat().st_size} bytes)")
        return str(duong_dan)

    def chay_onnx(self, duong_dan: str, X: np.ndarray) -> np.ndarray:
        """
        Chạy dự đoán với mô hình ONNX.

        Args:
            duong_dan: Đường dẫn file .onnx
            X: Dữ liệu đầu vào

        Returns:
            Kết quả dự đoán
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Cần cài đặt onnxruntime: pip install onnxruntime")

        X = np.asarray(X, dtype=np.float32)
        session = ort.InferenceSession(duong_dan)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        ket_qua = session.run([output_name], {input_name: X})
        return ket_qua[0]

    def kiem_tra_onnx(self, duong_dan: str) -> dict:
        """Kiểm tra thông tin mô hình ONNX."""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(duong_dan)
            inputs = [
                {"name": inp.name, "shape": inp.shape, "type": inp.type}
                for inp in session.get_inputs()
            ]
            outputs = [
                {"name": out.name, "shape": out.shape, "type": out.type}
                for out in session.get_outputs()
            ]
            return {"inputs": inputs, "outputs": outputs, "providers": session.get_providers()}
        except ImportError:
            raise ImportError("Cần cài đặt onnxruntime: pip install onnxruntime")
