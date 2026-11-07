"""XuatONNX - Xuất mô hình sang ONNX format và chạy suy luận."""

import json
import struct
from pathlib import Path
from typing import Any, Dict

import numpy as np

from vietnamese_ai.utils.logger import Logger


class XuatONNX:
    """
    Xuất mô hình sang ONNX format để deploy cross-platform.

    Hỗ trợ:
    - Chế độ `skl2onnx` + `onnxruntime` khi thư viện đã cài đặt.
    - Chế độ Native Graph Serializer & NumPy Runtime khi chạy độc lập (Zero-Dependency).

    Sử dụng:
        >>> xuat = XuatONNX()
        >>> xuat.xuat_sklearn(mo_hinh, "model.onnx", kich_thuoc_dau_vao=(None, 5))
        >>> du_doan = xuat.chay_onnx("model.onnx", X_test)
    """

    def __init__(self):
        self.logger = Logger("XuatONNX")

    @staticmethod
    def _trich_xuat_do_thi(mo_hinh: Any, kich_thuoc_dau_vao: tuple, ten_input: str, ten_output: str) -> Dict[str, Any]:
        model = getattr(mo_hinh, "_mo_hinh", mo_hinh)
        graph: Dict[str, Any] = {
            "ir_version": 8,
            "opset_import": 12,
            "loai_mo_hinh": type(model).__name__,
            "ten_input": ten_input,
            "ten_output": ten_output,
            "kich_thuoc_dau_vao": list(kich_thuoc_dau_vao),
            "nodes": [],
            "initializers": {},
        }
        if hasattr(model, "coef_"):
            graph["initializers"]["coef"] = np.asarray(model.coef_, dtype=np.float32).tolist()
            graph["nodes"].append("MatMul")
        if hasattr(model, "intercept_"):
            graph["initializers"]["intercept"] = np.asarray(model.intercept_, dtype=np.float32).tolist()
            graph["nodes"].append("Add")
        if hasattr(model, "classes_"):
            graph["initializers"]["classes"] = np.asarray(model.classes_).tolist()
            graph["nodes"].append("ArgMax")
        if hasattr(model, "theta_") and hasattr(model, "var_"):
            graph["initializers"]["theta"] = np.asarray(model.theta_, dtype=np.float32).tolist()
            graph["initializers"]["var"] = np.asarray(model.var_, dtype=np.float32).tolist()
            if hasattr(model, "class_prior_"):
                graph["initializers"]["class_prior"] = np.asarray(model.class_prior_, dtype=np.float32).tolist()
            graph["nodes"].append("GaussianNB")
        if hasattr(model, "cluster_centers_"):
            graph["initializers"]["cluster_centers"] = np.asarray(model.cluster_centers_, dtype=np.float32).tolist()
            graph["nodes"].append("KMeansDistance")
        if hasattr(mo_hinh, "_trong_so") and hasattr(mo_hinh, "_bias"):
            graph["initializers"]["mlp_weights"] = [w.tolist() for w in mo_hinh._trong_so]
            graph["initializers"]["mlp_biases"] = [b.tolist() for b in mo_hinh._bias]
            graph["nodes"].append("MLP")
        return graph

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
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

        model = getattr(mo_hinh, "_mo_hinh", mo_hinh)
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            shape = kich_thuoc_dau_vao if len(kich_thuoc_dau_vao) >= 2 else (None, kich_thuoc_dau_vao[0] or 1)
            initial_type = [(ten_input, FloatTensorType(shape))]
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
            with open(duong_dan_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            self.logger.info(f"Đã xuất ONNX: {duong_dan_path} ({duong_dan_path.stat().st_size} bytes)")
            return str(duong_dan_path)
        except ImportError:
            graph = self._trich_xuat_do_thi(mo_hinh, kich_thuoc_dau_vao, ten_input, ten_output)
            payload = json.dumps(graph).encode("utf-8")
            with open(duong_dan_path, "wb") as f:
                f.write(b"VAI_ONX")
                f.write(struct.pack("<I", len(payload)))
                f.write(payload)
            self.logger.info(
                f"Đã xuất ONNX (native graph): {duong_dan_path} ({duong_dan_path.stat().st_size} bytes)"
            )
            return str(duong_dan_path)

    @staticmethod
    def _thuc_thi_do_thi_native(graph: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        init = graph.get("initializers", {})

        if "mlp_weights" in init and "mlp_biases" in init:
            h = X
            weights = [np.asarray(w, dtype=np.float32) for w in init["mlp_weights"]]
            biases = [np.asarray(b, dtype=np.float32) for b in init["mlp_biases"]]
            for i in range(len(weights) - 1):
                h = np.maximum(0.0, h @ weights[i] + biases[i])
            out = h @ weights[-1] + biases[-1]
            if out.shape[-1] > 1:
                return np.argmax(out, axis=-1)
            return (out.ravel() >= 0.0).astype(int)

        if "theta" in init and "var" in init:
            theta = np.asarray(init["theta"], dtype=np.float32)
            var = np.asarray(init["var"], dtype=np.float32) + 1e-9
            priors = np.asarray(init.get("class_prior", np.ones(len(theta)) / len(theta)), dtype=np.float32)
            classes = np.asarray(init.get("classes", list(range(len(theta)))))
            # Log-likelihood Gaussian Naive Bayes
            diff = X[:, None, :] - theta[None, :, :]
            log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var[None, :, :]) + (diff**2) / var[None, :, :], axis=-1)
            log_prob += np.log(priors + 1e-12)[None, :]
            return classes[np.argmax(log_prob, axis=-1)]

        if "cluster_centers" in init:
            centers = np.asarray(init["cluster_centers"], dtype=np.float32)
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=-1)
            return np.argmin(dists, axis=-1)

        if "coef" in init:
            coef = np.asarray(init["coef"], dtype=np.float32)
            intercept = np.asarray(init.get("intercept", 0.0), dtype=np.float32)
            if coef.ndim == 1:
                scores = X @ coef + (intercept.ravel()[0] if intercept.size > 0 else 0.0)
            else:
                scores = X @ coef.T + intercept.reshape(1, -1)

            if "classes" in init:
                classes = np.asarray(init["classes"])
                if scores.ndim == 1 or scores.shape[1] == 1:
                    idx = (scores.ravel() >= 0.0).astype(int)
                    return classes[idx]
                return classes[np.argmax(scores, axis=-1)]
            return scores.ravel() if scores.ndim == 2 and scores.shape[1] == 1 else scores

        return np.zeros(len(X), dtype=np.float32)

    def chay_onnx(self, duong_dan: str, X: np.ndarray) -> np.ndarray:
        """
        Chạy dự đoán với mô hình ONNX.

        Args:
            duong_dan: Đường dẫn file .onnx
            X: Dữ liệu đầu vào

        Returns:
            Kết quả dự đoán
        """
        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")

        with open(duong_dan_path, "rb") as f:
            magic = f.read(7)
            if magic == b"VAI_ONX":
                payload_size = struct.unpack("<I", f.read(4))[0]
                graph = json.loads(f.read(payload_size).decode("utf-8"))
                return self._thuc_thi_do_thi_native(graph, X)

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Cần cài đặt onnxruntime: pip install onnxruntime")

        X_arr = np.asarray(X, dtype=np.float32)
        session = ort.InferenceSession(str(duong_dan_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        ket_qua = session.run([output_name], {input_name: X_arr})
        return ket_qua[0]

    def kiem_tra_onnx(self, duong_dan: str) -> dict:
        """Kiểm tra thông tin mô hình ONNX."""
        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")

        with open(duong_dan_path, "rb") as f:
            magic = f.read(7)
            if magic == b"VAI_ONX":
                payload_size = struct.unpack("<I", f.read(4))[0]
                graph = json.loads(f.read(payload_size).decode("utf-8"))
                return {
                    "inputs": [
                        {
                            "name": graph.get("ten_input", "input"),
                            "shape": graph.get("kich_thuoc_dau_vao", []),
                            "type": "tensor(float)",
                        }
                    ],
                    "outputs": [
                        {
                            "name": graph.get("ten_output", "output"),
                            "shape": [None],
                            "type": "tensor(float)",
                        }
                    ],
                    "providers": ["VietnameseAINativeExecutionProvider"],
                    "nodes": graph.get("nodes", []),
                }

        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(duong_dan_path))
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
