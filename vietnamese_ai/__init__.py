"""
Vietnamese AI Framework - Framework AI thuần tiếng Việt
=======================================================

Framework học máy thuần tiếng Việt, được thiết kế để đơn giản hóa
quá trình phát triển ứng dụng trí tuệ nhân tạo cho người Việt.

Sử dụng:
    >>> from vietnamese_ai import AutoML
    >>> auto = AutoML()
    >>> auto.fit(X_train, y_train)
    >>> du_doan = auto.predict(X_test)
"""

__version__ = "4.0.0"
__author__ = "EvoNet AI Team"

from vietnamese_ai.augmentation.text_augmenter import TangCuongVanBan
from vietnamese_ai.automl.auto_ml import AutoML
from vietnamese_ai.automl.nas import TimKiemKienTruc
from vietnamese_ai.cloud.deployment import CloudDeployment
from vietnamese_ai.cloud.marketplace import Marketplace
from vietnamese_ai.core.cross_validation import KiemDinhCheo
from vietnamese_ai.core.engine import Engine
from vietnamese_ai.core.hyperparameter import TimKiemThamSo
from vietnamese_ai.core.pipeline import Pipeline
from vietnamese_ai.deep_learning.mang_sau import MangSau
from vietnamese_ai.distributed.distributed import PhanTanHuanLuyen
from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer
from vietnamese_ai.embeddings.fasttext import FastTextTiengViet
from vietnamese_ai.embeddings.word2vec import Word2VecTiengViet
from vietnamese_ai.enterprise.audit import NhatKyHoatDong
from vietnamese_ai.enterprise.auth import HeThongXacThuc
from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem
from vietnamese_ai.export.onnx_export import XuatONNX
from vietnamese_ai.federated.learning import HocLienKet
from vietnamese_ai.hub.model_hub import ModelHub
from vietnamese_ai.interpretability.explainer import GiaiThichMoHinh
from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM
from vietnamese_ai.mobile.deployment import TriKhaiDiDong
from vietnamese_ai.models.base import BaseModel
from vietnamese_ai.models.classifier import PhanLoai
from vietnamese_ai.models.clustering import PhanCum
from vietnamese_ai.models.ensemble import MoHinhTapHop
from vietnamese_ai.models.neural_net import MangNron
from vietnamese_ai.models.regression import HoiQuy
from vietnamese_ai.nlp.pretrained.phobert import PhoBERTWrapper
from vietnamese_ai.nlp.sentiment import PhanTichCamXuc
from vietnamese_ai.plugins.plugin_manager import PluginManager
from vietnamese_ai.preprocessing.feature_engineering import TaoDacTrung
from vietnamese_ai.preprocessing.numerical import XuLySo
from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc
from vietnamese_ai.registry.model_registry import QuanLyMoHinh
from vietnamese_ai.saas.platform import NenTangDichVu
from vietnamese_ai.streaming.processor import XuLyStream
from vietnamese_ai.studio.builder import StudioKeoTha
from vietnamese_ai.timeseries.forecaster import DuDoanChuoiThoiGian
from vietnamese_ai.utils.io_utils import LuuTai
from vietnamese_ai.utils.logger import Logger
from vietnamese_ai.utils.metrics import Metrics
from vietnamese_ai.utils.validators import Validator
from vietnamese_ai.vision.image_classifier import PhanLoaiHinhAnh
from vietnamese_ai.web.app import UngDungWeb

__all__ = [
    "AutoML",
    "BaseModel",
    "CloudDeployment",
    "DuDoanChuoiThoiGian",
    "Engine",
    "FastTextTiengViet",
    "GiaiThichMoHinh",
    "HeThongXacThuc",
    "HocLienKet",
    "HoiQuy",
    "KiemDinhCheo",
    "Logger",
    "LuuTai",
    "MangNron",
    "MangSau",
    "Marketplace",
    "Metrics",
    "ModelHub",
    "MoHinhTapHop",
    "MultiGPUTrainer",
    "NenTangDichVu",
    "NhatKyHoatDong",
    "PhanCum",
    "PhanLoai",
    "PhanLoaiHinhAnh",
    "PhanTichCamXuc",
    "PhanTanHuanLuyen",
    "PhoBERTWrapper",
    "Pipeline",
    "PipelineThoiGianThuc",
    "PluginManager",
    "QuanLyMoHinh",
    "StudioKeoTha",
    "TangCuongVanBan",
    "TaoDacTrung",
    "TheoDoiThiNghiem",
    "TimKiemKienTruc",
    "TimKiemThamSo",
    "TriKhaiDiDong",
    "UngDungWeb",
    "Validator",
    "VietnameseLLM",
    "Word2VecTiengViet",
    "XuLySo",
    "XuLyStream",
    "XuLyVanBan",
    "XuatONNX",
]
