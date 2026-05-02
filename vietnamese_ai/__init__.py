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

__version__ = "1.0.0"
__author__ = "EvoNet AI Team"

from vietnamese_ai.core.engine import Engine
from vietnamese_ai.core.pipeline import Pipeline
from vietnamese_ai.core.cross_validation import KiemDinhCheo
from vietnamese_ai.core.hyperparameter import TimKiemThamSo
from vietnamese_ai.models.base import BaseModel
from vietnamese_ai.models.classifier import PhanLoai
from vietnamese_ai.models.regression import HoiQuy
from vietnamese_ai.models.clustering import PhanCum
from vietnamese_ai.models.ensemble import MoHinhTapHop
from vietnamese_ai.models.neural_net import MangNron
from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.preprocessing.numerical import XuLySo
from vietnamese_ai.preprocessing.feature_engineering import TaoDacTrung
from vietnamese_ai.embeddings.word2vec import Word2VecTiengViet
from vietnamese_ai.embeddings.fasttext import FastTextTiengViet
from vietnamese_ai.nlp.sentiment import PhanTichCamXuc
from vietnamese_ai.interpretability.explainer import GiaiThichMoHinh
from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem
from vietnamese_ai.automl.auto_ml import AutoML
from vietnamese_ai.augmentation.text_augmenter import TangCuongVanBan
from vietnamese_ai.utils.logger import Logger
from vietnamese_ai.utils.metrics import Metrics
from vietnamese_ai.utils.validators import Validator
from vietnamese_ai.utils.io_utils import LuuTai

__all__ = [
    "Engine",
    "Pipeline",
    "KiemDinhCheo",
    "TimKiemThamSo",
    "BaseModel",
    "PhanLoai",
    "HoiQuy",
    "PhanCum",
    "MoHinhTapHop",
    "MangNron",
    "XuLyVanBan",
    "XuLySo",
    "TaoDacTrung",
    "Word2VecTiengViet",
    "FastTextTiengViet",
    "PhanTichCamXuc",
    "GiaiThichMoHinh",
    "TheoDoiThiNghiem",
    "AutoML",
    "TangCuongVanBan",
    "Logger",
    "Metrics",
    "Validator",
    "LuuTai",
]
