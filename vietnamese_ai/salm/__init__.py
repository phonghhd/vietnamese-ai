"""Self-Adapting Language Models (SALM) cho tiếng Việt."""

import warnings

from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA
from vietnamese_ai.salm.self_consistency import SelfConsistency
from vietnamese_ai.salm.self_data import SinhDuLieuTuDong
from vietnamese_ai.salm.self_refine import SelfRefine
from vietnamese_ai.salm.test_time_training import TestTimeTraining

warnings.warn(
    "Gói 'vietnamese_ai.salm' đang ở trạng thái thử nghiệm/mô phỏng và được đánh dấu lỗi thời "
    "(deprecated) từ v31.2.0; dự kiến tách sang nhánh thử nghiệm ở v32.0.0. "
    "Với vòng lặp tự cải thiện có kiểm định, hãy dùng 'vietnamese_ai.fine_tuning.VongLapCanChinhAnToan'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SelfRefine",
    "SelfConsistency",
    "AdaptiveLoRA",
    "SinhDuLieuTuDong",
    "TestTimeTraining",
]
