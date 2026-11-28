"""Federated Learning - Học liên kết phân tán."""

import warnings

from vietnamese_ai.federated.learning import HocLienKet

from .rlhf import FederatedRLHF

warnings.warn(
    "Gói 'vietnamese_ai.federated' đang ở trạng thái mô phỏng đơn tiến trình và được đánh dấu lỗi thời "
    "(deprecated) từ v31.2.0; dự kiến tách sang nhánh thử nghiệm ở v32.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HocLienKet", "FederatedRLHF"]
