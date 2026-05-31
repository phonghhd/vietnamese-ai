"""LoggerCauTruc - Structured logging cho production."""

import json
import logging
import sys
import time
import traceback
from typing import Any, Dict, List, Optional


class LoggerCauTruc:
    """
    Structured logger cho production - output JSON format.

    Hỗ trợ:
    - JSON structured logging
    - Log levels (debug, info, warning, error, critical)
    - Context fields (request_id, user_id, model_name)
    - Performance logging (timing)
    - Error tracking with stacktrace

    Sử dụng:
        >>> logger = LoggerCauTruc(ten="vietnamese-ai", cap_do="INFO")
        >>> logger.info("Model loaded", {"model": "phobert", "time_ms": 150})
        >>> with logger.do_thoi_gian("inference"):
        ...     ket_qua = model.du_doan(X)
    """

    def __init__(
        self,
        ten: str = "vietnamese-ai",
        cap_do: str = "INFO",
        output: str = "stdout",
        context_mac_dinh: Optional[Dict[str, Any]] = None,
    ):
        self.ten = ten
        self.cap_do = getattr(logging, cap_do.upper(), logging.INFO)
        self.output = output
        self.context_mac_dinh = context_mac_dinh or {}

        self._logger = logging.getLogger(ten)
        self._logger.setLevel(self.cap_do)

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.cap_do)
            self._logger.addHandler(handler)

        self._context_stack: List[Dict[str, Any]] = []
        self._metrics: List[Dict[str, Any]] = []

    def debug(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log debug."""
        self._log(logging.DEBUG, msg, data)

    def info(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log info."""
        self._log(logging.INFO, msg, data)

    def warning(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log warning."""
        self._log(logging.WARNING, msg, data)

    def error(
        self,
        msg: str,
        data: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None,
    ) -> None:
        """Log error."""
        if exc_info and data is None:
            data = {}
        if exc_info and data is not None:
            data["error_type"] = type(exc_info).__name__
            data["error_message"] = str(exc_info)
            data["stacktrace"] = traceback.format_exc()

        self._log(logging.ERROR, msg, data)

    def critical(
        self,
        msg: str,
        data: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None,
    ) -> None:
        """Log critical."""
        if exc_info and data is None:
            data = {}
        if exc_info and data is not None:
            data["error_type"] = type(exc_info).__name__
            data["error_message"] = str(exc_info)
            data["stacktrace"] = traceback.format_exc()

        self._log(logging.CRITICAL, msg, data)

    def _log(self, level: int, msg: str, data: Optional[Dict[str, Any]]) -> None:
        """Ghi log structured."""
        log_entry = {
            "timestamp": time.time(),
            "level": logging.getLevelName(level),
            "logger": self.ten,
            "message": msg,
        }

        # Merge context
        if self.context_mac_dinh:
            log_entry.update(self.context_mac_dinh)

        if self._context_stack:
            for ctx in self._context_stack:
                log_entry.update(ctx)

        if data:
            log_entry["data"] = data

        # Output JSON
        log_json = json.dumps(log_entry, ensure_ascii=False, default=str)

        self._logger.log(level, log_json)

    class _TimerContext:
        """Context manager cho timing."""

        def __init__(self, logger: "LoggerCauTruc", operation: str):
            self.logger = logger
            self.operation = operation
            self.bat_dau = 0.0

        def __enter__(self) -> "LoggerCauTruc._TimerContext":
            self.bat_dau = time.time()
            return self

        def __exit__(self, *args: Any) -> None:
            elapsed = (time.time() - self.bat_dau) * 1000
            self.logger.info(
                f"{self.operation} hoàn tất",
                {"operation": self.operation, "time_ms": round(elapsed, 2)},
            )

    def do_thoi_gian(self, operation: str) -> "_TimerContext":
        """Context manager để đo thời gian."""
        return self._TimerContext(self, operation)

    def them_context(self, **kwargs: Any) -> None:
        """Thêm context cho tất cả log tiếp theo."""
        self._context_stack.append(kwargs)

    def xoa_context(self) -> None:
        """Xóa context gần nhất."""
        if self._context_stack:
            self._context_stack.pop()

    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status: int,
        time_ms: float,
    ) -> None:
        """Log HTTP request."""
        self.info(
            "HTTP Request",
            {
                "request_id": request_id,
                "method": method,
                "path": path,
                "status": status,
                "time_ms": round(time_ms, 2),
            },
        )

    def log_prediction(
        self,
        model_name: str,
        input_size: int,
        time_ms: float,
        confidence: Optional[float] = None,
    ) -> None:
        """Log model prediction."""
        data = {
            "model_name": model_name,
            "input_size": input_size,
            "time_ms": round(time_ms, 2),
        }
        if confidence is not None:
            data["confidence"] = round(confidence, 4)
        self.info("Prediction", data)

    def log_model_event(
        self,
        event: str,
        model_name: str,
        **kwargs: Any,
    ) -> None:
        """Log model lifecycle event."""
        data = {"event": event, "model_name": model_name, **kwargs}
        self.info(f"Model {event}", data)

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "ten": self.ten,
            "cap_do": logging.getLevelName(self.cap_do),
            "context_depth": len(self._context_stack),
            "context_mac_dinh": self.context_mac_dinh,
        }

    def __repr__(self) -> str:
        return f"LoggerCauTruc(ten='{self.ten}')"
