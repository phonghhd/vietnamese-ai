"""Logger - Trình ghi log tiếng Việt."""

import logging
import sys


class Logger:
    """
    Trình ghi log tiếng Việt cho framework.

    Hỗ trợ các mức: DEBUG, INFO, WARNING, ERROR

    Sử dụng:
        >>> logger = Logger("VietnameseAI")
        >>> logger.info("Bắt đầu huấn luyện")
        >>> logger.error("Lỗi xảy ra: ...")
    """

    MUC_LOG = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    def __init__(self, ten: str = "VietnameseAI", level: str = "INFO"):
        self.logger = logging.getLogger(ten)

        if not self.logger.handlers:
            self.logger.setLevel(self.MUC_LOG.get(level.upper(), logging.INFO))

            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, tin_nhan: str) -> None:
        self.logger.debug(tin_nhan)

    def info(self, tin_nhan: str) -> None:
        self.logger.info(tin_nhan)

    def warning(self, tin_nhan: str) -> None:
        self.logger.warning(tin_nhan)

    def error(self, tin_nhan: str) -> None:
        self.logger.error(tin_nhan)
