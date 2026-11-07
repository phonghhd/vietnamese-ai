"""Bộ Đóng Gói 1-Click Desktop Offline (Desktop Packager) - Vietnamese AI Framework."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, Union


class GoiDocLapDesktop:
    """
    Bộ Đóng Gói Ứng Dụng 1-Click Desktop & Offline (Standalone Portable Packager):
    - Tự động sinh file chạy độc lập cho Linux/macOS (`run_desktop.sh`) và Windows (`run_desktop.bat`).
    - Người dùng không cần cài đặt cấu hình phức tạp: chỉ cần nhấp đúp là khởi chạy Web Studio hoặc Chatbot.
    - Cấu hình sẵn các biến tối ưu hóa đa luồng CPU (OMP, MKL, NUMEXPR) để đạt tốc độ cao nhất.
    """

    @staticmethod
    def tao_launcher_sh(
        duong_dan_tep: Union[str, Path],
        che_do: str = "studio",
        port: int = 8080,
        host: str = "127.0.0.1",
        so_thread: int = 4,
    ) -> Path:
        """Sinh tệp thực thi `run_desktop.sh` cho Linux / macOS."""
        p = Path(duong_dan_tep)
        p.parent.mkdir(parents=True, exist_ok=True)

        lenh_args = f"studio --port {port} --host {host}" if che_do == "studio" else "chat"

        noi_dung = f"""#!/usr/bin/env bash
# ==============================================================================
# VIETNAMESE AI FRAMEWORK - 1-CLICK DESKTOP OFFLINE LAUNCHER
# Khởi chạy ứng dụng AI Tiếng Việt trực tiếp trên máy tính (Zero-GPU, Zero-Cost)
# ==============================================================================

set -e

# 1. Cấu hình tối ưu CPU Đa luồng cho Intel / AMD / Apple Silicon
export OMP_NUM_THREADS={so_thread}
export MKL_NUM_THREADS={so_thread}
export NUMEXPR_NUM_THREADS={so_thread}
export PYTHONUNBUFFERED=1

echo "================================================================="
echo "   VIETNAMESE AI DESKTOP - KHỞI ĐỘNG HỆ THỐNG AI TIẾNG VIỆT"
echo "================================================================="
echo "  * Chế độ hoạt động : {che_do.upper()}"
echo "  * Cổng máy chủ     : {port if che_do == 'studio' else 'CLI Terminal'}"
echo "  * CPU Threads      : {so_thread} threads"
echo "  * Trạng thái mạng  : OFFLINE 100% BẢO MẬT NỘI BỘ"
echo "================================================================="

# 2. Ưu tiên sử dụng virtualenv nếu có, fallback về python hệ thống
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "-> Đang khởi chạy với: $PYTHON_BIN"
if [ "{che_do}" = "studio" ]; then
    echo "-> Đang mở Web Dashboard tại: http://{host}:{port}"
    # Tự động mở trình duyệt nếu có lệnh xdg-open (Linux) hoặc open (macOS)
    if command -v open >/dev/null 2>&1; then
        (sleep 1 && open "http://{host}:{port}") &
    elif command -v xdg-open >/dev/null 2>&1; then
        (sleep 1 && xdg-open "http://{host}:{port}") &
    fi
fi

exec $PYTHON_BIN -m vietnamese_ai.cli.main {lenh_args}
"""
        p.write_text(noi_dung, encoding="utf-8")
        # Thêm quyền thực thi chmod +x
        try:
            st = os.stat(p)
            os.chmod(p, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass
        return p

    @staticmethod
    def tao_launcher_bat(
        duong_dan_tep: Union[str, Path],
        che_do: str = "studio",
        port: int = 8080,
        host: str = "127.0.0.1",
        so_thread: int = 4,
    ) -> Path:
        """Sinh tệp thực thi `run_desktop.bat` cho Windows Command Prompt."""
        p = Path(duong_dan_tep)
        p.parent.mkdir(parents=True, exist_ok=True)

        noi_dung = f"""@echo off
REM ==============================================================================
REM VIETNAMESE AI FRAMEWORK - 1-CLICK DESKTOP LAUNCHER CHO WINDOWS
REM ==============================================================================

set OMP_NUM_THREADS={so_thread}
set MKL_NUM_THREADS={so_thread}
set NUMEXPR_NUM_THREADS={so_thread}
set PYTHONUNBUFFERED=1

echo =================================================================
echo    VIETNAMESE AI DESKTOP - KHOI DONG HE THONG AI TIENG VIET
echo =================================================================
echo   * Che do hoat dong : {che_do.upper()}
echo   * Cong ket noi     : {port if che_do == 'studio' else 'Terminal'}
echo   * So luong Threads : {so_thread}
echo   * Bao mat du lieu  : 100% OFFLINE NOI BO
echo =================================================================

REM Uu tien Python virtualenv neu ton tai
set PYTHON_BIN=python
if exist .venv\\Scripts\\python.exe (
    set PYTHON_BIN=.venv\\Scripts\\python.exe
) else if exist venv\\Scripts\\python.exe (
    set PYTHON_BIN=venv\\Scripts\\python.exe
)

if "{che_do}"=="studio" (
    echo Dang mo trinh duyet tai http://{host}:{port}
    start "" "http://{host}:{port}"
    %PYTHON_BIN% -m vietnamese_ai.cli.main studio --port {port} --host {host}
) else (
    %PYTHON_BIN% -m vietnamese_ai.cli.main chat
)

pause
"""
        p.write_text(noi_dung, encoding="utf-8")
        return p

    def dong_goi_desktop(
        self,
        thu_muc_dich: Union[str, Path],
        che_do: str = "studio",
        port: int = 8080,
        so_thread: int = 4,
    ) -> Dict[str, Any]:
        """
        Đóng gói toàn bộ cấu trúc Portable Desktop độc lập vào thư mục chỉ định.
        """
        tm = Path(thu_muc_dich)
        tm.mkdir(parents=True, exist_ok=True)

        sh_path = tm / "run_desktop.sh"
        bat_path = tm / "run_desktop.bat"
        cfg_path = tm / "desktop_config.json"
        readme_path = tm / "README_DESKTOP.md"

        self.tao_launcher_sh(sh_path, che_do=che_do, port=port, so_thread=so_thread)
        self.tao_launcher_bat(bat_path, che_do=che_do, port=port, so_thread=so_thread)

        import vietnamese_ai
        ver = getattr(vietnamese_ai, "__version__", "30.1.0")

        config_data = {
            "app_name": "Vietnamese AI Desktop",
            "version": ver,
            "mode": che_do,
            "port": port,
            "threads": so_thread,
            "offline_only": True,
        }
        cfg_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8")

        readme_content = f"""# Vietnamese AI Framework - 1-Click Desktop Edition

Chào mừng bạn đến với phiên bản Desktop Tự hành 100% Offline của **Vietnamese AI Framework (v{ver})**.

### Cách khởi động cực kỳ đơn giản:
* **Trên Linux / macOS**:
  Mở Terminal tại thư mục này và gõ:
  ```bash
  ./run_desktop.sh
  ```
  *(Hoặc nhấp đúp chuột vào file `run_desktop.sh`)*.
* **Trên Windows**:
  Nhấp đúp chuột trực tiếp vào file **`run_desktop.bat`**.

Hệ thống sẽ tự động bật giao diện Web Studio Dashboard trực quan tại địa chỉ:
👉 **http://127.0.0.1:{port}**

### Tính năng vượt trội:
1. **Zero-GPU**: Chạy mượt mà trực tiếp trên chip CPU Intel, AMD hoặc Apple Silicon.
2. **Bảo mật tuyệt đối**: Dữ liệu tài liệu, ảnh quét, thông tin cá nhân hoàn toàn nằm trong máy của bạn, không gửi ra ngoài Internet.
3. **Đầy đủ công cụ**: Tích hợp sẵn Tường lửa AI, Trợ lý Pháp lý, Trích xuất CCCD/Hóa đơn, và Giám sát Huấn luyện.
"""
        readme_path.write_text(readme_content, encoding="utf-8")

        return {
            "thu_muc": str(tm),
            "file_sh": str(sh_path),
            "file_bat": str(bat_path),
            "file_cfg": str(cfg_path),
            "file_readme": str(readme_path),
            "che_do": che_do,
            "port": port,
        }


# English Aliases
DesktopPackager = GoiDocLapDesktop
StandalonePackager = GoiDocLapDesktop
