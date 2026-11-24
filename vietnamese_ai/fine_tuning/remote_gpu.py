"""Điều Phối Tinh Chỉnh GPU Từ Xa Đa Nhà Cung Cấp (Modal Serverless GPU, SSH GPU Nội Bộ & Subprocess Cục Bộ).

Triển khai vòng lặp Flywheel Local-First -> Cloud GPU Burst Fine-Tuning -> Edge Inference:
1. Chuẩn bị dữ liệu an toàn cục bộ (ẩn PII/Secrets qua `DataSanitizer` và lọc đầu độc qua `PhongThuDuLieu`).
2. Kiểm tra cổng đồng thuận gửi đám mây (`cho_phep_gui_cloud`) và trần ngân sách GPU (`MayTinhTien`).
3. Thực thi huấn luyện trên nhà cung cấp GPU đã chọn (`NhaCungCapModal`, `NhaCungCapSSH`, `NhaCungCapCucBo` hoặc tùy chỉnh).
4. Xác minh toàn vẹn SHA-256, ký số HMAC-SHA256 (`LuuTai`), qua Cổng Kiểm Định (`Eval Gate`) trước khi promote Registry và nạp lại vào `DongCoSuyLuanEdge`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from vietnamese_ai.monitor.cost_engine import MayTinhTien
from vietnamese_ai.security.data_defense import PhongThuDuLieu
from vietnamese_ai.security.data_sanitizer import DataSanitizer
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang
from vietnamese_ai.utils.io_utils import LuuTai
from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc
from vietnamese_ai.utils.logger import Logger

BANG_GIA_GPU_USD_GIO: Dict[str, float] = {
    "CPU": 0.05,
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}

TY_GIA_USD_VND: float = 25500.0


def _tinh_sha256_tep(duong_dan: Path) -> str:
    """Tính mã băm SHA-256 của một tệp nhị phân theo khối 64KB."""
    h = hashlib.sha256()
    with open(duong_dan, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CauHinhTinhChinhXa:
    """Cấu hình chuẩn cho một tác vụ tinh chỉnh mô hình trên GPU từ xa hoặc cục bộ."""

    mo_hinh: str = "vgpt2"
    nhiem_vu: str = "sft"
    loai_gpu: str = "T4"
    so_vong: int = 3
    rank_lora: int = 8
    toc_do_hoc: float = 2e-4
    dinh_dang_xuat: str = "gguf"
    quantization: str = "q8_0"
    timeout_giay: int = 1800
    ngan_sach_toi_da_usd: float = 2.0
    an_pii_truoc_khi_gui: bool = True
    loc_dau_doc_du_lieu: bool = True
    cho_phep_gui_cloud: bool = False
    che_do_offline_worker: bool = False
    ten_volume_cache: str = "vietnamese-ai-model-cache"
    ten_ung_dung_modal: str = "vietnamese-ai-gpu-trainer"
    chi_phi_gpu_usd_gio: Optional[float] = None

    def gia_gpu_moi_gio_usd(self) -> float:
        """Lấy đơn giá thuê GPU (USD/giờ) theo cấu hình hoặc bảng giá tham chiếu chuẩn."""
        if self.chi_phi_gpu_usd_gio is not None:
            return max(0.0, float(self.chi_phi_gpu_usd_gio))
        return BANG_GIA_GPU_USD_GIO.get(self.loai_gpu.upper(), BANG_GIA_GPU_USD_GIO["T4"])

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển cấu hình sang từ điển thuần JSON."""
        return asdict(self)


@dataclass
class KetQuaTinhChinhXa:
    """Kết quả chuẩn hóa trả về từ bất kỳ nhà cung cấp GPU từ xa nào."""

    nha_cung_cap: str
    ma_tac_vu: str
    trang_thai: str
    duong_dan_gguf: Optional[str] = None
    duong_dan_adapter: Optional[str] = None
    duong_dan_manifest: Optional[str] = None
    duong_dan_script: Optional[str] = None
    sha256_artifact: Optional[str] = None
    chu_ky_hmac: Optional[str] = None
    chi_so_huan_luyen: Dict[str, Any] = field(default_factory=dict)
    bao_cao_chi_phi: Dict[str, Any] = field(default_factory=dict)
    bao_cao_an_ninh_du_lieu: Dict[str, Any] = field(default_factory=dict)
    thoi_gian_giay: float = 0.0
    da_promote_registry: bool = False
    ket_qua_eval_gate: Optional[Dict[str, Any]] = None
    lenh_goi_y: Optional[str] = None
    chi_tiet_loi: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đối tượng kết quả sang dict để lưu log, hiển thị CLI hoặc trả JSON API."""
        return asdict(self)


def _sinh_noi_dung_worker_python() -> str:
    """Sinh mã nguồn worker Python độc lập thực thi huấn luyện thực tế bằng `BoTinhChinhMoHinh` và xuất GGUF/LoRA."""
    return '''"""Worker tinh chỉnh độc lập của Vietnamese AI Framework."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese AI Remote GPU Worker")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

    tuner = BoTinhChinhMoHinh(
        mo_hinh=str(cfg.get("mo_hinh", "vgpt2")),
        nhiem_vu=str(cfg.get("nhiem_vu", "sft")),
        dung_lora=True,
        rank_lora=int(cfg.get("rank_lora", 8)),
        che_do_offline=bool(cfg.get("che_do_offline_worker", False)),
    )
    kq_train = tuner.huan_luyen(
        samples,
        so_vong=int(cfg.get("so_vong", 3)),
        toc_do_hoc=float(cfg.get("toc_do_hoc", 2e-4)),
    )

    dinh_dang = str(cfg.get("dinh_dang_xuat", "gguf")).lower()
    quant = str(cfg.get("quantization", "q8_0"))
    gguf_file = None
    adapter_dir = None
    sha256_art = None

    if dinh_dang in ("lora", "both"):
        adapter_path = out_dir / "adapter"
        tuner.luu(str(adapter_path))
        adapter_dir = str(adapter_path)

    if dinh_dang in ("gguf", "both"):
        gguf_path = out_dir / "model_finetuned.gguf"
        tuner.xuat_gguf(str(gguf_path), quantization=quant)
        gguf_file = str(gguf_path)
        sha256_art = _sha256_file(gguf_path)
    elif adapter_dir is not None:
        cfg_file = Path(adapter_dir) / "finetuner_config.json"
        if cfg_file.exists():
            sha256_art = _sha256_file(cfg_file)

    manifest = {
        "trang_thai": "hoan_thanh",
        "mo_hinh": str(cfg.get("mo_hinh", "vgpt2")),
        "nhiem_vu": str(cfg.get("nhiem_vu", "sft")),
        "so_mau": len(samples),
        "quantization": quant,
        "duong_dan_gguf": gguf_file,
        "duong_dan_adapter": adapter_dir,
        "sha256_artifact": sha256_art,
        "chi_so_huan_luyen": kq_train if isinstance(kq_train, dict) else {"ket_qua": str(kq_train)},
        "thoi_gian_giay": round(time.perf_counter() - t0, 4),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"Loi worker huan luyen: {exc}\\n")
        sys.exit(1)
'''


class NhaCungCapGPUXa(ABC):
    """Giao diện trừu tượng chuẩn hóa mọi nhà cung cấp GPU từ xa (Modal, SSH GPU nội bộ, RunPod, cục bộ)."""

    ten_nha_cung_cap: str = "base"
    la_dam_may_ngoai: bool = True

    @abstractmethod
    def kiem_tra_san_sang(self) -> Dict[str, Any]:
        """Kiểm tra thư viện, công cụ dòng lệnh và thông tin xác thực của nhà cung cấp."""

    def uoc_tinh_chi_phi(
        self,
        cau_hinh: CauHinhTinhChinhXa,
        so_mau: int,
    ) -> Dict[str, Any]:
        """Ước tính thời gian và chi phí thuê GPU bằng `MayTinhTien` trước khi chạy thật."""
        gia_gio_usd = cau_hinh.gia_gpu_moi_gio_usd()
        gia_gio_vnd = gia_gio_usd * TY_GIA_USD_VND
        thoi_gian_uoc_tinh_giay = min(
            float(cau_hinh.timeout_giay),
            max(30.0, 20.0 + float(max(1, so_mau)) * float(max(1, cau_hinh.so_vong)) * 0.35),
        )
        chi_phi_uoc_tinh_usd = round((thoi_gian_uoc_tinh_giay / 3600.0) * gia_gio_usd, 4)
        chi_phi_toi_da_usd = round((float(cau_hinh.timeout_giay) / 3600.0) * gia_gio_usd, 4)
        bo_tinh = MayTinhTien(gia_thue_gpu_gio=gia_gio_vnd)
        _ = bo_tinh.lay_bao_cao_chi_phi()
        return {
            "nha_cung_cap": self.ten_nha_cung_cap,
            "loai_gpu": cau_hinh.loai_gpu,
            "gia_gpu_usd_gio": gia_gio_usd,
            "thoi_gian_uoc_tinh_giay": round(thoi_gian_uoc_tinh_giay, 2),
            "chi_phi_uoc_tinh_usd": chi_phi_uoc_tinh_usd,
            "chi_phi_uoc_tinh_vnd": round(chi_phi_uoc_tinh_usd * TY_GIA_USD_VND, 0),
            "chi_phi_toi_da_theo_timeout_usd": chi_phi_toi_da_usd,
            "ngan_sach_toi_da_usd": cau_hinh.ngan_sach_toi_da_usd,
            "vuot_ngan_sach": chi_phi_uoc_tinh_usd > cau_hinh.ngan_sach_toi_da_usd,
        }

    def tao_script_huan_luyen(
        self,
        thu_muc_dich: Path,
        ma_tac_vu: str,
    ) -> Path:
        """Ghi tệp worker Python độc lập phục vụ thực thi tại nút tính toán."""
        thu_muc_dich.mkdir(parents=True, exist_ok=True)
        script_path = thu_muc_dich / f"vai_worker_{ma_tac_vu}.py"
        script_path.write_text(_sinh_noi_dung_worker_python(), encoding="utf-8")
        return script_path

    @abstractmethod
    def thuc_thi_huan_luyen(
        self,
        tep_du_lieu_sach: Path,
        thu_muc_dich: Path,
        cau_hinh: CauHinhTinhChinhXa,
        ma_tac_vu: str,
        chay_that: bool = True,
    ) -> KetQuaTinhChinhXa:
        """Thực thi (hoặc dựng kế hoạch dry-run) quá trình huấn luyện và thu hồi artifact."""


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXTERNAL_DEPENDENCY,
    mo_ta=(
        "Nhà cung cấp Serverless GPU theo giây qua Modal (T4/L4/A10G/A100/H100), "
        "dùng Modal Volume để cache base model và tải về GGUF/LoRA có kiểm tra SHA-256."
    ),
    phu_thuoc_ngoai=["modal>=0.64.0", "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET hoặc ~/.modal.toml"],
    nhom_extra="cloud-gpu",
)
class NhaCungCapModal(NhaCungCapGPUXa):
    """Nhà cung cấp GPU đám mây serverless theo giây sử dụng nền tảng Modal (`modal.com`)."""

    ten_nha_cung_cap = "modal"
    la_dam_may_ngoai = True

    def __init__(
        self,
        ten_ung_dung: str = "vietnamese-ai-gpu-trainer",
        ten_volume_cache: str = "vietnamese-ai-model-cache",
    ) -> None:
        self.ten_ung_dung = ten_ung_dung
        self.ten_volume_cache = ten_volume_cache

    def kiem_tra_san_sang(self) -> Dict[str, Any]:
        """Kiểm tra gói Python `modal`, CLI `modal` và thông tin xác thực token Modal."""
        co_sdk = QuanLyPhuThuoc.da_cai("modal")
        modal_bin = shutil.which("modal")
        co_env_token = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
        co_file_token = (Path.home() / ".modal.toml").exists()
        co_xac_thuc = co_env_token or co_file_token
        return {
            "nha_cung_cap": self.ten_nha_cung_cap,
            "san_sang": bool((co_sdk or modal_bin) and co_xac_thuc),
            "co_modal_sdk": co_sdk,
            "co_modal_cli": modal_bin is not None,
            "co_xac_thuc_token": co_xac_thuc,
            "huong_dan": (
                'Cài đặt bằng: pip install "vietnamese-ai[cloud-gpu]" và đăng nhập bằng: modal token new'
                if not ((co_sdk or modal_bin) and co_xac_thuc)
                else "Modal Serverless GPU đã sẵn sàng."
            ),
        }

    def tao_modal_app_script(
        self,
        cau_hinh: CauHinhTinhChinhXa,
        thu_muc_dich: Path,
        ma_tac_vu: str,
    ) -> Path:
        """Sinh tệp ứng dụng Modal Python hoàn chỉnh, tự động gắn GPU, Volume cache và tải GGUF/LoRA về máy cục bộ."""
        thu_muc_dich.mkdir(parents=True, exist_ok=True)
        app_name = cau_hinh.ten_ung_dung_modal or self.ten_ung_dung
        vol_name = cau_hinh.ten_volume_cache or self.ten_volume_cache
        gpu_spec = cau_hinh.loai_gpu.upper()
        gpu_param = "None" if gpu_spec == "CPU" else repr(gpu_spec)
        script_content = f'''"""Ứng dụng Modal Serverless GPU sinh tự động bởi Vietnamese AI Framework."""

import hashlib
import json
import os
import tarfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import modal

app = modal.App({app_name!r})
cache_volume = modal.Volume.from_name({vol_name!r}, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.20.0,<3.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.7.0",
        "trl>=0.8.0",
        "datasets>=2.14.0",
        "vietnamese-ai>=31.0.3",
    )
)


@app.function(
    image=image,
    gpu={gpu_param},
    timeout={int(cau_hinh.timeout_giay)},
    volumes={{"/cache": cache_volume}},
)
def huan_luyen_tren_modal(du_lieu_jsonl: bytes, cau_hinh_json: str) -> Dict[str, Any]:
    os.environ["HF_HOME"] = "/cache/huggingface"
    t0 = time.perf_counter()
    cfg = json.loads(cau_hinh_json)
    job_dir = Path("/tmp/vai_modal_job")
    job_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        json.loads(line)
        for line in du_lieu_jsonl.decode("utf-8").splitlines()
        if line.strip()
    ]

    from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

    tuner = BoTinhChinhMoHinh(
        mo_hinh=str(cfg.get("mo_hinh", "vgpt2")),
        nhiem_vu=str(cfg.get("nhiem_vu", "sft")),
        dung_lora=True,
        rank_lora=int(cfg.get("rank_lora", 8)),
        che_do_offline=bool(cfg.get("che_do_offline_worker", False)),
    )
    kq_train = tuner.huan_luyen(
        samples,
        so_vong=int(cfg.get("so_vong", 3)),
        toc_do_hoc=float(cfg.get("toc_do_hoc", 2e-4)),
    )

    dinh_dang = str(cfg.get("dinh_dang_xuat", "gguf")).lower()
    quant = str(cfg.get("quantization", "q8_0"))
    gguf_bytes = b""
    adapter_tar_bytes = b""
    sha256_art = ""

    if dinh_dang in ("lora", "both"):
        adapter_dir = job_dir / "adapter"
        tuner.luu(str(adapter_dir))
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(str(adapter_dir), arcname="adapter")
        adapter_tar_bytes = buf.getvalue()
        sha256_art = hashlib.sha256(adapter_tar_bytes).hexdigest()

    if dinh_dang in ("gguf", "both"):
        gguf_path = job_dir / "model_finetuned.gguf"
        tuner.xuat_gguf(str(gguf_path), quantization=quant)
        gguf_bytes = gguf_path.read_bytes()
        sha256_art = hashlib.sha256(gguf_bytes).hexdigest()

    cache_volume.commit()
    manifest = {{
        "trang_thai": "hoan_thanh",
        "mo_hinh": str(cfg.get("mo_hinh", "vgpt2")),
        "nhiem_vu": str(cfg.get("nhiem_vu", "sft")),
        "so_mau": len(samples),
        "quantization": quant,
        "sha256_artifact": sha256_art,
        "chi_so_huan_luyen": kq_train if isinstance(kq_train, dict) else {{"ket_qua": str(kq_train)}},
        "thoi_gian_giay": round(time.perf_counter() - t0, 4),
    }}
    return {{
        "manifest": manifest,
        "gguf_bytes": gguf_bytes,
        "adapter_tar_bytes": adapter_tar_bytes,
    }}


@app.local_entrypoint()
def main(data_file: str, config_file: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    res = huan_luyen_tren_modal.remote(
        Path(data_file).read_bytes(),
        Path(config_file).read_text(encoding="utf-8"),
    )
    manifest = dict(res["manifest"])
    if res.get("gguf_bytes"):
        gguf_out = out_path / "model_finetuned.gguf"
        gguf_out.write_bytes(res["gguf_bytes"])
        manifest["duong_dan_gguf"] = str(gguf_out)
    if res.get("adapter_tar_bytes"):
        tar_out = out_path / "adapter.tar.gz"
        tar_out.write_bytes(res["adapter_tar_bytes"])
        manifest["duong_dan_adapter"] = str(tar_out)
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
'''
        script_path = thu_muc_dich / f"modal_job_{ma_tac_vu}.py"
        script_path.write_text(script_content, encoding="utf-8")
        return script_path

    def thuc_thi_huan_luyen(
        self,
        tep_du_lieu_sach: Path,
        thu_muc_dich: Path,
        cau_hinh: CauHinhTinhChinhXa,
        ma_tac_vu: str,
        chay_that: bool = True,
    ) -> KetQuaTinhChinhXa:
        """Sinh ứng dụng Modal và thực thi tác vụ GPU serverless (hoặc trả kế hoạch dry-run)."""
        thu_muc_dich.mkdir(parents=True, exist_ok=True)
        cfg_path = thu_muc_dich / f"config_{ma_tac_vu}.json"
        cfg_path.write_text(json.dumps(cau_hinh.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        script_path = self.tao_modal_app_script(cau_hinh, thu_muc_dich, ma_tac_vu)

        cmd = [
            "modal",
            "run",
            str(script_path),
            "--data-file",
            str(tep_du_lieu_sach),
            "--config-file",
            str(cfg_path),
            "--out-dir",
            str(thu_muc_dich),
        ]
        lenh_str = " ".join(shlex.quote(c) for c in cmd)

        if not chay_that:
            return KetQuaTinhChinhXa(
                nha_cung_cap=self.ten_nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="xem_truoc_dry_run",
                duong_dan_script=str(script_path),
                lenh_goi_y=lenh_str,
            )

        QuanLyPhuThuoc.nhap_tuy_chon("modal", nhom_extra="cloud-gpu", tinh_nang="Modal Serverless GPU")
        san_sang = self.kiem_tra_san_sang()
        if not san_sang["san_sang"]:
            raise RuntimeError(
                f"Nhà cung cấp Modal chưa sẵn sàng: {san_sang['huong_dan']}"
            )

        modal_bin = shutil.which("modal")
        exec_cmd = [modal_bin, *cmd[1:]] if modal_bin else [sys.executable, "-m", "modal", *cmd[1:]]
        t0 = time.perf_counter()
        proc = subprocess.run(
            exec_cmd,
            capture_output=True,
            text=True,
            timeout=cau_hinh.timeout_giay,
            check=False,
        )
        dt = round(time.perf_counter() - t0, 4)
        if proc.returncode != 0:
            return KetQuaTinhChinhXa(
                nha_cung_cap=self.ten_nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="that_bai",
                duong_dan_script=str(script_path),
                thoi_gian_giay=dt,
                lenh_goi_y=lenh_str,
                chi_tiet_loi=(proc.stderr or proc.stdout)[-1000:],
            )

        return _doc_va_xac_minh_ket_qua_manifest(
            nha_cung_cap=self.ten_nha_cung_cap,
            ma_tac_vu=ma_tac_vu,
            thu_muc_dich=thu_muc_dich,
            duong_dan_script=str(script_path),
            thoi_gian_giay=dt,
            lenh_goi_y=lenh_str,
        )


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXTERNAL_DEPENDENCY,
    mo_ta=(
        "Nhà cung cấp GPU từ xa qua SSH/SCP cho máy chủ nội bộ (On-Premise) hoặc đám mây GPU tại Việt Nam "
        "(Viettel/FPT/VNPT), không phụ thuộc bên thứ ba."
    ),
    phu_thuoc_ngoai=["ssh", "scp", "máy chủ GPU từ xa đã cài vietnamese-ai"],
    nhom_extra="core",
)
class NhaCungCapSSH(NhaCungCapGPUXa):
    """Nhà cung cấp GPU từ xa qua giao thức SSH/SCP (hỗ trợ máy chủ GPU on-premise tại Việt Nam)."""

    ten_nha_cung_cap = "ssh"
    la_dam_may_ngoai = False

    def __init__(
        self,
        host: str = "",
        user: Optional[str] = None,
        port: int = 22,
        khoa_ssh: Optional[str] = None,
        thu_muc_xa: str = "/tmp/vietnamese_ai_jobs",
        python_xa: str = "python3",
        la_dam_may_ngoai: bool = False,
    ) -> None:
        self.host = host.strip()
        self.user = user.strip() if user else None
        self.port = int(port)
        self.khoa_ssh = khoa_ssh
        self.thu_muc_xa = thu_muc_xa
        self.python_xa = python_xa
        self.la_dam_may_ngoai = bool(la_dam_may_ngoai)

    def _dich_ssh(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def _co_ssh_chung(self, la_scp: bool = False) -> List[str]:
        co = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-P" if la_scp else "-p",
            str(self.port),
        ]
        if self.khoa_ssh:
            co.extend(["-i", self.khoa_ssh])
        return co

    def kiem_tra_san_sang(self) -> Dict[str, Any]:
        """Kiểm tra công cụ `ssh`/`scp` trên máy cục bộ và thông số kết nối máy chủ GPU đích."""
        co_ssh = shutil.which("ssh") is not None
        co_scp = shutil.which("scp") is not None
        co_host = bool(self.host)
        return {
            "nha_cung_cap": self.ten_nha_cung_cap,
            "san_sang": co_ssh and co_scp and co_host,
            "co_ssh": co_ssh,
            "co_scp": co_scp,
            "host": self.host,
            "port": self.port,
            "la_dam_may_ngoai": self.la_dam_may_ngoai,
            "huong_dan": (
                "Cần cung cấp địa chỉ máy chủ GPU (`host`) và công cụ ssh/scp."
                if not (co_ssh and co_scp and co_host)
                else f"Sẵn sàng kết nối SSH tới {self._dich_ssh()}:{self.port}."
            ),
        }

    def tao_ke_hoach_lenh(
        self,
        tep_du_lieu_sach: Path,
        cfg_path: Path,
        script_path: Path,
        thu_muc_dich: Path,
        ma_tac_vu: str,
    ) -> Dict[str, List[str]]:
        """Tạo chuỗi lệnh SSH/SCP an toàn (chống command injection bằng `shlex.quote`)."""
        target = self._dich_ssh() or "gpu-server"
        remote_job_dir = f"{self.thu_muc_xa.rstrip('/')}/{ma_tac_vu}"
        remote_out_dir = f"{remote_job_dir}/out"

        cmd_mkdir = ["ssh", *self._co_ssh_chung(False), target, f"mkdir -p {shlex.quote(remote_out_dir)}"]
        cmd_upload = [
            "scp",
            *self._co_ssh_chung(True),
            str(tep_du_lieu_sach),
            str(cfg_path),
            str(script_path),
            f"{target}:{remote_job_dir}/",
        ]
        remote_run = (
            f"{shlex.quote(self.python_xa)} {shlex.quote(remote_job_dir + '/' + script_path.name)} "
            f"--data {shlex.quote(remote_job_dir + '/' + tep_du_lieu_sach.name)} "
            f"--config {shlex.quote(remote_job_dir + '/' + cfg_path.name)} "
            f"--out-dir {shlex.quote(remote_out_dir)}"
        )
        cmd_train = ["ssh", *self._co_ssh_chung(False), target, remote_run]
        cmd_download = [
            "scp",
            "-r",
            *self._co_ssh_chung(True),
            f"{target}:{remote_out_dir}/.",
            str(thu_muc_dich),
        ]
        return {
            "mkdir": cmd_mkdir,
            "upload": cmd_upload,
            "train": cmd_train,
            "download": cmd_download,
        }

    def thuc_thi_huan_luyen(
        self,
        tep_du_lieu_sach: Path,
        thu_muc_dich: Path,
        cau_hinh: CauHinhTinhChinhXa,
        ma_tac_vu: str,
        chay_that: bool = True,
    ) -> KetQuaTinhChinhXa:
        """Tải dữ liệu sạch lên máy chủ GPU qua SCP, huấn luyện qua SSH và kéo GGUF/LoRA về máy cục bộ."""
        thu_muc_dich.mkdir(parents=True, exist_ok=True)
        cfg_path = thu_muc_dich / f"config_{ma_tac_vu}.json"
        cfg_path.write_text(json.dumps(cau_hinh.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        script_path = self.tao_script_huan_luyen(thu_muc_dich, ma_tac_vu)
        ke_hoach = self.tao_ke_hoach_lenh(tep_du_lieu_sach, cfg_path, script_path, thu_muc_dich, ma_tac_vu)
        lenh_str = " && ".join(" ".join(shlex.quote(x) for x in step) for step in ke_hoach.values())

        if not chay_that:
            return KetQuaTinhChinhXa(
                nha_cung_cap=self.ten_nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="xem_truoc_dry_run",
                duong_dan_script=str(script_path),
                lenh_goi_y=lenh_str,
            )

        san_sang = self.kiem_tra_san_sang()
        if not san_sang["san_sang"]:
            raise RuntimeError(f"Nhà cung cấp SSH chưa sẵn sàng: {san_sang['huong_dan']}")

        t0 = time.perf_counter()
        for buoc, cmd in ke_hoach.items():
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cau_hinh.timeout_giay,
                check=False,
            )
            if proc.returncode != 0:
                return KetQuaTinhChinhXa(
                    nha_cung_cap=self.ten_nha_cung_cap,
                    ma_tac_vu=ma_tac_vu,
                    trang_thai="that_bai",
                    duong_dan_script=str(script_path),
                    thoi_gian_giay=round(time.perf_counter() - t0, 4),
                    lenh_goi_y=lenh_str,
                    chi_tiet_loi=f"Lỗi bước '{buoc}': {(proc.stderr or proc.stdout)[-800:]}",
                )

        dt = round(time.perf_counter() - t0, 4)
        return _doc_va_xac_minh_ket_qua_manifest(
            nha_cung_cap=self.ten_nha_cung_cap,
            ma_tac_vu=ma_tac_vu,
            thu_muc_dich=thu_muc_dich,
            duong_dan_script=str(script_path),
            thoi_gian_giay=dt,
            lenh_goi_y=lenh_str,
        )


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Nhà cung cấp thực thi tinh chỉnh trên tiến trình cục bộ cách ly (CPU hoặc CUDA GPU cục bộ), "
        "dùng chung giao thức worker/manifest/GGUF với Modal và SSH."
    ),
    nhom_extra="core",
)
class NhaCungCapCucBo(NhaCungCapGPUXa):
    """Nhà cung cấp thực thi vòng lặp tinh chỉnh thực tế trong tiến trình con độc lập ngay tại máy cục bộ."""

    ten_nha_cung_cap = "local"
    la_dam_may_ngoai = False

    def kiem_tra_san_sang(self) -> Dict[str, Any]:
        """Kiểm tra môi trường Python cục bộ và trạng thái CUDA GPU."""
        hw = QuanLyPhuThuoc.phat_hien_phan_cung_cai_dat()
        return {
            "nha_cung_cap": self.ten_nha_cung_cap,
            "san_sang": True,
            "co_cuda_gpu": hw["co_cuda_gpu"],
            "so_nhan_cpu": hw["so_nhan_cpu"],
            "huong_dan": "Tiến trình huấn luyện cục bộ sẵn sàng (không gửi dữ liệu ra mạng).",
        }

    def uoc_tinh_chi_phi(
        self,
        cau_hinh: CauHinhTinhChinhXa,
        so_mau: int,
    ) -> Dict[str, Any]:
        """Chi phí thuê đám mây bằng 0 khi chạy trên phần cứng cục bộ."""
        kq = super().uoc_tinh_chi_phi(cau_hinh, so_mau)
        kq["gia_gpu_usd_gio"] = 0.0
        kq["chi_phi_uoc_tinh_usd"] = 0.0
        kq["chi_phi_uoc_tinh_vnd"] = 0.0
        kq["chi_phi_toi_da_theo_timeout_usd"] = 0.0
        kq["vuot_ngan_sach"] = False
        return kq

    def thuc_thi_huan_luyen(
        self,
        tep_du_lieu_sach: Path,
        thu_muc_dich: Path,
        cau_hinh: CauHinhTinhChinhXa,
        ma_tac_vu: str,
        chay_that: bool = True,
    ) -> KetQuaTinhChinhXa:
        """Chạy worker tinh chỉnh thật trong tiến trình con độc lập và xuất GGUF/LoRA có ký số HMAC."""
        thu_muc_dich.mkdir(parents=True, exist_ok=True)
        cfg_dict = cau_hinh.to_dict()
        cfg_dict["che_do_offline_worker"] = True
        cfg_path = thu_muc_dich / f"config_{ma_tac_vu}.json"
        cfg_path.write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        script_path = self.tao_script_huan_luyen(thu_muc_dich, ma_tac_vu)

        cmd = [
            sys.executable,
            str(script_path),
            "--data",
            str(tep_du_lieu_sach),
            "--config",
            str(cfg_path),
            "--out-dir",
            str(thu_muc_dich),
        ]
        lenh_str = " ".join(shlex.quote(c) for c in cmd)

        if not chay_that:
            return KetQuaTinhChinhXa(
                nha_cung_cap=self.ten_nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="xem_truoc_dry_run",
                duong_dan_script=str(script_path),
                lenh_goi_y=lenh_str,
            )

        t0 = time.perf_counter()
        env = dict(os.environ)
        goc_du_an = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = f"{goc_du_an}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cau_hinh.timeout_giay,
            env=env,
            check=False,
        )
        dt = round(time.perf_counter() - t0, 4)
        if proc.returncode != 0:
            return KetQuaTinhChinhXa(
                nha_cung_cap=self.ten_nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="that_bai",
                duong_dan_script=str(script_path),
                thoi_gian_giay=dt,
                lenh_goi_y=lenh_str,
                chi_tiet_loi=(proc.stderr or proc.stdout)[-1000:],
            )

        return _doc_va_xac_minh_ket_qua_manifest(
            nha_cung_cap=self.ten_nha_cung_cap,
            ma_tac_vu=ma_tac_vu,
            thu_muc_dich=thu_muc_dich,
            duong_dan_script=str(script_path),
            thoi_gian_giay=dt,
            lenh_goi_y=lenh_str,
        )


def _doc_va_xac_minh_ket_qua_manifest(
    nha_cung_cap: str,
    ma_tac_vu: str,
    thu_muc_dich: Path,
    duong_dan_script: str,
    thoi_gian_giay: float,
    lenh_goi_y: str,
) -> KetQuaTinhChinhXa:
    """Đọc `manifest.json` trả về từ worker, đối chiếu SHA-256 và ký toàn vẹn HMAC-SHA256 cục bộ."""
    manifest_path = thu_muc_dich / "manifest.json"
    if not manifest_path.exists():
        return KetQuaTinhChinhXa(
            nha_cung_cap=nha_cung_cap,
            ma_tac_vu=ma_tac_vu,
            trang_thai="that_bai",
            duong_dan_script=duong_dan_script,
            thoi_gian_giay=thoi_gian_giay,
            lenh_goi_y=lenh_goi_y,
            chi_tiet_loi="Không tìm thấy tệp manifest.json sau khi hoàn tất tinh chỉnh.",
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gguf_local = thu_muc_dich / "model_finetuned.gguf"
    duong_dan_gguf = str(gguf_local) if gguf_local.exists() else manifest.get("duong_dan_gguf")
    adapter_local = thu_muc_dich / "adapter.tar.gz"
    if not adapter_local.exists():
        adapter_local = thu_muc_dich / "adapter"
    duong_dan_adapter = str(adapter_local) if adapter_local.exists() else manifest.get("duong_dan_adapter")

    sha_mong_doi = manifest.get("sha256_artifact")
    sha_thuc_te: Optional[str] = None
    chu_ky_hmac: Optional[str] = None

    artifact_chinh: Optional[Path] = None
    if duong_dan_gguf and Path(duong_dan_gguf).is_file():
        artifact_chinh = Path(duong_dan_gguf)
    elif duong_dan_adapter and Path(duong_dan_adapter).is_file():
        artifact_chinh = Path(duong_dan_adapter)

    if artifact_chinh is not None:
        sha_thuc_te = _tinh_sha256_tep(artifact_chinh)
        if sha_mong_doi and sha_thuc_te != sha_mong_doi:
            return KetQuaTinhChinhXa(
                nha_cung_cap=nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="that_bai",
                duong_dan_gguf=duong_dan_gguf,
                duong_dan_adapter=duong_dan_adapter,
                duong_dan_manifest=str(manifest_path),
                duong_dan_script=duong_dan_script,
                sha256_artifact=sha_thuc_te,
                thoi_gian_giay=thoi_gian_giay,
                lenh_goi_y=lenh_goi_y,
                chi_tiet_loi=(
                    f"Cảnh báo toàn vẹn dữ liệu: SHA-256 artifact tải về ({sha_thuc_te}) "
                    f"không khớp manifest ({sha_mong_doi})."
                ),
            )
        chu_ky_hmac = LuuTai.ky_file(str(artifact_chinh))

    return KetQuaTinhChinhXa(
        nha_cung_cap=nha_cung_cap,
        ma_tac_vu=ma_tac_vu,
        trang_thai="hoan_thanh",
        duong_dan_gguf=duong_dan_gguf,
        duong_dan_adapter=duong_dan_adapter,
        duong_dan_manifest=str(manifest_path),
        duong_dan_script=duong_dan_script,
        sha256_artifact=sha_thuc_te or sha_mong_doi,
        chu_ky_hmac=chu_ky_hmac,
        chi_so_huan_luyen=manifest.get("chi_so_huan_luyen", {}),
        thoi_gian_giay=thoi_gian_giay,
        lenh_goi_y=lenh_goi_y,
    )


class BoDieuPhoiTinhChinhXa:
    """
    Bộ Điều Phối Tinh Chỉnh GPU Đa Nhà Cung Cấp (Multi-Provider Remote GPU Flywheel Orchestrator).

    Hỗ trợ cắm nóng nhiều nhà cung cấp (`modal`, `ssh`, `local`, hoặc provider tùy chỉnh),
    tự động áp dụng cổng làm sạch PII/Backdoor cục bộ, kiểm soát trần ngân sách GPU,
    xác minh SHA-256 + HMAC-SHA256 và cổng kiểm định chất lượng (`Eval Gate`) trước khi promote.
    """

    def __init__(
        self,
        thu_muc_lam_viec: Union[str, Path] = "vai_remote_jobs",
        nha_cung_cap: Optional[Dict[str, NhaCungCapGPUXa]] = None,
    ) -> None:
        self.thu_muc_lam_viec = Path(thu_muc_lam_viec)
        self.thu_muc_lam_viec.mkdir(parents=True, exist_ok=True)
        self.logger = Logger("RemoteGPUFlywheel")
        self._nha_cung_cap: Dict[str, NhaCungCapGPUXa] = {
            "modal": NhaCungCapModal(),
            "ssh": NhaCungCapSSH(),
            "local": NhaCungCapCucBo(),
        }
        if nha_cung_cap:
            for ten, prov in nha_cung_cap.items():
                self._nha_cung_cap[ten.strip().lower()] = prov

    def dang_ky_nha_cung_cap(self, ten: str, provider: NhaCungCapGPUXa) -> None:
        """Đăng ký thêm hoặc ghi đè một nhà cung cấp GPU tùy chỉnh (ví dụ RunPod, Vertex AI, FPT GPU)."""
        self._nha_cung_cap[ten.strip().lower()] = provider

    def lay_nha_cung_cap(self, ten: str) -> NhaCungCapGPUXa:
        """Lấy đối tượng nhà cung cấp GPU theo tên chuẩn hóa."""
        khoa = ten.strip().lower()
        if khoa not in self._nha_cung_cap:
            hop_le = ", ".join(sorted(self._nha_cung_cap.keys()))
            raise ValueError(f"Nhà cung cấp GPU '{ten}' không tồn tại. Các lựa chọn hợp lệ: {hop_le}")
        return self._nha_cung_cap[khoa]

    def danh_sach_nha_cung_cap(self) -> Dict[str, Dict[str, Any]]:
        """Trả về trạng thái sẵn sàng của tất cả nhà cung cấp GPU đã đăng ký."""
        return {ten: prov.kiem_tra_san_sang() for ten, prov in self._nha_cung_cap.items()}

    def chuan_bi_du_lieu_an_toan(
        self,
        nguon_du_lieu: Union[str, Path, List[Dict[str, Any]]],
        cau_hinh: CauHinhTinhChinhXa,
        thu_muc_dich: Path,
        ma_tac_vu: str,
    ) -> Tuple[Path, int, Dict[str, Any]]:
        """
        Đọc, lọc mẫu đầu độc (Sleeper Backdoor / Split-Payload) và ẩn PII/Secrets cục bộ
        trước khi xuất thành tệp JSONL sạch sẵn sàng gửi đi huấn luyện.
        """
        ban_ghi: List[Dict[str, Any]] = []
        if isinstance(nguon_du_lieu, (str, Path)):
            p = Path(nguon_du_lieu)
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu huấn luyện: {p}")
            raw_text = p.read_text(encoding="utf-8").strip()
            if p.suffix.lower() == ".json" and raw_text.startswith("["):
                ban_ghi = [dict(x) if isinstance(x, dict) else {"text": str(x)} for x in json.loads(raw_text)]
            else:
                for dong in raw_text.splitlines():
                    if dong.strip():
                        obj = json.loads(dong)
                        ban_ghi.append(obj if isinstance(obj, dict) else {"text": str(obj)})
        elif isinstance(nguon_du_lieu, list):
            for item in nguon_du_lieu:
                ban_ghi.append(dict(item) if isinstance(item, dict) else {"text": str(item)})
        else:
            raise ValueError("nguon_du_lieu phải là đường dẫn tệp JSON/JSONL hoặc danh sách bản ghi.")

        tong_goc = len(ban_ghi)
        bc_phong_thu: Dict[str, Any] = {}
        du_lieu_hien_tai: List[Dict[str, Any]] = ban_ghi

        if cau_hinh.loc_dau_doc_du_lieu and du_lieu_hien_tai:
            du_lieu_loc, bc_phong_thu = PhongThuDuLieu.quet_va_lam_sach_dataset(
                du_lieu_hien_tai, loai_bo_mau_doc=True
            )
            du_lieu_hien_tai = list(du_lieu_loc)

        bc_pii: Dict[str, Any] = {
            "tong_mau_kiem_tra": len(du_lieu_hien_tai),
            "so_mau_dau_doc_bi_loai": 0,
            "so_mau_da_an_pii": 0,
            "an_toan": True,
        }
        if cau_hinh.an_pii_truoc_khi_gui and du_lieu_hien_tai:
            du_lieu_sach, bc_pii = DataSanitizer.quet_dau_doc_du_lieu(
                du_lieu_hien_tai,
                an_pii=True,
                loai_bo_mau_doc=cau_hinh.loc_dau_doc_du_lieu,
            )
            du_lieu_hien_tai = list(du_lieu_sach)

        thu_muc_dich.mkdir(parents=True, exist_ok=True)
        tep_sach = thu_muc_dich / f"dataset_sanitized_{ma_tac_vu}.jsonl"
        with open(tep_sach, "w", encoding="utf-8") as f:
            for rec in du_lieu_hien_tai:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        bao_cao_an_ninh = {
            "tong_mau_ban_dau": tong_goc,
            "so_mau_sau_lam_sach": len(du_lieu_hien_tai),
            "so_mau_bi_loai_tong": tong_goc - len(du_lieu_hien_tai),
            "so_mau_da_an_pii": int(bc_phong_thu.get("so_mau_da_an_pii", 0))
            + int(bc_pii.get("so_mau_da_an_pii", 0)),
            "phat_hien_bom_hen_gio": bool(bc_phong_thu.get("phat_hien_bom_hen_gio", False)),
            "sha256_du_lieu_sach": _tinh_sha256_tep(tep_sach),
            "tep_du_lieu_sach": str(tep_sach),
        }
        return tep_sach, len(du_lieu_hien_tai), bao_cao_an_ninh

    def chuan_bi_du_lieu_tu_workspace(
        self,
        thu_muc_workspace: Union[str, Path],
        ma_workspace: Optional[str] = None,
        duong_dan_xuat: Optional[Union[str, Path]] = None,
        an_pii: bool = True,
    ) -> Dict[str, Any]:
        """Xuất thẳng phản hồi Thumbs Up/Down + câu trả lời sửa từ `QuanLyPhienChatWorkspace` thành tệp DPO JSONL sạch cho GPU Flywheel."""
        from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace

        out_path = Path(duong_dan_xuat) if duong_dan_xuat else (self.thu_muc_lam_viec / "workspace_dpo_feedback.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ql = QuanLyPhienChatWorkspace(thu_muc_du_lieu=thu_muc_workspace, bat_xac_thuc=False)
        try:
            return ql.xuat_tap_du_lieu_dpo(
                duong_dan_jsonl=out_path,
                ma_workspace=ma_workspace,
                an_pii=an_pii,
            )
        finally:
            ql.dong()

    def kiem_dinh_eval_gate(
        self,
        ket_qua: KetQuaTinhChinhXa,
        bo_cau_hoi_kiem_thu: Optional[List[str]] = None,
        diem_toi_thieu: float = 0.5,
        ham_danh_gia: Optional[Callable[[KetQuaTinhChinhXa], float]] = None,
        su_dung_eval_gate_4_truc: bool = False,
        ban_goc: Optional[Any] = None,
        gioi_han_mau_4_truc: int = 6,
        tro_ly_eval: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Cổng kiểm định trước khi promote model về Production:
        1. Xác thực chữ ký toàn vẹn HMAC-SHA256 của artifact tải về.
        2. Nạp thực tế tệp GGUF qua `DongCoSuyLuanEdge` và kiểm tra đầu ra an toàn không bị tường lửa chặn.
        3. Khi bật `su_dung_eval_gate_4_truc=True` (hoặc truyền `ban_goc`), tự động chấm hồi quy 4 trục qua `CongKiemDinhMoHinh`
           (trích dẫn pháp lý, hiệu lực luật, chống ảo giác/PII, kháng Prompt Injection).
        4. So sánh điểm đánh giá với ngưỡng `diem_toi_thieu`.
        """
        artifact = ket_qua.duong_dan_gguf or ket_qua.duong_dan_adapter
        if not artifact or not Path(artifact).is_file():
            return {
                "dat_chuan": False,
                "ly_do": "Không có artifact tệp hợp lệ để kiểm định.",
                "diem": 0.0,
            }

        hmac_ok = LuuTai.xac_thuc_file(artifact)
        if not hmac_ok:
            return {
                "dat_chuan": False,
                "ly_do": "Chữ ký toàn vẹn HMAC-SHA256 của artifact không hợp lệ.",
                "hmac_hop_le": False,
                "diem": 0.0,
            }

        if ham_danh_gia is not None:
            diem = float(ham_danh_gia(ket_qua))
            return {
                "dat_chuan": diem >= diem_toi_thieu,
                "hmac_hop_le": True,
                "diem": round(diem, 4),
                "diem_toi_thieu": diem_toi_thieu,
                "che_do": "ham_danh_gia_tuy_chinh",
            }

        cau_hoi_list = bo_cau_hoi_kiem_thu or [
            "Xin chào, hãy giới thiệu ngắn gọn về hệ thống AI tiếng Việt.",
            "Thời giờ làm việc bình thường theo quy định pháp luật là bao nhiêu?",
        ]

        gguf_valid = False
        so_tensor_gguf = 0
        edge_obj: Optional[Any] = None
        if ket_qua.duong_dan_gguf and Path(ket_qua.duong_dan_gguf).is_file():
            from vietnamese_ai.edge.native_inference import (
                DongCoSuyLuanEdge,
                GGUFMemoryMappedStreamer,
            )

            streamer = GGUFMemoryMappedStreamer(ket_qua.duong_dan_gguf)
            thong_tin_gguf = streamer.lay_thong_tin()
            streamer.close()
            gguf_valid = True
            so_tensor_gguf = int(thong_tin_gguf.get("so_tensor", 0))
            edge_obj = DongCoSuyLuanEdge(duong_dan_mo_hinh=ket_qua.duong_dan_gguf, do_dai_toi_da=32)

        if su_dung_eval_gate_4_truc or ban_goc is not None:
            from vietnamese_ai.fine_tuning.eval_gate import CongKiemDinhMoHinh

            ung_vien_eval: Any = tro_ly_eval
            if ung_vien_eval is None:
                if edge_obj is not None:
                    def _edge_fn(cau_hoi: str, _e: Any = edge_obj) -> str:
                        r = _e.sinh_kem_thong_ke(cau_hoi, do_dai_toi_da=32)
                        return str(r.get("van_ban", ""))

                    ung_vien_eval = _edge_fn
                else:
                    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

                    ung_vien_eval = TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=True)

            gate = CongKiemDinhMoHinh()
            kq_4 = gate.so_sanh_va_quyet_dinh(
                ung_vien=ung_vien_eval,
                ban_goc=ban_goc,
                gioi_han_mau=max(1, int(gioi_han_mau_4_truc)),
                nguong_dat=diem_toi_thieu,
            )
            kq_4_dict = kq_4.to_dict()
            diem_tong = float(kq_4_dict["diem_ung_vien"].get("diem_tong_hop", 0.0))
            return {
                "dat_chuan": bool(kq_4_dict["dat_chuan"]),
                "quyet_dinh": kq_4_dict["quyet_dinh"],
                "ly_do": kq_4_dict["ly_do"],
                "hmac_hop_le": True,
                "gguf_hop_le": gguf_valid,
                "so_tensor_gguf": so_tensor_gguf,
                "diem": round(diem_tong, 4),
                "diem_toi_thieu": diem_toi_thieu,
                "che_do": "eval_gate_4_truc",
                "bao_cao_4_truc": kq_4_dict,
            }

        if edge_obj is not None:
            so_dat = 0
            for q in cau_hoi_list:
                kq_s = edge_obj.sinh_kem_thong_ke(q, do_dai_toi_da=16)
                if kq_s.get("an_toan") and str(kq_s.get("van_ban", "")).strip():
                    so_dat += 1
            diem = so_dat / max(1, len(cau_hoi_list))
            return {
                "dat_chuan": diem >= diem_toi_thieu,
                "hmac_hop_le": True,
                "gguf_hop_le": True,
                "so_tensor_gguf": so_tensor_gguf,
                "so_cau_dat": so_dat,
                "tong_so_cau": len(cau_hoi_list),
                "diem": round(diem, 4),
                "diem_toi_thieu": diem_toi_thieu,
            }

        return {
            "dat_chuan": True,
            "hmac_hop_le": True,
            "diem": 1.0,
            "diem_toi_thieu": diem_toi_thieu,
        }

    def chay_vong_lap_flywheel(
        self,
        nguon_du_lieu: Union[str, Path, List[Dict[str, Any]]],
        ten_nha_cung_cap: str = "modal",
        cau_hinh: Optional[CauHinhTinhChinhXa] = None,
        chay_that: bool = False,
        chay_eval_gate: bool = True,
        diem_toi_thieu_eval: float = 0.5,
        ham_danh_gia_eval: Optional[Callable[[KetQuaTinhChinhXa], float]] = None,
        quan_ly_mo_hinh: Optional[Any] = None,
        ten_dang_ky_registry: Optional[str] = None,
        su_dung_eval_gate_4_truc: bool = False,
        ban_goc_eval: Optional[Any] = None,
        gioi_han_mau_4_truc: int = 6,
    ) -> KetQuaTinhChinhXa:
        """
        Thực thi trọn vẹn vòng lặp Flywheel Local -> Remote GPU -> Eval Gate -> Edge Artifact:
        1. Làm sạch PII/Backdoor cục bộ.
        2. Kiểm tra đồng thuận đám mây (`cho_phep_gui_cloud`) & trần ngân sách GPU (`MayTinhTien`).
        3. Gửi tác vụ lên nhà cung cấp (`modal` / `ssh` / `local` / tùy chỉnh) hoặc xuất kế hoạch dry-run.
        4. Xác minh SHA-256, ký HMAC-SHA256, chạy Eval Gate và promote vào `QuanLyMoHinh` nếu đạt chuẩn.
        """
        cfg = cau_hinh or CauHinhTinhChinhXa()
        provider = self.lay_nha_cung_cap(ten_nha_cung_cap)
        ma_tac_vu = f"{provider.ten_nha_cung_cap}_{uuid.uuid4().hex[:8]}"
        thu_muc_tac_vu = self.thu_muc_lam_viec / ma_tac_vu

        tep_sach, so_mau_sach, bc_an_ninh = self.chuan_bi_du_lieu_an_toan(
            nguon_du_lieu=nguon_du_lieu,
            cau_hinh=cfg,
            thu_muc_dich=thu_muc_tac_vu,
            ma_tac_vu=ma_tac_vu,
        )
        if so_mau_sach == 0:
            return KetQuaTinhChinhXa(
                nha_cung_cap=provider.ten_nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="bi_chan_bao_mat",
                bao_cao_an_ninh_du_lieu=bc_an_ninh,
                chi_tiet_loi="Không còn mẫu dữ liệu hợp lệ nào sau khi lọc bảo mật và khử độc.",
            )

        bc_chi_phi = provider.uoc_tinh_chi_phi(cfg, so_mau=so_mau_sach)
        if bc_chi_phi.get("vuot_ngan_sach"):
            return KetQuaTinhChinhXa(
                nha_cung_cap=provider.ten_nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="bi_chan_ngan_sach",
                bao_cao_chi_phi=bc_chi_phi,
                bao_cao_an_ninh_du_lieu=bc_an_ninh,
                chi_tiet_loi=(
                    f"Chi phí ước tính ({bc_chi_phi['chi_phi_uoc_tinh_usd']} USD) vượt ngân sách tối đa "
                    f"({cfg.ngan_sach_toi_da_usd} USD)."
                ),
            )

        if chay_that and provider.la_dam_may_ngoai and not cfg.cho_phep_gui_cloud:
            return KetQuaTinhChinhXa(
                nha_cung_cap=provider.ten_nha_cung_cap,
                ma_tac_vu=ma_tac_vu,
                trang_thai="bi_chan_bao_mat",
                bao_cao_chi_phi=bc_chi_phi,
                bao_cao_an_ninh_du_lieu=bc_an_ninh,
                chi_tiet_loi=(
                    f"Cổng bảo vệ dữ liệu (Nghị định 13/2023/NĐ-CP): Nhà cung cấp '{provider.ten_nha_cung_cap}' "
                    "thuộc đám mây ngoài. Hãy xác nhận rõ ràng bằng `cho_phep_gui_cloud=True` (hoặc cờ `--allow-cloud`)."
                ),
            )

        kq = provider.thuc_thi_huan_luyen(
            tep_du_lieu_sach=tep_sach,
            thu_muc_dich=thu_muc_tac_vu,
            cau_hinh=cfg,
            ma_tac_vu=ma_tac_vu,
            chay_that=chay_that,
        )
        kq.bao_cao_chi_phi = bc_chi_phi
        kq.bao_cao_an_ninh_du_lieu = bc_an_ninh

        if kq.trang_thai == "hoan_thanh" and chay_eval_gate:
            kq_eval = self.kiem_dinh_eval_gate(
                kq,
                diem_toi_thieu=diem_toi_thieu_eval,
                ham_danh_gia=ham_danh_gia_eval,
                su_dung_eval_gate_4_truc=su_dung_eval_gate_4_truc,
                ban_goc=ban_goc_eval,
                gioi_han_mau_4_truc=gioi_han_mau_4_truc,
            )
            kq.ket_qua_eval_gate = kq_eval
            if kq_eval.get("dat_chuan") and quan_ly_mo_hinh is not None:
                ten_reg = ten_dang_ky_registry or f"{cfg.mo_hinh}_{provider.ten_nha_cung_cap}"
                phien_ban = f"remote_{ma_tac_vu}"
                try:
                    quan_ly_mo_hinh.dang_ky(
                        {
                            "duong_dan_gguf": kq.duong_dan_gguf,
                            "duong_dan_adapter": kq.duong_dan_adapter,
                            "sha256": kq.sha256_artifact,
                            "nha_cung_cap": provider.ten_nha_cung_cap,
                        },
                        ten=ten_reg,
                        version=phien_ban,
                        metrics={"eval_gate_score": kq_eval.get("diem", 0.0)},
                        ghi_chu=f"Tinh chỉnh từ xa ({provider.ten_nha_cung_cap}/{cfg.loai_gpu}) đã qua Eval Gate",
                    )
                    quan_ly_mo_hinh.promote(ten_reg, phien_ban, "production")
                    kq.da_promote_registry = True
                except Exception as exc:
                    self.logger.warning(f"Không thể đăng ký vào Model Registry: {exc}")

        return kq


RemoteFineTuneConfig = CauHinhTinhChinhXa
RemoteFineTuneResult = KetQuaTinhChinhXa
RemoteGPUProvider = NhaCungCapGPUXa
ModalGPUProvider = NhaCungCapModal
SSHRemoteGPUProvider = NhaCungCapSSH
LocalSubprocessGPUProvider = NhaCungCapCucBo
RemoteFineTuneOrchestrator = BoDieuPhoiTinhChinhXa

__all__ = [
    "BANG_GIA_GPU_USD_GIO",
    "TY_GIA_USD_VND",
    "CauHinhTinhChinhXa",
    "KetQuaTinhChinhXa",
    "NhaCungCapGPUXa",
    "NhaCungCapModal",
    "NhaCungCapSSH",
    "NhaCungCapCucBo",
    "BoDieuPhoiTinhChinhXa",
    "RemoteFineTuneConfig",
    "RemoteFineTuneResult",
    "RemoteGPUProvider",
    "ModalGPUProvider",
    "SSHRemoteGPUProvider",
    "LocalSubprocessGPUProvider",
    "RemoteFineTuneOrchestrator",
]
