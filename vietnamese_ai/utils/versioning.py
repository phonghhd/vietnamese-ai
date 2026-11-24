"""Quản Lý Phiên Bản Semantic Versioning 2.0.0 & Đóng Gói Tách Gói PyPI Độc Lập (`vietnamese_ai.utils.versioning`).

Hỗ trợ:
- Phân tích, so sánh và nâng phiên bản tuân thủ nghiêm ngặt chuẩn SemVer 2.0.0 (`MAJOR.MINOR.PATCH[-prerelease][+build]`).
- Kiểm tra tính đồng bộ phiên bản giữa mã nguồn (`__init__.py`), `pyproject.toml`, `setup.py` và các gói con độc lập.
- Sinh cấu hình `pyproject.toml` chuẩn PEP 621 cho các gói phân phối PyPI tách rời (`vietnamese-ai-nlp`,
  `vietnamese-ai-edge`, `vietnamese-ai-legal`, `vietnamese-ai-zkml`, `vietnamese-ai-classic-ml`) trong khi
  `vietnamese-ai` đóng vai trò meta-package hợp nhất.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

PHIEN_BAN_CHUAN = "31.0.3"

DANH_MUC_GOI_DOC_LAP: Dict[str, Dict[str, Any]] = {
    "vietnamese-ai": {
        "vai_tro": "meta-package",
        "mo_ta": "Meta-package hợp nhất Vietnamese AI Framework (lõi tối thiểu + tất cả nhóm mở rộng tùy chọn)",
        "phien_ban": PHIEN_BAN_CHUAN,
        "namespaces": ["vietnamese_ai"],
        "phu_thuoc": ["numpy>=1.20.0,<3.0.0"],
        "phu_thuoc_tuy_chon": {
            "nlp": ["underthesea>=6.0"],
            "edge": ["numpy>=1.20.0,<3.0.0"],
            "legal": [
                "pypdf>=3.0.0",
                "python-docx>=1.0.0",
                "pdfplumber>=0.10.0",
                "pillow>=9.0.0",
                "pytesseract>=0.3.10",
            ],
            "zkml": ["numpy>=1.20.0,<3.0.0"],
            "classic-ml": ["scikit-learn>=1.0.0", "pandas>=1.3.0"],
        },
    },
    "vietnamese-ai-nlp": {
        "vai_tro": "sub-package",
        "mo_ta": "Lõi NLP tiếng Việt: BoTachTuTiengVietSieuToc, BoNhungVanBanSieuToc, GanNhanTuLoai, NER & PhoBERT",
        "phien_ban": PHIEN_BAN_CHUAN,
        "namespaces": ["vietnamese_ai.nlp", "vietnamese_ai.preprocessing.text"],
        "phu_thuoc": ["numpy>=1.20.0,<3.0.0"],
        "phu_thuoc_tuy_chon": {
            "vietnamese-nlp": ["underthesea>=6.0"],
            "phobert": ["transformers>=4.30.0", "torch>=2.0.0"],
        },
    },
    "vietnamese-ai-edge": {
        "vai_tro": "sub-package",
        "mo_ta": "Động cơ suy luận Zero-GPU Edge CPU: GGUF mmap, Q2_K/Q3_K/Q4_K/Q4_0/Q4_1/Q8_0, Swarm CPU LAN",
        "phien_ban": PHIEN_BAN_CHUAN,
        "namespaces": ["vietnamese_ai.edge", "vietnamese_ai.core.edge"],
        "phu_thuoc": ["numpy>=1.20.0,<3.0.0"],
        "phu_thuoc_tuy_chon": {
            "local-llm": ["llama-cpp-python>=0.2.0"],
            "export": ["onnxruntime>=1.16.0", "gguf>=0.6.0"],
        },
    },
    "vietnamese-ai-legal": {
        "vai_tro": "sub-package",
        "mo_ta": "RAG Hành chính & Pháp lý Việt Nam: Hierarchy-Preserving Chunking, OCR/PDF/DOCX Parser & Legal Prompts",
        "phien_ban": PHIEN_BAN_CHUAN,
        "namespaces": ["vietnamese_ai.rag.legal", "vietnamese_ai.legal", "vietnamese_ai.legal.rag"],
        "phu_thuoc": ["numpy>=1.20.0,<3.0.0"],
        "phu_thuoc_tuy_chon": {
            "ocr-pdf": [
                "pypdf>=3.0.0",
                "python-docx>=1.0.0",
                "pdfplumber>=0.10.0",
                "pillow>=9.0.0",
                "pytesseract>=0.3.10",
            ],
        },
    },
    "vietnamese-ai-zkml": {
        "vai_tro": "sub-package",
        "mo_ta": "Mật mã học xác thực suy luận AI: CayMerkleSuyLuan, Schnorr/Fiat-Shamir RFC 5114 & KiemToanMatMa",
        "phien_ban": PHIEN_BAN_CHUAN,
        "namespaces": ["vietnamese_ai.crypto", "vietnamese_ai.zkml"],
        "phu_thuoc": ["numpy>=1.20.0,<3.0.0"],
        "phu_thuoc_tuy_chon": {},
    },
    "vietnamese-ai-classic-ml": {
        "vai_tro": "sub-package",
        "mo_ta": "Gói mở rộng Học máy cổ điển tiếng Việt (PhanLoai, HoiQuy, PhanCum, MangNron)",
        "phien_ban": PHIEN_BAN_CHUAN,
        "namespaces": ["vietnamese_ai.classic_ml", "vietnamese_ai.models"],
        "phu_thuoc": ["numpy>=1.20.0,<3.0.0", "scikit-learn>=1.0.0", "pandas>=1.3.0"],
        "phu_thuoc_tuy_chon": {
            "tabular": ["xgboost>=2.0.0", "lightgbm>=4.0.0"],
        },
    },
}


@dataclass(frozen=True)
class PhienBanSemVer:
    """Biểu diễn phiên bản tuân thủ nghiêm ngặt Semantic Versioning 2.0.0."""

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def phan_tich(cls, chuoi_phien_ban: str) -> "PhienBanSemVer":
        """Phân tích chuỗi SemVer 2.0.0; ném `ValueError` nếu không đúng định dạng chuẩn."""
        if not isinstance(chuoi_phien_ban, str):
            raise ValueError("Chuỗi phiên bản SemVer phải có kiểu str")
        match = _SEMVER_RE.match(chuoi_phien_ban.strip())
        if not match:
            raise ValueError(
                f"Phiên bản '{chuoi_phien_ban}' không tuân thủ chuẩn Semantic Versioning 2.0.0 (MAJOR.MINOR.PATCH)"
            )
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    @classmethod
    def hop_le(cls, chuoi_phien_ban: str) -> bool:
        """Kiểm tra nhanh một chuỗi có hợp lệ chuẩn SemVer 2.0.0 hay không."""
        try:
            cls.phan_tich(chuoi_phien_ban)
            return True
        except ValueError:
            return False

    def _khoa_so_sanh(self) -> Tuple[int, int, int, int, Tuple[Tuple[int, Union[int, str]], ...]]:
        """Tạo khóa so sánh đúng quy tắc ưu tiên SemVer 2.0.0 (prerelease thấp hơn bản chính thức)."""
        if self.prerelease is None:
            return (self.major, self.minor, self.patch, 1, ())
        phan_pre: List[Tuple[int, Union[int, str]]] = []
        for token in self.prerelease.split("."):
            if token.isdigit():
                phan_pre.append((0, int(token)))
            else:
                phan_pre.append((1, token))
        return (self.major, self.minor, self.patch, 0, tuple(phan_pre))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PhienBanSemVer):
            return NotImplemented
        return self._khoa_so_sanh() < other._khoa_so_sanh()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, PhienBanSemVer):
            return NotImplemented
        return self._khoa_so_sanh() <= other._khoa_so_sanh()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, PhienBanSemVer):
            return NotImplemented
        return self._khoa_so_sanh() > other._khoa_so_sanh()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, PhienBanSemVer):
            return NotImplemented
        return self._khoa_so_sanh() >= other._khoa_so_sanh()

    def tang_patch(self) -> "PhienBanSemVer":
        """Tăng số bản vá lỗi (PATCH) cho các sửa lỗi tương thích ngược."""
        return PhienBanSemVer(self.major, self.minor, self.patch + 1)

    def tang_minor(self) -> "PhienBanSemVer":
        """Tăng số bản tính năng (MINOR) và đặt lại PATCH = 0."""
        return PhienBanSemVer(self.major, self.minor + 1, 0)

    def tang_major(self) -> "PhienBanSemVer":
        """Tăng số bản chính (MAJOR) cho thay đổi phá vỡ API và đặt lại MINOR/PATCH = 0."""
        return PhienBanSemVer(self.major + 1, 0, 0)

    def tuong_thich_api(self, phien_ban_khac: "PhienBanSemVer") -> bool:
        """Kiểm tra tương thích API công khai theo SemVer (cùng MAJOR khi MAJOR >= 1)."""
        if self.major == 0 or phien_ban_khac.major == 0:
            return self.major == phien_ban_khac.major and self.minor == phien_ban_khac.minor
        return self.major == phien_ban_khac.major

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += f"-{self.prerelease}"
        if self.build:
            base += f"+{self.build}"
        return base


class QuanLyPhatHanhGoi:
    """Quản lý quy chuẩn SemVer, đối chiếu đồng bộ phiên bản và xuất manifest cho các gói PyPI tách rời."""

    @staticmethod
    def danh_sach_goi() -> Dict[str, Dict[str, Any]]:
        """Trả về bản sao danh mục meta-package và các gói con độc lập."""
        return {k: dict(v) for k, v in DANH_MUC_GOI_DOC_LAP.items()}

    @staticmethod
    def tao_noi_dung_pyproject(ten_goi: str, phien_ban: Optional[str] = None) -> str:
        """Tạo nội dung tệp `pyproject.toml` chuẩn PEP 621 cho một gói con tách rời."""
        if ten_goi not in DANH_MUC_GOI_DOC_LAP:
            hop_le = ", ".join(sorted(DANH_MUC_GOI_DOC_LAP.keys()))
            raise KeyError(f"Không tìm thấy gói '{ten_goi}' trong danh mục. Hợp lệ: {hop_le}")

        thong_tin = DANH_MUC_GOI_DOC_LAP[ten_goi]
        ver = str(PhienBanSemVer.phan_tich(phien_ban or str(thong_tin["phien_ban"])))
        deps_lines = "\n".join(f'    "{dep}",' for dep in thong_tin["phu_thuoc"])

        opt_blocks: List[str] = []
        for extra_name, extra_deps in thong_tin.get("phu_thuoc_tuy_chon", {}).items():
            inner = "\n".join(f'    "{dep}",' for dep in extra_deps)
            opt_blocks.append(f"{extra_name} = [\n{inner}\n]")
        opt_section = ""
        if opt_blocks:
            opt_section = "\n[project.optional-dependencies]\n" + "\n".join(opt_blocks) + "\n"

        includes = ", ".join(f'"{ns}*"' for ns in thong_tin["namespaces"])
        return (
            "[build-system]\n"
            'requires = ["setuptools>=68.0", "wheel"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            "[project]\n"
            f'name = "{ten_goi}"\n'
            f'version = "{ver}"\n'
            f'description = "{thong_tin["mo_ta"]}"\n'
            'license = {text = "MIT"}\n'
            'requires-python = ">=3.8"\n'
            "dependencies = [\n"
            f"{deps_lines}\n"
            "]\n"
            f"{opt_section}\n"
            "[tool.setuptools.packages.find]\n"
            'where = ["../.."]\n'
            f"include = [{includes}]\n"
        )

    @classmethod
    def xuat_pyproject_goi_con(
        cls,
        ten_goi: str,
        thu_muc_goc: Union[str, Path],
        phien_ban: Optional[str] = None,
    ) -> Path:
        """Xuất tệp `pyproject.toml` cho gói con độc lập vào thư mục đích."""
        dich = Path(thu_muc_goc) / ten_goi / "pyproject.toml"
        dich.parent.mkdir(parents=True, exist_ok=True)
        dich.write_text(cls.tao_noi_dung_pyproject(ten_goi, phien_ban=phien_ban), encoding="utf-8")
        return dich

    @classmethod
    def kiem_tra_tuan_thu_semver(cls, thu_muc_du_an: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Kiểm tra tính hợp lệ SemVer 2.0.0 và sự đồng bộ phiên bản trên toàn dự án."""
        import vietnamese_ai

        goc = Path(thu_muc_du_an) if thu_muc_du_an else Path(__file__).resolve().parents[2]
        ver_init = vietnamese_ai.__version__
        semver_init = PhienBanSemVer.phan_tich(ver_init)

        cac_nguon: Dict[str, str] = {"vietnamese_ai.__version__": ver_init}
        pyproject_path = goc / "pyproject.toml"
        if pyproject_path.exists():
            m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_path.read_text(encoding="utf-8"), re.MULTILINE)
            if m:
                cac_nguon["pyproject.toml"] = m.group(1)

        setup_path = goc / "setup.py"
        if setup_path.exists():
            m = re.search(r'version\s*=\s*"([^"]+)"', setup_path.read_text(encoding="utf-8"))
            if m:
                cac_nguon["setup.py"] = m.group(1)

        packages_dir = goc / "packages"
        if packages_dir.exists():
            for manifest in sorted(packages_dir.glob("*/pyproject.toml")):
                m = re.search(r'^version\s*=\s*"([^"]+)"', manifest.read_text(encoding="utf-8"), re.MULTILINE)
                if m:
                    cac_nguon[f"packages/{manifest.parent.name}/pyproject.toml"] = m.group(1)

        cargo_path = goc / "rust_core" / "Cargo.toml"
        if cargo_path.exists():
            m = re.search(r'^version\s*=\s*"([^"]+)"', cargo_path.read_text(encoding="utf-8"), re.MULTILINE)
            if m:
                cac_nguon["rust_core/Cargo.toml"] = m.group(1)

        tat_ca_hop_le = all(PhienBanSemVer.hop_le(v) for v in cac_nguon.values())
        dong_bo = len(set(cac_nguon.values())) == 1
        return {
            "hop_le_semver": tat_ca_hop_le,
            "dong_bo_phien_ban": dong_bo,
            "phien_ban_chuan": str(semver_init),
            "major": semver_init.major,
            "minor": semver_init.minor,
            "patch": semver_init.patch,
            "cac_nguon": cac_nguon,
        }

    @classmethod
    def kiem_tra_san_sang_phat_hanh(
        cls,
        thu_muc_du_an: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Kiểm định toàn diện trước khi phát hành TestPyPI/PyPI (SemVer, phụ thuộc, type stubs, Rust abi3 và tài liệu)."""
        import vietnamese_ai
        from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc

        goc = Path(thu_muc_du_an) if thu_muc_du_an else Path(__file__).resolve().parents[2]
        bc_semver = cls.kiem_tra_tuan_thu_semver(goc)
        bc_deps = QuanLyPhuThuoc.kiem_tra_xung_dot_phu_thuoc()

        pyi_path = goc / "vietnamese_ai" / "__init__.pyi"
        thieu_stub: List[str] = []
        if pyi_path.exists():
            pyi_txt = pyi_path.read_text(encoding="utf-8")
            for sym in getattr(vietnamese_ai, "__all__", []):
                if f"as {sym}" not in pyi_txt and f"{sym}:" not in pyi_txt:
                    thieu_stub.append(sym)
        else:
            thieu_stub = list(getattr(vietnamese_ai, "__all__", []))

        cargo_path = goc / "rust_core" / "Cargo.toml"
        rust_abi3_ok = cargo_path.exists() and "abi3-py39" in cargo_path.read_text(encoding="utf-8")

        tep_bat_buoc = [
            "README.md",
            "SECURITY.md",
            "pyproject.toml",
            "setup.py",
            "benchmarks/Dockerfile.benchmark",
            "benchmarks/datasets/legal_vn_corpus.jsonl",
            "vietnamese_ai/rag/legal/data/legal_vn_corpus.jsonl",
            "vietnamese_ai/rag/legal/data/legal_vn_qa.jsonl",
        ]
        thieu_tep = [t for t in tep_bat_buoc if not (goc / t).is_file()]

        san_sang = (
            bool(bc_semver["hop_le_semver"])
            and bool(bc_semver["dong_bo_phien_ban"])
            and bool(bc_deps["hop_le"])
            and len(thieu_stub) == 0
            and rust_abi3_ok
            and len(thieu_tep) == 0
        )
        return {
            "san_sang_phat_hanh": san_sang,
            "phien_ban": bc_semver["phien_ban_chuan"],
            "semver": bc_semver,
            "phu_thuoc": bc_deps,
            "so_bieu_tuong_cong_khai": len(getattr(vietnamese_ai, "__all__", [])),
            "thieu_type_stub": thieu_stub,
            "rust_abi3_san_sang": rust_abi3_ok,
            "thieu_tep_bat_buoc": thieu_tep,
            "lenh_dong_goi_goi_y": "python -m build && twine check dist/*",
        }


SemVerVersion = PhienBanSemVer
ReleasePackageManager = QuanLyPhatHanhGoi

__all__ = [
    "PHIEN_BAN_CHUAN",
    "DANH_MUC_GOI_DOC_LAP",
    "PhienBanSemVer",
    "SemVerVersion",
    "QuanLyPhatHanhGoi",
    "ReleasePackageManager",
]
