"""QuanLyKhoTriThucPhapLy (LegalKnowledgeHub) - Quản Lý, Dựng & Tải Kho Tri Thức Pháp Lý `.vaidb` Theo Lĩnh Vực (`vai pull-kb`)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from vietnamese_ai.rag.legal.auto_ingest import (
    META_KEY_CLAUSES,
    META_KEY_RELATIONS,
    META_KEY_VALIDITY,
    BoNapVanBanPhapQuyTuDong,
)
from vietnamese_ai.rag.legal.citation_graph import DoThiThamChieuPhapLy
from vietnamese_ai.rag.legal.open_corpus import (
    tai_bo_phap_ly_mo,
    tai_bo_phap_ly_thoi_gian,
    tai_canh_do_thi_phap_ly_thoi_gian,
)
from vietnamese_ai.rag.legal.validity_tracker import (
    CO_SO_HIEU_LUC_PHAP_LUAT_VN,
    CO_SO_PHIEN_BAN_DIEU_KHOAN_VN,
)
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@dataclass(frozen=True)
class ThongTinGoiTriThucPhapLy:
    """Siêu dữ liệu chuẩn cho một gói kho tri thức pháp lý `.vaidb` phát hành theo lĩnh vực."""

    ma_goi: str
    ten_hien_thi: str
    phien_ban: str
    linh_vuc: str
    mo_ta: str
    tu_khoa_linh_vuc: Tuple[str, ...] = ()
    gom_du_lieu_thoi_gian: bool = True
    url_mac_dinh: str = ""
    sha256_ky_vong: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển thông tin gói tri thức sang từ điển JSON."""
        data = asdict(self)
        data["tu_khoa_linh_vuc"] = list(self.tu_khoa_linh_vuc)
        return data


DANH_MUC_KHO_PHAP_LY_VN: Dict[str, ThongTinGoiTriThucPhapLy] = {
    "vn-business-core-2025": ThongTinGoiTriThucPhapLy(
        ma_goi="vn-business-core-2025",
        ten_hien_thi="Kho Pháp Lý Doanh Nghiệp Việt Nam Toàn Diện 2025 (160 điều khoản + Đồ thị thời gian)",
        phien_ban="2025.1",
        linh_vuc="Tổng hợp Doanh nghiệp",
        mo_ta=(
            "Gồm toàn bộ 100 điều khoản pháp lý mở và 60 điều khoản đa phiên bản hiệu lực 2013–2025 "
            "(Lao động, BHXH, Thuế GTGT, Hóa đơn 123/70, Đất đai, Nhà ở, Đấu thầu, Dữ liệu cá nhân) "
            "kèm HNSW và 38 cạnh đồ thị tham chiếu."
        ),
        tu_khoa_linh_vuc=(),
        gom_du_lieu_thoi_gian=True,
        url_mac_dinh="https://github.com/phonghhd/vietnamese-ai/releases/download/v31.3.0/vn-business-core-2025.vaidb",
    ),
    "vn-labor-bhxh-2025": ThongTinGoiTriThucPhapLy(
        ma_goi="vn-labor-bhxh-2025",
        ten_hien_thi="Kho Luật Lao Động & Bảo Hiểm Xã Hội 2025 (BLLĐ 2012/2019, BHXH 2014→2024)",
        phien_ban="2025.1",
        linh_vuc="Lao động - BHXH",
        mo_ta=(
            "Chuyên sâu Bộ luật Lao động 45/2019/QH14, Nghị định 145/2020/NĐ-CP, Luật BHXH 58/2014/QH13 "
            "và 41/2024/QH15 (giảm điều kiện lương hưu còn 15 năm từ 01/07/2025), Nghị định 158/2025/NĐ-CP."
        ),
        tu_khoa_linh_vuc=("lao động", "bảo hiểm xã hội", "bhxh", "lương hưu", "thử việc"),
        gom_du_lieu_thoi_gian=True,
        url_mac_dinh="https://github.com/phonghhd/vietnamese-ai/releases/download/v31.3.0/vn-labor-bhxh-2025.vaidb",
    ),
    "vn-tax-invoice-2025": ThongTinGoiTriThucPhapLy(
        ma_goi="vn-tax-invoice-2025",
        ten_hien_thi="Kho Luật Thuế GTGT & Hóa Đơn Điện Tử 2025 (NĐ 123/70, TT 32/2025, GTGT 2024)",
        phien_ban="2025.1",
        linh_vuc="Thuế & Hóa đơn",
        mo_ta=(
            "Chuyên sâu Nghị định 123/2020/NĐ-CP & 70/2025/NĐ-CP (bãi bỏ hủy hóa đơn từ 01/06/2025), "
            "Thông tư 32/2025/TT-BTC, Luật Thuế GTGT 48/2024/QH15, Nghị định 181/2025/NĐ-CP và Quản lý thuế."
        ),
        tu_khoa_linh_vuc=("thuế", "hóa đơn", "gtgt", "chứng từ", "quyết toán"),
        gom_du_lieu_thoi_gian=True,
        url_mac_dinh="https://github.com/phonghhd/vietnamese-ai/releases/download/v31.3.0/vn-tax-invoice-2025.vaidb",
    ),
    "vn-land-housing-2025": ThongTinGoiTriThucPhapLy(
        ma_goi="vn-land-housing-2025",
        ten_hien_thi="Kho Luật Đất Đai, Nhà Ở & Bất Động Sản 2025 (31/2024, 27/2023, 29/2023)",
        phien_ban="2025.1",
        linh_vuc="Đất đai - Nhà ở - BĐS",
        mo_ta=(
            "Chuyên sâu Luật Đất đai 45/2013 → 31/2024/QH15, Nghị định 101/102/2024, Luật Nhà ở 27/2023/QH15, "
            "Luật Kinh doanh BĐS 29/2023/QH15 (trần đặt cọc 5%, chung cư mini, bảng giá đất hằng năm)."
        ),
        tu_khoa_linh_vuc=("đất đai", "nhà ở", "bất động sản", "sổ đỏ", "sổ hồng", "chung cư"),
        gom_du_lieu_thoi_gian=True,
        url_mac_dinh="https://github.com/phonghhd/vietnamese-ai/releases/download/v31.3.0/vn-land-housing-2025.vaidb",
    ),
    "vn-data-digital-2025": ThongTinGoiTriThucPhapLy(
        ma_goi="vn-data-digital-2025",
        ten_hien_thi="Kho Luật Dữ Liệu Cá Nhân, An Ninh Mạng & Giao Dịch Điện Tử 2025",
        phien_ban="2025.1",
        linh_vuc="Dữ liệu & Giao dịch số",
        mo_ta=(
            "Chuyên sâu Nghị định 13/2023/NĐ-CP (72 giờ thông báo vi phạm, chuyển dữ liệu ra nước ngoài), "
            "Luật An ninh mạng 24/2018 & NĐ 53/2022, Luật Giao dịch điện tử 20/2023 & NĐ 137/2024."
        ),
        tu_khoa_linh_vuc=("dữ liệu", "an ninh mạng", "giao dịch điện tử", "chữ ký", "công nghệ"),
        gom_du_lieu_thoi_gian=True,
        url_mac_dinh="https://github.com/phonghhd/vietnamese-ai/releases/download/v31.3.0/vn-data-digital-2025.vaidb",
    ),
}


def _tinh_sha256_tep(duong_dan: Path) -> str:
    """Tính mã băm SHA-256 theo luồng khối 64KB cho tệp `.vaidb`."""
    h = hashlib.sha256()
    with duong_dan.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _kiem_tra_url_tai_kb(url: str) -> str:
    """Xác thực URL tải gói `.vaidb` hợp lệ (chỉ cho phép `https://` hoặc loopback khi kiểm thử)."""
    parsed = urllib.parse.urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if scheme == "https" and host:
        return url.strip()
    if scheme == "http" and host in ("127.0.0.1", "localhost", "::1"):
        return url.strip()
    raise ValueError(f"URL tải kho tri thức không an toàn hoặc không hợp lệ: {url}")


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Trình quản lý, dựng cục bộ và tải kho tri thức pháp lý `.vaidb` dựng sẵn theo lĩnh vực "
        "kèm HNSW, đồ thị hiệu lực thời gian và xác minh SHA-256 (`vai pull-kb`)."
    ),
)
class QuanLyKhoTriThucPhapLy:
    """Quản lý tải, tự dựng cục bộ, đóng gói từ thư mục doanh nghiệp và kiểm chứng SHA-256 cho kho `.vaidb`."""

    def __init__(self, thu_muc_luu: Optional[Union[str, Path]] = None) -> None:
        self.thu_muc_luu = (
            Path(thu_muc_luu) if thu_muc_luu is not None else (Path.home() / ".vietnamese_ai" / "kb")
        )
        self.thu_muc_luu.mkdir(parents=True, exist_ok=True)

    def danh_sach_goi(self) -> List[Dict[str, Any]]:
        """Liệt kê toàn bộ gói tri thức pháp lý chuẩn cùng trạng thái sẵn sàng cục bộ."""
        ket_qua: List[Dict[str, Any]] = []
        for ma_goi, info in DANH_MUC_KHO_PHAP_LY_VN.items():
            vaidb_path = self.thu_muc_luu / f"{ma_goi}.vaidb"
            manifest_path = Path(f"{vaidb_path}.manifest.json")
            item = info.to_dict()
            item["da_cai_cuc_bo"] = vaidb_path.is_file()
            item["duong_dan"] = str(vaidb_path)
            item["kich_thuoc_kb"] = round(vaidb_path.stat().st_size / 1024.0, 1) if vaidb_path.is_file() else 0.0
            if manifest_path.is_file():
                try:
                    m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    item["sha256_cuc_bo"] = m_data.get("sha256")
                    item["tong_dieu_khoan"] = m_data.get("tong_dieu_khoan", 0)
                except Exception:
                    pass
            ket_qua.append(item)
        return ket_qua

    @staticmethod
    def _khop_bo_loc_linh_vuc(item: Dict[str, Any], tu_khoa: Sequence[str]) -> bool:
        """Kiểm tra một bản ghi điều khoản có thuộc phạm vi lĩnh vực mục tiêu hay không."""
        if not tu_khoa:
            return True
        ghep = " ".join(
            str(item.get(k, ""))
            for k in ("linh_vuc", "ten_van_ban", "van_ban", "tieu_de", "so_hieu", "noi_dung")
        ).lower()
        return any(tk.lower() in ghep for tk in tu_khoa)

    def dung_goi_cuc_bo(
        self,
        ma_goi: str,
        duong_dan_dich: Optional[Union[str, Path]] = None,
        xay_dung_hnsw: bool = True,
        che_do_nhung: str = "onnx_viet",
        duong_dan_onnx_embed: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Dựng 100% offline một gói `.vaidb` chuẩn hóa từ bộ pháp lý mở + đa phiên bản thời gian + HNSW + manifest SHA-256."""
        if ma_goi not in DANH_MUC_KHO_PHAP_LY_VN:
            raise ValueError(
                f"Không tìm thấy gói tri thức '{ma_goi}'. Các gói hợp lệ: {', '.join(DANH_MUC_KHO_PHAP_LY_VN)}"
            )

        from vietnamese_ai.rag.rag_pipeline import RAGPipeline

        info = DANH_MUC_KHO_PHAP_LY_VN[ma_goi]
        dich = Path(duong_dan_dich) if duong_dan_dich is not None else (self.thu_muc_luu / f"{ma_goi}.vaidb")
        dich.parent.mkdir(parents=True, exist_ok=True)
        if dich.exists():
            dich.unlink()

        ban_ghi_gop: List[Dict[str, Any]] = []
        da_co_id: set[str] = set()

        if info.gom_du_lieu_thoi_gian:
            for item in tai_bo_phap_ly_thoi_gian():
                if self._khop_bo_loc_linh_vuc(item, info.tu_khoa_linh_vuc):
                    ma = str(item.get("id", ""))
                    if ma and ma not in da_co_id:
                        da_co_id.add(ma)
                        ban_ghi_gop.append(dict(item))

        for item in tai_bo_phap_ly_mo():
            if self._khop_bo_loc_linh_vuc(item, info.tu_khoa_linh_vuc):
                ma = str(item.get("id", ""))
                if not ma or ma in da_co_id:
                    ma = f"mo_{ma}"
                da_co_id.add(ma)
                ban_sao = dict(item)
                ban_sao["id"] = ma
                ban_ghi_gop.append(ban_sao)

        rag = RAGPipeline(
            backend="vaidb",
            duong_dan_vaidb=str(dich),
            che_do_nhung=che_do_nhung,
            duong_dan_onnx_embed=str(duong_dan_onnx_embed) if duong_dan_onnx_embed else None,
            che_do_rerank="onnx_viet",
            che_do_hanh_chinh=True,
        )

        so_hieu_co: set[str] = set()
        try:
            for item in ban_ghi_gop:
                so_hieu = str(item.get("so_hieu", ""))
                if so_hieu:
                    so_hieu_co.add(so_hieu)
                ten_vb = str(item.get("ten_van_ban") or item.get("van_ban") or so_hieu)
                dieu = str(item.get("dieu", ""))
                khoan = str(item.get("khoan", ""))
                tieu_de = str(item.get("tieu_de", ""))
                noi_dung = str(item.get("noi_dung", ""))
                khoan_tag = f" {khoan}" if khoan else ""
                van_ban_nap = f"[{ten_vb} - {so_hieu}] {dieu}{khoan_tag}: {tieu_de}\n{noi_dung}".strip()
                rag.them_tai_lieu(
                    str(item["id"]),
                    van_ban_nap,
                    metadata={
                        "van_ban": ten_vb,
                        "so_hieu": so_hieu,
                        "dieu": dieu,
                        "khoan": khoan,
                        "tieu_de": tieu_de,
                        "linh_vuc": str(item.get("linh_vuc", info.linh_vuc)),
                        "hieu_luc_tu": item.get("hieu_luc_tu"),
                        "het_hieu_luc_tu": item.get("het_hieu_luc_tu"),
                        "trang_thai": item.get("trang_thai", "hieu_luc"),
                    },
                    che_do_hanh_chinh=False,
                )

            hnsw_info: Optional[Dict[str, Any]] = None
            if xay_dung_hnsw and len(ban_ghi_gop) >= 2 and hasattr(rag.csdl_vector, "xay_dung_chi_muc_hnsw"):
                hnsw_info = rag.csdl_vector.xay_dung_chi_muc_hnsw(
                    m=12,
                    ef_construction=64,
                    ef_search=48,
                    luu_xuong_dia=True,
                )

            conn = getattr(rag.csdl_vector, "_conn", None)
            if conn is not None:
                ho_so_list = [
                    hs.to_dict()
                    for hs in CO_SO_HIEU_LUC_PHAP_LUAT_VN
                    if (not info.tu_khoa_linh_vuc) or (hs.so_hieu in so_hieu_co)
                ]
                phien_ban_list = [
                    pb.to_dict()
                    for pb in CO_SO_PHIEN_BAN_DIEU_KHOAN_VN
                    if (not info.tu_khoa_linh_vuc) or (pb.so_hieu in so_hieu_co)
                ]
                do_thi_mac_dinh = DoThiThamChieuPhapLy()
                quan_he_list = [
                    qh.to_dict()
                    for qh in do_thi_mac_dinh._canh
                    if (not info.tu_khoa_linh_vuc) or (qh.nguon in so_hieu_co or qh.dich in so_hieu_co)
                ]
                BoNapVanBanPhapQuyTuDong._ghi_sieu_du_lieu_phap_ly_sql(
                    conn, ho_so_list, phien_ban_list, quan_he_list
                )
                canh_thoi_gian = [
                    c
                    for c in tai_canh_do_thi_phap_ly_thoi_gian()
                    if (not info.tu_khoa_linh_vuc)
                    or (str(c.get("nguon", "")) in da_co_id or str(c.get("dich", "")) in da_co_id)
                ]
                conn.execute(
                    "INSERT OR REPLACE INTO vai_meta(khoa, gia_tri) VALUES (?, ?)",
                    ("legal_temporal_clause_edges_json", json.dumps(canh_thoi_gian, ensure_ascii=False)),
                )
                conn.commit()
        finally:
            rag.dong()

        sha256_val = _tinh_sha256_tep(dich)
        manifest_path = Path(f"{dich}.manifest.json")
        manifest = {
            "ma_goi": info.ma_goi,
            "ten_hien_thi": info.ten_hien_thi,
            "phien_ban": info.phien_ban,
            "linh_vuc": info.linh_vuc,
            "ten_tep": dich.name,
            "sha256": sha256_val,
            "kich_thuoc_bytes": dich.stat().st_size,
            "tong_dieu_khoan": len(ban_ghi_gop),
            "che_do_nhung": che_do_nhung,
            "nguon": "local_build",
            "ngay_tao_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "trang_thai": "da_dung_thanh_cong",
            "ma_goi": info.ma_goi,
            "duong_dan_vaidb": str(dich),
            "duong_dan_manifest": str(manifest_path),
            "sha256": sha256_val,
            "kich_thuoc_kb": round(dich.stat().st_size / 1024.0, 1),
            "tong_dieu_khoan": len(ban_ghi_gop),
            "hnsw": hnsw_info,
        }

    def tai_hoac_dung_goi(
        self,
        ma_goi: str,
        duong_dan_dich: Optional[Union[str, Path]] = None,
        uu_tien_cuc_bo: bool = True,
        url_tuy_chinh: Optional[str] = None,
        sha256_ky_vong: Optional[str] = None,
        chay_that: bool = True,
        timeout_giay: float = 60.0,
    ) -> Dict[str, Any]:
        """Tải gói `.vaidb` từ URL từ xa (kèm kiểm tra SHA-256) hoặc tự dựng cục bộ 100% offline."""
        info = DANH_MUC_KHO_PHAP_LY_VN.get(ma_goi)
        dich = Path(duong_dan_dich) if duong_dan_dich is not None else (self.thu_muc_luu / f"{ma_goi}.vaidb")

        if url_tuy_chinh or (not uu_tien_cuc_bo):
            url_nguon = url_tuy_chinh or (info.url_mac_dinh if info is not None else "")
            if not url_nguon:
                raise ValueError(f"Không có URL từ xa cho gói '{ma_goi}'.")
            url_sach = _kiem_tra_url_tai_kb(url_nguon)
            sha_mong_doi = sha256_ky_vong or (info.sha256_ky_vong if info is not None else None)
            if not chay_that:
                return {
                    "trang_thai": "xem_truoc_dry_run",
                    "ma_goi": ma_goi,
                    "url": url_sach,
                    "duong_dan_vaidb": str(dich),
                    "sha256_ky_vong": sha_mong_doi,
                }
            dich.parent.mkdir(parents=True, exist_ok=True)
            tmp_part = dich.with_suffix(".vaidb.part")
            req = urllib.request.Request(url_sach, headers={"User-Agent": "vietnamese-ai-kb-hub/31.3.0"})
            with urllib.request.urlopen(req, timeout=float(timeout_giay)) as resp, tmp_part.open("wb") as out_f:
                while True:
                    buf = resp.read(65536)
                    if not buf:
                        break
                    out_f.write(buf)

            sha_thuc = _tinh_sha256_tep(tmp_part)
            if sha_mong_doi and sha_thuc.lower() != sha_mong_doi.strip().lower():
                tmp_part.unlink(missing_ok=True)
                raise ValueError(
                    f"Sai lệch mã băm SHA-256 cho gói '{ma_goi}': kỳ vọng {sha_mong_doi}, thực tế {sha_thuc}."
                )

            header = tmp_part.read_bytes()[:16]
            if not header.startswith(b"SQLite format 3\x00"):
                tmp_part.unlink(missing_ok=True)
                raise ValueError(f"Tệp tải về cho gói '{ma_goi}' không phải cơ sở dữ liệu SQLite `.vaidb` hợp lệ.")

            tmp_part.replace(dich)
            manifest_path = Path(f"{dich}.manifest.json")
            manifest = {
                "ma_goi": ma_goi,
                "ten_tep": dich.name,
                "sha256": sha_thuc,
                "kich_thuoc_bytes": dich.stat().st_size,
                "url": url_sach,
                "nguon": "remote_download",
                "ngay_tai_utc": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "trang_thai": "da_tai_thanh_cong",
                "ma_goi": ma_goi,
                "duong_dan_vaidb": str(dich),
                "duong_dan_manifest": str(manifest_path),
                "sha256": sha_thuc,
                "kich_thuoc_kb": round(dich.stat().st_size / 1024.0, 1),
                "url": url_sach,
            }

        if info is None:
            raise ValueError(
                f"Không tìm thấy gói tri thức '{ma_goi}'. Các gói hợp lệ: {', '.join(DANH_MUC_KHO_PHAP_LY_VN)}"
            )
        if not chay_that:
            return {
                "trang_thai": "xem_truoc_dry_run",
                "ma_goi": ma_goi,
                "nguon": "local_build",
                "duong_dan_vaidb": str(dich),
            }
        return self.dung_goi_cuc_bo(ma_goi=ma_goi, duong_dan_dich=dich)

    def dong_goi_tu_thu_muc(
        self,
        thu_muc_nguon: Union[str, Path],
        ma_goi: str,
        duong_dan_dich: Optional[Union[str, Path]] = None,
        tach_khoan_dieu_dai: bool = True,
        nguong_tach_khoan_tu: int = 180,
        xay_dung_hnsw: bool = True,
        che_do_nhung: str = "onnx_viet",
        duong_dan_onnx_embed: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Đóng gói toàn bộ văn bản pháp quy / quy chế doanh nghiệp từ thư mục thành 1 tệp `.vaidb` + `.manifest.json`."""
        dich = Path(duong_dan_dich) if duong_dan_dich is not None else (self.thu_muc_luu / f"{ma_goi}.vaidb")
        pipeline = BoNapVanBanPhapQuyTuDong(
            che_do_nhung=che_do_nhung,
            duong_dan_onnx_embed=duong_dan_onnx_embed,
        )
        kq = pipeline.dong_goi_kho_vaidb(
            nguon_dau_vao=thu_muc_nguon,
            duong_dan_vaidb=dich,
            xay_dung_hnsw=xay_dung_hnsw,
            tach_khoan_dieu_dai=tach_khoan_dieu_dai,
            nguong_tach_khoan_tu=nguong_tach_khoan_tu,
            xuat_manifest=True,
        )
        kq["ma_goi"] = ma_goi
        kq["trang_thai"] = "da_dong_goi_thanh_cong"
        return kq

    @staticmethod
    def xac_minh_goi(duong_dan_vaidb: Union[str, Path]) -> Dict[str, Any]:
        """Kiểm chứng tính toàn vẹn SHA-256 và cấu trúc SQLite `.vaidb` đối chiếu với tệp `.manifest.json`."""
        p = Path(duong_dan_vaidb)
        if not p.is_file():
            return {"hop_le": False, "ly_do": f"Không tìm thấy tệp `.vaidb`: {p}"}

        sha_thuc = _tinh_sha256_tep(p)
        manifest_path = Path(f"{p}.manifest.json")
        sha_manifest: Optional[str] = None
        if manifest_path.is_file():
            try:
                m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                sha_manifest = str(m_data.get("sha256", "")).strip() or None
            except Exception:
                return {"hop_le": False, "ly_do": f"Tệp manifest hỏng định dạng JSON: {manifest_path}"}

        khop_sha = (sha_manifest is None) or (sha_thuc.lower() == sha_manifest.lower())
        so_ban_ghi = 0
        co_meta_phap_ly = False
        try:
            conn = sqlite3.connect(str(p))
            try:
                row = conn.execute("SELECT COUNT(*) FROM vai_chunks").fetchone()
                so_ban_ghi = int(row[0]) if row else 0
                meta_rows = conn.execute(
                    "SELECT COUNT(*) FROM vai_meta WHERE khoa IN (?, ?, ?)",
                    (META_KEY_VALIDITY, META_KEY_CLAUSES, META_KEY_RELATIONS),
                ).fetchone()
                co_meta_phap_ly = bool(meta_rows and int(meta_rows[0]) > 0)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return {"hop_le": False, "ly_do": f"Lỗi cấu trúc SQLite `.vaidb`: {exc}", "sha256": sha_thuc}

        return {
            "hop_le": bool(khop_sha and so_ban_ghi > 0),
            "duong_dan_vaidb": str(p),
            "sha256_thuc_te": sha_thuc,
            "sha256_manifest": sha_manifest,
            "khop_manifest_sha256": khop_sha,
            "so_ban_ghi": so_ban_ghi,
            "co_sieu_du_lieu_phap_ly": co_meta_phap_ly,
        }


LegalKnowledgeHub = QuanLyKhoTriThucPhapLy

__all__ = [
    "ThongTinGoiTriThucPhapLy",
    "DANH_MUC_KHO_PHAP_LY_VN",
    "QuanLyKhoTriThucPhapLy",
    "LegalKnowledgeHub",
]
