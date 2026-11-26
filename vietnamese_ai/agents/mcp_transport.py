"""MayChuMCPThucTe (ProductionMCPServer) - Máy Chủ MCP Chuẩn `stdio` & `HTTP/SSE` Cắm Thẳng Vào Cursor, VS Code & Claude Desktop."""

from __future__ import annotations

import json
import queue
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union
from urllib.parse import parse_qs, urlparse

from vietnamese_ai.agents.mcp import MCPServer
from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

MAX_MCP_BODY_BYTES = 16 * 1024 * 1024


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Máy chủ Model Context Protocol (MCP JSON-RPC 2.0) hỗ trợ giao vận chuẩn `stdio` và `HTTP/SSE`, "
        "tích hợp sẵn 6 công cụ Vietnamese AI (tra cứu pháp lý, kiểm tra hiệu lực, đồ thị luật, rà soát hợp đồng, "
        "quét bảo mật PII/Prompt Injection, suy luận Edge LLM) sẵn sàng cắm vào Cursor, VS Code và Claude Desktop."
    ),
    nhom_extra="core",
)
class MayChuMCPThucTe(MCPServer):
    """
    Máy chủ MCP sẵn sàng sản xuất (`stdio` & `HTTP/SSE`) của Vietnamese AI Framework.

    Đăng ký sẵn 6 công cụ chuẩn hóa kèm bảo vệ `TuongLuaBaoVeFramework`:
    - `tra_cuu_phap_ly`: Tra cứu điều khoản luật Việt Nam (100 điều khoản mở / `.vaidb`).
    - `kiem_tra_hieu_luc`: Kiểm tra hiệu lực văn bản luật hoặc quét trích dẫn hết hạn trong hợp đồng.
    - `do_thi_tham_chieu_luat`: Truy vết văn bản thay thế hiện hành, Nghị định/Thông tư hướng dẫn và sơ đồ Mermaid.
    - `ra_soat_hop_dong`: So sánh hợp đồng đối tác với hợp đồng chuẩn và phát hiện rủi ro pháp lý.
    - `quet_bao_mat_du_lieu`: Quét Prompt Injection & ẩn PII tiếng Việt (CCCD, SĐT, Email).
    - `hoi_edge_llm`: Chạy suy luận Zero-GPU Edge CPU cục bộ.
    """

    def __init__(
        self,
        ten: str = "vietnamese-ai-mcp",
        phien_ban: str = "31.1.0",
        duong_dan_vaidb: Optional[Union[str, Path]] = None,
        nap_kho_phap_ly_mo: bool = True,
        bat_tuong_lua: bool = True,
        quan_ly_plugin: Optional[Any] = None,
        tro_ly_phap_ly: Optional[Any] = None,
        bat_cong_cu_mo_rong: bool = False,
        bat_cong_cu_phap_ly_thoi_gian: bool = False,
    ) -> None:
        super().__init__(ten=ten, phien_ban=phien_ban, bat_tuong_lua=bat_tuong_lua)
        self.duong_dan_vaidb = str(duong_dan_vaidb) if duong_dan_vaidb else None
        self.nap_kho_phap_ly_mo = nap_kho_phap_ly_mo
        self.quan_ly_plugin = quan_ly_plugin
        self._tro_ly_phap_ly: Optional[Any] = tro_ly_phap_ly
        self._bo_so_sanh_hd: Optional[Any] = None
        self._dong_co_edge: Optional[Any] = None
        self._dong_co_phap_ly_thoi_gian: Optional[Any] = None
        self._dang_ky_cong_cu_mac_dinh()
        if bat_cong_cu_mo_rong:
            self.dang_ky_cong_cu_nang_cao()
        if bat_cong_cu_phap_ly_thoi_gian:
            self.dang_ky_cong_cu_phap_ly_thoi_gian()
        if quan_ly_plugin is not None:
            self.gan_quan_ly_plugin(quan_ly_plugin)
        self._dang_ky_tai_nguyen_mac_dinh()

    def gan_quan_ly_plugin(self, quan_ly_plugin: Any) -> int:
        """Tự động khám phá và đăng ký toàn bộ công cụ từ hệ sinh thái V-Plugin 2.0 (`QuanLyPlugin`) lên MCP Server."""
        self.quan_ly_plugin = quan_ly_plugin
        plugins_cong_cu = getattr(quan_ly_plugin, "cac_plugin_cong_cu", None)
        if not isinstance(plugins_cong_cu, dict):
            return 0

        da_dang_ky = 0
        for plugin in plugins_cong_cu.values():
            if not hasattr(plugin, "dang_ky_cong_cu"):
                continue
            for t_info in plugin.dang_ky_cong_cu():
                if not isinstance(t_info, dict) or not t_info.get("ten"):
                    continue
                t_name = str(t_info["ten"])
                if t_name in self._cong_cu:
                    continue

                def _tao_ham(_pl: Any = plugin, _tn: str = t_name) -> Any:
                    def _run(**kwargs: Any) -> str:
                        res = _pl.thuc_thi_cong_cu(_tn, **kwargs)
                        return json.dumps(res, ensure_ascii=False, default=str) if isinstance(res, (dict, list)) else str(res)

                    return _run

                cc = CongCu(
                    ten=t_name,
                    mo_ta=str(t_info.get("mo_ta", f"Công cụ V-Plugin {t_name}")),
                    ham_thuc_thi=_tao_ham(),
                )
                t_params = t_info.get("tham_so")
                if isinstance(t_params, dict) and t_params:
                    cc.tham_so = {
                        str(k): {"kieu": "str", "bat_buoc": False, "mo_ta": str(v)}
                        for k, v in t_params.items()
                    }
                self.dang_ky_cong_cu(cc)
                da_dang_ky += 1
        return da_dang_ky

    def _lay_tro_ly_phap_ly(self) -> Any:
        """Khởi tạo lười `TroLyPhapLy` dùng chung cho các công cụ pháp lý MCP."""
        if self._tro_ly_phap_ly is None:
            from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

            self._tro_ly_phap_ly = TroLyPhapLy(
                nap_kho_mo=self.nap_kho_phap_ly_mo,
                duong_dan_vaidb=self.duong_dan_vaidb,
                dung_onnx_semantic=True,
            )
        return self._tro_ly_phap_ly

    def _lay_bo_so_sanh_hd(self) -> Any:
        """Khởi tạo lười `BoSoSanhRaSoatHopDong`."""
        if self._bo_so_sanh_hd is None:
            from vietnamese_ai.rag.legal.contract_review import BoSoSanhRaSoatHopDong

            self._bo_so_sanh_hd = BoSoSanhRaSoatHopDong()
        return self._bo_so_sanh_hd

    def _dang_ky_cong_cu_mac_dinh(self) -> None:
        """Đăng ký 6 công cụ MCP cốt lõi của hệ sinh thái Vietnamese AI."""

        def _tra_cuu_phap_ly(cau_hoi: str, ngay_tra_cuu: str = "") -> str:
            ngay = ngay_tra_cuu.strip() or None
            kq = self._lay_tro_ly_phap_ly().tra_cuu(cau_hoi, ngay_tra_cuu=ngay)
            return json.dumps(kq, ensure_ascii=False)

        def _kiem_tra_hieu_luc(
            so_hieu_hoac_van_ban: str,
            ngay_tra_cuu: str = "",
            dieu: str = "",
            khoan: str = "",
        ) -> str:
            ngay = ngay_tra_cuu.strip() or None
            tl = self._lay_tro_ly_phap_ly()
            if dieu.strip():
                kq = tl.tra_cuu_dieu_khoan_theo_ngay(
                    so_hieu=so_hieu_hoac_van_ban,
                    dieu=dieu.strip(),
                    ngay_tra_cuu=ngay,
                    khoan=khoan.strip() or None,
                )
            else:
                kq = tl.kiem_tra_hieu_luc(so_hieu_hoac_van_ban, ngay_tra_cuu=ngay)
            return json.dumps(kq, ensure_ascii=False)

        def _do_thi_tham_chieu_luat(
            so_hieu: str,
            ngay_tra_cuu: str = "",
            chi_lay_con_hieu_luc: bool = True,
        ) -> str:
            ngay = ngay_tra_cuu.strip() or None
            tl = self._lay_tro_ly_phap_ly()
            kq = tl.tra_cuu_van_ban_lien_quan(
                so_hieu,
                ngay_tra_cuu=ngay,
                chi_lay_con_hieu_luc=bool(chi_lay_con_hieu_luc),
            )
            kq["mermaid"] = tl.do_thi_tham_chieu.xuat_do_thi_mermaid(so_hieu_trung_tam=so_hieu)
            return json.dumps(kq, ensure_ascii=False)

        def _ra_soat_hop_dong(
            hop_dong_doi_tac: str,
            hop_dong_chuan: str = "",
            ngay_tra_cuu: str = "",
            xuat_track_changes_docx: str = "",
        ) -> str:
            chuan = hop_dong_chuan if hop_dong_chuan.strip() else None
            ngay = ngay_tra_cuu.strip() or None
            engine = self._lay_bo_so_sanh_hd()
            kq = engine.so_sanh_va_ra_soat(
                hop_dong_doi_tac=hop_dong_doi_tac,
                hop_dong_chuan=chuan,
                ngay_tra_cuu=ngay,
            )
            if xuat_track_changes_docx.strip():
                out_p = engine.xuat_bao_cao_track_changes_docx(kq, xuat_track_changes_docx.strip())
                kq["track_changes_docx_path"] = str(out_p)
            return json.dumps(kq, ensure_ascii=False)

        def _quet_bao_mat_du_lieu(van_ban: str) -> str:
            from vietnamese_ai.security.data_sanitizer import DataSanitizer
            from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

            fw = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
            kt = fw.kiem_tra_dau_vao(van_ban)
            sach = DataSanitizer.lam_sach(van_ban)
            return json.dumps(
                {
                    "an_toan": kt["an_toan"],
                    "ly_do": kt.get("ly_do", "An toàn"),
                    "van_ban_da_an_pii": sach,
                },
                ensure_ascii=False,
            )

        def _hoi_edge_llm(cau_hoi: str, do_dai_toi_da: int = 96) -> str:
            from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge

            if self._dong_co_edge is None:
                self._dong_co_edge = DongCoSuyLuanEdge(do_dai_toi_da=int(do_dai_toi_da))
            stats = self._dong_co_edge.sinh_kem_thong_ke(cau_hoi, do_dai_toi_da=int(do_dai_toi_da))
            return json.dumps(stats, ensure_ascii=False)

        self.dang_ky_cong_cu(
            CongCu(
                ten="tra_cuu_phap_ly",
                mo_ta="Tra cứu điều khoản pháp luật Việt Nam (100 điều khoản mở & kho .vaidb) kèm kiểm tra hiệu lực và chống ảo giác.",
                ham_thuc_thi=_tra_cuu_phap_ly,
            )
        )
        self.dang_ky_cong_cu(
            CongCu(
                ten="kiem_tra_hieu_luc",
                mo_ta="Kiểm tra hiệu lực một số hiệu văn bản luật Việt Nam hoặc quét toàn bộ hợp đồng tìm trích dẫn hết hạn.",
                ham_thuc_thi=_kiem_tra_hieu_luc,
            )
        )
        self.dang_ky_cong_cu(
            CongCu(
                ten="do_thi_tham_chieu_luat",
                mo_ta="Truy vấn Đồ thị Tham chiếu Pháp lý Việt Nam: văn bản thay thế hiện hành, Nghị định/Thông tư hướng dẫn và sơ đồ Mermaid.",
                ham_thuc_thi=_do_thi_tham_chieu_luat,
            )
        )
        self.dang_ky_cong_cu(
            CongCu(
                ten="ra_soat_hop_dong",
                mo_ta="Rà soát rủi ro pháp lý, so sánh hợp đồng đối tác với hợp đồng chuẩn (Contract Diff) và cảnh báo điều khoản bất lợi.",
                ham_thuc_thi=_ra_soat_hop_dong,
            )
        )
        self.dang_ky_cong_cu(
            CongCu(
                ten="quet_bao_mat_du_lieu",
                mo_ta="Kiểm tra Prompt Injection bằng Tường lửa AI và ẩn tự động dữ liệu nhạy cảm PII (SĐT, CCCD, Email, API key).",
                ham_thuc_thi=_quet_bao_mat_du_lieu,
            )
        )
        self.dang_ky_cong_cu(
            CongCu(
                ten="hoi_edge_llm",
                mo_ta="Chạy suy luận LLM tiếng Việt cục bộ Zero-GPU trên CPU qua DongCoSuyLuanEdge.",
                ham_thuc_thi=_hoi_edge_llm,
            )
        )

    def dang_ky_cong_cu_nang_cao(self) -> int:
        """Đăng ký thêm 3 công cụ MCP chuyên sâu Hướng B (Point-in-Time Điều luật, Tách từ Viterbi/FST, HNSW .vaidb)."""
        da_them = 0

        def _tra_cuu_dieu_khoan(
            so_hieu: str,
            dieu: str,
            ngay_tra_cuu: str = "",
            khoan: str = "",
        ) -> str:
            kq = self._lay_tro_ly_phap_ly().tra_cuu_dieu_khoan_theo_ngay(
                so_hieu=so_hieu,
                dieu=dieu,
                ngay_tra_cuu=ngay_tra_cuu.strip() or None,
                khoan=khoan.strip() or None,
            )
            return json.dumps(kq, ensure_ascii=False)

        def _tach_tu_tieng_viet(
            van_ban: str,
            phuong_phap: str = "viterbi",
            chuan_hoa_dau: bool = True,
            noi_tu_ghep: bool = True,
        ) -> str:
            from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc

            tok = BoTachTuTiengVietSieuToc()
            vb_chuan = tok.chuan_hoa_dau_thanh(van_ban) if chuan_hoa_dau else tok.chuan_hoa_unicode(van_ban)
            tokens = tok.tach_tu(
                van_ban,
                noi_tu_ghep=bool(noi_tu_ghep),
                chuan_hoa_dau=bool(chuan_hoa_dau),
                phuong_phap=phuong_phap,
            )
            return json.dumps(
                {
                    "van_ban_chuan_hoa": vb_chuan,
                    "phuong_phap": phuong_phap,
                    "so_token": len(tokens),
                    "tokens": tokens,
                },
                ensure_ascii=False,
            )

        def _xay_dung_hnsw_vaidb(
            duong_dan_vaidb: str = "",
            m: int = 12,
            ef_construction: int = 64,
            ef_search: int = 48,
        ) -> str:
            path_db = duong_dan_vaidb.strip() or self.duong_dan_vaidb
            if path_db:
                from vietnamese_ai.rag.vaidb_store import KhoTriThucVAIDB

                with KhoTriThucVAIDB(duong_dan=path_db) as db:
                    info = db.xay_dung_chi_muc_hnsw(
                        m=int(m),
                        ef_construction=int(ef_construction),
                        ef_search=int(ef_search),
                    )
                    return json.dumps({"hnsw": info, "thong_ke": db.thong_ke()}, ensure_ascii=False)
            info_tl = self._lay_tro_ly_phap_ly().xay_dung_chi_muc_hnsw(
                m=int(m),
                ef_construction=int(ef_construction),
                ef_search=int(ef_search),
            )
            return json.dumps({"hnsw": info_tl}, ensure_ascii=False)

        ds_moi = [
            CongCu(
                ten="tra_cuu_dieu_khoan_theo_ngay",
                mo_ta="Tra cứu chính xác phiên bản hiệu lực theo mốc thời gian (Point-in-Time) của một Điều/Khoản luật Việt Nam.",
                ham_thuc_thi=_tra_cuu_dieu_khoan,
            ),
            CongCu(
                ten="tach_tu_tieng_viet",
                mo_ta="Chuẩn hóa dấu thanh âm vị học tiếng Việt (FST) và tách từ ghép tối ưu toàn cục Viterbi DAG.",
                ham_thuc_thi=_tach_tu_tieng_viet,
            ),
            CongCu(
                ten="xay_dung_chi_muc_hnsw_vaidb",
                mo_ta="Xây dựng và lưu bền vững đồ thị chỉ mục vector HNSW bên trong tệp kho tri thức `.vaidb`.",
                ham_thuc_thi=_xay_dung_hnsw_vaidb,
            ),
        ]
        for cc in ds_moi:
            if cc.ten not in self._cong_cu:
                self.dang_ky_cong_cu(cc)
                da_them += 1
        return da_them

    def dang_ky_cong_cu_phap_ly_thoi_gian(self) -> int:
        """Đăng ký 2 công cụ MCP v31.1.0 cho ViLegal-TemporalRAG và Kiểm chứng trích dẫn 3 tầng."""
        da_them = 0

        def _lay_dong_co_tg() -> Any:
            if self._dong_co_phap_ly_thoi_gian is None:
                from vietnamese_ai.rag.legal.temporal_rag import (
                    tao_dong_co_phap_ly_thoi_gian_mac_dinh,
                )

                self._dong_co_phap_ly_thoi_gian = tao_dong_co_phap_ly_thoi_gian_mac_dinh()
            return self._dong_co_phap_ly_thoi_gian

        def _tra_cuu_phap_ly_thoi_gian(
            cau_hoi: str,
            ngay_tra_cuu: str = "",
            che_do: str = "full",
            top_k: int = 5,
        ) -> str:
            kq = _lay_dong_co_tg().tra_loi_phap_ly(
                cau_hoi=cau_hoi,
                ngay_tra_cuu=ngay_tra_cuu.strip() or None,
                top_k=max(1, min(20, int(top_k))),
                che_do=che_do,
            )
            return json.dumps(kq, ensure_ascii=False)

        def _kiem_chung_trich_dan_phap_ly(
            cau_tra_loi: str,
            nguon_json: str = "[]",
            ngay_tra_cuu: str = "",
            cau_hoi: str = "",
        ) -> str:
            from vietnamese_ai.rag.legal.citation_verifier import BoKiemChungTrichDanPhapLy

            nguon = json.loads(nguon_json) if nguon_json.strip() else []
            if not isinstance(nguon, list):
                nguon = []
            verifier = BoKiemChungTrichDanPhapLy()
            kq = verifier.kiem_chung(
                cau_tra_loi=cau_tra_loi,
                nguon_truy_hoi=nguon,
                ngay_tra_cuu=ngay_tra_cuu.strip() or None,
                cau_hoi=cau_hoi.strip() or None,
            ).to_dict()
            return json.dumps(kq, ensure_ascii=False)

        for cc in (
            CongCu(
                ten="tra_cuu_phap_ly_thoi_gian",
                mo_ta="Truy hồi đa bước theo mốc thời gian & đồ thị điều khoản kèm kiểm chứng 3 tầng (ViLegal-TemporalRAG).",
                ham_thuc_thi=_tra_cuu_phap_ly_thoi_gian,
            ),
            CongCu(
                ten="kiem_chung_trich_dan_phap_ly",
                mo_ta="Kiểm chứng trích dẫn pháp lý 3 tầng (Grounding, Temporal Consistency, Numeric Consistency).",
                ham_thuc_thi=_kiem_chung_trich_dan_phap_ly,
            ),
        ):
            if cc.ten not in self._cong_cu:
                self.dang_ky_cong_cu(cc)
                da_them += 1
        return da_them

    def _dang_ky_tai_nguyen_mac_dinh(self) -> None:
        """Đăng ký các tài nguyên MCP cơ bản cung cấp siêu dữ liệu hệ thống."""
        self.dang_ky_tai_nguyen(
            uri="vai://legal/open-corpus-summary",
            ten="Tóm tắt Kho Pháp lý Mở Việt Nam",
            noi_dung="Kho pháp lý mở tích hợp sẵn 100 điều khoản thực tế và 35 câu QA chuẩn thuộc 15+ bộ luật/nghị định Việt Nam.",
            mime_type="text/plain",
        )
        self.dang_ky_tai_nguyen(
            uri="vai://system/info",
            ten="Thông tin Vietnamese AI MCP Server",
            noi_dung=json.dumps(
                {
                    "ten": self.ten,
                    "phien_ban": self.phien_ban,
                    "duong_dan_vaidb": self.duong_dan_vaidb,
                    "so_cong_cu": len(self._cong_cu),
                },
                ensure_ascii=False,
            ),
            mime_type="application/json",
        )

    def xu_ly_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mở rộng `MCPServer.xu_ly_jsonrpc` để hỗ trợ chuẩn `ping` và thông báo `notifications/*`."""
        method = str(request.get("method", ""))
        req_id = request.get("id")
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method.startswith("notifications/"):
            return {}
        return super().xu_ly_jsonrpc(request)

    @staticmethod
    def _doc_chinh_xac_bytes_text(src: TextIO, so_byte: int) -> str:
        """Đọc đúng `so_byte` byte UTF-8 từ luồng văn bản (hỗ trợ cả `StringIO` đa byte tiếng Việt)."""
        if so_byte <= 0:
            return ""
        chunks: List[str] = []
        da_doc_bytes = 0
        while da_doc_bytes < so_byte:
            ch = src.read(1)
            if not ch:
                break
            chunks.append(ch)
            da_doc_bytes += len(ch.encode("utf-8"))
        return "".join(chunks)

    def chay_stdio(
        self,
        in_stream: Optional[TextIO] = None,
        out_stream: Optional[TextIO] = None,
    ) -> int:
        """Chạy vòng lặp giao vận MCP chuẩn `stdio` (hỗ trợ cả JSON-RPC từng dòng lẫn khung `Content-Length` chuẩn UTF-8 bytes)."""
        src = in_stream or sys.stdin
        dst = out_stream or sys.stdout
        da_xu_ly = 0
        bin_in = getattr(src, "buffer", None)

        while True:
            use_content_length = False
            if bin_in is not None:
                raw_line = bin_in.readline()
                if not raw_line:
                    break
                stripped_b = raw_line.strip()
                if not stripped_b:
                    continue
                if stripped_b.lower().startswith(b"content-length:"):
                    use_content_length = True
                    try:
                        length = int(stripped_b.split(b":", 1)[1].strip())
                    except ValueError:
                        continue
                    while True:
                        hdr_b = bin_in.readline()
                        if not hdr_b or hdr_b.strip() == b"":
                            break
                    if length < 0 or length > MAX_MCP_BODY_BYTES:
                        continue
                    buf = bytearray()
                    while len(buf) < length:
                        piece = bin_in.read(length - len(buf))
                        if not piece:
                            break
                        buf.extend(piece)
                    raw_payload = bytes(buf).decode("utf-8", errors="replace")
                else:
                    raw_payload = stripped_b.decode("utf-8", errors="replace")
            else:
                line = src.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.lower().startswith("content-length:"):
                    use_content_length = True
                    try:
                        length = int(stripped.split(":", 1)[1].strip())
                    except ValueError:
                        continue
                    while True:
                        hdr = src.readline()
                        if not hdr or hdr.strip() == "":
                            break
                    if length < 0 or length > MAX_MCP_BODY_BYTES:
                        continue
                    raw_payload = self._doc_chinh_xac_bytes_text(src, length)
                else:
                    raw_payload = stripped

            try:
                req = json.loads(raw_payload)
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"JSON không hợp lệ: {exc}"},
                }
                self._ghi_thong_diep_stdio(dst, err_resp, use_content_length)
                continue

            resp = self.xu_ly_jsonrpc(req)
            da_xu_ly += 1
            if resp and req.get("id") is not None:
                self._ghi_thong_diep_stdio(dst, resp, use_content_length)

        return da_xu_ly

    @staticmethod
    def _ghi_thong_diep_stdio(dst: TextIO, payload: Dict[str, Any], dung_content_length: bool) -> None:
        """Ghi phản hồi JSON-RPC ra luồng chuẩn `stdout`."""
        body = json.dumps(payload, ensure_ascii=False)
        if dung_content_length:
            byte_len = len(body.encode("utf-8"))
            dst.write(f"Content-Length: {byte_len}\r\n\r\n{body}")
        else:
            dst.write(body + "\n")
        dst.flush()

    def tao_http_server(self, host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
        """Tạo máy chủ HTTP + SSE (`GET /sse`, `POST /messages`, `POST /rpc`, `GET /health`) chuẩn MCP."""
        mcp_instance = self
        sse_sessions: Dict[str, queue.Queue[Optional[str]]] = {}
        sse_lock = threading.Lock()

        class _MCPHTTPHandler(BaseHTTPRequestHandler):
            def _gui_json(self, obj: Dict[str, Any], status: int = 200) -> None:
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)
                if path == "/health":
                    self._gui_json(
                        {
                            "trang_thai": "ok",
                            "ten": mcp_instance.ten,
                            "phien_ban": mcp_instance.phien_ban,
                            "so_cong_cu": len(mcp_instance._cong_cu),
                            "giao_van": "http-sse",
                        }
                    )
                    return
                if path == "/sse":
                    accept = self.headers.get("Accept", "").lower()
                    yeu_cau_stream = "text/event-stream" in accept or qs.get("stream", [""])[0] == "1"
                    if not yeu_cau_stream:
                        payload = b"event: endpoint\ndata: /messages\n\n"
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        self.wfile.flush()
                        return

                    session_id = uuid.uuid4().hex
                    ev_queue: queue.Queue[Optional[str]] = queue.Queue()
                    with sse_lock:
                        sse_sessions[session_id] = ev_queue
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    try:
                        ep_msg = f"event: endpoint\ndata: /messages?session_id={session_id}\n\n".encode("utf-8")
                        self.wfile.write(ep_msg)
                        self.wfile.flush()
                        while True:
                            try:
                                item = ev_queue.get(timeout=15.0)
                            except queue.Empty:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                                continue
                            if item is None:
                                break
                            self.wfile.write(f"event: message\ndata: {item}\n\n".encode("utf-8"))
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        with sse_lock:
                            sse_sessions.pop(session_id, None)
                    return
                self.send_error(404, "Not Found")

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)
                if path not in ("/messages", "/rpc"):
                    self.send_error(404, "Not Found")
                    return
                raw_len = self.headers.get("Content-Length", "0")
                try:
                    length = int(raw_len)
                except (TypeError, ValueError):
                    self._gui_json(
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Content-Length không hợp lệ"}},
                        status=400,
                    )
                    return
                if length < 0 or length > MAX_MCP_BODY_BYTES:
                    self._gui_json(
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Kích thước gói tin vượt giới hạn"}},
                        status=413 if length > MAX_MCP_BODY_BYTES else 400,
                    )
                    return
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    req = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    self._gui_json(
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}},
                        status=400,
                    )
                    return
                resp = mcp_instance.xu_ly_jsonrpc(req)
                out_obj = resp if resp else {"jsonrpc": "2.0", "result": {"ok": True}}
                session_id = qs.get("session_id", [""])[0]
                if session_id:
                    with sse_lock:
                        q_ref = sse_sessions.get(session_id)
                    if q_ref is not None and resp:
                        q_ref.put(json.dumps(resp, ensure_ascii=False))
                self._gui_json(out_obj)

            def log_message(self, format: str, *args: Any) -> None:
                return

        server = ThreadingHTTPServer((host, int(port)), _MCPHTTPHandler)
        orig_shutdown = server.shutdown

        def _shutdown_with_sse() -> None:
            with sse_lock:
                for q_ref in list(sse_sessions.values()):
                    q_ref.put(None)
            orig_shutdown()

        server.shutdown = _shutdown_with_sse  # type: ignore[method-assign]
        return server

    def chay_sse(self, host: str = "127.0.0.1", port: int = 8090) -> None:
        """Khởi động máy chủ MCP HTTP/SSE lắng nghe lâu dài."""
        httpd = self.tao_http_server(host=host, port=port)
        actual_port = httpd.server_port
        print(f"Vietnamese AI MCP Server (HTTP/SSE) đang chạy tại http://{host}:{actual_port}")
        print(f"  SSE Endpoint:     http://{host}:{actual_port}/sse")
        print(f"  JSON-RPC Endpoint: http://{host}:{actual_port}/messages")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã dừng Vietnamese AI MCP Server.")
        finally:
            httpd.server_close()

    @staticmethod
    def xuat_cau_hinh_ide(
        ten_ide: str = "cursor",
        duong_dan_vaidb: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sinh khối cấu hình JSON sẵn sàng dán vào Cursor, Claude Desktop hoặc VS Code."""
        args: List[str] = ["-m", "vietnamese_ai.cli.main", "mcp-server", "--transport", "stdio"]
        if duong_dan_vaidb:
            args.extend(["--vaidb", duong_dan_vaidb])

        entry = {
            "command": sys.executable or "python",
            "args": args,
        }
        ide = ten_ide.strip().lower()
        if ide == "vscode":
            return {"servers": {"vietnamese-ai": {"type": "stdio", **entry}}}
        return {"mcpServers": {"vietnamese-ai": entry}}

    def dong(self) -> None:
        """Giải phóng tài nguyên `.vaidb` nếu đã mở."""
        if self._tro_ly_phap_ly is not None and hasattr(self._tro_ly_phap_ly, "rag"):
            self._tro_ly_phap_ly.rag.dong()


ProductionMCPServer = MayChuMCPThucTe

__all__ = [
    "MayChuMCPThucTe",
    "ProductionMCPServer",
]
