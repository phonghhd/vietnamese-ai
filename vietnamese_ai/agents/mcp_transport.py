"""MayChuMCPThucTe (ProductionMCPServer) - Máy Chủ MCP Chuẩn `stdio` & `HTTP/SSE` Cắm Thẳng Vào Cursor, VS Code & Claude Desktop."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from vietnamese_ai.agents.mcp import MCPServer
from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


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
        phien_ban: str = "31.0.3",
        duong_dan_vaidb: Optional[Union[str, Path]] = None,
        nap_kho_phap_ly_mo: bool = True,
        bat_tuong_lua: bool = True,
        quan_ly_plugin: Optional[Any] = None,
        tro_ly_phap_ly: Optional[Any] = None,
    ) -> None:
        super().__init__(ten=ten, phien_ban=phien_ban, bat_tuong_lua=bat_tuong_lua)
        self.duong_dan_vaidb = str(duong_dan_vaidb) if duong_dan_vaidb else None
        self.nap_kho_phap_ly_mo = nap_kho_phap_ly_mo
        self.quan_ly_plugin = quan_ly_plugin
        self._tro_ly_phap_ly: Optional[Any] = tro_ly_phap_ly
        self._bo_so_sanh_hd: Optional[Any] = None
        self._dong_co_edge: Optional[Any] = None
        self._dang_ky_cong_cu_mac_dinh()
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

        def _tra_cuu_phap_ly(cau_hoi: str) -> str:
            kq = self._lay_tro_ly_phap_ly().tra_cuu(cau_hoi)
            return json.dumps(kq, ensure_ascii=False)

        def _kiem_tra_hieu_luc(so_hieu_hoac_van_ban: str, ngay_tra_cuu: str = "") -> str:
            ngay = ngay_tra_cuu.strip() or None
            kq = self._lay_tro_ly_phap_ly().kiem_tra_hieu_luc(so_hieu_hoac_van_ban, ngay_tra_cuu=ngay)
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

        def _ra_soat_hop_dong(hop_dong_doi_tac: str, hop_dong_chuan: str = "") -> str:
            chuan = hop_dong_chuan if hop_dong_chuan.strip() else None
            kq = self._lay_bo_so_sanh_hd().so_sanh_va_ra_soat(
                hop_dong_doi_tac=hop_dong_doi_tac,
                hop_dong_chuan=chuan,
            )
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

    def chay_stdio(
        self,
        in_stream: Optional[TextIO] = None,
        out_stream: Optional[TextIO] = None,
    ) -> int:
        """Chạy vòng lặp giao vận MCP chuẩn `stdio` (hỗ trợ cả JSON-RPC từng dòng lẫn khung `Content-Length`)."""
        src = in_stream or sys.stdin
        dst = out_stream or sys.stdout
        da_xu_ly = 0

        while True:
            line = src.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            use_content_length = False
            if stripped.lower().startswith("content-length:"):
                use_content_length = True
                length = int(stripped.split(":", 1)[1].strip())
                while True:
                    hdr = src.readline()
                    if not hdr or hdr.strip() == "":
                        break
                raw_payload = src.read(length)
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
                path = self.path.split("?")[0]
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
                self.send_error(404, "Not Found")

            def do_POST(self) -> None:
                path = self.path.split("?")[0]
                if path not in ("/messages", "/rpc"):
                    self.send_error(404, "Not Found")
                    return
                length = int(self.headers.get("Content-Length", 0))
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
                self._gui_json(resp if resp else {"jsonrpc": "2.0", "result": {"ok": True}})

            def log_message(self, format: str, *args: Any) -> None:
                return

        return ThreadingHTTPServer((host, int(port)), _MCPHTTPHandler)

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
