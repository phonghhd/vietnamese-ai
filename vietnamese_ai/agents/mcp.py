"""Giao thức MCP (Model Context Protocol - JSON-RPC 2.0) cho hệ thống Tác tử (Agents) của Vietnamese AI."""

from typing import Any, Callable, Dict, List, Optional, Union

from vietnamese_ai.agents.tools import CongCu


class MCPServer:
    """
    Máy chủ chuẩn MCP (Model Context Protocol - JSON-RPC 2.0) cho phép xuất bản
    các `CongCu` (Tools), Tài nguyên (Resources) và Prompt của `vietnamese-ai`
    để mọi MCP Client hoặc IDE/Agent bên ngoài có thể khám phá và gọi trực tiếp.
    """

    def __init__(
        self,
        ten: str = "vietnamese-ai-mcp",
        phien_ban: str = "1.0.0",
        bat_tuong_lua: bool = True,
    ):
        from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

        self.ten = ten
        self.phien_ban = phien_ban
        self.bat_tuong_lua = bat_tuong_lua
        self.tuong_lua = (
            TuongLuaBaoVeFramework(gioi_han_req_moi_phut=10000) if bat_tuong_lua else None
        )
        self._cong_cu: Dict[str, CongCu] = {}
        self._tai_nguyen: Dict[str, Dict[str, Any]] = {}

    def dang_ky_cong_cu(self, cong_cu_obj: Union[CongCu, Callable]) -> None:
        """Đăng ký một `CongCu` hoặc hàm Python vào MCP Server."""
        if not isinstance(cong_cu_obj, CongCu):
            ten = getattr(cong_cu_obj, "__name__", "tool")
            mo_ta = getattr(cong_cu_obj, "__doc__", "") or f"Công cụ {ten}"
            cong_cu_obj = CongCu(ten=ten, mo_ta=mo_ta.strip(), ham_thuc_thi=cong_cu_obj)
        self._cong_cu[cong_cu_obj.ten] = cong_cu_obj

    def dang_ky_tai_nguyen(
        self, uri: str, ten: str, noi_dung: str, mime_type: str = "text/plain"
    ) -> None:
        """Đăng ký một tài nguyên (Resource) tĩnh hoặc động vào MCP Server."""
        self._tai_nguyen[uri] = {
            "uri": uri,
            "name": ten,
            "mimeType": mime_type,
            "text": noi_dung,
        }

    def danh_sach_tools_mcp(self) -> List[Dict[str, Any]]:
        """Trả về danh sách công cụ theo chuẩn JSON Schema của giao thức MCP (`tools/list`)."""
        ket_qua = []
        for ten, cc in self._cong_cu.items():
            info = cc.to_dict() if hasattr(cc, "to_dict") else {"tham_so": getattr(cc, "tham_so", {})}
            props = {}
            required = []
            for p_name, p_meta in info.get("tham_so", {}).items():
                kieu_py = str(p_meta.get("kieu", "str")).lower()
                json_type = "string"
                if "int" in kieu_py:
                    json_type = "integer"
                elif "float" in kieu_py:
                    json_type = "number"
                elif "bool" in kieu_py:
                    json_type = "boolean"
                props[p_name] = {"type": json_type}
                if p_meta.get("bat_buoc", False):
                    required.append(p_name)

            ket_qua.append(
                {
                    "name": ten,
                    "description": cc.mo_ta,
                    "inputSchema": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                }
            )
        return ket_qua

    def xu_ly_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý thông điệp JSON-RPC 2.0 chuẩn MCP (`initialize`, `tools/list`, `tools/call`, `resources/*`)."""
        req_id = request.get("id", 1)
        method = request.get("method", "")
        params = request.get("params", {}) or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.ten, "version": self.phien_ban},
                    "capabilities": {"tools": {}, "resources": {}},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.danh_sach_tools_mcp()},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {}) or {}
            if tool_name not in self._cong_cu:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Không tìm thấy công cụ MCP: {tool_name}"},
                }
            if self.tuong_lua is not None and isinstance(tool_args, dict):
                hop_le, ly_do = self.tuong_lua.kiem_tra_lenh_cong_cu(tool_name, tool_args)
                if not hop_le:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": f"[Bị chặn bởi Tường lửa MCP]: {ly_do}"}],
                            "isError": True,
                        },
                    }
            try:
                cc_obj = self._cong_cu[tool_name]
                out = cc_obj.chay(**tool_args) if hasattr(cc_obj, "chay") else cc_obj._ham_thuc_thi(**tool_args)
                van_ban_ra = str(out)
                if self.tuong_lua is not None:
                    van_ban_ra = self.tuong_lua.loc_dau_ra(van_ban_ra)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": van_ban_ra}],
                        "isError": False,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Lỗi: {e}"}],
                        "isError": True,
                    },
                }

        if method == "resources/list":
            res_list = [
                {"uri": r["uri"], "name": r["name"], "mimeType": r["mimeType"]}
                for r in self._tai_nguyen.values()
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": res_list}}

        if method == "resources/read":
            uri = params.get("uri", "")
            if uri not in self._tai_nguyen:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Không tìm thấy tài nguyên: {uri}"},
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"contents": [self._tai_nguyen[uri]]},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Phương thức MCP không được hỗ trợ: {method}"},
        }


class MCPClient:
    """
    Trình khách MCP (Model Context Protocol Client) kết nối với `MCPServer` hoặc endpoint JSON-RPC
    và chuyển đổi các công cụ MCP thành `CongCu` để `TacTu` sử dụng trực tiếp.
    """

    def __init__(self, server: MCPServer):
        self.server = server
        self._req_counter = 0

    def _gui_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._req_counter += 1
        resp = self.server.xu_ly_jsonrpc(
            {"jsonrpc": "2.0", "id": self._req_counter, "method": method, "params": params or {}}
        )
        if "error" in resp:
            raise RuntimeError(resp["error"].get("message", "Lỗi MCP"))
        return resp.get("result", {})

    def khoi_tao(self) -> Dict[str, Any]:
        """Gửi bắt tay `initialize` tới MCP Server."""
        return self._gui_rpc("initialize")

    def danh_sach_cong_cu(self) -> List[Dict[str, Any]]:
        """Lấy danh sách công cụ từ MCP Server (`tools/list`)."""
        return self._gui_rpc("tools/list").get("tools", [])

    def goi_cong_cu(self, ten: str, tham_so: Optional[Dict[str, Any]] = None) -> str:
        """Gọi thực thi một công cụ trên MCP Server (`tools/call`)."""
        res = self._gui_rpc("tools/call", {"name": ten, "arguments": tham_so or {}})
        contents = res.get("content", [])
        return "\n".join(c.get("text", "") for c in contents if c.get("type") == "text")

    def doc_tai_nguyen(self, uri: str) -> str:
        """Đọc nội dung tài nguyên từ MCP Server (`resources/read`)."""
        res = self._gui_rpc("resources/read", {"uri": uri})
        contents = res.get("contents", [])
        return contents[0].get("text", "") if contents else ""

    def chuyen_thanh_cong_cu(self) -> List[CongCu]:
        """Chuyển đổi toàn bộ công cụ trên MCP Server thành danh sách `CongCu` của `vietnamese-ai`."""
        mcp_tools = self.danh_sach_cong_cu()
        ket_qua: List[CongCu] = []
        for t in mcp_tools:
            t_name = t["name"]
            t_desc = t.get("description", f"MCP Tool {t_name}")

            def _make_runner(tool_name: str) -> Callable:
                def _runner(**kwargs: Any) -> str:
                    return self.goi_cong_cu(tool_name, kwargs)

                return _runner

            ket_qua.append(CongCu(ten=t_name, mo_ta=t_desc, ham_thuc_thi=_make_runner(t_name)))
        return ket_qua


class MCPBridge:
    """Cầu nối hai chiều giữa `TacTu` (Agent) và hệ sinh thái `MCP`."""

    @staticmethod
    def gan_mcp_vao_tac_tu(tac_tu: Any, nguon_mcp: Union[MCPServer, MCPClient]) -> int:
        """Gắn toàn bộ công cụ từ `MCPServer` hoặc `MCPClient` vào một `TacTu`."""
        client = nguon_mcp if isinstance(nguon_mcp, MCPClient) else MCPClient(nguon_mcp)
        cac_cong_cu = client.chuyen_thanh_cong_cu()
        for cc in cac_cong_cu:
            tac_tu.cong_cu[cc.ten] = cc
        return len(cac_cong_cu)

    @staticmethod
    def tao_mcp_server_tu_tac_tu(tac_tu: Any, ten_server: str = "tac-tu-mcp") -> MCPServer:
        """Xuất bản toàn bộ công cụ hiện có của một `TacTu` thành một `MCPServer` chuẩn."""
        server = MCPServer(ten=ten_server)
        for cc in getattr(tac_tu, "cong_cu", {}).values():
            server.dang_ky_cong_cu(cc)
        return server
