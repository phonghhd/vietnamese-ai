import json
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, List


class KieuDang:
    LIGHT = "light"
    DARK = "dark"


class Component:
    """Lớp cơ sở cho mọi thành phần giao diện V-UI."""

    def __init__(self):
        self.id = "comp_" + uuid.uuid4().hex[:8]

    def render_html(self) -> str:
        return ""

    def render_js(self) -> str:
        return ""

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        """Đăng ký các hàm callback Python vào thư mục API của Server."""
        pass


class UIApp:
    """
    Micro-Framework cốt lõi. Gộp các Components lại và tự động chạy Server.
    """

    def __init__(self, tieu_de: str = "V-Neural Studio", theme: str = KieuDang.DARK):
        self.tieu_de = tieu_de
        self.theme = theme
        self.components: List[Component] = []
        self.api_handlers: Dict[str, Callable] = {}

    def them_cot(self, *components: Component):
        """Thêm các component vào một hàng ngang (cột)."""
        self.components.extend(components)
        for comp in components:
            comp.dang_ky_api(self.api_handlers)

    def _sinh_html(self) -> str:
        bg_color = (
            "bg-gray-900 text-white" if self.theme == KieuDang.DARK else "bg-gray-50 text-gray-900"
        )

        html_blocks = [c.render_html() for c in self.components]
        js_blocks = [c.render_js() for c in self.components]

        html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>{self.tieu_de}</title>
    <link rel="manifest" href="/manifest.json">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        /* Tùy chỉnh thanh cuộn */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: #4B5563; border-radius: 4px; }}
    </style>
</head>
<body class="{bg_color} min-h-screen p-8 font-sans">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            {self.tieu_de}
        </h1>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            {"".join(html_blocks)}
        </div>
    </div>

    <script>
        // JS Core để gọi API Python
        async function goi_python(endpoint, data) {{
            const res = await fetch('/api/' + endpoint, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(data)
            }});
            return await res.json();
        }}

        {"".join(js_blocks)}

        // PWA Service Worker Registration
        if ('serviceWorker' in navigator) {{
            navigator.serviceWorker.register('/sw.js').then(reg => {{
                console.log('PWA Service Worker registered!', reg.scope);
            }}).catch(err => console.error('PWA Error:', err));
        }}
    </script>
</body>
</html>
"""
        return html

    def chay(self, port: int = 8080):
        """Chạy server Zero-Dependency (chỉ dùng chuẩn Python)."""
        app = self
        html_content = self._sinh_html().encode("utf-8")

        class VUIHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(html_content)
                elif self.path == "/manifest.json":
                    manifest = {
                        "name": app.tieu_de,
                        "short_name": "V-UI",
                        "start_url": "/",
                        "display": "standalone",
                        "background_color": "#111827",
                        "theme_color": "#111827",
                    }
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(manifest).encode("utf-8"))
                elif self.path == "/sw.js":
                    sw_code = """
                    const CACHE_NAME = 'v-ui-cache-v1';
                    self.addEventListener('install', event => {
                        event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(['/'])));
                    });
                    self.addEventListener('fetch', event => {
                        event.respondWith(caches.match(event.request).then(res => res || fetch(event.request)));
                    });
                    """
                    self.send_response(200)
                    self.send_header("Content-type", "application/javascript")
                    self.end_headers()
                    self.wfile.write(sw_code.encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path.startswith("/api/"):
                    endpoint = self.path[5:]
                    if endpoint in app.api_handlers:
                        content_len = int(self.headers.get("Content-Length", 0))
                        body = self.rfile.read(content_len)
                        try:
                            data = json.loads(body)
                            # Gọi hàm Python đã đăng ký
                            ket_qua = app.api_handlers[endpoint](data)

                            self.send_response(200)
                            self.send_header("Content-type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps(ket_qua).encode("utf-8"))
                        except Exception as e:
                            self.send_response(500)
                            self.end_headers()
                            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                    else:
                        self.send_response(404)
                        self.end_headers()

            def log_message(self, format, *args):
                pass  # Tắt log rác

        server = HTTPServer(("0.0.0.0", port), VUIHandler)
        print(f"🚀 V-UI đang chạy tại: http://localhost:{port}")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã tắt V-UI.")
            server.server_close()
