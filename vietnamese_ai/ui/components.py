from typing import Any, Callable, Dict

from vietnamese_ai.ui.core import Component


class GiaoDienChat(Component):
    """
    Khung Chat tương tác với LLM hoặc Agent.
    Tự động cuộn, hiển thị trạng thái đang nghĩ (Typing...).
    """
    def __init__(self, xu_ly_tin_nhan: Callable[[str], str], chieu_cao: str = "500px"):
        super().__init__()
        self.xu_ly_tin_nhan = xu_ly_tin_nhan
        self.chieu_cao = chieu_cao
        self.api_endpoint = f"chat_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="flex flex-col bg-gray-800 rounded-xl border border-gray-700 shadow-lg overflow-hidden">
            <div class="bg-gray-900 p-4 border-b border-gray-700 font-semibold text-blue-400">
                Agentic Chat
            </div>
            <div id="{self.id}_messages" class="flex-1 p-4 overflow-y-auto" style="height: {self.chieu_cao};">
                <!-- Tin nhắn sẽ hiện ở đây -->
                <div class="mb-4 text-center text-gray-500 text-sm">Bắt đầu cuộc trò chuyện...</div>
            </div>
            <div class="p-4 bg-gray-900 border-t border-gray-700">
                <form id="{self.id}_form" class="flex gap-2">
                    <input type="text" id="{self.id}_input"
                           class="flex-1 bg-gray-800 text-white rounded-lg px-4 py-2 border border-gray-700 focus:outline-none focus:border-blue-500 transition-colors"
                           placeholder="Nhập tin nhắn..." autocomplete="off">
                    <button type="submit"
                            class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors">
                        Gửi
                    </button>
                </form>
            </div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        document.getElementById('{self.id}_form').addEventListener('submit', async (e) => {{
            e.preventDefault();
            const input_el = document.getElementById('{self.id}_input');
            const messages_el = document.getElementById('{self.id}_messages');
            const text = input_el.value.trim();
            if(!text) return;

            if(messages_el.innerHTML.includes("Bắt đầu cuộc trò chuyện")) {{
                messages_el.innerHTML = "";
            }}

            messages_el.innerHTML += `
                <div class="mb-4 flex justify-end">
                    <div class="bg-blue-600 text-white px-4 py-2 rounded-2xl rounded-tr-sm max-w-[80%] shadow">
                        ${{text}}
                    </div>
                </div>`;
            input_el.value = '';
            messages_el.scrollTop = messages_el.scrollHeight;

            const loading_id = 'load_' + Date.now();
            messages_el.innerHTML += `
                <div id="${{loading_id}}" class="mb-4 flex justify-start">
                    <div class="bg-gray-700 text-gray-300 px-4 py-2 rounded-2xl rounded-tl-sm shadow flex gap-1">
                        <span class="animate-bounce">.</span>
                        <span class="animate-bounce" style="animation-delay: 0.1s">.</span>
                        <span class="animate-bounce" style="animation-delay: 0.2s">.</span>
                    </div>
                </div>`;
            messages_el.scrollTop = messages_el.scrollHeight;

            try {{
                const res = await goi_python('{self.api_endpoint}', {{ "text": text }});
                document.getElementById(loading_id).remove();

                // Parse Markdown and Highlight code
                let html_content = marked.parse(res.reply);

                messages_el.innerHTML += `
                    <div class="mb-4 flex justify-start">
                        <div class="bg-gray-700 text-white px-4 py-2 rounded-2xl rounded-tl-sm max-w-[90%] shadow border border-gray-600 prose prose-invert">
                            ${{html_content}}
                        </div>
                    </div>`;

                // Apply Highlight.js to new blocks
                messages_el.querySelectorAll('pre code').forEach((block) => {{
                    hljs.highlightElement(block);
                }});

                messages_el.scrollTop = messages_el.scrollHeight;
            }} catch(e) {{
                document.getElementById(loading_id).remove();
                messages_el.innerHTML += `
                    <div class="mb-4 text-center text-red-400 text-sm">Lỗi kết nối tới Agent.</div>`;
            }}
        }});
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            # Gọi hàm python của người dùng truyền vào
            ket_qua = self.xu_ly_tin_nhan(data.get("text", ""))
            return {"reply": ket_qua}
        thu_muc_api[self.api_endpoint] = handler

class BangThongKe(Component):
    """
    Hiển thị thông số (metrics) thời gian thực. (VD: Giá trị Token, Tốc độ Node).
    """
    def __init__(self, tieu_de: str, ham_lay_du_lieu: Callable[[], Dict[str, Any]], cap_nhat_sau: int = 2):
        super().__init__()
        self.tieu_de = tieu_de
        self.ham_lay_du_lieu = ham_lay_du_lieu
        self.cap_nhat_sau = cap_nhat_sau
        self.api_endpoint = f"stat_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6 flex flex-col">
            <h3 class="text-xl font-bold mb-4 text-purple-400 flex items-center justify-between">
                {self.tieu_de}
                <span id="{self.id}_dot" class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            </h3>
            <div id="{self.id}_content" class="grid grid-cols-2 gap-4 flex-1">
                <div class="text-gray-400 text-center py-8 col-span-2">Đang tải dữ liệu...</div>
            </div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        async function fetchStats_{self.id}() {{
            try {{
                const res = await goi_python('{self.api_endpoint}', {{}});
                const content = document.getElementById('{self.id}_content');
                let html = '';
                for (const [key, val] of Object.entries(res.data)) {{
                    html += `
                        <div class="bg-gray-900 rounded-lg p-4 border border-gray-700">
                            <div class="text-sm text-gray-400 mb-1">${{key}}</div>
                            <div class="text-2xl font-bold text-white">${{val}}</div>
                        </div>
                    `;
                }}
                content.innerHTML = html;
                document.getElementById('{self.id}_dot').classList.remove("bg-red-500");
                document.getElementById('{self.id}_dot').classList.add("bg-green-500");
            }} catch(e) {{
                document.getElementById('{self.id}_dot').classList.remove("bg-green-500");
                document.getElementById('{self.id}_dot').classList.add("bg-red-500");
            }}
        }}

        // Chạy lần đầu tiên
        fetchStats_{self.id}();
        // Cập nhật định kỳ
        setInterval(fetchStats_{self.id}, {self.cap_nhat_sau * 1000});
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            return {"data": self.ham_lay_du_lieu()}
        thu_muc_api[self.api_endpoint] = handler

class KhuVucTaiTaiLieu(Component):
    """Khu vực kéo thả file (Upload). Hỗ trợ PDF, CSV."""
    def __init__(self, ham_xu_ly_file: Callable[[str, str], str]):
        super().__init__()
        self.ham_xu_ly_file = ham_xu_ly_file
        self.api_endpoint = f"upload_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-pink-400">Tải Tài Liệu Lên RAG</h3>
            <div id="{self.id}_dropzone" class="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-pink-500 transition-colors cursor-pointer bg-gray-900">
                <p class="text-gray-400">Kéo thả file vào đây hoặc click để chọn file</p>
                <p class="text-xs text-gray-500 mt-2">Hỗ trợ .txt, .csv, .pdf (dưới dạng base64)</p>
                <input type="file" id="{self.id}_file" class="hidden">
            </div>
            <div id="{self.id}_status" class="mt-4 text-sm text-gray-400"></div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        const dropzone_{self.id} = document.getElementById('{self.id}_dropzone');
        const file_input_{self.id} = document.getElementById('{self.id}_file');
        const status_{self.id} = document.getElementById('{self.id}_status');

        dropzone_{self.id}.addEventListener('click', () => file_input_{self.id}.click());

        dropzone_{self.id}.addEventListener('dragover', (e) => {{
            e.preventDefault();
            dropzone_{self.id}.classList.add('border-pink-500');
        }});

        dropzone_{self.id}.addEventListener('dragleave', (e) => {{
            e.preventDefault();
            dropzone_{self.id}.classList.remove('border-pink-500');
        }});

        async function handleFile_{self.id}(file) {{
            if (!file) return;
            status_{self.id}.innerHTML = `<span class="text-yellow-400">Đang xử lý ${{file.name}}...</span>`;

            const reader = new FileReader();
            reader.onload = async (e) => {{
                const base64_content = e.target.result;
                try {{
                    const res = await goi_python('{self.api_endpoint}', {{
                        name: file.name,
                        content: base64_content
                    }});
                    status_{self.id}.innerHTML = `<span class="text-green-400">Thành công: ${{res.msg}}</span>`;
                }} catch(err) {{
                    status_{self.id}.innerHTML = `<span class="text-red-400">Lỗi upload.</span>`;
                }}
            }};
            reader.readAsDataURL(file);
        }}

        dropzone_{self.id}.addEventListener('drop', (e) => {{
            e.preventDefault();
            dropzone_{self.id}.classList.remove('border-pink-500');
            handleFile_{self.id}(e.dataTransfer.files[0]);
        }});

        file_input_{self.id}.addEventListener('change', (e) => {{
            handleFile_{self.id}(e.target.files[0]);
        }});
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            msg = self.ham_xu_ly_file(data.get("name", ""), data.get("content", ""))
            return {"msg": msg}
        thu_muc_api[self.api_endpoint] = handler

class CameraVaMicro(Component):
    """Truy cập Webcam và Micro trực tiếp từ trình duyệt (Multi-modal)."""
    def __init__(self, ham_xu_ly_anh: Callable[[str], str]):
        super().__init__()
        self.ham_xu_ly_anh = ham_xu_ly_anh
        self.api_endpoint = f"vision_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-indigo-400">Vision AI (Webcam)</h3>
            <div class="bg-black rounded-lg overflow-hidden mb-4 relative aspect-video flex items-center justify-center">
                <video id="{self.id}_video" autoplay playsinline class="w-full h-full object-cover hidden"></video>
                <div id="{self.id}_placeholder" class="text-gray-500">Camera đang tắt</div>
                <canvas id="{self.id}_canvas" class="hidden"></canvas>
            </div>
            <div class="flex gap-2">
                <button id="{self.id}_start" class="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex-1">
                    Bật Camera
                </button>
                <button id="{self.id}_capture" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex-1 hidden">
                    Phân tích Ảnh
                </button>
            </div>
            <div id="{self.id}_result" class="mt-4 text-sm text-gray-300"></div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        const video_{self.id} = document.getElementById('{self.id}_video');
        const canvas_{self.id} = document.getElementById('{self.id}_canvas');
        const startBtn_{self.id} = document.getElementById('{self.id}_start');
        const captureBtn_{self.id} = document.getElementById('{self.id}_capture');
        const placeholder_{self.id} = document.getElementById('{self.id}_placeholder');
        const result_{self.id} = document.getElementById('{self.id}_result');
        let stream_{self.id} = null;

        startBtn_{self.id}.addEventListener('click', async () => {{
            if (!stream_{self.id}) {{
                try {{
                    stream_{self.id} = await navigator.mediaDevices.getUserMedia({{ video: true }});
                    video_{self.id}.srcObject = stream_{self.id};
                    video_{self.id}.classList.remove('hidden');
                    placeholder_{self.id}.classList.add('hidden');
                    captureBtn_{self.id}.classList.remove('hidden');
                    startBtn_{self.id}.innerText = "Tắt Camera";
                }} catch(err) {{
                    alert("Không thể truy cập Camera. Hãy kiểm tra quyền (Permissions).");
                }}
            }} else {{
                stream_{self.id}.getTracks().forEach(track => track.stop());
                stream_{self.id} = null;
                video_{self.id}.classList.add('hidden');
                placeholder_{self.id}.classList.remove('hidden');
                captureBtn_{self.id}.classList.add('hidden');
                startBtn_{self.id}.innerText = "Bật Camera";
            }}
        }});

        captureBtn_{self.id}.addEventListener('click', async () => {{
            if (!stream_{self.id}) return;
            canvas_{self.id}.width = video_{self.id}.videoWidth;
            canvas_{self.id}.height = video_{self.id}.videoHeight;
            canvas_{self.id}.getContext('2d').drawImage(video_{self.id}, 0, 0);

            const base64_image = canvas_{self.id}.toDataURL('image/jpeg');
            result_{self.id}.innerHTML = `<span class="text-yellow-400 animate-pulse">Agent đang phân tích hình ảnh...</span>`;

            try {{
                const res = await goi_python('{self.api_endpoint}', {{ image: base64_image }});
                result_{self.id}.innerHTML = `<div class="bg-gray-700 p-3 rounded-lg border border-gray-600"><b>AI:</b> ${{res.reply}}</div>`;
            }} catch(err) {{
                result_{self.id}.innerHTML = `<span class="text-red-400">Lỗi phân tích.</span>`;
            }}
        }});
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            return {"reply": self.ham_xu_ly_anh(data.get("image", ""))}
        thu_muc_api[self.api_endpoint] = handler

class BieuDoTriThuc(Component):
    """Trực quan hóa GraphRAG bằng Vis-Network."""
    def __init__(self, ham_lay_graph: Callable[[], Dict[str, Any]]):
        super().__init__()
        self.ham_lay_graph = ham_lay_graph
        self.api_endpoint = f"graph_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6 flex flex-col" style="height: 500px;">
            <h3 class="text-xl font-bold mb-4 text-emerald-400 flex justify-between items-center">
                Bàn Cờ Tri Thức (GraphRAG)
                <button id="{self.id}_refresh" class="text-sm bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded">Làm mới</button>
            </h3>
            <div id="{self.id}_network" class="flex-1 bg-gray-900 rounded-lg border border-gray-700"></div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        let network_{self.id} = null;

        async function loadGraph_{self.id}() {{
            try {{
                const res = await goi_python('{self.api_endpoint}', {{}});
                const container = document.getElementById('{self.id}_network');

                const data = {{
                    nodes: new vis.DataSet(res.nodes),
                    edges: new vis.DataSet(res.edges)
                }};

                const options = {{
                    nodes: {{
                        shape: 'dot',
                        size: 16,
                        font: {{ color: '#fff' }},
                        color: {{ background: '#10B981', border: '#059669' }}
                    }},
                    edges: {{
                        font: {{ color: '#9CA3AF', align: 'middle' }},
                        color: '#4B5563',
                        arrows: 'to'
                    }},
                    physics: {{
                        forceAtlas2Based: {{ gravitationalConstant: -50, centralGravity: 0.01, springLength: 100, springConstant: 0.08 }},
                        maxVelocity: 50,
                        solver: 'forceAtlas2Based',
                        timestep: 0.35,
                        stabilization: {{ iterations: 150 }}
                    }}
                }};

                if (network_{self.id}) network_{self.id}.destroy();
                network_{self.id} = new vis.Network(container, data, options);
            }} catch(e) {{
                console.error("Lỗi tải Graph:", e);
            }}
        }}

        document.getElementById('{self.id}_refresh').addEventListener('click', loadGraph_{self.id});
        setTimeout(loadGraph_{self.id}, 500); // Tải sau khi UI render
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            return self.ham_lay_graph()
        thu_muc_api[self.api_endpoint] = handler

class TroLyGiongNoi(Component):
    """Trợ lý Voice AI bằng Web Speech API (TTS & STT)."""
    def __init__(self, ham_xu_ly_giong_noi: Callable[[str], str]):
        super().__init__()
        self.ham_xu_ly_giong_noi = ham_xu_ly_giong_noi
        self.api_endpoint = f"voice_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-orange-400">Trợ Lý Giọng Nói (Native AI)</h3>
            <div class="flex gap-4">
                <button id="{self.id}_mic" class="bg-red-600 hover:bg-red-700 text-white p-4 rounded-full flex-shrink-0 transition-transform hover:scale-105 active:scale-95 shadow-lg shadow-red-900/50">
                    🎤
                </button>
                <div class="flex-1 bg-gray-900 rounded-lg p-4 text-gray-300 font-mono text-sm border border-gray-700 relative overflow-hidden">
                    <div id="{self.id}_text" class="absolute inset-0 p-4 overflow-y-auto">Bấm Micro để nói...</div>
                    <div id="{self.id}_wave" class="absolute bottom-0 left-0 h-1 bg-red-500 w-full transform scale-x-0 transition-transform origin-left"></div>
                </div>
            </div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        const micBtn_{self.id} = document.getElementById('{self.id}_mic');
        const textDisplay_{self.id} = document.getElementById('{self.id}_text');
        const wave_{self.id} = document.getElementById('{self.id}_wave');

        let recognition = null;
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.lang = 'vi-VN';
            recognition.interimResults = true;

            recognition.onstart = () => {{
                wave_{self.id}.classList.remove('scale-x-0');
                wave_{self.id}.classList.add('animate-pulse', 'scale-x-100');
                micBtn_{self.id}.classList.replace('bg-red-600', 'bg-orange-500');
                textDisplay_{self.id}.innerText = "Đang nghe...";
            }};

            recognition.onresult = (e) => {{
                const transcript = Array.from(e.results)
                    .map(r => r[0].transcript)
                    .join('');
                textDisplay_{self.id}.innerText = transcript;

                if(e.results[0].isFinal) {{
                    wave_{self.id}.classList.remove('animate-pulse');
                    textDisplay_{self.id}.innerText += " (Đang xử lý...)";
                    goi_python('{self.api_endpoint}', {{text: transcript}}).then(res => {{
                        textDisplay_{self.id}.innerText = res.reply;
                        // TTS
                        const utterance = new SpeechSynthesisUtterance(res.reply);
                        utterance.lang = 'vi-VN';
                        window.speechSynthesis.speak(utterance);
                    }});
                }}
            }};

            recognition.onend = () => {{
                wave_{self.id}.classList.add('scale-x-0');
                micBtn_{self.id}.classList.replace('bg-orange-500', 'bg-red-600');
            }};

            micBtn_{self.id}.addEventListener('click', () => {{
                try {{ recognition.start(); }} catch(e) {{}}
            }});
        }} else {{
            textDisplay_{self.id}.innerText = "Trình duyệt của bạn không hỗ trợ Web Speech API.";
        }}
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            return {"reply": self.ham_xu_ly_giong_noi(data.get("text", ""))}
        thu_muc_api[self.api_endpoint] = handler

class ChiaSeManHinh(Component):
    """Chia sẻ màn hình cho Agent (Screen Copilot)."""
    def __init__(self, ham_xu_ly_man_hinh: Callable[[str], str]):
        super().__init__()
        self.ham_xu_ly_man_hinh = ham_xu_ly_man_hinh
        self.api_endpoint = f"screen_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-cyan-400">Agent Copilot (Screen Share)</h3>
            <button id="{self.id}_share" class="w-full bg-cyan-600 hover:bg-cyan-700 text-white px-4 py-3 rounded-lg font-bold shadow-lg shadow-cyan-900/50 flex justify-center items-center gap-2 transition-transform hover:scale-105 active:scale-95">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>
                Bắt đầu Copilot
            </button>
            <div id="{self.id}_preview" class="mt-4 hidden relative rounded overflow-hidden border border-gray-600">
                <video id="{self.id}_video" autoplay class="w-full h-auto opacity-50"></video>
                <canvas id="{self.id}_canvas" class="hidden"></canvas>
                <div class="absolute inset-0 flex items-center justify-center">
                    <span id="{self.id}_status" class="bg-black/70 text-cyan-400 px-3 py-1 rounded text-sm animate-pulse font-mono">Đang quan sát...</span>
                </div>
            </div>
            <div id="{self.id}_reply" class="mt-4 text-sm text-gray-300 font-mono bg-gray-900 p-3 rounded hidden border-l-2 border-cyan-500"></div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        const shareBtn_{self.id} = document.getElementById('{self.id}_share');
        const preview_{self.id} = document.getElementById('{self.id}_preview');
        const video_{self.id} = document.getElementById('{self.id}_video');
        const canvas_{self.id} = document.getElementById('{self.id}_canvas');
        const reply_{self.id} = document.getElementById('{self.id}_reply');

        let screenStream_{self.id} = null;
        let captureInterval_{self.id} = null;

        shareBtn_{self.id}.addEventListener('click', async () => {{
            if (!screenStream_{self.id}) {{
                try {{
                    screenStream_{self.id} = await navigator.mediaDevices.getDisplayMedia({{ video: true }});
                    video_{self.id}.srcObject = screenStream_{self.id};
                    preview_{self.id}.classList.remove('hidden');
                    reply_{self.id}.classList.remove('hidden');
                    shareBtn_{self.id}.innerHTML = "Dừng Copilot";
                    shareBtn_{self.id}.classList.replace('bg-cyan-600', 'bg-red-600');

                    // Chụp ảnh mỗi 5 giây
                    captureInterval_{self.id} = setInterval(async () => {{
                        canvas_{self.id}.width = video_{self.id}.videoWidth;
                        canvas_{self.id}.height = video_{self.id}.videoHeight;
                        canvas_{self.id}.getContext('2d').drawImage(video_{self.id}, 0, 0);
                        const b64 = canvas_{self.id}.toDataURL('image/jpeg', 0.5); // Nén 50%

                        try {{
                            const res = await goi_python('{self.api_endpoint}', {{image: b64}});
                            reply_{self.id}.innerHTML = "<b>Copilot:</b> " + res.reply;
                        }} catch(e) {{}}
                    }}, 5000);

                    // Bắt sự kiện khi user tự tắt share qua browser UI
                    screenStream_{self.id}.getVideoTracks()[0].onended = stopShare_{self.id};
                }} catch(err) {{
                    alert("Không thể chia sẻ màn hình.");
                }}
            }} else {{
                stopShare_{self.id}();
            }}
        }});

        function stopShare_{self.id}() {{
            if(screenStream_{self.id}) {{
                screenStream_{self.id}.getTracks().forEach(t => t.stop());
                screenStream_{self.id} = null;
            }}
            clearInterval(captureInterval_{self.id});
            preview_{self.id}.classList.add('hidden');
            reply_{self.id}.classList.add('hidden');
            shareBtn_{self.id}.innerHTML = "Bắt đầu Copilot";
            shareBtn_{self.id}.classList.replace('bg-red-600', 'bg-cyan-600');
        }}
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            return {"reply": self.ham_xu_ly_man_hinh(data.get("image", ""))}
        thu_muc_api[self.api_endpoint] = handler

class DinhViGPS(Component):
    """Lấy vị trí GPS của thiết bị."""
    def __init__(self, ham_xu_ly_vi_tri: Callable[[float, float], str]):
        super().__init__()
        self.ham_xu_ly_vi_tri = ham_xu_ly_vi_tri
        self.api_endpoint = f"gps_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-lime-400">GPS Context</h3>
            <button id="{self.id}_btn" class="w-full bg-lime-600 hover:bg-lime-700 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                Cập nhật Vị trí cho Agent
            </button>
            <div id="{self.id}_status" class="mt-2 text-sm text-gray-400 text-center font-mono">Chưa rõ vị trí</div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        document.getElementById('{self.id}_btn').addEventListener('click', () => {{
            const st = document.getElementById('{self.id}_status');
            st.innerText = "Đang tìm vệ tinh...";
            if (navigator.geolocation) {{
                navigator.geolocation.getCurrentPosition(
                    async (pos) => {{
                        st.innerText = `Lat: ${{pos.coords.latitude.toFixed(4)}}, Lng: ${{pos.coords.longitude.toFixed(4)}}`;
                        const res = await goi_python('{self.api_endpoint}', {{
                            lat: pos.coords.latitude,
                            lng: pos.coords.longitude
                        }});
                        st.innerText += `\\n👉 ${{res.reply}}`;
                    }},
                    (err) => {{ st.innerText = "Bị từ chối GPS."; }}
                );
            }} else {{
                st.innerText = "Không hỗ trợ GPS.";
            }}
        }});
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            return {"reply": self.ham_xu_ly_vi_tri(data.get("lat", 0.0), data.get("lng", 0.0))}
        thu_muc_api[self.api_endpoint] = handler

class BoNhoTrinhDuyet(Component):
    """Component ẩn dùng IndexedDB làm Vector/Cache Storage."""
    def __init__(self):
        super().__init__()

    def render_html(self) -> str:
        return f"""<div id="{self.id}_db" class="hidden"></div>"""

    def render_js(self) -> str:
        return f"""
        // Khởi tạo IndexedDB "VNeuralDB"
        const dbReq_{self.id} = indexedDB.open('VNeuralDB', 1);
        dbReq_{self.id}.onupgradeneeded = (e) => {{
            const db = e.target.result;
            if (!db.objectStoreNames.contains('cache')) {{
                db.createObjectStore('cache', {{keyPath: 'id'}});
            }}
        }};
        dbReq_{self.id}.onsuccess = (e) => {{
            console.log("VNeural IndexedDB (0MB) Ready!");
            // Sẵn sàng lưu Vector Embedding dưới dạng Float32Array ở Client.
        }};
        """

class TrinhChieu3D(Component):
    """Trình chiếu 3D sử dụng Three.js."""
    def __init__(self):
        super().__init__()

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6 flex flex-col h-[400px]">
            <h3 class="text-xl font-bold mb-4 text-purple-400">Trình chiếu 3D (WebGL)</h3>
            <div id="{self.id}_canvas" class="flex-1 w-full bg-black rounded-lg overflow-hidden relative cursor-move"></div>
            <p class="text-xs text-gray-500 mt-2 text-center">Xoay bằng chuột. GPU Rendering.</p>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        // Basic Three.js setup
        const container_{self.id} = document.getElementById('{self.id}_canvas');
        const scene_{self.id} = new THREE.Scene();
        scene_{self.id}.background = new THREE.Color(0x111827); // tailwind gray-900

        const camera_{self.id} = new THREE.PerspectiveCamera(75, container_{self.id}.clientWidth / container_{self.id}.clientHeight, 0.1, 1000);
        camera_{self.id}.position.z = 5;

        const renderer_{self.id} = new THREE.WebGLRenderer({{ antialias: true }});
        renderer_{self.id}.setSize(container_{self.id}.clientWidth, container_{self.id}.clientHeight);
        container_{self.id}.appendChild(renderer_{self.id}.domElement);

        // Tạo một khối xoay (Demo hình cầu wireframe cho AI)
        const geometry_{self.id} = new THREE.IcosahedronGeometry(2, 1);
        const material_{self.id} = new THREE.MeshBasicMaterial({{ color: 0xA78BFA, wireframe: true }}); // purple-400
        const sphere_{self.id} = new THREE.Mesh(geometry_{self.id}, material_{self.id});
        scene_{self.id}.add(sphere_{self.id});

        function animate_{self.id}() {{
            requestAnimationFrame(animate_{self.id});
            sphere_{self.id}.rotation.x += 0.005;
            sphere_{self.id}.rotation.y += 0.01;
            renderer_{self.id}.render(scene_{self.id}, camera_{self.id});
        }}
        animate_{self.id}();

        // Xử lý resize
        window.addEventListener('resize', () => {{
            camera_{self.id}.aspect = container_{self.id}.clientWidth / container_{self.id}.clientHeight;
            camera_{self.id}.updateProjectionMatrix();
            renderer_{self.id}.setSize(container_{self.id}.clientWidth, container_{self.id}.clientHeight);
        }});

        // Xử lý xoay chuột đơn giản
        let isDragging_{self.id} = false;
        let previousMousePosition_{self.id} = {{ x: 0, y: 0 }};

        container_{self.id}.addEventListener('mousedown', () => isDragging_{self.id} = true);
        window.addEventListener('mouseup', () => isDragging_{self.id} = false);
        container_{self.id}.addEventListener('mousemove', (e) => {{
            if(isDragging_{self.id}) {{
                const deltaMove = {{
                    x: e.offsetX - previousMousePosition_{self.id}.x,
                    y: e.offsetY - previousMousePosition_{self.id}.y
                }};
                sphere_{self.id}.rotation.y += deltaMove.x * 0.01;
                sphere_{self.id}.rotation.x += deltaMove.y * 0.01;
            }}
            previousMousePosition_{self.id} = {{ x: e.offsetX, y: e.offsetY }};
        }});
        """

class TinhToanGPU(Component):
    """Sử dụng sức mạnh CPU/GPU Trình duyệt để tính toán Tensor."""
    def __init__(self, ham_xu_ly_ket_qua: Callable[[float], str]):
        super().__init__()
        self.ham_xu_ly_ket_qua = ham_xu_ly_ket_qua
        self.api_endpoint = f"gpu_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-fuchsia-400">WebGPU/JS Tensor Compute</h3>
            <button id="{self.id}_btn" class="w-full bg-fuchsia-600 hover:bg-fuchsia-700 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                Thử Offload Tính toán 10 triệu Vector
            </button>
            <div id="{self.id}_status" class="mt-2 text-sm text-gray-400 text-center font-mono">Chưa chạy</div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        document.getElementById('{self.id}_btn').addEventListener('click', async () => {{
            const st = document.getElementById('{self.id}_status');
            st.innerHTML = "Đang chạy vòng lặp 10 triệu phép tính Float trên Client...";

            // Chạy tính toán nặng để block Main Thread hoặc dùng Web Worker
            setTimeout(async () => {{
                let sum = 0;
                for(let i=0; i<10000000; i++) {{
                    sum += Math.sin(i) * Math.cos(i);
                }}

                const res = await goi_python('{self.api_endpoint}', {{ result: sum }});
                st.innerHTML = `<span class="text-green-400">Tính xong! Server nói: ${{res.reply}}</span>`;
            }}, 100);
        }});
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            return {"reply": self.ham_xu_ly_ket_qua(data.get("result", 0.0))}
        thu_muc_api[self.api_endpoint] = handler

class DieuKhienIoT(Component):
    """Giao tiếp với phần cứng Bluetooth BLE."""
    def __init__(self):
        super().__init__()

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-blue-400">IoT Bluetooth (WebBLE)</h3>
            <button id="{self.id}_btn" class="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                Quét Thiết Bị IoT Xung Quanh
            </button>
            <div id="{self.id}_status" class="mt-2 text-sm text-gray-400 text-center font-mono">Yêu cầu HTTPS/Localhost trên Chrome</div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        document.getElementById('{self.id}_btn').addEventListener('click', async () => {{
            const st = document.getElementById('{self.id}_status');
            if (navigator.bluetooth) {{
                try {{
                    const device = await navigator.bluetooth.requestDevice({{
                        acceptAllDevices: true
                    }});
                    st.innerHTML = `<span class="text-blue-400">Đã kết nối: ${{device.name || 'Thiết bị lạ'}}</span>`;
                }} catch(err) {{
                    st.innerHTML = `<span class="text-red-400">Hủy quét hoặc lỗi.</span>`;
                }}
            }} else {{
                st.innerText = "Trình duyệt không hỗ trợ Web Bluetooth.";
            }}
        }});
        """

class TrangThaiPin(Component):
    """Lấy trạng thái năng lượng để tối ưu AI."""
    def __init__(self, ham_canh_bao_pin: Callable[[float], str]):
        super().__init__()
        self.ham_canh_bao_pin = ham_canh_bao_pin
        self.api_endpoint = f"battery_{self.id}"

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-yellow-400">Năng lượng Hệ thống</h3>
            <div class="flex items-center gap-4">
                <div class="flex-1 bg-gray-700 h-6 rounded-full overflow-hidden relative">
                    <div id="{self.id}_bar" class="h-full bg-green-500 w-0 transition-all duration-1000"></div>
                </div>
                <div id="{self.id}_text" class="font-mono text-yellow-400 font-bold">--%</div>
            </div>
            <div id="{self.id}_reply" class="mt-4 text-sm text-gray-300"></div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        async function updateBattery_{self.id}(battery) {{
            const bar = document.getElementById('{self.id}_bar');
            const text = document.getElementById('{self.id}_text');
            const pct = Math.round(battery.level * 100);

            bar.style.width = pct + '%';
            text.innerText = pct + '%';

            if (pct < 20) bar.classList.replace('bg-green-500', 'bg-red-500');
            else bar.classList.replace('bg-red-500', 'bg-green-500');

            const res = await goi_python('{self.api_endpoint}', {{ level: pct }});
            document.getElementById('{self.id}_reply').innerText = res.reply;
        }}

        if ('getBattery' in navigator) {{
            navigator.getBattery().then(battery => {{
                updateBattery_{self.id}(battery);
                battery.addEventListener('levelchange', () => updateBattery_{self.id}(battery));
            }});
        }} else {{
            document.getElementById('{self.id}_text').innerText = "N/A";
        }}
        """

    def dang_ky_api(self, thu_muc_api: Dict[str, Callable]) -> None:
        def handler(data: dict):
            return {"reply": self.ham_canh_bao_pin(data.get("level", 100))}
        thu_muc_api[self.api_endpoint] = handler

class KinhThucTeAo(Component):
    """Trải nghiệm WebXR (Kính VR)."""
    def __init__(self):
        super().__init__()

    def render_html(self) -> str:
        return f"""
        <div class="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-6">
            <h3 class="text-xl font-bold mb-4 text-rose-400">WebXR (Kính VR)</h3>
            <button id="{self.id}_btn" class="w-full bg-rose-600 hover:bg-rose-700 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                Kích Hoạt Môi Trường VR
            </button>
            <div id="{self.id}_status" class="mt-2 text-sm text-gray-400 text-center font-mono">Đang kiểm tra thiết bị...</div>
        </div>
        """

    def render_js(self) -> str:
        return f"""
        if ('xr' in navigator) {{
            navigator.xr.isSessionSupported('immersive-vr').then((supported) => {{
                if (supported) {{
                    document.getElementById('{self.id}_status').innerText = "VR Sẵn sàng!";
                }} else {{
                    document.getElementById('{self.id}_status').innerText = "Thiết bị không hỗ trợ VR.";
                    document.getElementById('{self.id}_btn').disabled = true;
                    document.getElementById('{self.id}_btn').classList.add('opacity-50');
                }}
            }});
        }} else {{
            document.getElementById('{self.id}_status').innerText = "Trình duyệt không hỗ trợ WebXR.";
            document.getElementById('{self.id}_btn').disabled = true;
            document.getElementById('{self.id}_btn').classList.add('opacity-50');
        }}
        """
