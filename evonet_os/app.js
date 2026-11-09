// --- Đồng Hồ & Trạng Thái ---
function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    document.getElementById('clock').textContent = timeString;
}
setInterval(updateClock, 1000);
updateClock();

// --- Quản Lý Cửa Sổ (Window Management) ---
function toggleWindow(id) {
    const win = document.getElementById(id);
    if (win) {
        win.classList.toggle('hidden');
        if (!win.classList.contains('hidden')) {
            bringToFront(win);
        }
    }
}

function bringToFront(win) {
    document.querySelectorAll('.os-window').forEach(w => {
        w.style.zIndex = w.style.zIndex > 10 ? parseInt(w.style.zIndex) - 1 : 10;
    });
    win.style.zIndex = 100;
}

// --- Logic Di Chuyển Cửa Sổ (Draggable) ---
let isDragging = false;
let currentWindow = null;
let offsetX = 0, offsetY = 0;

document.addEventListener('mousedown', (e) => {
    // Chỉ kích hoạt khi kéo từ thanh tiêu đề (window-header)
    const header = e.target.closest('.window-header');
    if (header) {
        const win = header.closest('.draggable');
        if (win) {
            isDragging = true;
            currentWindow = win;
            
            bringToFront(currentWindow);
            
            const rect = currentWindow.getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;
        }
    }
});

document.addEventListener('mousemove', (e) => {
    if (isDragging && currentWindow) {
        currentWindow.style.left = `${e.clientX - offsetX}px`;
        currentWindow.style.top = `${e.clientY - offsetY}px`;
    }
});

document.addEventListener('mouseup', () => {
    isDragging = false;
    currentWindow = null;
});

// --- API Kéo Thả (Drag & Drop từ Dock lên Canvas) ---
function drag(ev, type) {
    ev.dataTransfer.setData("nodeType", type);
    ev.dataTransfer.effectAllowed = "copy";
}

function allowDrop(ev) {
    ev.preventDefault();
    ev.dataTransfer.dropEffect = "copy";
}

function drop(ev) {
    ev.preventDefault();
    const nodeType = ev.dataTransfer.getData("nodeType");
    if (!nodeType) return;

    // Tính toán toạ độ tương đối trên Canvas
    const canvasContainer = document.getElementById('canvas-container');
    const canvasRect = canvasContainer.getBoundingClientRect();
    const x = ev.clientX - canvasRect.left;
    const y = ev.clientY - canvasRect.top;

    createNewNode(nodeType, x, y, canvasContainer);
}

// --- Khởi tạo Node Động ---
function createNewNode(type, x, y, container) {
    const node = document.createElement('div');
    node.className = 'os-window draggable';
    
    // Căn giữa node tại vị trí chuột thả
    node.style.left = `${x - 120}px`; 
    node.style.top = `${y - 20}px`;
    node.style.zIndex = 50;
    
    let title = 'Tác Tử AI';
    let icon = 'fa-robot';
    let content = '<p>Engine: <strong>Default</strong></p>';
    
    if (type === 'rag') {
        title = 'Kho Dữ Liệu RAG';
        icon = 'fa-database';
        content = '<p>Vector DB: <strong>Qdrant</strong></p><button class="os-btn primary">Nhúng Dữ Liệu</button>';
    } else if (type === 'security') {
        title = 'Blue Team Sandbox';
        icon = 'fa-shield-alt';
        content = '<p>Mức an toàn: <strong>Cấp độ 5</strong></p>';
    } else if (type === 'agent') {
        title = 'Swarm Node';
        icon = 'fa-user-astronaut';
        content = '<p>Trạng thái: <strong>Sẵn sàng</strong></p>';
    }
    
    node.innerHTML = `
        <div class="window-header">
            <span class="window-title"><i class="fas ${icon}"></i> ${title}</span>
            <div class="window-controls">
                <div class="control-btn close" onclick="this.closest('.os-window').remove()"></div>
            </div>
        </div>
        <div class="window-content">
            <div class="node-port input-port" title="Nhận dữ liệu"></div>
            ${content}
            <div class="node-port output-port" title="Xuất dữ liệu"></div>
        </div>
    `;
    
    container.appendChild(node);
    bringToFront(node);
    
    // Animation sinh động (Pop-up effect)
    node.animate([
        { transform: 'scale(0.8)', opacity: 0 },
        { transform: 'scale(1.05)', opacity: 1 },
        { transform: 'scale(1)', opacity: 1 }
    ], { duration: 300, easing: 'cubic-bezier(0.16, 1, 0.3, 1)' });
}

// --- SVG Interactive Wiring & DAG ---
let drawingWire = false;
let startPort = null;
let currentPath = null;
const svgCanvas = document.getElementById('connection-lines');
let connections = []; // Mảng chứa sơ đồ DAG

document.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('node-port')) {
        e.stopPropagation();
        drawingWire = true;
        startPort = e.target;
        startPort.style.background = '#10b981';
        
        currentPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        currentPath.classList.add('svg-wire', 'drawing');
        svgCanvas.appendChild(currentPath);
    }
});

document.addEventListener('mousemove', (e) => {
    if (drawingWire && startPort && currentPath) {
        const startRect = startPort.getBoundingClientRect();
        const canvasRect = svgCanvas.getBoundingClientRect();
        
        const startX = startRect.left + startRect.width/2 - canvasRect.left;
        const startY = startRect.top + startRect.height/2 - canvasRect.top;
        const endX = e.clientX - canvasRect.left;
        const endY = e.clientY - canvasRect.top;
        
        // Vẽ đường cong Bezier
        const cpX = (startX + endX) / 2;
        currentPath.setAttribute('d', \`M \${startX} \${startY} C \${cpX} \${startY}, \${cpX} \${endY}, \${endX} \${endY}\`);
    }
});

document.addEventListener('mouseup', (e) => {
    if (drawingWire) {
        drawingWire = false;
        const endPort = e.target.closest('.node-port');
        
        if (endPort && endPort !== startPort && 
            (startPort.classList.contains('output-port') && endPort.classList.contains('input-port') ||
             startPort.classList.contains('input-port') && endPort.classList.contains('output-port'))) {
            
            // Kết nối thành công
            currentPath.classList.remove('drawing');
            endPort.style.background = '#10b981';
            
            // Lưu vào cấu trúc DAG (Mock)
            connections.push({
                from: startPort.closest('.os-window').querySelector('.window-title').innerText,
                to: endPort.closest('.os-window').querySelector('.window-title').innerText
            });
            console.log("Đã kết nối:", connections);
        } else {
            // Hủy dây nối nếu thả chuột ra ngoài hoặc kết nối sai cổng
            currentPath.remove();
            startPort.style.background = '';
        }
        currentPath = null;
        startPort = null;
    }
});

// Hàm cập nhật lại các dây đã nối khi cửa sổ bị di chuyển
function updateWires() {
    // Trong thực tế, cần lặp qua mảng connections và tính toán lại thuộc tính 'd' của các thẻ <path>.
    // Ở đây ta có thể dùng requestAnimationFrame nếu muốn render real-time toàn bộ dây.
}

// --- Visual Compiler (Sinh mã JSON) ---
function exportToEvoNet() {
    const nodes = document.querySelectorAll('.os-window:not(.hidden)');
    const workflow = {
        nodes: nodes.length,
        connections: connections,
        engine: "v25.0",
        timestamp: new Date().toISOString()
    };
    return JSON.stringify(workflow, null, 2);
}

// Bắt sự kiện nút "Khởi động Swarm" (đang hardcode trên Agent Điều phối v21)
document.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON' && e.target.innerText.includes('Khởi động')) {
        toggleWindow('terminal-window');
        mockWebSocketStream();
    }
});

// --- Terminal Live Streaming (WebSocket Mock) ---
function appendLog(message, type = 'info') {
    const term = document.getElementById('terminal-output');
    if (!term) return;
    
    const div = document.createElement('div');
    div.className = \`log-line \${type}\`;
    div.innerText = message;
    term.appendChild(div);
    term.scrollTop = term.scrollHeight; // Auto-scroll
}

let isStreaming = false;
function mockWebSocketStream() {
    if (isStreaming) return;
    isStreaming = true;
    
    appendLog("> Kết nối WebSocket tới wss://evonet.ai/stream...", "info");
    
    setTimeout(() => appendLog("✅ [WSS] Đã kết nối với Lõi Python (v25.0)", "success"), 500);
    
    const jsonStr = exportToEvoNet();
    setTimeout(() => appendLog(\`> Gửi Graph JSON:\\n\${jsonStr}\`, "info"), 1000);
    
    setTimeout(() => appendLog("⚙️ [Server] Đang biên dịch Đồ thị DAG...", "system"), 1500);
    setTimeout(() => appendLog("🤖 [Agent] Tác tử Điều Phối đã nhận lệnh. Đang spawn Agent con...", "info"), 2500);
    setTimeout(() => appendLog("🔌 [MCP] Gọi Webhook từ xa để trích xuất dữ liệu Node.js.", "system"), 3500);
    setTimeout(() => appendLog("✅ [Hoàn thành] Swarm Workflow chạy thành công!", "success"), 5000);
    
    setTimeout(() => { isStreaming = false; }, 5100);
}

// =========================================================================
// --- Trợ Lý Thoại Thời Gian Thực (Web Audio API & WebRTC Voice Bridge) ---
// =========================================================================

let voiceSocket = null;
let audioContext = null;
let mediaStream = null;
let audioInputNode = null;
let audioProcessorNode = null;
let analyserNode = null;
let isRecording = false;
let animationFrameId = null;

// Khởi tạo Canvas vẽ dạng sóng âm thanh
function initWaveformVisualizer() {
    const canvas = document.getElementById('voice-waveform');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    function drawIdle() {
        ctx.fillStyle = '#0b1120';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.beginPath();
        const midY = canvas.height / 2;
        ctx.moveTo(0, midY);
        for (let x = 0; x < canvas.width; x += 10) {
            ctx.lineTo(x, midY);
        }
        ctx.stroke();
    }
    drawIdle();
}

// Bắt đầu hoặc dừng ghi âm microphone
async function toggleVoiceRecording() {
    const btn = document.getElementById('btn-toggle-mic');
    const statusText = document.getElementById('voice-status');
    const pulseRing = document.getElementById('voice-ring');

    if (!isRecording) {
        try {
            // 1. Kết nối WebSocket tới MayChuWebRTCVoice
            connectVoiceWebSocket();

            // 2. Mở Microphone qua Web Audio API
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            audioInputNode = audioContext.createMediaStreamSource(mediaStream);

            // Analyser cho Visualizer
            analyserNode = audioContext.createAnalyser();
            analyserNode.fftSize = 256;
            audioInputNode.connect(analyserNode);

            // ScriptProcessorNode thu âm và chuyển đổi sang PCM 16-bit
            audioProcessorNode = audioContext.createScriptProcessor(2048, 1, 1);
            audioInputNode.connect(audioProcessorNode);
            audioProcessorNode.connect(audioContext.destination);

            audioProcessorNode.onaudioprocess = (e) => {
                if (!isRecording || !voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) return;
                const inputFloat32 = e.inputBuffer.getChannelData(0);
                
                // Chuyển đổi Float32 [-1.0, 1.0] sang PCM 16-bit Little Endian
                const pcm16 = new Int16Array(inputFloat32.length);
                for (let i = 0; i < inputFloat32.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputFloat32[i]));
                    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                voiceSocket.send(pcm16.buffer);
            };

            isRecording = true;
            btn.innerHTML = '<i class="fas fa-stop"></i> Dừng Microphone';
            btn.classList.add('danger');
            statusText.innerText = 'Trạng thái: Đang lắng nghe giọng nói...';
            pulseRing.className = 'voice-pulse-ring recording';

            // Kích hoạt vẽ trực quan sóng âm
            renderLiveWaveform();
            appendLog("🎙️ [Voice AI] Microphone kích hoạt. Đang truyền phát âm thanh thời gian thực...", "info");

        } catch (err) {
            console.error("Lỗi truy cập microphone:", err);
            statusText.innerText = 'Lỗi: ' + err.message;
            appendLog("❌ [Voice AI] Lỗi truy cập Microphone: " + err.message, "error");
        }
    } else {
        stopVoiceRecording();
    }
}

function stopVoiceRecording() {
    isRecording = false;
    const btn = document.getElementById('btn-toggle-mic');
    const statusText = document.getElementById('voice-status');
    const pulseRing = document.getElementById('voice-ring');

    if (btn) {
        btn.innerHTML = '<i class="fas fa-microphone"></i> Bật Microphone';
        btn.classList.remove('danger');
    }
    if (statusText) statusText.innerText = 'Trạng thái: Đã dừng ghi âm';
    if (pulseRing) pulseRing.className = 'voice-pulse-ring';

    if (audioProcessorNode) {
        audioProcessorNode.disconnect();
        audioProcessorNode = null;
    }
    if (audioInputNode) {
        audioInputNode.disconnect();
        audioInputNode = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    initWaveformVisualizer();
    appendLog("⏹️ [Voice AI] Đã dừng thu âm microphone.", "info");
}

// Kết nối WebSocket tới máy chủ Python MayChuWebRTCVoice
function connectVoiceWebSocket() {
    if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) return;

    voiceSocket = new WebSocket('ws://127.0.0.1:8765');
    voiceSocket.binaryType = 'arraybuffer';

    voiceSocket.onopen = () => {
        appendLog("🔗 [Voice AI] WebSocket Voice Bridge kết nối thành công tới ws://127.0.0.1:8765", "success");
    };

    voiceSocket.onmessage = async (event) => {
        if (typeof event.data === 'string') {
            const data = JSON.parse(event.data);
            handleVoiceServerMessage(data);
        } else if (event.data instanceof ArrayBuffer) {
            // Nhận chunk PCM 16-bit tổng hợp từ TTS và phát ra loa
            playPcmAudioChunk(event.data);
        }
    };

    voiceSocket.onerror = (err) => {
        appendLog("⚠️ [Voice AI] WebSocket cảnh báo / mô phỏng cục bộ", "system");
    };

    voiceSocket.onclose = () => {
        appendLog("🔌 [Voice AI] WebSocket Voice Bridge đã đóng.", "info");
    };
}

// Xử lý thông điệp JSON từ Server
function handleVoiceServerMessage(data) {
    const userTranscriptElem = document.getElementById('user-transcript');
    const aiResponseElem = document.getElementById('ai-response');
    const statusText = document.getElementById('voice-status');
    const pulseRing = document.getElementById('voice-ring');

    switch (data.loai) {
        case 'vad_bat_dau':
            if (statusText) statusText.innerText = 'Trạng thái: Đang nói... (VAD phát hiện âm)';
            break;
        case 'nhan_dien_giong_noi':
            if (userTranscriptElem) userTranscriptElem.innerText = data.van_ban || '...';
            if (statusText) statusText.innerText = `Trạng thái: Đã nhận diện (Độ tin cậy: ${Math.round((data.do_tin_cay || 0.9) * 100)}%)`;
            break;
        case 'phan_hoi_van_ban':
            if (aiResponseElem) aiResponseElem.innerText = data.cau_tra_loi || '';
            break;
        case 'bat_dau_phat_am_thanh':
            if (statusText) statusText.innerText = 'Trạng thái: AI đang trả lời bằng giọng nói 6 thanh điệu...';
            if (pulseRing) pulseRing.className = 'voice-pulse-ring speaking';
            break;
        case 'ket_thuc_phat_am_thanh':
            setTimeout(() => {
                if (statusText && isRecording) statusText.innerText = 'Trạng thái: Đang lắng nghe...';
                if (pulseRing && isRecording) pulseRing.className = 'voice-pulse-ring recording';
            }, 1000);
            break;
    }
}

// Phát mẩu âm thanh PCM 16-bit nhận từ TTS
function playPcmAudioChunk(arrayBuffer) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    }

    const int16Array = new Int16Array(arrayBuffer);
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
    }

    const audioBuffer = audioContext.createBuffer(1, float32Array.length, 16000);
    audioBuffer.copyToChannel(float32Array, 0);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}

// Vẽ dao động sóng âm thời gian thực trên Canvas
function renderLiveWaveform() {
    const canvas = document.getElementById('voice-waveform');
    if (!canvas || !analyserNode) return;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        if (!isRecording) return;
        animationFrameId = requestAnimationFrame(draw);

        analyserNode.getByteTimeDomainData(dataArray);

        ctx.fillStyle = '#0b1120';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2;
        ctx.strokeStyle = '#38bdf8';
        ctx.beginPath();

        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }
    draw();
}

// Thử giọng nói TTS trực tiếp
function testAIVoice() {
    const sampleText = "Xin chào, tôi là trợ lý AI tiếng Việt của bạn!";
    appendLog(`🔊 [Voice AI] Kiểm thử phát giọng nói: "${sampleText}"`, "info");
    
    const aiResponseElem = document.getElementById('ai-response');
    if (aiResponseElem) aiResponseElem.innerText = sampleText;

    if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
        voiceSocket.send(JSON.stringify({
            loai: "cau_hoi_text",
            noi_dung: sampleText
        }));
    } else {
        // Fallback Web Speech API nếu chưa bật máy chủ Python
        if ('speechSynthesis' in window) {
            const utter = new SpeechSynthesisUtterance(sampleText);
            utter.lang = 'vi-VN';
            window.speechSynthesis.speak(utter);
        }
    }
}

// Khởi chạy visualizer trạng thái ban đầu khi tải trang
window.addEventListener('DOMContentLoaded', () => {
    initWaveformVisualizer();
});

// ==========================================
// Cụm Swarm CPU & Tailscale Hub Handlers
// ==========================================

function copySwarmJoinCommand() {
    const pin = document.getElementById('swarm-pin-display')?.innerText.replace(/\s+/g, '') || "829410";
    const ip = document.getElementById('ts-ip-badge')?.innerText || "100.85.12.1:9999";
    const cmd = `vietnamese-ai swarm join ${ip} --pin ${pin}`;
    
    if (navigator.clipboard) {
        navigator.clipboard.writeText(cmd);
        appendLog(`📋 [Swarm Hub] Đã sao chép lệnh gia nhập: "${cmd}"`, "success");
        alert(`Đã sao chép lệnh gia nhập cụm:\n\n${cmd}\n\nHãy dán lệnh này vào Terminal của máy con để tự động ghép cụm.`);
    } else {
        prompt("Sao chép lệnh gia nhập cụm:", cmd);
    }
}

function simulateNewDeviceRequest() {
    const modal = document.getElementById('swarm-approval-modal');
    if (modal) {
        modal.classList.remove('hidden');
        appendLog("🔔 [Swarm Zero-Trust] Phát hiện thiết bị mới 'Mini-PC-Office' yêu cầu ghép cụm...", "warning");
    }
}

function resolveDeviceApproval(allowed) {
    const modal = document.getElementById('swarm-approval-modal');
    if (modal) modal.classList.add('hidden');

    if (allowed) {
        appendLog("✅ [Swarm Zero-Trust] Đã phê duyệt 'Mini-PC-Office' gia nhập cụm. Cấp Session Token thành công!", "success");
        const list = document.getElementById('swarm-node-list');
        const count = document.getElementById('swarm-node-count');
        if (list) {
            const newCard = document.createElement('div');
            newCard.className = 'swarm-node-card worker';
            newCard.innerHTML = `
                <div class="node-icon"><i class="fas fa-microchip"></i></div>
                <div class="node-info">
                    <div class="node-name">💻 Mini-PC-Office (Worker / Đã duyệt)</div>
                    <div class="node-meta">IP: 100.85.12.8 • Shard 2 (Lớp 24 - 31) • 4 Core CPU • Ping: 15ms</div>
                </div>
                <div class="node-status active">Đã xác thực</div>
            `;
            list.appendChild(newCard);
        }
        if (count) count.innerText = "3 Máy";
    } else {
        appendLog("⛔ [Swarm Zero-Trust] Đã từ chối thiết bị 'Mini-PC-Office'. Đóng socket và chặn IP.", "error");
    }
}

let funnelEnabled = false;
function toggleTailscaleFunnel() {
    funnelEnabled = !funnelEnabled;
    const badge = document.getElementById('ts-ip-badge');
    const label = document.getElementById('ts-status-label');
    
    if (funnelEnabled) {
        if (badge) badge.innerText = "https://evonet-ai.ts.net";
        if (label) label.innerText = "Tailscale Funnel: Public HTTPS";
        appendLog("🌐 [Tailscale Funnel] Đã kích hoạt đường hầm Public Web: https://evonet-ai.ts.net", "success");
        alert("🎉 Đã bật Tailscale Funnel!\n\nWebsite EvoNet OS và Voice AI hiện đã mở công khai tại:\nhttps://evonet-ai.ts.net\n\nBạn có thể mở link này trên điện thoại để trải nghiệm trực tiếp.");
    } else {
        if (badge) badge.innerText = "100.85.12.1:9999";
        if (label) label.innerText = "Tailscale: Đang hoạt động";
        appendLog("🔒 [Tailscale Funnel] Đã đóng đường hầm Public Web về mạng nội bộ Tailnet.", "info");
    }
}
