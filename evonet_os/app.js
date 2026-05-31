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
