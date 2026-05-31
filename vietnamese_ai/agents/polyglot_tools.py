"""
Polyglot Tools / Model Context Protocol (MCP)
Cho phép Agent của EvoNet (Python) gọi các hàm được viết bằng Node.js, PHP, hoặc các ngôn ngữ khác thông qua Webhook.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any

from vietnamese_ai.agents.tools import CongCu

logger = logging.getLogger("PolyglotTools")

class CongCuTuXa(CongCu):
    """
    Công cụ từ xa (Remote Tool).
    Khi Agent kích hoạt, thay vì chạy code Python, nó sẽ gửi HTTP POST tới một Webhook.
    Điều này cho phép Tác tử AI (Python) điều khiển các máy chủ Node.js/PHP từ xa.
    """
    def __init__(self, ten: str, mo_ta: str, webhook_url: str, tham_so: Dict[str, str] = None):
        self.webhook_url = webhook_url
        
        # Hàm nội bộ sẽ thực hiện thao tác Fetch (Webhook Call)
        def remote_executor(**kwargs) -> str:
            payload = json.dumps({"tool": ten, "parameters": kwargs}).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url, 
                data=payload, 
                headers={'Content-Type': 'application/json', 'User-Agent': 'EvoNet-MCP/25.0'}
            )
            
            try:
                # Giao tiếp HTTP với độ trễ tối đa 10s
                with urllib.request.urlopen(req, timeout=10.0) as response:
                    result = response.read().decode('utf-8')
                    return result
            except urllib.error.URLError as e:
                logger.error(f"Lỗi gọi công cụ từ xa '{ten}' tại {webhook_url}: {e}")
                return f"[Lỗi MCP] Không thể kết nối tới máy chủ chứa công cụ (Node.js/PHP): {e}"
                
        super().__init__(ten=ten, mo_ta=mo_ta, ham_thuc_thi=remote_executor)
        if tham_so is not None:
            self.tham_so = tham_so
