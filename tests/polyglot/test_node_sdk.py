import subprocess
import os

def test_nodejs_sdk():
    """
    Test giả lập chạy SDK Node.js thông qua luồng subprocess của Python.
    Đảm bảo SDK Node.js EvoNetAI khởi tạo và fetch thành công.
    """
    sdk_dir = os.path.join(os.path.dirname(__file__), "../../sdks/nodejs")
    
    # Chạy npm test (vốn dĩ gọi node test.js)
    result = subprocess.run(
        ["node", "test.js"],
        cwd=sdk_dir,
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Node.js test failed: {result.stderr}"
    assert "EvoNetAI Node.js SDK Mock Tests Passed!" in result.stdout
