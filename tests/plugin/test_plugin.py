import os
import sys
import tempfile
import pytest
from vietnamese_ai.plugin import QuanLyPlugin
from vietnamese_ai.sandbox import LoiAnNinh

def tao_file_plugin(duong_dan: str, ma_nguon: str):
    with open(duong_dan, "w", encoding="utf-8") as f:
        f.write(ma_nguon)

def test_nap_va_go_plugin_hop_le():
    quan_ly = QuanLyPlugin()
    
    ma_nguon_hop_le = """
from vietnamese_ai.plugin import PluginCoSo

class PluginChaoHoi(PluginCoSo):
    ten = "Plugin Chào Hỏi"
    phien_ban = "1.0.0"
    
    def khoi_dong(self):
        self.trang_thai_hoat_dong = True
        return True
        
    def thuc_thi(self, ten="Người Lạ"):
        return {"loi_chao": f"Xin chào {ten}!"}
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "plugin_chao_hoi.py")
        tao_file_plugin(file_path, ma_nguon_hop_le)
        
        # 1. Nạp Plugin
        plugin = quan_ly.nap_plugin(file_path)
        assert plugin is not None
        assert plugin.ten == "Plugin Chào Hỏi"
        assert plugin.trang_thai_hoat_dong is True
        
        # 2. Thực thi logic
        kq = plugin.thuc_thi(ten="AI")
        assert kq["loi_chao"] == "Xin chào AI!"
        
        # 3. Gỡ Plugin (Hot Unload)
        ten_module = "plugin_chao_hoi"
        quan_ly.go_plugin(ten_module)
        assert ten_module not in quan_ly.cac_plugin_dang_chay
        assert ten_module not in sys.modules

def test_chan_plugin_doc_hai():
    quan_ly = QuanLyPlugin()
    
    ma_nguon_doc = """
import os # Dòng này sẽ bị AST bắt

from vietnamese_ai.plugin import PluginCoSo

class PluginHacker(PluginCoSo):
    ten = "Hack"
    def thuc_thi(self):
        os.system('echo hack')
        return {}
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "plugin_doc_hai.py")
        tao_file_plugin(file_path, ma_nguon_doc)
        
        # Nếu không có quyền admin (bo_qua_kiem_duyet=False), sẽ dính LoiAnNinh
        with pytest.raises(LoiAnNinh):
            quan_ly.nap_plugin(file_path)
            
        # Kiểm tra tính năng bỏ qua kiểm duyệt
        plugin_cho_phep = quan_ly.nap_plugin(file_path, bo_qua_kiem_duyet=True)
        assert plugin_cho_phep is not None
        assert plugin_cho_phep.ten == "Hack"
