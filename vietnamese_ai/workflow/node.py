"""
NutCongViec (Workflow Node) - Đại diện cho một khối chức năng độc lập
trong một quy trình làm việc đồ thị (DAG).
"""

from typing import Callable, List, Dict, Any


class NutCongViec:
    """
    Nút công việc trong DAG Engine.
    Mỗi nút nhận các biến đầu vào, chạy hàm xử lý và trả về các biến đầu ra.
    """

    def __init__(
        self, 
        id_nut: str, 
        ham_xu_ly: Callable, 
        dau_vao: List[str] = None, 
        dau_ra: List[str] = None
    ):
        """
        Khởi tạo Nút.
        
        Args:
            id_nut: Tên định danh duy nhất của nút (vd: 'doc_file', 'dich_thuat').
            ham_xu_ly: Hàm Python sẽ được thực thi. Hàm này phải nhận các kwargs khớp với `dau_vao`
                       và trả về một dict chứa các key khớp với `dau_ra`.
            dau_vao: Danh sách tên các biến mà nút này cần để chạy.
            dau_ra: Danh sách tên các biến mà nút này sẽ tạo ra.
        """
        self.id_nut = id_nut
        self.ham_xu_ly = ham_xu_ly
        self.dau_vao = dau_vao or []
        self.dau_ra = dau_ra or []

    def chay(self, du_lieu_dau_vao: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi nút với dữ liệu được cung cấp.
        
        Args:
            du_lieu_dau_vao: Dictionary chứa tất cả dữ liệu từ các nút trước đó.
            
        Returns:
            Dictionary chứa kết quả đầu ra của nút này.
        """
        # Trích xuất đúng các biến mà hàm xử lý cần
        kwargs = {}
        for key in self.dau_vao:
            if key not in du_lieu_dau_vao:
                raise ValueError(f"Nút '{self.id_nut}' thiếu dữ liệu đầu vào bắt buộc: '{key}'")
            kwargs[key] = du_lieu_dau_vao[key]
            
        # Chạy hàm
        ket_qua_tra_ve = self.ham_xu_ly(**kwargs)
        
        # Hàm xử lý có thể trả về một giá trị duy nhất (nếu chỉ có 1 đầu ra)
        # Hoặc trả về một dict (nếu có nhiều đầu ra)
        ket_qua_cuoi = {}
        if len(self.dau_ra) == 1:
            # Nếu hàm trả về thẳng 1 giá trị không phải dict
            if not isinstance(ket_qua_tra_ve, dict) or self.dau_ra[0] not in ket_qua_tra_ve:
                ket_qua_cuoi[self.dau_ra[0]] = ket_qua_tra_ve
            else:
                ket_qua_cuoi[self.dau_ra[0]] = ket_qua_tra_ve[self.dau_ra[0]]
        elif len(self.dau_ra) > 1:
            if not isinstance(ket_qua_tra_ve, dict):
                raise TypeError(f"Nút '{self.id_nut}' phải trả về dict khi có nhiều đầu ra.")
            for key in self.dau_ra:
                if key not in ket_qua_tra_ve:
                    raise ValueError(f"Nút '{self.id_nut}' không sinh ra biến đầu ra: '{key}'")
                ket_qua_cuoi[key] = ket_qua_tra_ve[key]
                
        return ket_qua_cuoi

    def __repr__(self):
        return f"NutCongViec(id='{self.id_nut}', in={self.dau_vao}, out={self.dau_ra})"
