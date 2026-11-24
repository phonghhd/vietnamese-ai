"""Tập Dữ Liệu Tiếng Việt Đa Miền Phục Vụ Benchmark Tái Lập (`vietnamese_ai.benchmark.datasets_vn`).

Mở rộng phạm vi đánh giá ngoài văn bản pháp luật sang 5 lĩnh vực thực tế tại Việt Nam:
1. Pháp lý & Hành chính
2. Tài chính & Ngân hàng
3. Y tế & Sức khỏe
4. Khoa học Công nghệ & AI
5. Thương mại Điện tử & Dịch vụ Khách hàng
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

TAP_DU_LIEU_DA_MIEN_VN: Dict[str, List[Dict[str, str]]] = {
    "phap_ly_hanh_chinh": [
        {
            "id": "pl_01",
            "tieu_de": "Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân",
            "noi_dung": (
                "Điều 1. Phạm vi điều chỉnh. Nghị định này quy định về bảo vệ dữ liệu cá nhân "
                "và trách nhiệm của cơ quan, tổ chức, cá nhân trong hoạt động xử lý dữ liệu tại Việt Nam."
            ),
        },
        {
            "id": "pl_02",
            "tieu_de": "Công văn hướng dẫn chữ ký số và thủ tục hành chính",
            "noi_dung": (
                "Ủy ban nhân dân thành phố Hà Nội triển khai tiếp nhận hồ sơ trực tuyến mức độ 4 "
                "áp dụng chữ ký số chuyên dùng công vụ cho toàn bộ quy trình hành chính."
            ),
        },
    ],
    "tai_chinh_ngan_hang": [
        {
            "id": "tc_01",
            "tieu_de": "Quy trình thẩm định tín dụng doanh nghiệp nhỏ và vừa",
            "noi_dung": (
                "Ngân hàng kiểm tra báo cáo tài chính, mã số thuế, dòng tiền kinh doanh và lịch sử "
                "trả nợ trên hệ thống thông tin tín dụng quốc gia trước khi phê duyệt hạn mức vay vốn."
            ),
        },
        {
            "id": "tc_02",
            "tieu_de": "Cảnh báo giao dịch bất thường và phòng chống rửa tiền",
            "noi_dung": (
                "Các giao dịch chuyển khoản giá trị từ 500.000.000 đồng trở lên trong ngày được hệ thống "
                "giám sát tự động đánh dấu để xác thực danh tính khách hàng theo quy định của Ngân hàng Nhà nước."
            ),
        },
    ],
    "y_te_suc_khoe": [
        {
            "id": "yt_01",
            "tieu_de": "Hướng dẫn quản lý hồ sơ bệnh án điện tử",
            "noi_dung": (
                "Bệnh viện lưu trữ kết quả xét nghiệm máu, chẩn đoán hình ảnh và đơn thuốc trong hồ sơ "
                "bệnh án điện tử được mã hóa nhằm đảm bảo quyền riêng tư của người bệnh."
            ),
        },
        {
            "id": "yt_02",
            "tieu_de": "Khuyến cáo phòng chống sốt xuất huyết mùa mưa",
            "noi_dung": (
                "Bộ Y tế khuyến cáo người dân diệt lăng quăng, ngủ màn và đến cơ sở y tế ngay khi có "
                "triệu chứng sốt cao liên tục trên 39 độ kéo dài hơn hai ngày."
            ),
        },
    ],
    "cong_nghe_ai": [
        {
            "id": "cn_01",
            "tieu_de": "Suy luận mô hình ngôn ngữ lượng tử hóa trên CPU biên",
            "noi_dung": (
                "Kỹ thuật lượng tử hóa Q4_K và ánh xạ bộ nhớ mmap cho phép chạy mô hình ngôn ngữ lớn "
                "trực tiếp trên máy tính văn phòng không cần GPU rời với mức tiêu thụ RAM thấp."
            ),
        },
        {
            "id": "cn_02",
            "tieu_de": "Kiến trúc truy hồi kết hợp cho xử lý ngôn ngữ tự nhiên tiếng Việt",
            "noi_dung": (
                "Hệ thống kết hợp tách từ ghép tiếng Việt với tìm kiếm ngữ nghĩa vector và điểm BM25 "
                "giúp nâng cao độ chính xác trích dẫn tài liệu chuyên ngành."
            ),
        },
    ],
    "thuong_mai_dien_tu": [
        {
            "id": "tm_01",
            "tieu_de": "Chính sách đổi trả và bảo hành thiết bị điện tử",
            "noi_dung": (
                "Khách hàng được đổi mới sản phẩm trong vòng 15 ngày nếu phát sinh lỗi phần cứng từ "
                "nhà sản xuất và hưởng chế độ bảo hành chính hãng 12 tháng trên toàn quốc."
            ),
        },
        {
            "id": "tm_02",
            "tieu_de": "Tối ưu giao hàng nội thành trong ngày",
            "noi_dung": (
                "Đơn hàng đặt trước 14 giờ tại thành phố Hồ Chí Minh và Đà Nẵng được giao hỏa tốc "
                "trong vòng 4 giờ kèm thông báo trạng thái vận đơn theo thời gian thực."
            ),
        },
    ],
}

BO_TRUY_HOI_DA_MIEN_VN: List[Dict[str, Any]] = [
    {
        "mien": "phap_ly_hanh_chinh",
        "cau_hoi": "Nghị định 13/2023/NĐ-CP quy định về vấn đề gì?",
        "id_ky_vong": "pl_01",
    },
    {
        "mien": "tai_chinh_ngan_hang",
        "cau_hoi": "Giao dịch chuyển khoản giá trị bao nhiêu bị hệ thống giám sát tự động đánh dấu?",
        "id_ky_vong": "tc_02",
    },
    {
        "mien": "y_te_suc_khoe",
        "cau_hoi": "Triệu chứng sốt xuất huyết nào cần đến cơ sở y tế ngay?",
        "id_ky_vong": "yt_02",
    },
    {
        "mien": "cong_nghe_ai",
        "cau_hoi": "Kỹ thuật nào giúp chạy mô hình ngôn ngữ trên CPU văn phòng tiết kiệm RAM?",
        "id_ky_vong": "cn_01",
    },
    {
        "mien": "thuong_mai_dien_tu",
        "cau_hoi": "Thời hạn đổi mới sản phẩm và bảo hành chính hãng là bao lâu?",
        "id_ky_vong": "tm_01",
    },
]

BO_KIEM_THU_NER_VN: List[Dict[str, Any]] = [
    {
        "van_ban": (
            "Giám đốc Nguyễn Văn Minh tại Công ty Cổ phần Công nghệ Việt (MST: 0101248141) "
            "làm việc tại Hà Nội."
        ),
        "loai_ky_vong": {"CHUC_DANH", "NGUOI", "TO_CHUC", "MA_SO_THUE", "DIA_DANH"},
    },
    {
        "van_ban": (
            "Liên hệ email hotro@doanhnghiep.vn hoặc số 0912345678 trước ngày 15/06/2025 "
            "theo Nghị định số 13/2023/NĐ-CP."
        ),
        "loai_ky_vong": {"EMAIL", "SO_DIEN_THOAI", "NGAY_THANG", "VAN_BAN_PHAP_LUAT"},
    },
    {
        "van_ban": (
            "Khách hàng Trần Thị Lan có CCCD số 001098012345 thanh toán 5.000.000 đồng tại Đà Nẵng."
        ),
        "loai_ky_vong": {"NGUOI", "CCCD", "TIEN_TE", "DIA_DANH"},
    },
]

BO_KIEM_THU_CAM_XUC_VN: List[Dict[str, str]] = [
    {"van_ban": "Sản phẩm rất tốt, giao hàng nhanh và chất lượng tuyệt vời", "nhan": "positive"},
    {"van_ban": "Dịch vụ hỗ trợ nhiệt tình, phần mềm hoạt động ổn định và rất hay", "nhan": "positive"},
    {"van_ban": "Chất lượng quá tệ, hàng bị lỗi và dịch vụ chăm sóc khách hàng rất kém", "nhan": "negative"},
    {"van_ban": "Ứng dụng hay bị chậm, trải nghiệm xấu và gây thất vọng", "nhan": "negative"},
    {"van_ban": "Hôm nay trời có mây, nhiệt độ phòng duy trì ở mức 26 độ", "nhan": "neutral"},
]

def _doc_jsonl(duong_dan: Path) -> List[Dict[str, Any]]:
    """Đọc tệp JSONL thành danh sách từ điển chuẩn UTF-8."""
    if not duong_dan.exists():
        return []
    ket_qua: List[Dict[str, Any]] = []
    for dong in duong_dan.read_text(encoding="utf-8").splitlines():
        sach = dong.strip()
        if sach:
            ket_qua.append(dict(json.loads(sach)))
    return ket_qua


def _duong_dan_bo_du_lieu_ngoai(ten_tep: str) -> Path:
    """Tìm tệp dữ liệu pháp lý mở trong `benchmarks/datasets/` hoặc bản đóng gói sẵn trong wheel `vietnamese_ai/rag/legal/data/`."""
    goc = Path(__file__).resolve().parents[2]
    ngoai = goc / "benchmarks" / "datasets" / ten_tep
    if ngoai.is_file():
        return ngoai
    return Path(__file__).resolve().parents[1] / "rag" / "legal" / "data" / ten_tep


def tai_bo_phap_ly_mo(duong_dan: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Tải bộ dữ liệu đánh giá pháp luật Việt Nam công khai dạng JSONL ngoài mã nguồn (100 điều khoản)."""
    path = Path(duong_dan) if duong_dan is not None else _duong_dan_bo_du_lieu_ngoai("legal_vn_corpus.jsonl")
    du_lieu = _doc_jsonl(path)
    if du_lieu:
        return du_lieu
    return list(TAP_DU_LIEU_DA_MIEN_VN["phap_ly_hanh_chinh"])


def tai_bo_hoi_dap_phap_ly_mo(duong_dan: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Tải bộ câu hỏi - căn cứ chuẩn đánh giá Legal RAG ngoài mã nguồn (35 cặp hỏi-đáp)."""
    path = Path(duong_dan) if duong_dan is not None else _duong_dan_bo_du_lieu_ngoai("legal_vn_qa.jsonl")
    du_lieu = _doc_jsonl(path)
    if du_lieu:
        return du_lieu
    return [q for q in BO_TRUY_HOI_DA_MIEN_VN if q.get("mien") == "phap_ly_hanh_chinh"]


__all__ = [
    "TAP_DU_LIEU_DA_MIEN_VN",
    "BO_TRUY_HOI_DA_MIEN_VN",
    "BO_KIEM_THU_NER_VN",
    "BO_KIEM_THU_CAM_XUC_VN",
    "tai_bo_phap_ly_mo",
    "tai_bo_hoi_dap_phap_ly_mo",
]
