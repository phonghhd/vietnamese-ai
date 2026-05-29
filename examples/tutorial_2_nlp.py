"""
Tutorial 2: Xử lý văn bản tiếng Việt
======================================

Hướng dẫn: tách từ, TF-IDF, sentiment, NER, tóm tắt, dịch, chính tả.
"""

# === 1. Xử lý văn bản cơ bản ===

from vietnamese_ai import XuLyVanBan

xl = XuLyVanBan()

van_ban = "Trí tuệ nhân tạo (AI) đang thay đổi cách con người làm việc và học tập."

# Tách từ
tu = xl.tach_tu(van_ban)
print(f"Tách từ: {tu}")

# Chuẩn hóa
chuan = xl.chuan_hoa("  Hello   World  ")
print(f"Chuẩn hóa: '{chuan}'")

# Loại bỏ từ dừng
van_ban_khong_dung = xl.loai_bo_tu_dung("tôi là một sinh viên")
print(f"Không từ dừng: {van_ban_khong_dung}")

# Trích xuất từ khóa
tu_khoa = xl.trich_xuat_tu_khoa(van_ban, top_n=3)
print(f"Từ khóa: {tu_khoa}")

# === 2. TF-IDF ===

van_ban_list = [
    "Trí tuệ nhân tạo đang phát triển mạnh",
    "Học máy là một phần của trí tuệ nhân tạo",
    "Deep learning rất hiệu quả cho nhận dạng hình ảnh",
]
tfidf = xl.ma_hoa_tfidf(van_ban_list)
print(f"\nTF-IDF shape: {tfidf.shape}")

# Tạo từ điển
tu_dien = xl.tao_tu_dien(van_ban_list)
print(f"Từ điển: {len(tu_dien)} từ")

# === 3. Nhận diện thực thể (NER) ===

from vietnamese_ai import NhanDienThucThe

ner = NhanDienThucThe(su_dung_underthesea=False)

van_ban = "Nguyễn Văn A sống tại Hà Nội từ 01/01/2024. Liên hệ: test@gmail.com hoặc 0912345678"
entities = ner.nhan_dien(van_ban)
print("\nThực thể tìm thấy:")
for e in entities:
    print(f"  {e['loai']}: {e['van_ban']}")

# === 4. Phân tích cảm xúc ===

from vietnamese_ai import PhanTichCamXuc

ptcx = PhanTichCamXuc(che_do="tu_dien")
for van_ban in [
    "Sản phẩm rất tốt, tôi rất hài lòng",
    "Dịch vụ quá tệ, thất vọng nặng nề",
    "Bình thường, không có gì đặc biệt",
]:
    ket_qua = ptcx.phan_tich(van_ban)
    print(f"\n'{van_ban[:40]}...' → {ket_qua.get('nhan', 'N/A')}")

# === 5. Tóm tắt văn bản ===

from vietnamese_ai import TomTatVanBan

tt = TomTatVanBan(che_do="extractive")
van_ban_dai = (
    "Trí tuệ nhân tạo đang thay đổi thế giới. "
    "Nó được ứng dụng trong y tế, giáo dục, và giao thông. "
    "Tại Việt Nam, AI đang phát triển mạnh mẽ. "
    "Nhiều startup AI đã ra đời. "
    "Chính phủ cũng có chính sách hỗ trợ phát triển AI."
)
ket_qua = tt.tom_tat(van_ban_dai, so_cau=2)
print(f"\nTóm tắt: {ket_qua['tom_tat']}")

# === 6. Kiểm tra chính tả ===

from vietnamese_ai import KiemTraChinhTa

kt = KiemTraChinhTa()
kt.them_tu_dien({"xin", "chào", "thế", "giới", "học", "máy"})
ket_qua = kt.kiem_tra("xin chao the gioi")
print(f"\nLỗi chính tả: {ket_qua['so_loi']} lỗi")

print("\n✓ Tutorial 2 hoàn tất!")
