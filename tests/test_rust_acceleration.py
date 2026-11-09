"""Kiểm thử Bộ Tăng Tốc Native Rust Core (Pillar 3: Rust Native Acceleration) - Vietnamese AI Framework."""

import numpy as np

import vietnamese_ai.nlp.native_tokenizer as nt_module
from vietnamese_ai.nlp.native_tokenizer import (
    BoNhungVanBanSieuToc,
    BoTachTuTiengVietSieuToc,
    co_ho_tro_rust,
)


def test_co_ho_tro_rust():
    """Kiểm tra cờ hỗ trợ Rust phản ánh chính xác trạng thái môi trường."""
    ho_tro = co_ho_tro_rust()
    assert isinstance(ho_tro, bool)


def test_rust_fast_clean_spaces():
    """Kiểm tra loại bỏ ký tự rác zero-width và gom cụm khoảng trắng."""
    tokenizer = BoTachTuTiengVietSieuToc()
    raw = "Hà Nội \u200b  là  \ufeff thủ   đô   Việt Nam  \u200c "
    sach = tokenizer.chuan_hoa_unicode(raw)
    assert "\u200b" not in sach
    assert "\ufeff" not in sach
    assert "\u200c" not in sach
    assert "  " not in sach
    assert "Hà Nội là thủ đô Việt Nam" in sach


def test_rust_fast_token_count():
    """Kiểm tra đếm số từ siêu tốc qua Rust Rayon đa luồng và Python fallback."""
    tokenizer = BoTachTuTiengVietSieuToc()
    text = "Trí tuệ nhân tạo Việt Nam đang trên đà bứt phá công nghệ mạnh mẽ. " * 20
    count_rust = tokenizer.dem_so_tu(text, su_dung_rust=True)
    count_py = tokenizer.dem_so_tu(text, su_dung_rust=False)

    assert count_rust > 0
    assert count_py > 0
    # Đếm số lượng từ tương đương nhau
    assert abs(count_rust - count_py) <= 50


def test_rust_fast_keyword_search():
    """Kiểm tra tìm kiếm từ khóa tiếng Việt đa luồng/SIMD."""
    tokenizer = BoTachTuTiengVietSieuToc()
    van_ban = "Việt Nam là quốc gia độc lập. Đất nước Việt Nam anh hùng. Con người Việt Nam mến khách."
    vi_tri = tokenizer.tim_tu_khoa_nhanh(van_ban, "Việt Nam")

    assert len(vi_tri) == 3
    for idx in vi_tri:
        assert van_ban[idx : idx + len("Việt Nam")] == "Việt Nam"

    # Trường hợp từ khóa không tồn tại
    assert tokenizer.tim_tu_khoa_nhanh(van_ban, "Sao Hỏa") == []
    # Trường hợp chuỗi rỗng
    assert tokenizer.tim_tu_khoa_nhanh("", "Việt Nam") == []
    assert tokenizer.tim_tu_khoa_nhanh(van_ban, "") == []


def test_rust_fast_batch_cosine_similarity():
    """Kiểm tra tính toán Cosine Similarity hàng loạt trên đa lõi CPU."""
    embedder = BoNhungVanBanSieuToc(so_chieu=64)
    v1 = embedder.nhung("Hà Nội nghìn năm văn hiến")
    v2 = embedder.nhung("Thủ đô Hà Nội thân yêu")
    v3 = embedder.nhung("Trí tuệ nhân tạo và học máy")

    ma_tran = [v1, v2, v3]
    sims = embedder.tim_kiem_tuong_dong_hang_loat(v1, ma_tran)

    assert len(sims) == 3
    # v1 tương đồng tuyệt đối với chính nó (cosine ~ 1.0)
    assert abs(sims[0] - 1.0) < 1e-4
    # v1 và v2 gần nghĩa hơn v1 và v3
    assert sims[1] > sims[2]


def test_fallback_thuan_python_khi_khong_co_rust(monkeypatch):
    """Đảm bảo 100% Zero-Dependency: Hệ thống fallback mượt mà nếu không có thư viện Rust."""
    monkeypatch.setattr(nt_module, "_rust_core", None)

    tokenizer = BoTachTuTiengVietSieuToc()
    embedder = BoNhungVanBanSieuToc(so_chieu=64)

    # Clean spaces fallback
    s = tokenizer.chuan_hoa_unicode("Xin   chào    Việt Nam")
    assert s == "Xin chào Việt Nam"

    # Token count fallback
    cnt = tokenizer.dem_so_tu("Trí tuệ nhân tạo Việt Nam")
    assert cnt > 0

    # Keyword search fallback
    pos = tokenizer.tim_tu_khoa_nhanh("Hà Nội là thủ đô Hà Nội", "Hà Nội")
    assert len(pos) == 2

    # Batch cosine similarity fallback
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sims = embedder.tim_kiem_tuong_dong_hang_loat(q, m)
    assert len(sims) == 2
    assert abs(sims[0] - 1.0) < 1e-5
    assert abs(sims[1] - 0.0) < 1e-5


def test_rust_batch_cosine_ma_tran_rong():
    """Kiểm tra xử lý biên ma trận rỗng."""
    embedder = BoNhungVanBanSieuToc(so_chieu=32)
    q = embedder.nhung("Kiểm tra")
    sims = embedder.tim_kiem_tuong_dong_hang_loat(q, [])
    assert len(sims) == 0
