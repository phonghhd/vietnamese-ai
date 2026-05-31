import time

import pytest

from vietnamese_ai.workflow import DongChayDAG, LoiVongLap, NutCongViec


def test_dag_chay_tuan_tu():
    engine = DongChayDAG()

    thu_tu_chay_thuc_te = []

    def buoc_1():
        thu_tu_chay_thuc_te.append(1)
        return {"du_lieu_1": "xin chao"}

    def buoc_2(du_lieu_1):
        thu_tu_chay_thuc_te.append(2)
        return {"du_lieu_2": du_lieu_1 + " the gioi"}

    def buoc_3(du_lieu_2):
        thu_tu_chay_thuc_te.append(3)
        return {"ket_qua": len(du_lieu_2)}

    engine.them_nut(NutCongViec("n1", buoc_1, dau_ra=["du_lieu_1"]))
    engine.them_nut(NutCongViec("n2", buoc_2, dau_vao=["du_lieu_1"], dau_ra=["du_lieu_2"]))
    engine.them_nut(NutCongViec("n3", buoc_3, dau_vao=["du_lieu_2"], dau_ra=["ket_qua"]))

    ket_qua_cuoi = engine.thuc_thi()

    assert thu_tu_chay_thuc_te == [1, 2, 3]
    assert ket_qua_cuoi["ket_qua"] == len("xin chao the gioi")


def test_dag_phat_hien_vong_lap():
    engine = DongChayDAG()

    # A chờ C
    engine.them_nut(NutCongViec("A", lambda c: {"a": 1}, dau_vao=["c"], dau_ra=["a"]))
    # B chờ A
    engine.them_nut(NutCongViec("B", lambda a: {"b": 1}, dau_vao=["a"], dau_ra=["b"]))
    # C chờ B -> Tạo thành vòng lặp A->B->C->A
    engine.them_nut(NutCongViec("C", lambda b: {"c": 1}, dau_vao=["b"], dau_ra=["c"]))

    with pytest.raises(LoiVongLap):
        engine.sap_xep_topo()


def test_dag_chay_song_song():
    engine = DongChayDAG()

    nhat_ky = []

    def doc_nguon():
        nhat_ky.append("BAT_DAU_NGUON")
        time.sleep(0.1)
        return {"van_ban": "Hello"}

    def dich_anh_viet(van_ban):
        nhat_ky.append("BAT_DAU_DICH_VIET")
        time.sleep(0.2)
        nhat_ky.append("XONG_DICH_VIET")
        return {"viet": "Xin chào"}

    def dich_anh_phap(van_ban):
        nhat_ky.append("BAT_DAU_DICH_PHAP")
        time.sleep(0.2)
        nhat_ky.append("XONG_DICH_PHAP")
        return {"phap": "Bonjour"}

    def tong_hop(viet, phap):
        nhat_ky.append("TONG_HOP")
        return {"ket_qua": f"{viet} - {phap}"}

    engine.them_nut(NutCongViec("Nguon", doc_nguon, dau_ra=["van_ban"]))
    engine.them_nut(NutCongViec("Dich_Viet", dich_anh_viet, dau_vao=["van_ban"], dau_ra=["viet"]))
    engine.them_nut(NutCongViec("Dich_Phap", dich_anh_phap, dau_vao=["van_ban"], dau_ra=["phap"]))
    engine.them_nut(NutCongViec("Tong_Hop", tong_hop, dau_vao=["viet", "phap"], dau_ra=["ket_qua"]))

    start_time = time.time()
    kq = engine.thuc_thi()
    end_time = time.time()

    assert kq["ket_qua"] == "Xin chào - Bonjour"

    # Do 2 nút dịch chạy song song (sleep 0.2s) và nút nguồn chạy 0.1s
    # Tổng thời gian phải nhỏ hơn 0.1 + 0.2 + 0.2 = 0.5s
    assert end_time - start_time < 0.45

    # Nút nguồn chắc chắn phải chạy đầu tiên
    assert nhat_ky[0] == "BAT_DAU_NGUON"
    # Tổng hợp chắc chắn phải ở cuối
    assert nhat_ky[-1] == "TONG_HOP"
