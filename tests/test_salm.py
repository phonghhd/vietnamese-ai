"""Tests cho SALM (Self-Adapting Language Models) v11.0."""

import numpy as np


class TestSelfRefine:
    def test_khoi_tao(self):
        from vietnamese_ai.salm.self_refine import SelfRefine

        refine = SelfRefine(ham_sinh=lambda p: "output")
        assert refine.so_vong_toi_da == 3
        assert refine.nguong_chat_luong == 0.8

    def test_chay_co_ban(self):
        from vietnamese_ai.salm.self_refine import SelfRefine

        call_count = [0]

        def mock_sinh(prompt):
            call_count[0] += 1
            return f"Output cho prompt {call_count[0]}"

        refine = SelfRefine(
            ham_sinh=mock_sinh,
            so_vong_toi_da=2,
            nguong_chat_luong=0.99,
        )
        ket_qua = refine.chay("Test prompt")
        assert "output_cuoi" in ket_qua
        assert ket_qua["so_vong"] >= 1

    def test_dung_khi_dat_nguong(self):
        from vietnamese_ai.salm.self_refine import SelfRefine

        def mock_danh_gia(prompt, output):
            return {"diem": 0.95}

        refine = SelfRefine(
            ham_sinh=lambda p: "good output",
            ham_danh_gia=mock_danh_gia,
            so_vong_toi_da=10,
            nguong_chat_luong=0.9,
        )
        ket_qua = refine.chay("Test")
        assert ket_qua["dat_nguong"] is True
        assert ket_qua["so_vong"] == 1

    def test_thong_ke(self):
        from vietnamese_ai.salm.self_refine import SelfRefine

        refine = SelfRefine(ham_sinh=lambda p: "out")
        refine.chay("test")
        stats = refine.thong_ke()
        assert stats["so_lan_chay"] == 1

    def test_lich_su(self):
        from vietnamese_ai.salm.self_refine import SelfRefine

        refine = SelfRefine(ham_sinh=lambda p: "out", so_vong_toi_da=1)
        refine.chay("test")
        lich_su = refine.lay_lich_su()
        assert len(lich_su) == 1


class TestSelfConsistency:
    def test_khoi_tao(self):
        from vietnamese_ai.salm.self_consistency import SelfConsistency

        sc = SelfConsistency(ham_sinh=lambda p: "answer", so_luong=3)
        assert sc.so_luong == 3

    def test_chay_co_ban(self):
        from vietnamese_ai.salm.self_consistency import SelfConsistency

        sc = SelfConsistency(
            ham_sinh=lambda p: "42",
            so_luong=5,
        )
        ket_qua = sc.chay("2 + 2 = ?")
        assert ket_qua["dap_an"] == "42"
        assert ket_qua["so_luong_paths"] == 5
        assert ket_qua["ty_le_dong_nhat"] == 1.0

    def test_da_dang_paths(self):
        from vietnamese_ai.salm.self_consistency import SelfConsistency

        answers = ["42", "42", "43", "42", "44"]

        def mock_sinh(prompt):
            return answers.pop(0)

        sc = SelfConsistency(ham_sinh=mock_sinh, so_luong=5)
        ket_qua = sc.chay("What is 2+2?")
        assert ket_qua["dap_an"] == "42"
        assert ket_qua["ty_le_dong_nhat"] == 0.6

    def test_chain_of_thought(self):
        from vietnamese_ai.salm.self_consistency import SelfConsistency

        sc = SelfConsistency(ham_sinh=lambda p: "Bước 1: ...\nTrả lời: 42", so_luong=3)
        ket_qua = sc.chay("2+2=?", che_do="cot")
        assert "42" in ket_qua["dap_an"]

    def test_thong_ke(self):
        from vietnamese_ai.salm.self_consistency import SelfConsistency

        sc = SelfConsistency(ham_sinh=lambda p: "answer", so_luong=3)
        sc.chay("test")
        stats = sc.thong_ke()
        assert stats["so_lan_chay"] == 1


class TestAdaptiveLoRA:
    def test_khoi_tao(self):
        from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA

        al = AdaptiveLoRA(che_do="keyword")
        assert al.che_do == "keyword"
        assert len(al.danh_sach_adapters()) == 0

    def test_dang_ky_va_chon(self):
        from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA

        al = AdaptiveLoRA(che_do="keyword")
        al.dang_ky_adapter("math", "math_adapter", keywords=["tính", "cộng", "số"])
        al.dang_ky_adapter("code", "code_adapter", keywords=["code", "function", "python"])

        chon = al.chon_adapter("Tính tổng 2 số")
        assert len(chon) == 1
        assert chon[0]["ten"] == "math"

    def test_chon_theo_keyword(self):
        from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA

        al = AdaptiveLoRA(che_do="keyword")
        al.dang_ky_adapter("a", "adapter_a", keywords=["hello", "world"])
        al.dang_ky_adapter("b", "adapter_b", keywords=["code", "python"])

        chon = al.chon_adapter("Write python code", top_k=2)
        assert chon[0]["ten"] == "b"

    def test_embedding_mode(self):
        from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA

        al = AdaptiveLoRA(che_do="embedding")
        al.dang_ky_adapter("a", "adapter_a", embedding=np.array([1.0, 0.0, 0.0]))
        al.dang_ky_adapter("b", "adapter_b", embedding=np.array([0.0, 1.0, 0.0]))

        al.dang_ky_ham_embed(lambda text: np.array([0.9, 0.1, 0.0]))
        chon = al.chon_adapter("test")
        assert chon[0]["ten"] == "a"

    def test_ket_hop_adapters(self):
        from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA

        al = AdaptiveLoRA(che_do="keyword")
        al.dang_ky_adapter("a", "adapter_a", keywords=["hello"])
        al.dang_ky_adapter("b", "adapter_b", keywords=["world"])

        trong_so = al.ket_hop_adapters("hello world")
        assert "a" in trong_so
        assert "b" in trong_so
        assert abs(sum(trong_so.values()) - 1.0) < 0.01

    def test_xoa_adapter(self):
        from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA

        al = AdaptiveLoRA()
        al.dang_ky_adapter("a", "adapter")
        assert al.xoa_adapter("a") is True
        assert len(al.danh_sach_adapters()) == 0

    def test_thong_ke(self):
        from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA

        al = AdaptiveLoRA()
        al.dang_ky_adapter("a", "adapter", keywords=["test"])
        al.chon_adapter("test")
        stats = al.thong_ke()
        assert stats["so_adapters"] == 1


class TestSinhDuLieuTuDong:
    def test_khoi_tao(self):
        from vietnamese_ai.salm.self_data import SinhDuLieuTuDong

        sd = SinhDuLieuTuDong(ham_sinh=lambda p: "output")
        assert sd.nguong_chat_luong == 0.5

    def test_sinh_instruction(self):
        from vietnamese_ai.salm.self_data import SinhDuLieuTuDong

        call_count = [0]

        def mock_sinh(prompt):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return "Tạo bài toán cộng đơn giản"
            return "2 + 2 = 4"

        sd = SinhDuLieuTuDong(ham_sinh=mock_sinh, nguong_chat_luong=0.3)
        sd.them_giong_mau("Tính tổng", "Kết quả là 10")
        du_lieu = sd.sinh(3, loai="instruction")
        assert len(du_lieu) > 0

    def test_sinh_qa(self):
        from vietnamese_ai.salm.self_data import SinhDuLieuTuDong

        sd = SinhDuLieuTuDong(
            ham_sinh=lambda p: "Q: AI là gì?\nA: Trí tuệ nhân tạo", nguong_chat_luong=0.3
        )
        sd.them_giong_mau("ML là gì?", "Học máy")
        du_lieu = sd.sinh(2, loai="qa")
        assert len(du_lieu) > 0

    def test_giong_mac_dinh(self):
        from vietnamese_ai.salm.self_data import SinhDuLieuTuDong

        sd = SinhDuLieuTuDong(ham_sinh=lambda p: "output text", nguong_chat_luong=0.3)
        du_lieu = sd.sinh(2, loai="instruction")
        assert len(du_lieu) > 0

    def test_xoa_du_lieu(self):
        from vietnamese_ai.salm.self_data import SinhDuLieuTuDong

        sd = SinhDuLieuTuDong(ham_sinh=lambda p: "out", nguong_chat_luong=0.0)
        sd.them_giong_mau("test", "output")
        sd.sinh(2)
        assert len(sd.lay_du_lieu_da_sinh()) > 0
        sd.xoa_du_lieu()
        assert len(sd.lay_du_lieu_da_sinh()) == 0

    def test_thong_ke(self):
        from vietnamese_ai.salm.self_data import SinhDuLieuTuDong

        sd = SinhDuLieuTuDong(ham_sinh=lambda p: "out")
        sd.them_giong_mau("test", "output")
        stats = sd.thong_ke()
        assert stats["so_giong_mau"] == 1


class TestTestTimeTraining:
    def test_khoi_tao(self):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.salm.test_time_training import TestTimeTraining

        model = PhanLoai()
        ttt = TestTimeTraining(model, che_do="entropy_minimization")
        assert ttt.che_do == "entropy_minimization"
        assert ttt.da_thich_ung is False

    def test_thich_ung(self):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.salm.test_time_training import TestTimeTraining

        X_train = np.random.randn(50, 4)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X_train, y_train)

        X_test = np.random.randn(10, 4)
        ttt = TestTimeTraining(model, che_do="entropy_minimization", so_buoc_mac_dinh=3)
        ket_qua = ttt.thich_ung(X_test)

        assert "loss_cuoi" in ket_qua
        assert ket_qua["so_buoc"] == 3
        assert ttt.da_thich_ung is True

    def test_phuc_hoi_trong_so(self):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.salm.test_time_training import TestTimeTraining

        X_train = np.random.randn(50, 4)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X_train, y_train)

        ttt = TestTimeTraining(model, so_buoc_mac_dinh=2)
        ttt.luu_trong_so_goc()

        X_test = np.random.randn(10, 4)
        ttt.thich_ung(X_test)
        assert ttt.da_thich_ung is True

        ttt.phuc_hoi_trong_so()
        assert ttt.da_thich_ung is False

    def test_du_doan(self):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.salm.test_time_training import TestTimeTraining

        X_train = np.random.randn(50, 4)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X_train, y_train)

        ttt = TestTimeTraining(model, so_buoc_mac_dinh=1)
        X_test = np.random.randn(5, 4)
        ttt.thich_ung(X_test)
        du_doan = ttt.du_doan(X_test)
        assert len(du_doan) == 5

    def test_contrastive_mode(self):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.salm.test_time_training import TestTimeTraining

        X_train = np.random.randn(50, 4)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X_train, y_train)

        ttt = TestTimeTraining(model, che_do="contrastive", so_buoc_mac_dinh=2)
        X_test = np.random.randn(10, 4)
        ket_qua = ttt.thich_ung(X_test)
        assert ket_qua["so_buoc"] == 2

    def test_masked_prediction(self):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.salm.test_time_training import TestTimeTraining

        X_train = np.random.randn(50, 4)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = PhanLoai(thuat_toan="logistic")
        model.huan_luyen(X_train, y_train)

        ttt = TestTimeTraining(model, che_do="masked_prediction", so_buoc_mac_dinh=2)
        X_test = np.random.randn(10, 4)
        ket_qua = ttt.thich_ung(X_test)
        assert ket_qua["so_buoc"] == 2

    def test_thong_ke(self):
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.salm.test_time_training import TestTimeTraining

        model = PhanLoai()
        ttt = TestTimeTraining(model)
        stats = ttt.thong_ke()
        assert stats["che_do"] == "entropy_minimization"
        assert stats["da_thich_ung"] is False


class TestSALMIntegration:
    def test_version(self):
        import vietnamese_ai

        assert vietnamese_ai.__version__ == "11.0.1"

    def test_imports(self):
        from vietnamese_ai import (
            AdaptiveLoRA,
            SelfConsistency,
            SelfRefine,
            SinhDuLieuTuDong,
            TestTimeTraining,
        )

        assert SelfRefine is not None
        assert SelfConsistency is not None
        assert AdaptiveLoRA is not None
        assert SinhDuLieuTuDong is not None
        assert TestTimeTraining is not None

    def test_salm_full_flow(self):
        from vietnamese_ai.salm import AdaptiveLoRA, SelfConsistency, SelfRefine

        refine = SelfRefine(ham_sinh=lambda p: f"Refined: {p[:20]}", so_vong_toi_da=2)
        sc = SelfConsistency(ham_sinh=lambda p: "42", so_luong=3)
        al = AdaptiveLoRA(che_do="keyword")
        al.dang_ky_adapter("math", "adapter", keywords=["tính"])

        r1 = refine.chay("Test prompt")
        r2 = sc.chay("2+2=?")
        r3 = al.chon_adapter("Tính tổng")

        assert r1["so_vong"] >= 1
        assert r2["dap_an"] == "42"
        assert r3[0]["ten"] == "math"
