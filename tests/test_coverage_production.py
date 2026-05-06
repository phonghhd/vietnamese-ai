import time

import pytest


class TestKiemTraSucKhoe:
    def test_creation(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        assert ktsk is not None

    def test_dang_ky_check(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        ktsk.dang_ky_check("test", lambda: {"status": "healthy"})
        assert "test" in ktsk._checks

    def test_kiem_tra(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        ktsk.dang_ky_check("test", lambda: {"status": "healthy"})
        result = ktsk.kiem_tra()
        assert "chi_tiet" in result
        assert "test" in result["chi_tiet"]

    def test_ready(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        ktsk.dang_ky_check("test", lambda: {"status": "ok"})
        assert ktsk.ready() is True

    def test_live(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        assert ktsk.live() is True

    def test_dang_ky_checks_mac_dinh(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        ktsk.dang_ky_checks_mac_dinh()
        assert len(ktsk._checks) > 0

    def test_lich_su(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        ktsk.dang_ky_check("test", lambda: {"status": "healthy"})
        ktsk.kiem_tra()
        assert len(ktsk.lich_su()) > 0

    def test_ty_le_healthy(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        ktsk.dang_ky_check("ok", lambda: {"status": "ok"})
        result = ktsk.kiem_tra()
        assert result["trang_thai"] == "healthy"

    def test_check_failing(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        ktsk.dang_ky_check("fail", lambda: {"status": "unhealthy"})
        result = ktsk.kiem_tra()
        assert "chi_tiet" in result

    def test_check_exception(self):
        from vietnamese_ai.production.health import KiemTraSucKhoe

        ktsk = KiemTraSucKhoe()
        ktsk.dang_ky_check("error", lambda: 1 / 0)
        result = ktsk.kiem_tra()
        assert "chi_tiet" in result


class TestMachCat:
    def test_creation(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat()
        assert mc is not None

    def test_cho_phep_closed(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat()
        assert mc.cho_phep() is True

    def test_ghi_nhan_thanh_cong(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat()
        mc.ghi_nhan_thanh_cong()

    def test_ghi_nhan_loi(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat(so_loi_toi_da=2)
        mc.ghi_nhan_loi()
        mc.ghi_nhan_loi()
        assert mc.cho_phep() is False

    def test_thuc_hien(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat()
        result = mc.thuc_hien(lambda: 42)
        assert result == 42

    def test_context_manager(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat()
        with mc:
            pass
        assert mc is not None

    def test_reset(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat(so_loi_toi_da=1)
        mc.ghi_nhan_loi()
        assert mc.cho_phep() is False
        mc.reset()
        assert mc.cho_phep() is True

    def test_lay_thong_ke(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat()
        tk = mc.lay_thong_ke()
        assert "trang_thai" in tk

    def test_half_open(self):
        from vietnamese_ai.production.circuit_breaker import MachCat

        mc = MachCat(so_loi_toi_da=1, timeout_phuc_hoi=0.01)
        mc.ghi_nhan_loi()
        time.sleep(0.02)
        assert mc.cho_phep() is True


class TestLoggerCauTruc:
    def test_creation(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc("test")
        assert logger is not None

    def test_log_levels(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc("test")
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")
        logger.critical("critical msg")

    def test_them_context(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc("test")
        logger.them_context(request_id="123")
        logger.info("test")
        logger.xoa_context()

    def test_do_thoi_gian(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc("test")
        with logger.do_thoi_gian("test_op"):
            time.sleep(0.01)

    def test_log_request(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc("test")
        logger.log_request("req1", "GET", "/api/test", 200, 100.0)

    def test_log_prediction(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc("test")
        logger.log_prediction("model1", [1, 2], 0.5)

    def test_log_model_event(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc("test")
        logger.log_model_event("loaded", "model1", version="1.0")

    def test_thong_ke(self):
        from vietnamese_ai.production.logging import LoggerCauTruc

        logger = LoggerCauTruc("test")
        logger.info("test")
        tk = logger.thong_ke()
        assert "ten" in tk


class TestQuanLyMetrics:
    def test_creation(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        assert qm is not None

    def test_counter(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        qm.counter("requests")
        qm.counter("requests")
        assert qm.lay_counter("requests") == 2

    def test_gauge(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        qm.gauge("cpu", 75.5)
        assert qm.lay_gauge("cpu") == 75.5

    def test_histogram(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        for i in range(100):
            qm.histogram("latency", float(i))
        stats = qm.lay_histogram_stats("latency")
        assert "mean" in stats
        assert "p50" in stats
        assert "p99" in stats

    def test_timer(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        with qm.timer("operation"):
            time.sleep(0.01)
        stats = qm.lay_histogram_stats("operation")
        assert stats["count"] == 1

    def test_export_prometheus(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        qm.counter("requests")
        output = qm.export_prometheus()
        assert "requests" in output

    def test_export_json(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        qm.counter("requests")
        output = qm.export_json()
        assert isinstance(output, dict)

    def test_reset(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        qm.counter("requests")
        qm.reset()
        assert qm.lay_counter("requests") == 0

    def test_thong_ke(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        qm.counter("requests")
        tk = qm.thong_ke()
        assert "so_counters" in tk

    def test_nonexistent_counter(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        assert qm.lay_counter("nonexistent") == 0

    def test_nonexistent_gauge(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        assert qm.lay_gauge("nonexistent") == 0.0

    def test_nonexistent_histogram(self):
        from vietnamese_ai.production.metrics import QuanLyMetrics

        qm = QuanLyMetrics()
        stats = qm.lay_histogram_stats("nonexistent")
        assert stats["count"] == 0


class TestLamNongModel:
    def test_creation(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        assert lnm is not None

    def test_dang_ky_model(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        lnm.dang_ky_model("model1", lambda: "model_instance")
        assert "model1" in lnm.danh_sach_models()

    def test_lay_model(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        lnm.dang_ky_model("model1", lambda: "model_instance")
        model = lnm.lay_model("model1")
        assert model is not None

    def test_lay_model_nonexistent(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        with pytest.raises(ValueError):
            lnm.lay_model("nonexistent")

    def test_lam_nong(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        lnm.dang_ky_model("model1", lambda: "model_instance")
        lnm.lam_nong("model1")
        assert lnm.trang_thai_model("model1")["da_warmup"] is True

    def test_lam_nong_tat_ca(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        lnm.dang_ky_model("m1", lambda: "model1")
        lnm.dang_ky_model("m2", lambda: "model2")
        lnm.lam_nong_tat_ca()
        assert lnm.trang_thai_model("m1")["da_warmup"] is True

    def test_xoa_model(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        lnm.dang_ky_model("model1", lambda: "model_instance")
        lnm.xoa_model("model1")
        assert "model1" not in lnm.danh_sach_models()

    def test_thong_ke(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        lnm.dang_ky_model("model1", lambda: "model_instance")
        tk = lnm.thong_ke()
        assert "so_models" in tk

    def test_auto_refresh(self):
        from vietnamese_ai.production.warmup import LamNongModel

        lnm = LamNongModel()
        lnm.bat_dau_auto_refresh()
        time.sleep(0.2)
        lnm.dung_auto_refresh()
