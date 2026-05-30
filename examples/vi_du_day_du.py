"""Ví dụ đầy đủ sử dụng Vietnamese AI Framework."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietnamese_ai.core.engine import Engine
from vietnamese_ai.core.pipeline import Pipeline
from vietnamese_ai.datalake.sample_data import DuLieuMau
from vietnamese_ai.models.classifier import PhanLoai
from vietnamese_ai.models.clustering import PhanCum
from vietnamese_ai.models.neural_net import MangNron
from vietnamese_ai.models.regression import HoiQuy
from vietnamese_ai.preprocessing.numerical import XuLySo
from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger
from vietnamese_ai.utils.metrics import Metrics


def main():
    logger = Logger("ViDu")
    logger.info("=" * 60)
    logger.info("VI DU SU DUNG VIETNAMESE AI FRAMEWORK")
    logger.info("=" * 60)

    # ==========================================
    # 1. PHÂN LOẠI
    # ==========================================
    logger.info("\n--- 1. PHAN LOAI ---")
    X_pl, y_pl = DuLieuMau.phan_loai_don_gian(so_mau=400, so_dac_trung=5)
    X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X_pl, y_pl)

    # So sánh nhiều thuật toán
    engine = Engine()
    cac_thuat_toan = ["logistic", "knn", "rung_ngau_nhien", "gradient_boosting"]

    for tt in cac_thuat_toan:
        pl = PhanLoai(thuat_toan=tt)
        engine.huan_luyen(pl, X_train, y_train, ten_mo_hinh=f"PhanLoai({tt})")
        diem = pl.danh_gia(X_test, y_test)
        logger.info(f"  {tt}: do_chinh_xac = {diem:.4f}")

    # ==========================================
    # 2. HỒI QUY
    # ==========================================
    logger.info("\n--- 2. HOI QUY ---")
    X_hq, y_hq, ten_dac_trung = DuLieuMau.du_lieu_thuc_te_don_gian(so_mau=500)
    X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X_hq, y_hq)

    for tt in ["tuyen_tinh", "ridge", "rung_ngau_nhien"]:
        hq = HoiQuy(thuat_toan=tt)
        hq.huan_luyen(X_train, y_train)
        bc = hq.bao_cao(X_test, y_test)
        logger.info(f"  {tt}: MSE={bc['mse']:.2f}, R2={bc['r2']:.4f}")

    # ==========================================
    # 3. PHÂN CỤM
    # ==========================================
    logger.info("\n--- 3. PHAN CUM ---")
    X_pc, y_pc = DuLieuMau.phan_cum_don_gian(so_mau=300, so_cum=3)

    pc = PhanCum(so_cum=3)
    pc.huan_luyen(X_pc)
    pc.du_doan(X_pc)
    diem = pc.danh_gia(X_pc)
    logger.info(f"  KMeans: silhouette = {diem:.4f}")

    # ==========================================
    # 4. MẠNG NƠ-RON
    # ==========================================
    logger.info("\n--- 4. MANG NO-RON ---")
    X_nn, y_nn = DuLieuMau.phan_loai_don_gian(so_mau=300, so_dac_trung=10)
    X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X_nn, y_nn)

    mang = MangNron(lop_an=[32, 16], so_vong=100, toc_do_hoc=0.01)
    mang.huan_luyen(X_train, y_train)
    diem = mang.danh_gia(X_test, y_test)
    logger.info(f"  MangNron: do_chinh_xac = {diem:.4f}")

    # ==========================================
    # 5. XỬ LÝ VĂN BẢN
    # ==========================================
    logger.info("\n--- 5. XU LY VAN BAN ---")
    xl = XuLyVanBan()
    van_ban = DuLieuMau.van_ban_tieng_viet()

    for vb in van_ban[:3]:
        tu_khoa = xl.trich_xuat_tu_khoa(vb, top_n=3)
        logger.info(f"  '{vb[:50]}...' -> Tu khoa: {tu_khoa}")

    tfidf = xl.ma_hoa_tfidf(van_ban)
    logger.info(f"  Ma tran TF-IDF: {tfidf.shape}")

    # ==========================================
    # 6. PIPELINE
    # ==========================================
    logger.info("\n--- 6. PIPELINE ---")
    pipe = Pipeline()
    pipe.them_buoc("chuan_hoa", XuLySo())
    pipe.them_buoc("phan_loai", PhanLoai(thuat_toan="rung_ngau_nhien"))

    X_pl2, y_pl2 = DuLieuMau.phan_loai_don_gian(so_mau=200, so_dac_trung=4)
    X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X_pl2, y_pl2)

    pipe.fit(X_train, y_train)
    du_doan = pipe.predict(X_test)
    do_chinh_xac = Metrics.do_chinh_xac(y_test, du_doan)
    logger.info(f"  Pipeline: do_chinh_xac = {do_chinh_xac:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("HOAN TAT VI DU!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
