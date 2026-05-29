"""
Tutorial 1: Bắt đầu với Vietnamese AI Framework
================================================

Hướng dẫn cơ bản: phân loại, hồi quy, AutoML, pipeline.
"""

import numpy as np

# === 1. Phân loại (Classification) ===
from vietnamese_ai import KiemDinhCheo, PhanLoai, XuLySo

# Tạo dữ liệu mẫu
np.random.seed(42)
X_train = np.random.randn(200, 4)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
X_test = np.random.randn(50, 4)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

# Huấn luyện với nhiều thuật toán
for thuat_toan in ["logistic", "knn", "svm", "rung_ngau_nhien"]:
    model = PhanLoai(thuat_toan=thuat_toan)
    model.huan_luyen(X_train, y_train)
    diem = model.danh_gia(X_test, y_test)
    print(f"{thuat_toan}: accuracy = {diem:.3f}")

# Cross-validation
model = PhanLoai(thuat_toan="logistic")
kdc = KiemDinhCheo(so_fold=5)
ket_qua = kdc.chay(model, X_train, y_train)
print(f"\nCross-validation: {ket_qua['diem_trung_binh']:.3f} ± {ket_qua['do_lech_chuan']:.3f}")

# === 2. Hồi quy (Regression) ===

from vietnamese_ai import HoiQuy

X_reg = np.random.randn(200, 3)
y_reg = X_reg[:, 0] * 2 + X_reg[:, 1] * 3 + np.random.randn(200) * 0.5

for thuat_toan in ["tuyen_tinh", "ridge", "lasso"]:
    model = HoiQuy(thuat_toan=thuat_toan)
    model.huan_luyen(X_reg[:150], y_reg[:150])
    diem = model.danh_gia(X_reg[150:], y_reg[150:])
    print(f"{thuat_toan}: MSE = {diem:.3f}")

# === 3. AutoML ===

from vietnamese_ai import AutoML

auto = AutoML()
auto.fit(X_train, y_train)
diem = auto.danh_gia(X_test, y_test)
print(f"\nAutoML: accuracy = {diem:.3f}")
print(f"Best model: {auto._mo_hinh_tot_nhat}")

# === 4. Pipeline ===

from vietnamese_ai import Pipeline

pipe = Pipeline(ten="demo_pipeline")
pipe.them_buoc("scale", XuLySo())
pipe.them_buoc("model", PhanLoai(thuat_toan="logistic"))
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test[:5])
print(f"\nPipeline predictions: {pred}")

# === 5. Lưu/tải mô hình ===

import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    duong_dan = os.path.join(tmpdir, "model.pkl")
    pipe.luu(duong_dan)
    pipe_loaded = Pipeline.tai(duong_dan)
    pred2 = pipe_loaded.predict(X_test[:5])
    print(f"Loaded predictions: {pred2}")
    print(f"Match: {np.array_equal(pred, pred2)}")

print("\n✓ Tutorial 1 hoàn tất!")
