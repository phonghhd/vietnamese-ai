# Phạm Vi Kiểm Toán Mật Mã Học Độc Lập (Third-Party Cryptography & zkML Audit Brief)

Tài liệu này được biên soạn dành cho **đơn vị kiểm toán mật mã học độc lập bên thứ ba** nhằm đánh giá chính thức độ an toàn, tính đúng đắn toán học và các giả định tin cậy của phân hệ `vietnamese_ai.crypto` / `vietnamese_ai.zkml` và xác thực nút mạng Swarm CPU.

---

## 1. Tệp Mã Nguồn Trong Phạm Vi Kiểm Toán (In-Scope Files)

| Tệp nguồn | Thành phần | Chức năng mật mã |
|---|---|---|
| `vietnamese_ai/security/verifiable_inference.py` | `CayMerkleSuyLuan`, `XacThucSuyLuanMaoHiem`, `THONG_SO_NHOM_SCHNORR` | Merkle Tree tách miền RFC 6962 (`0x00`/`0x01`), cam kết Schnorr/Fiat-Shamir trên RFC 5114 MODP-2048-256, ký HMAC-SHA256 |
| `vietnamese_ai/crypto/audit.py` | `KiemToanMatMa` | Bộ kiểm định bất biến toán học & phát hiện giả mạo tự động |
| `vietnamese_ai/edge/swarm_auth.py` | `XacThucCumSwarm` | Bắt tay challenge-response HMAC, mã PIN ngẫu nhiên, chống brute-force IP, session token |
| `vietnamese_ai/utils/io_utils.py` | `LuuTai`, `_RestrictedUnpickler` | Ký tệp mô hình HMAC-SHA256 và giới hạn giải tuần tự |

---

## 2. Đặc Tả Mật Mã Hiện Tại

1. **Nhóm Schnorr (`THONG_SO_NHOM_SCHNORR`)**:
   - Chuẩn: **RFC 5114 Section 2.3** (2048-bit prime $p$, 256-bit prime subgroup order $q \mid (p - 1)$, generator $g$ cấp $q$).
   - Kiểm tra thành viên nhóm con bắt buộc trong `xac_thuc_cam_ket` và `xac_minh_cam_ket`: $1 < y < p$, $1 < a < p$, $y^q \equiv 1 \pmod p$, $a^q \equiv 1 \pmod p$, $0 < e < q$, $0 \le z < q$.
2. **Sinh khóa & Nonce**:
   - Khóa mặc định: `secrets.token_hex(32)` (OS CSPRNG).
   - Nonce $k$: kết hợp 256-bit `secrets.token_bytes(32)` và `HMAC-SHA256(secret, b"zkml1-nonce:" || payload || entropy)` rút gọn vào $[1, q - 1]$.
3. **Biến đổi Fiat-Shamir tách miền**:
   - Tensor commitment: `SHA-256("zkml1-tensor:" || a_hex || ":" || y_hex || ":" || tensor_hash)`.
   - Layer commitment: `SHA-256("zkml1-layer:" || r_hex || ":" || pub_key || ":" || payload_hex)`.
   - Bắt buộc đối chiếu `hmac.compare_digest(pub_key.lower(), proof_pub.lower())` trong `xac_minh_cam_ket`.
4. **Merkle Tree (`CayMerkleSuyLuan`)**:
   - Nút lá: `SHA-256(0x00 || data)`.
   - Nút nội bộ: `SHA-256(0x01 || left_hex || ":" || right_hex)`.

---

## 3. Các Câu Hỏi Trọng Tâm Cho Đơn Vị Kiểm Toán Độc Lập

1. **Rò rỉ kênh phụ thời gian (Timing Side-Channel)**: Đánh giá rủi ro từ phép lũy thừa big-int `pow(g, k, p)` của CPython trong môi trường chia sẻ CPU đa người dùng và lộ trình chuyển sang backend C/Rust thời gian hằng (`constant-time`).
2. **Độ lệch phân bố modulo (`mod (q - 1)`)**: Đánh giá độ chệch thống kê khi rút gọn đầu ra SHA-256 256-bit theo `q - 1` (256-bit với bit cao nhất bằng 1 — cần kiểm tra liệu rejection sampling hay mở rộng 512-bit HKDF là cần thiết cho mức an toàn tuyệt đối).
3. **Chuyển tiếp kiến trúc sang Full Circuit zkML**: Khuyến nghị tích hợp hệ thống chứng minh mạch số học (Halo2 / Groth16 / Plonky3 / EZKL) cho bài toán cần chứng minh đúng trọng số chứ không chỉ toàn vẹn định danh nút & tensor hash.
4. **An ninh giao thức LAN Swarm (`swarm_auth.py`)**: Đánh giá chống phát lại (replay), bảo mật kênh truyền TCP (khuyến nghị bọc TLS/WireGuard/Tailscale thay vì TCP thuần trên mạng không tin cậy).

---

## 4. Lệnh Chạy Tự Kiểm Toán Trước Khi Bàn Giao Kiểm Định

```bash
python -c "
import json
from vietnamese_ai.crypto import KiemToanMatMa
bc = KiemToanMatMa.chay_kiem_toan_tu_dong()
print(json.dumps(bc, indent=2, ensure_ascii=False))
assert bc['tong_quan_dat'] is True
"
```
