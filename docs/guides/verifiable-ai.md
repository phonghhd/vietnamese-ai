# 🛡️ Verifiable AI, zkML & Mật Mã Học (`vietnamese_ai.crypto` / `vietnamese_ai.zkml`)

Vietnamese AI Framework cung cấp hệ thống **Xác Thực Suy Luận Mật Mã Học (Verifiable AI Inference)** kết hợp **Cây Merkle tách miền (RFC 6962 style)** và **lược đồ cam kết phi tương tác Schnorr / Fiat-Shamir** trên nhóm con cấp nguyên tố chuẩn **RFC 5114 Section 2.3 (MODP-2048-256)**.

> [!IMPORTANT]
> **Trạng thái & Phạm vi mật mã học**: Các thành phần này được gắn nhãn `THU_NGHIEM` (`Experimental`) và cần được kiểm toán độc lập bởi đơn vị mật mã bên thứ ba trước khi dùng cho các hệ thống trọng yếu. Xem chi tiết phần [Giả định an toàn & Giới hạn](#4-gia-inh-an-toan-gioi-han-quan-trong).

---

## 1. Lựa Chọn Tham Số Mật Mã (Cryptographic Choices)

| Thành phần | Lựa chọn triển khai | Lý do bảo mật |
|---|---|---|
| **Nhóm đại số Schnorr** | RFC 5114 Section 2.3 (MODP-2048-256) | Trường nguyên tố $p$ 2048-bit (~112-bit bảo mật NFS) cùng nhóm con cấp nguyên tố $q$ 256-bit ($q \mid (p-1)$, 128-bit bảo mật Pollard rho). |
| **Kiểm tra nhóm con** | $1 < x < p$ và $x^q \equiv 1 \pmod p$ | Ngăn tấn công nhóm con nhỏ (small-subgroup confinement) và phần tử bậc thấp ($p - 1$). |
| **Nguồn ngẫu nhiên** | `secrets` (OS CSPRNG) + HMAC-SHA256 hedged nonce | Loại bỏ hoàn toàn entropy yếu theo `time.time()`; bảo đảm nonce $k$ đều trong $[1, q-1]$ tránh rò rỉ khóa bí mật $x$. |
| **Hàm băm & Fiat-Shamir** | SHA-256 tách miền (`zkml1-tensor:` / `zkml1-layer:`) | Ràng buộc chặt $(a, y, H(\text{tensor}))$ hoặc $(R, y, \text{payload})$ để chống giả mạo chứng từ không có khóa bí mật. |
| **Merkle Tree** | SHA-256 tách miền lá `0x00` / nút nội bộ `0x01` | Chống tấn công tiền ảnh bậc hai (second-preimage attack) trên cây Merkle. |
| **So sánh chuỗi băm / MAC** | `hmac.compare_digest` | Giảm thiểu rò rỉ kênh phụ thời gian khi đối chiếu hash và chữ ký HMAC-SHA256. |

### Công thức toán học Schnorr / Fiat-Shamir

Với khóa bí mật $x \in [1, q-1]$, khóa công khai $y = g^x \bmod p$, và nonce ngẫu nhiên $k \in [1, q-1]$:

$$a = g^k \bmod p$$
$$e = \left(\text{int}(\text{SHA256}(\text{domain} \parallel a \parallel y \parallel H(\text{tensor}))) \bmod (q - 1)\right) + 1$$
$$z = (k + e \cdot x) \bmod q$$

Bộ xác minh kiểm tra:
1. $0 < e < q$, $0 \le z < q$, $1 < y < p$, $1 < a < p$
2. $y^q \equiv 1 \pmod p$ và $a^q \equiv 1 \pmod p$ (thành viên nhóm con cấp $q$)
3. Tính lại $e$ từ $(a, y, H(\text{tensor}))$ (khi xác thực nghiêm ngặt hoặc có `tensor_hash`)
4. $g^z \equiv a \cdot y^e \pmod p$

---

## 2. Hướng Dẫn Sử Dụng (API Chính Xác)

Bạn có thể import trực tiếp từ `vietnamese_ai.crypto`, `vietnamese_ai.zkml` hoặc `vietnamese_ai.security.verifiable_inference`.

### A. Cây Merkle Suy Luận (`CayMerkleSuyLuan`)

```python
import numpy as np
from vietnamese_ai.crypto import CayMerkleSuyLuan

tensor_activations = [
    np.random.randn(128).astype(np.float32),
    np.random.randn(128).astype(np.float32),
    np.random.randn(128).astype(np.float32),
    np.random.randn(128).astype(np.float32),
]

cay_merkle = CayMerkleSuyLuan(tensor_activations)
root_hash = cay_merkle.lay_goc_merkle()
proof_path = cay_merkle.sinh_bang_chung(index_la=2)

hop_le = cay_merkle.kiem_tra_bang_chung(
    du_lieu=tensor_activations[2],
    index_la=2,
    proof_path=proof_path,
    root=root_hash,
)
assert hop_le is True
```

### B. Cam Kết Schnorr/Fiat-Shamir (`XacThucSuyLuanMaoHiem`)

```python
import numpy as np
from vietnamese_ai.zkml import XacThucSuyLuanMaoHiem

verifier = XacThucSuyLuanMaoHiem()
activation_in = np.random.randn(1, 128).astype(np.float32)
activation_out = np.random.randn(1, 128).astype(np.float32)

cam_ket_tensor = verifier.tao_cam_ket_zkml(activation_out, layer_id=1, node_id="worker_1")
assert XacThucSuyLuanMaoHiem.xac_thuc_cam_ket_nghiem_ngat(cam_ket_tensor, tensor=activation_out) is True

cam_ket_layer = verifier.sinh_cam_ket_layer(
    node_id="worker_1",
    layer_id=1,
    tensor_dau_vao=activation_in,
    tensor_dau_ra=activation_out,
)
assert XacThucSuyLuanMaoHiem.xac_minh_cam_ket(
    cam_ket_layer,
    tensor_dau_vao=activation_in,
    tensor_dau_ra=activation_out,
) is True
```

### C. Chạy Tự Kiểm Toán Mật Mã (`KiemToanMatMa`)

```python
from vietnamese_ai.crypto import KiemToanMatMa

bao_cao = KiemToanMatMa.chay_kiem_toan_tu_dong()
assert bao_cao["tong_quan_dat"] is True
```

---

## 3. Lệnh CLI Kiểm Tra Nhanh

```bash
vai verify --tamper-check
```

---

## 4. Giả Định An Toàn & Giới Hạn Quan Trọng

1. **Phạm vi chứng minh (Commitment vs. Full Circuit zkML)**:
   - `XacThucSuyLuanMaoHiem` chứng minh rằng nút sở hữu khóa bí mật tương ứng với `public_key` đã ký cam kết lên `input_hash`, `output_hash`, và `merkle_root` của tensor kích hoạt.
   - Nó **KHÔNG** chứng minh toán học từng phép nhân ma trận trong mạng neural bằng mạch số học (như Groth16, PLONK, Halo2, hay STARK). Nếu một nút độc hại sở hữu khóa hợp lệ cố ý chạy sai trọng số nhưng tự ký lên kết quả sai đó, cần kết hợp thêm cơ chế đối chứng lặp lại (redundant execution / spot-checking) giữa nhiều nút độc lập.
2. **Quản lý khóa & Danh tính nút**:
   - Người vận hành phải phân phối và ghim (pin) danh sách `public_key` tin cậy của các nút worker hợp pháp; việc xác minh chữ ký chỉ có giá trị nếu `public_key` nằm trong tập nút đã đăng ký.
3. **Kênh phụ thời gian trên Python**:
   - Phép lũy thừa modulo `pow()` của Python big-int không đảm bảo thời gian hằng tuyệt đối chống lại kẻ tấn công có quyền đo thời gian cục bộ độ phân giải cao.
4. **Yêu cầu xác thực nghiêm ngặt**:
   - Luôn truyền `tensor_hash` vào `xac_thuc_cam_ket(...)` / `xac_nhan_cam_ket_chuoi(...)` hoặc gọi `xac_thuc_cam_ket_nghiem_ngat(...)` / `xac_minh_cam_ket(...)` trong môi trường sản xuất để bắt buộc đối chiếu lại thách thức Fiat-Shamir `e`.

---

## 5. Checklist Sẵn Sàng Cho Kiểm Toán Độc Lập Bên Thứ Ba

- [x] Chuẩn hóa tham số nhóm mật mã theo chuẩn mở đã công bố (RFC 5114 Section 2.3: 2048-bit $p$, 256-bit prime $q$).
- [x] Loại bỏ nguồn entropy không an toàn (`time.time()`), thay bằng `secrets` (OS CSPRNG) kết hợp hedged HMAC-SHA256.
- [x] Bổ sung kiểm tra thành viên nhóm con ($y^q \equiv 1 \pmod p$, $a^q \equiv 1 \pmod p$) và miền giá trị $(e, z)$.
- [x] Bổ sung kiểm tra khớp `public_key` giữa payload và proof string trong `xac_minh_cam_ket`.
- [x] Bổ sung tách miền RFC 6962 (`0x00` / `0x01`) trong `CayMerkleSuyLuan`.
- [x] Cung cấp bộ tự kiểm toán tự động `KiemToanMatMa.chay_kiem_toan_tu_dong()`.
- [ ] Hoàn tất kiểm toán mã nguồn mật mã bởi đơn vị bảo mật độc lập (Third-Party Cryptography Audit).
- [ ] Tích hợp backend constant-time C/Rust hoặc mạch zk-SNARK/STARK thực tế cho các lớp tính toán trọng yếu (lộ trình Giai đoạn 3).
