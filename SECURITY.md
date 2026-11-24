# Security Policy — Vietnamese AI Framework

## 1. Supported Versions

| Version | Supported | Chính sách cập nhật |
| --- | --- | --- |
| `31.0.x` | :white_check_mark: | Hỗ trợ đầy đủ (tính năng, sửa lỗi, bản vá bảo mật) |
| `30.x` | :white_check_mark: | Chỉ phát hành bản vá bảo mật mức Cao / Nghiêm trọng |
| `< 30.0` | :x: | Không còn hỗ trợ |

Framework tuân thủ nghiêm ngặt **Semantic Versioning 2.0.0** (`MAJOR.MINOR.PATCH`). Mọi bản vá bảo mật tương thích ngược sẽ được phát hành ở mức `PATCH`.

---

## 2. Coordinated Vulnerability Disclosure (Quy Trình Báo Cáo Lỗ Hổng)

Nếu bạn phát hiện lỗ hổng bảo mật, vui lòng **KHÔNG** mở GitHub Issue hoặc Pull Request công khai trước khi có bản vá.

### Kênh tiếp nhận & Thông tin cần cung cấp
- **Email bảo mật**: `huynhduongphong9@gmail.com` (tiêu đề: `[SECURITY] <mô tả ngắn>`)
- **GitHub Security Advisories**: Sử dụng chức năng *Report a vulnerability* (Private Vulnerability Reporting) trên repository khi có sẵn.
- **Nội dung báo cáo**: Phiên bản `vietnamese-ai`, phiên bản Python/OS, tệp/hàm bị ảnh hưởng, các bước tái hiện (PoC tối giản), mức độ tác động và đề xuất khắc phục (nếu có).

### Cam kết thời gian phản hồi (SLA)

| Mức độ (CVSS v3.1) | Xác nhận tiếp nhận | Đánh giá & phân loại (Triage) | Mục tiêu phát hành bản vá |
| --- | --- | --- | --- |
| **Critical (9.0 – 10.0)** | ≤ 24 giờ | ≤ 48 giờ | ≤ 7 ngày |
| **High (7.0 – 8.9)** | ≤ 48 giờ | ≤ 5 ngày | ≤ 14 ngày |
| **Medium (4.0 – 6.9)** | ≤ 48 giờ | ≤ 7 ngày | ≤ 30 ngày |
| **Low (0.1 – 3.9)** | ≤ 72 giờ | ≤ 14 ngày | Bản phát hành định kỳ tiếp theo (≤ 90 ngày) |

### Phạm vi bảo mật (Scope)
- **Remote Code Execution (RCE)**: Thực thi mã tùy ý qua tool calling, sandbox escape, `eval`/`exec` hoặc nạp mô hình.
- **Path Traversal & SSRF**: Truy cập tệp ngoài thư mục cho phép hoặc gửi yêu cầu tới mạng nội bộ qua Agent tools.
- **Unsafe Deserialization**: Vượt qua `_RestrictedUnpickler` hoặc bỏ qua chữ ký HMAC-SHA256 khi nạp mô hình.
- **Cryptographic & zkML Weaknesses**: Giả mạo cam kết Schnorr/Fiat-Shamir, tấn công nhóm con nhỏ, tái sử dụng nonce, xung đột tiền ảnh Merkle Tree hoặc vượt qua xác thực Swarm TCP LAN.
- **Authentication / Authorization Bypass**: Bỏ qua kiểm tra API key, phân quyền RBAC hoặc Swarm PIN/HMAC handshake.
- **Data Leakage & DoS**: Rò rỉ PII, khóa bí mật, System Prompt hoặc tấn công làm cạn kiệt tài nguyên (memory/CPU exhaustion).

### Ngoài phạm vi (Out of Scope)
- Lỗ hổng xuất phát thuần túy từ thư viện bên thứ ba (vui lòng báo cáo đồng thời cho upstream và thông báo cho chúng tôi để cập nhật giới hạn phiên bản).
- Tấn công social engineering hoặc yêu cầu quyền truy cập vật lý trực tiếp vào máy chủ.

---

## 3. Các Lớp Bảo Mật Được Triển Khai & Bản Chất Kỹ Thuật

| Phân hệ | Cơ chế kỹ thuật triển khai | Trạng thái | Giới hạn & Yêu cầu Production |
| --- | --- | --- | --- |
| **AI Firewall & DLP** (`TuongLuaAI`, `TuongLuaBaoVeFramework`, `DataSanitizer`) | Tự phát triển (Regex, Aho-Corasick, chuẩn hóa Unicode/zero-width/Base64, quét PII/Secret) | Stable | Lớp phòng thủ chiều sâu bước đầu; không thể chặn 100% prompt injection ngữ nghĩa phức tạp. |
| **Safe Math Evaluator** (`danh_gia_bieu_thuc_toan`, `cong_cu_may_tinh`) | Phân tích cây cú pháp AST thuần với danh sách trắng toán tử & hàm toán học (`math`), **không dùng `eval()`** | Stable | Chỉ đánh giá biểu thức số học; chặn thuộc tính dunder, import và truy cập hệ thống. |
| **V-Sandbox (`ThucThiDocLap`) & Agent Sandbox (`MoiTruongCachLy`)** | Sandbox mức ứng dụng: kiểm tra AST tĩnh + tiến trình con cô lập (`python -I` / `multiprocessing`) + restricted builtins + giới hạn thời gian | Experimental / App-level | **Không thay thế cách ly cấp nhân OS/container.** Bắt buộc bọc trong container không đặc quyền trên production (xem Mục 5). |
| **zkML & Verifiable AI** (`CayMerkleSuyLuan`, `XacThucSuyLuanMaoHiem`) | Merkle Tree tách miền RFC 6962 (`0x00`/`0x01`) + Schnorr/Fiat-Shamir trên RFC 5114 (2048-bit $p$, 256-bit $q$) + OS CSPRNG (`secrets`) | Experimental (cần kiểm toán độc lập) | Chứng thực danh tính nút và tính toàn vẹn băm tensor/Merkle root; **không** thay thế full-circuit zk-SNARK/STARK. |
| **Swarm Zero-Trust LAN** (`XacThucCumSwarm`) | Challenge-response HMAC-SHA256, mã PIN ngẫu nhiên, khóa IP brute-force, session token có thời hạn | Stable | Trên mạng WAN/Internet **bắt buộc** chạy qua kênh mã hóa lớp vận chuyển (Tailscale / WireGuard / TLS). |
| **Red Teaming & Blue Team** (`RedTeamSimulator`, `RedTeamAgent`, `BlueTeamAgent`) | Bộ kịch bản mô phỏng nội bộ & khung điều phối tác tử | Experimental / Orchestration | Hỗ trợ kiểm thử hồi quy nội bộ; **không** thay thế kiểm định độc lập (Garak, PyRIT, pentest). |
| **Model Integrity** (`LuuTai`, `_RestrictedUnpickler`) | Ký/xác thực HMAC-SHA256 (`VAI_MODEL_SIGNING_KEY`) + danh sách trắng module an toàn khi giải tuần tự | Stable | Không nạp tệp `.pkl` không rõ nguồn gốc ngay cả khi bật unpickler giới hạn. |

---

## 4. Kiểm Toán Mã Nguồn Mật Mã Học (zkML Third-Party Audit Readiness)

Các module mật mã học (`vietnamese_ai/security/verifiable_inference.py`, `vietnamese_ai/crypto/audit.py`, `vietnamese_ai/edge/swarm_auth.py`) đang ở trạng thái **sẵn sàng bàn giao kiểm toán độc lập bởi chuyên gia mật mã học**.

- Tài liệu phạm vi kiểm toán chi tiết: [`docs/guides/crypto-audit-scope.md`](docs/guides/crypto-audit-scope.md).
- Chạy tự kiểm toán bất biến toán học trước khi triển khai:

```python
from vietnamese_ai.crypto import KiemToanMatMa

bao_cao = KiemToanMatMa.chay_kiem_toan_tu_dong()
assert bao_cao["tong_quan_dat"] is True
```

- Khi xác thực cam kết trên production, **luôn** gọi `XacThucSuyLuanMaoHiem.xac_thuc_cam_ket_nghiem_ngat(cam_ket, tensor=...)` hoặc `XacThucSuyLuanMaoHiem.xac_minh_cam_ket(...)` và đối chiếu `public_key` với danh sách khóa công khai tin cậy đã đăng ký.

---

## 5. Khuyến Nghị Triển Khai An Toàn Cho Người Dùng (Production Hardening)

### A. Tuyệt Đối Không Dùng `eval()` / `exec()` Trong Tác Tử

```python
from vietnamese_ai.agents import TacTu, cong_cu_may_tinh, danh_gia_bieu_thuc_toan

ket_qua = danh_gia_bieu_thuc_toan("15 * 8 + sqrt(16)")
tac_tu = TacTu(
    llm=llm,
    danh_sach_cong_cu=[cong_cu_may_tinh],
    bat_tuong_lua=True,
    loc_thong_tin_nhay_cam=True,
)
```

### B. Sandbox Hóa Tác Tử Tự Thực Thi Mã (Defense-in-Depth)

Khi bắt buộc cho phép tác tử chạy mã Python sinh động trên môi trường nhiều người dùng:
1. **Tầng ứng dụng**: Dùng `ThucThiDocLap(timeout_giay=5)` hoặc `MoiTruongCachLy.thuc_thi(..., timeout_giay=5)` kết hợp `bat_tuong_lua=True` và Human-in-the-Loop (`yeu_cau_xac_nhan=True` + `ham_xac_nhan`).
2. **Tầng hệ điều hành / Container (Bắt buộc cho production)**: Chạy worker thực thi mã bên trong container không đặc quyền, tắt mạng và giới hạn chặt tài nguyên:

```bash
docker run --rm \
  --network=none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --pids-limit=64 \
  --memory=256m \
  --cpus=0.5 \
  --user=65534:65534 \
  vietnamese-ai-worker:31.0.2
```

3. Đối với mức cô lập cao nhất trên hạ tầng đám mây đa khách hàng, sử dụng runtime microVM (`gVisor` / `Kata Containers` / `Firecracker`).

### C. Bảo Mật Chuỗi Cung Ứng & Ký Số Mô Hình

- Dùng `requirements.txt` cho production (chỉ phụ thuộc `numpy`) và tách riêng `requirements-dev.txt` cho phát triển.
- Kiểm tra tự động không xung đột phụ thuộc và đồng bộ SemVer trong CI/CD:

```python
from vietnamese_ai.utils import QuanLyPhatHanhGoi, QuanLyPhuThuoc

assert QuanLyPhuThuoc.kiem_tra_xung_dot_phu_thuoc()["hop_le"] is True
assert QuanLyPhatHanhGoi.kiem_tra_tuan_thu_semver()["dong_bo_phien_ban"] is True
```

- Đặt khóa bí mật qua biến môi trường (`VAI_API_KEY`, `VAI_MODEL_SIGNING_KEY`, `VAI_SWARM_PIN`), tuyệt đối không hardcode vào mã nguồn.

```python
from vietnamese_ai.utils.io_utils import LuuTai, tai_an_toan

LuuTai.ky_file("model.pkl")
if LuuTai.xac_thuc_file("model.pkl"):
    mo_hinh = tai_an_toan("model.pkl", cho_phep_mo_hinh=True)
```

---

## 6. External Toolchain Requirements (Yêu Cầu Công Cụ Ngoài)

- **C++ JIT & Kernel Fusion (`EvoJITCompiler`, `EvoKernelCPU`, `FusedDenseNormReLU`)**: Bắt buộc `g++` / `clang++` hỗ trợ C++17 (tùy chọn `-fopenmp`).
- **Rust Core Acceleration (`rust_core/`)**: Cần `rustc` + `cargo` + `maturin` khi tự biên dịch `vietnamese_ai_rust.so`.
- **Native llama.cpp Bridge (`CauNoiCtypesNative`)**: Cần `libllama.so` / `libllama.dylib` / `llama.dll` hoặc binary `llama-cli`.
- **Mobile Native Deployment (`TriKhaiDiDong`, `SDKGenerator`)**: Cần `tensorflow`/`ai-edge-litert` + Android SDK/NDK (`.tflite`), `coremltools` + Xcode trên macOS (`.mlmodel`), ExecuTorch (`.pte`), Flutter SDK / React Native CLI.
- **Vision OCR (`TrichXuatVanBanAnh`)**: Cần binary `tesseract-ocr` hệ thống khi dùng `pytesseract`.
- **GPU Acceleration (`UnslothWrapper`, `HuanLuyenPyTorch`)**: Cần NVIDIA CUDA Toolkit + PyTorch CUDA tương thích.

## Acknowledgments

Chúng tôi trân trọng cảm ơn cộng đồng và các nhà nghiên cứu bảo mật đã đóng góp giúp nâng cao mức độ an toàn của framework.
