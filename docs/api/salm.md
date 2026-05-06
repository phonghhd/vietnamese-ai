# API Reference: Self-Adapting Language Models (SALM)

## SelfRefine

```python
class SelfRefine:
    """Tự cải thiện output qua nhiều vòng lặp."""

    def __init__(
        self,
        ham_sinh: Callable[[str], str],
        ham_danh_gia: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        ham_feedback: Optional[Callable[[str, str], str]] = None,
        ham_refine: Optional[Callable[[str, str], str]] = None,
        so_vong_toi_da: int = 3,
        nguong_chat_luong: float = 0.8,
        giam_diem_dung: float = 0.01,
    )

    def chay(self, prompt: str, nhiet_do: float = 0.7) -> Dict[str, Any]
    def chay_nhieu(self, prompts: List[str], nhiet_do: float = 0.7) -> List[Dict[str, Any]]
    def lay_lich_su(self) -> List[Dict[str, Any]]
    def thong_ke(self) -> Dict[str, Any]
```

### Returns: `chay()`

| Key | Type | Mô tả |
|---|---|---|
| `output_cuoi` | `str` | Output cuối cùng |
| `lich_su_vong` | `List[Dict]` | Lịch sử mỗi vòng |
| `diem_cuoi` | `float` | Điểm cuối cùng |
| `so_vong` | `int` | Số vòng đã chạy |
| `tong_thoi_gian` | `float` | Thời gian (giây) |
| `dat_nguong` | `bool` | Có đạt ngưỡng không |

---

## SelfConsistency

```python
class SelfConsistency:
    """Multiple reasoning paths với majority voting."""

    def __init__(
        self,
        ham_sinh: Callable[[str], str],
        so_luong: int = 5,
        ham_trich_xuat: Optional[Callable[[str], str]] = None,
        ham_danh_gia: Optional[Callable[[str, str, str], float]] = None,
    )

    def chay(self, prompt: str, che_do: str = "truc_tiep", nhiet_do: float = 0.7) -> Dict[str, Any]
    def chay_nhieu(self, prompts: List[str], che_do: str = "truc_tiep") -> List[Dict[str, Any]]
    def lay_lich_su(self) -> List[Dict[str, Any]]
    def thong_ke(self) -> Dict[str, Any]
```

### Returns: `chay()`

| Key | Type | Mô tả |
|---|---|---|
| `dap_an` | `str` | Đáp án (majority vote) |
| `so_luong_paths` | `int` | Số paths đã tạo |
| `ty_le_dong_nhat` | `float` | Tỷ lệ paths cùng đáp án |
| `cac_paths` | `List[Dict]` | Chi tiết từng path |
| `phan_phoi` | `Dict` | Phân phối đáp án |
| `diem` | `float` | Điểm tin cậy |

---

## AdaptiveLoRA

```python
class AdaptiveLoRA:
    """Tự chọn LoRA adapter theo task."""

    def __init__(self, che_do: str = "keyword", trong_so_mac_dinh: float = 1.0)

    def dang_ky_adapter(self, ten: str, adapter: Any, keywords: Optional[List[str]] = None, embedding: Optional[np.ndarray] = None, trong_so: float = 1.0) -> None
    def dang_ky_ham_embed(self, ham: Callable[[str], np.ndarray]) -> None
    def chon_adapter(self, input_text: str, top_k: int = 1) -> List[Dict[str, Any]]
    def ket_hop_adapters(self, input_text: str) -> Dict[str, float]
    def xoa_adapter(self, ten: str) -> bool
    def danh_sach_adapters(self) -> List[str]
    def thong_ke_su_dung(self) -> Dict[str, int]
    def thong_ke(self) -> Dict[str, Any]
```

### Parameters: `dang_ky_adapter()`

| Param | Type | Mô tả |
|---|---|---|
| `ten` | `str` | Tên adapter |
| `adapter` | `Any` | LoRA adapter object |
| `keywords` | `List[str]` | Từ khóa liên quan |
| `embedding` | `np.ndarray` | Vector biểu diễn task |
| `trong_so` | `float` | Trọng số ưu tiên |

---

## SinhDuLieuTuDong

```python
class SinhDuLieuTuDong:
    """Sinh dữ liệu huấn luyện tự động."""

    def __init__(self, ham_sinh: Callable[[str], str], ham_danh_gia: Optional[Callable] = None, nguong_chat_luong: float = 0.5, toi_da_lap: int = 3)

    def them_giong_mau(self, instruction: str, output: str, input_text: str = "") -> None
    def them_nhieu_giong_mau(self, mau_list: List[Dict[str, str]]) -> None
    def sinh(self, so_luong: int, loai: str = "instruction", chu_de: str = "") -> List[Dict[str, str]]
    def lay_du_lieu_da_sinh(self) -> List[Dict[str, str]]
    def xoa_du_lieu(self) -> None
    def thong_ke(self) -> Dict[str, Any]
```

### Parameters: `sinh()`

| Param | Type | Mô tả |
|---|---|---|
| `so_luong` | `int` | Số mẫu cần sinh |
| `loai` | `str` | `instruction`, `qa`, `classification`, `completion` |
| `chu_de` | `str` | Chủ đề cụ thể |

---

## TestTimeTraining

```python
class TestTimeTraining:
    """Adapt model tại inference time."""

    def __init__(self, model: Any, che_do: str = "entropy_minimization", toc_do_hoc: float = 0.001, so_buoc_mac_dinh: int = 5)

    def luu_trong_so_goc(self) -> None
    def phuc_hoi_trong_so(self) -> None
    def thich_ung(self, X: np.ndarray, so_buoc: Optional[int] = None) -> Dict[str, Any]
    def du_doan(self, X: np.ndarray) -> np.ndarray
    def du_doan_xac_suat(self, X: np.ndarray) -> np.ndarray
    def thong_ke(self) -> Dict[str, Any]
```

### Parameters: `__init__()`

| Param | Type | Default | Mô tả |
|---|---|---|---|
| `model` | `Any` | — | Model cần adapt |
| `che_do` | `str` | `entropy_minimization` | Chiến lược TTT |
| `toc_do_hoc` | `float` | `0.001` | Learning rate |
| `so_buoc_mac_dinh` | `int` | `5` | Số bước adapt |

### Returns: `thich_ung()`

| Key | Type | Mô tả |
|---|---|---|
| `loss_cuoi` | `float` | Loss cuối cùng |
| `loss_dau` | `float` | Loss ban đầu |
| `giam_loss` | `float` | Mức giảm loss |
| `so_buoc` | `int` | Số bước đã chạy |
| `lich_su_loss` | `List[float]` | Loss theo từng bước |
