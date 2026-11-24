# Edge Inference & Swarm CPU API Reference (`vietnamese_ai.edge`)

Gói suy luận biên Zero-GPU tải lười chuẩn PEP 562 (cũng tái xuất phần lõi tại `vietnamese_ai.core.edge`).

## `BoGiaiLuongTuQ4` / `BoGiaiLuongTuGGUF`

```python
class BoGiaiLuongTuQ4:
    @staticmethod
    def giai_q8_0(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray
    @staticmethod
    def giai_q4_0(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray
    @staticmethod
    def giai_q4_1(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray
    @staticmethod
    def giai_q2_k(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray
    @staticmethod
    def giai_q3_k(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray
    @staticmethod
    def giai_q4_k(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray
    @staticmethod
    def giai_f16(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray
    @staticmethod
    def giai_f32(raw_bytes: bytes, so_phan_tu: int) -> np.ndarray
    @classmethod
    def giai_luong_tu(raw_bytes: bytes, so_phan_tu: int, loai_quant: str) -> np.ndarray
    @classmethod
    def nhan_ma_tran_luong_tu(
        x: np.ndarray,
        raw_bytes: bytes,
        shape_w: Tuple[int, int],
        loai_quant: str = "q4_0",
        kich_thuoc_tile: int = 64,
    ) -> np.ndarray
```

## `GGUFMemoryMappedStreamer`, `BoTachTuBPEGGUF` & `BoGiaiMaTransformerEdge`

```python
class GGUFMemoryMappedStreamer:
    def __init__(duong_dan_tep: Union[str, Path]) -> None
    def danh_sach_tensor() -> List[str]
    def doc_tensor_tho(ten_tensor: str) -> Optional[Dict[str, Any]]
    def doc_tensor(ten_tensor: str) -> Optional[np.ndarray]
    def lay_thong_tin() -> Dict[str, Any]
    def close() -> None

class BoTachTuBPEGGUF:
    @classmethod
    def tu_streamer(streamer: GGUFMemoryMappedStreamer) -> Optional[BoTachTuBPEGGUF]
    def ma_hoa(van_ban: str) -> List[int]
    def giai_ma(ids: List[int]) -> str

class BoGiaiMaTransformerEdge:
    def __init__(d_model: int = 128, so_dau: int = 4, d_ff: int = 256, seed: int = 42, so_dau_kv: Optional[int] = None, so_lop: int = 1) -> None
    def nap_trong_so_tu_gguf(streamer: GGUFMemoryMappedStreamer, giu_luong_tu: bool = True) -> int
    def buoc_suy_luan(x_step: np.ndarray, vi_tri: int = 0) -> np.ndarray
    def sinh_tu_gguf_bpe(prompt: str, streamer: GGUFMemoryMappedStreamer, so_token_toi_da: int = 16, nhiet_do: float = 0.0) -> Dict[str, Any]
    def sinh_stream_tu_gguf_bpe(prompt: str, streamer: GGUFMemoryMappedStreamer, so_token_toi_da: int = 16, nhiet_do: float = 0.0)
```

Hỗ trợ nhận diện đa kiến trúc LLM (`KIEN_TRUC_LLM_HO_TRO`): `llama`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `mistral`, `phi3`, `deepseek2`, `gpt2`, `vietnamese_ai`. Khi nạp tệp GGUF v3 chứa `tokenizer.ggml.tokens` và `token_embd.weight`, `DongCoSuyLuanEdge` tự động căn chỉnh kiến trúc `BoGiaiMaTransformerEdge` và phát trực tiếp luồng token BPE.

## `DongCoSuyLuanEdge`, `CauNoiCtypesNative` & `QuanLyMoHinhEdge`

```python
class QuanLyMoHinhEdge:
    def __init__(thu_muc_luu: Optional[Union[str, Path]] = None) -> None
    def chon_muc_luong_tu_theo_ram(ten_mo_hinh: str, ram_kha_dung_gb: Optional[float] = None) -> Dict[str, Any]
    def goi_y_mo_hinh_theo_may(ram_kha_dung_gb: Optional[float] = None) -> List[Dict[str, Any]]
    def tai_mo_hinh(ten_hoac_nguon: str, muc_luong_tu: str = "auto", chay_that: bool = True, sha256_mong_doi: Optional[str] = None, ...) -> Dict[str, Any]
    def danh_sach_da_tai() -> List[Dict[str, Any]]
    def tim_duong_dan_mo_hinh(ten_hoac_duong_dan: Union[str, Path]) -> Optional[Path]
    def xoa_mo_hinh(khoa_hoac_ten: str) -> bool

def do_ram_he_thong_gb() -> Dict[str, Any]
def dinh_dang_prompt_chat(cau_hoi: str, he_thong: str = ..., chat_template: str = "chatml", ngu_canh: Optional[str] = None) -> str

class DongCoSuyLuanEdge:
    def __init__(
        duong_dan_mo_hinh: Optional[Union[str, Path]] = None,
        so_luong_thread: int = 4,
        nguong_an_toan: bool = True,
        do_dai_toi_da: int = 256,
        nhiet_do: float = 0.7,
        uu_tien_native_c: bool = False,
        n_ctx: int = 4096,
        chat_template: Optional[str] = None,
        rag_pipeline: Optional[Any] = None,
    ) -> None
    def nap_tai_lieu_rag(duong_dan: Union[str, Path], che_do_hanh_chinh: bool = True, toi_uu_ivf: bool = False) -> Dict[str, Any]
    def hoi_kem_rag(cau_hoi: str, top_k: int = 3, do_dai_toi_da: Optional[int] = None, nhiet_do: Optional[float] = None) -> Dict[str, Any]
    def sinh_stream(prompt: str, do_dai_toi_da: Optional[int] = None, nhiet_do: Optional[float] = None)
    def sinh_van_ban(prompt: str, do_dai_toi_da: Optional[int] = None, nhiet_do: Optional[float] = None) -> str
    def sinh_kem_thong_ke(prompt: str, do_dai_toi_da: Optional[int] = None, nhiet_do: Optional[float] = None) -> Dict[str, Any]
    def du_doan(X: Any, **kwargs: Any) -> Any
```

`CauNoiCtypesNative` tự động chọn tầng thực thi tối ưu: (1) `llama-cpp-python` in-process mmap + streaming (`pip install "vietnamese-ai[local-llm]"`), (2) binary `llama-cli`, (3) `ctypes` `libllama.so` hoặc fallback `DongCoSuyLuanEdge`.
CLI: `vai pull --list`, `vai pull qwen2.5-3b-q4`, `vai run qwen2.5-3b-q4 --rag nghi_dinh_13.docx --admin --prompt "..."`.

## Kiến trúc CPU & Toolchain Build Native (`vietnamese_ai.edge.native_build`)

```python
def thong_tin_kien_truc_cpu() -> Dict[str, Any]

class KiemTraMoiTruongNative:
    @staticmethod
    def kiem_tra_toolchain() -> Dict[str, Any]
    @staticmethod
    def cau_hinh_setuptools_extension(
        ten_extension: str,
        cac_tep_nguon: List[str],
        bat_openmp: bool = True,
        duong_dan_include: Optional[List[str]] = None,
    ) -> Dict[str, Any]
```

## Cụm Swarm CPU, TLS/mTLS & TCP LAN (`NutSwarmCPU`, `DieuPhoiSwarmCPU`)

```python
def tao_ssl_context_server(certfile: str, keyfile: str, cafile: Optional[str] = None, yeu_cau_mtls: bool = False) -> ssl.SSLContext
def tao_ssl_context_client(cafile: Optional[str] = None, certfile: Optional[str] = None, keyfile: Optional[str] = None, kiem_tra_hostname: bool = True) -> ssl.SSLContext

class NutSwarmCPU:
    def nap_trong_so_gguf(duong_dan_gguf: str) -> Dict[str, Any]
    def xuat_phan_manh_gguf(duong_dan_xuat: str, quantization: str = "q8_0") -> str
    def khoi_dong_server_lan(host: str = "127.0.0.1", port: int = 8890, ma_pin: Optional[str] = None, ssl_context: Optional[ssl.SSLContext] = None, ...) -> None
    def dung_server_lan() -> None
    def lan_truyen_tien(hidden_states: np.ndarray, buoc: int = 0, tao_proof: bool = True)

class DieuPhoiSwarmCPU:
    def khoi_tao_cum_gia_lap(so_node: int = 4, so_core_moi_node: int = 4, ...) -> List[NutSwarmCPU]
    def them_node_mang_vat_ly(node_id: str, endpoint_url: str, dai_layer: Tuple[int, int], ssl_context: Optional[ssl.SSLContext] = None, ...) -> NutSwarmCPU
    def lan_truyen_pipeline(hidden_states: np.ndarray, tao_proof: Optional[bool] = None)
    def suy_luan_swarm(prompt: str, so_token_sinh: int = 16, xac_minh_proof: bool = True) -> Dict[str, Any]
```

Hỗ trợ CLI trực tiếp: `vai swarm --host --tls-cert server.pem --tls-key server.key --tls-ca ca.pem --mtls`.
