# Crypto & zkML API Reference (`vietnamese_ai.crypto` / `vietnamese_ai.zkml`)

Gói mật mã học xác thực suy luận (tải lười PEP 562), triển khai nhóm Schnorr chuẩn **RFC 5114 Section 2.3 (MODP-2048-256)** và Merkle Tree tách miền RFC 6962 (`0x00`/`0x01`).

> [!IMPORTANT]
> Gắn nhãn `EXPERIMENTAL`: khuyến nghị kiểm toán mật mã độc lập bởi bên thứ ba trước khi triển khai cho hệ thống trọng yếu. Xem thêm [Verifiable AI Guide](../guides/verifiable-ai.md) và [Crypto Audit Scope](../guides/crypto-audit-scope.md).

## `CayMerkleSuyLuan`

```python
class CayMerkleSuyLuan:
    def __init__(danh_sach_leaf_data: Optional[Sequence[Union[str, bytes, np.ndarray]]] = None) -> None
    @staticmethod
    def bam_du_lieu(du_lieu: Union[str, bytes, np.ndarray]) -> str
    def them_la(du_lieu: Union[str, bytes, np.ndarray]) -> str
    def xay_dung_cay() -> str
    def lay_goc_merkle() -> str
    def lay_root_hash() -> str
    def sinh_merkle_proof(index_la: int) -> Dict[str, Any]
    def sinh_bang_chung(index_la: int) -> List[Dict[str, str]]
    def kiem_tra_bang_chung(
        du_lieu: Union[str, bytes, np.ndarray],
        index_la: int,
        proof_path: List[Dict[str, str]],
        root: str,
    ) -> bool
    @classmethod
    def xac_minh_merkle_proof(
        root_ky_vong: str,
        leaf_hash: str,
        proof_path: List[Dict[str, str]],
    ) -> bool
```

Alias: `InferenceMerkleTree`.

## `XacThucSuyLuanMaoHiem`

```python
class XacThucSuyLuanMaoHiem:
    def __init__(private_key: Optional[str] = None) -> None
    @property
    def khoa_bi_mat() -> int
    @property
    def khoa_cong_khai() -> int
    @staticmethod
    def bam_tensor(tensor: np.ndarray) -> str
    def tao_cam_ket_zkml(tensor: np.ndarray, layer_id: int = 0, node_id: str = "node_local") -> Dict[str, Any]
    @classmethod
    def xac_thuc_cam_ket(y: int, a: int, e: int, z: int, tensor_hash: Optional[str] = None) -> bool
    @classmethod
    def xac_nhan_cam_ket_chuoi(chuoi_cam_ket: str, tensor_hash: Optional[str] = None) -> bool
    @classmethod
    def xac_thuc_cam_ket_nghiem_ngat(cam_ket: Dict[str, Any], tensor: Optional[np.ndarray] = None) -> bool
    def ky_ket_qua(thong_diep: str) -> str
    def xac_thuc_chu_ky(thong_diep: str, chu_ky: str) -> bool
    def sinh_cam_ket_layer(
        node_id: str,
        layer_id: int,
        tensor_dau_vao: np.ndarray,
        tensor_dau_ra: np.ndarray,
    ) -> Dict[str, Any]
    @classmethod
    def xac_minh_cam_ket(
        cam_ket: Dict[str, Any],
        tensor_dau_vao: Optional[np.ndarray] = None,
        tensor_dau_ra: Optional[np.ndarray] = None,
    ) -> bool
```

Alias: `VerifiableInference`. Hằng số tham số nhóm: `THONG_SO_NHOM_SCHNORR`.

## `KiemToanMatMa`

```python
class KiemToanMatMa:
    @staticmethod
    def kiem_tra_tham_so_nhom() -> Dict[str, Any]
    @staticmethod
    def kiem_tra_chong_tan_cong_nhom_con_nho() -> Dict[str, Any]
    @staticmethod
    def kiem_tra_chong_gia_mao_fiat_shamir() -> Dict[str, Any]
    @staticmethod
    def kiem_tra_tach_mien_merkle() -> Dict[str, Any]
    @classmethod
    def chay_kiem_toan_tu_dong() -> Dict[str, Any]
```

Alias: `CryptoAudit`.
