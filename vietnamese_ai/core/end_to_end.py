"""QuyTrinhDauCuoi (HeSinhThaiAI / EndToEndWorkflow) - Cầu nối đầu-cuối hợp nhất toàn bộ Vietnamese AI Framework."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.core.engine import Engine
from vietnamese_ai.core.pipeline import Pipeline
from vietnamese_ai.datalake.dataset_manager import HoDuLieu
from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem
from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh
from vietnamese_ai.production.health import KiemTraSucKhoe
from vietnamese_ai.production.metrics import QuanLyMetrics
from vietnamese_ai.prompts.guardrails import LuongAnToan
from vietnamese_ai.rag.rag_pipeline import RAGPipeline
from vietnamese_ai.registry.model_registry import QuanLyMoHinh
from vietnamese_ai.serving.batch_server import MayChuBatch
from vietnamese_ai.utils.logger import Logger


class QuyTrinhDauCuoi:
    """
    Cầu nối Đầu-Cuối (End-to-End Orchestrator) hợp nhất mọi phân hệ của Vietnamese AI Framework.

    Kết nối liền mạch:
    1. Dữ liệu & Phiên bản hóa (`HoDuLieu` - DataLake DVC)
    2. Tri thức & Truy xuất ngữ nghĩa (`RAGPipeline` - Hybrid Search & Reranking)
    3. Huấn luyện & Tinh chỉnh đa năng (`Engine` + `BoTinhChinhMoHinh` + `Pipeline` + `TheoDoiThiNghiem`)
    4. Quản lý phiên bản & Xuất bản (`QuanLyMoHinh` + `XuatGGUF` + `XuatONNX`)
    5. Tác tử thông minh & Kiểm duyệt an toàn (`TacTu` + `CongCu` + `LuongAnToan`)
    6. Phục vụ & Giám sát Production (`MayChuBatch` + `KiemTraSucKhoe` + `QuanLyMetrics`)
    """

    def __init__(
        self,
        ten_du_an: str = "VietnameseAI_E2E",
        thu_muc_lam_viec: str = "vai_workspace",
        tu_cam: Optional[List[str]] = None,
        chan_pii: bool = True,
        bat_tuong_lua: bool = True,
    ):
        from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

        self.ten_du_an = ten_du_an
        self.thu_muc = Path(thu_muc_lam_viec)
        self.thu_muc.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(ten_du_an)

        self.ho_du_lieu = HoDuLieu(duong_dan_luu_tru=str(self.thu_muc / "datalake"))
        self.theo_doi = TheoDoiThiNghiem(
            ten_thu_nghiem=ten_du_an,
            luu_tai=str(self.thu_muc / "experiments"),
        )
        self.quan_ly_mo_hinh = QuanLyMoHinh(duong_dan=str(self.thu_muc / "registry"))
        self.engine = Engine(
            ten=ten_du_an,
            theo_doi_thi_nghiem=self.theo_doi,
            quan_ly_mo_hinh=self.quan_ly_mo_hinh,
        )
        self.luong_an_toan = LuongAnToan(tu_cam=tu_cam or [], chan_pii=chan_pii)
        self.bat_tuong_lua = bat_tuong_lua
        self.tuong_lua = (
            TuongLuaBaoVeFramework(
                tu_dong_an_pii_dau_ra=chan_pii, gioi_han_req_moi_phut=10000
            )
            if bat_tuong_lua
            else None
        )
        self.metrics = QuanLyMetrics()
        self.suc_khoe = KiemTraSucKhoe()

        self.rag: Optional[RAGPipeline] = None
        self.mo_hinh_hien_tai: Optional[Any] = None
        self.tac_tu: Optional[TacTu] = None
        self.may_chu_batch: Optional[MayChuBatch] = None
        self._dataset_hien_tai: Optional[str] = None

    def nap_du_lieu(
        self,
        nguon_du_lieu: Any,
        ten_dataset: str = "train_data.jsonl",
        tin_nhan_commit: str = "Nạp dữ liệu vào quy trình đầu-cuối",
    ) -> Dict[str, Any]:
        """
        Bước 1: Nạp dữ liệu vào `HoDuLieu` (DataLake) và tạo mã băm phiên bản SHA-256.
        Chấp nhận đường dẫn file (str/Path), danh sách bản ghi (`List[Dict]`), hoặc danh sách câu (`List[str]`).
        """
        if not ten_dataset.endswith((".jsonl", ".csv", ".json")):
            ten_dataset = f"{ten_dataset}.jsonl"

        if isinstance(nguon_du_lieu, (str, Path)) and Path(nguon_du_lieu).exists():
            ma_bam = self.ho_du_lieu.nhap_du_lieu(
                str(nguon_du_lieu), ten_dataset, tin_nhan_commit=tin_nhan_commit
            )
            so_ban_ghi = sum(len(b) for b in self.ho_du_lieu.doc_luong_du_lieu(ten_dataset))
        elif isinstance(nguon_du_lieu, list):
            records: List[Dict[str, Any]] = []
            for item in nguon_du_lieu:
                if isinstance(item, dict):
                    records.append(item)
                else:
                    records.append({"text": str(item)})
            ma_bam = self.ho_du_lieu.nhap_danh_sach(
                records, ten_dataset, tin_nhan_commit=tin_nhan_commit
            )
            so_ban_ghi = len(records)
        else:
            raise ValueError("nguon_du_lieu phải là đường dẫn file hợp lệ hoặc danh sách dữ liệu.")

        self._dataset_hien_tai = ten_dataset
        return {
            "ten_dataset": ten_dataset,
            "ma_bam_sha256": ma_bam,
            "so_ban_ghi": so_ban_ghi,
        }

    def xay_dung_rag(
        self,
        tai_lieu: Optional[Any] = None,
        ten_dataset: Optional[str] = None,
        truong_van_ban: str = "text",
    ) -> Dict[str, Any]:
        """
        Bước 2: Khởi tạo và lập chỉ mục tri thức cho `RAGPipeline` từ DataLake hoặc danh sách tài liệu.
        """
        ham_embed = None
        if isinstance(self.mo_hinh_hien_tai, BoTinhChinhMoHinh) and self.mo_hinh_hien_tai._hf_wrapper is not None:
            ham_embed = self.mo_hinh_hien_tai._hf_wrapper

        if self.rag is None:
            self.rag = RAGPipeline(ham_embed=ham_embed)

        so_tai_lieu = 0
        if tai_lieu is not None:
            if isinstance(tai_lieu, list):
                for idx, doc in enumerate(tai_lieu):
                    if isinstance(doc, dict):
                        text = str(
                            doc.get(
                                truong_van_ban,
                                doc.get("output", doc.get("instruction", doc.get("van_ban", ""))),
                            )
                        )
                        doc_id = str(doc.get("id", f"doc_{idx}"))
                    else:
                        text = str(doc)
                        doc_id = f"doc_{idx}"
                    if text.strip():
                        self.rag.them_tai_lieu(doc_id, text)
                        so_tai_lieu += 1
        ds_name = ten_dataset or self._dataset_hien_tai
        if ds_name and tai_lieu is None:
            so_tai_lieu += self.ho_du_lieu.nap_vao_rag(
                ds_name, self.rag, truong_van_ban=truong_van_ban
            )

        return {
            "so_tai_lieu_da_nap": so_tai_lieu,
            "trang_thai": "san_sang",
        }

    def huan_luyen_mo_hinh(
        self,
        mo_hinh: Any = "vgpt2",
        nhiem_vu: str = "sft",
        du_lieu_train: Optional[Any] = None,
        dung_lora: bool = True,
        rank_lora: int = 8,
        so_vong: int = 3,
        toc_do_hoc: float = 2e-3,
        ten_dang_ky: Optional[str] = None,
        promote_production: bool = True,
        che_do_offline: bool = True,
    ) -> Dict[str, Any]:
        """
        Bước 3: Tinh chỉnh / Huấn luyện mô hình qua `BoTinhChinhMoHinh` và `Engine`,
        tự động ghi nhận vào `TheoDoiThiNghiem` và đăng ký vào `QuanLyMoHinh`.
        """
        if du_lieu_train is None:
            if not self._dataset_hien_tai:
                raise RuntimeError("Chưa có dữ liệu huấn luyện. Hãy gọi nap_du_lieu() hoặc truyền du_lieu_train.")
            du_lieu_train = []
            for batch in self.ho_du_lieu.doc_luong_du_lieu(self._dataset_hien_tai):
                du_lieu_train.extend(batch)

        if isinstance(mo_hinh, BoTinhChinhMoHinh):
            tuner = mo_hinh
        else:
            tuner = BoTinhChinhMoHinh(
                mo_hinh=mo_hinh,
                nhiem_vu=nhiem_vu,
                dung_lora=dung_lora,
                rank_lora=rank_lora,
                che_do_offline=che_do_offline,
            )

        ten_model_reg = ten_dang_ky or f"{self.ten_du_an}_{nhiem_vu}"
        ket_qua_engine = self.engine.huan_luyen(
            tuner,
            du_lieu_train,
            ten_mo_hinh=ten_model_reg,
            dang_ky_registry=True,
            so_vong=so_vong,
            toc_do_hoc=toc_do_hoc,
        )

        if promote_production and "version_registry" in ket_qua_engine:
            self.quan_ly_mo_hinh.promote(
                ten_model_reg, ket_qua_engine["version_registry"], "production"
            )

        self.mo_hinh_hien_tai = tuner
        if self.rag is not None and tuner._hf_wrapper is not None:
            self.rag.ham_sinh = lambda cau_hoi, ngu_canh: tuner.sinh_van_ban(cau_hoi)

        return ket_qua_engine

    def tao_tac_tu(
        self,
        danh_sach_cong_cu: Optional[List[CongCu]] = None,
        max_iterations: int = 3,
    ) -> TacTu:
        """
        Bước 4: Khởi tạo `TacTu` (Agent) kết nối trực tiếp với mô hình đã tinh chỉnh và `RAGPipeline`.
        """
        if self.mo_hinh_hien_tai is None:
            self.mo_hinh_hien_tai = BoTinhChinhMoHinh(
                mo_hinh="vgpt2", nhiem_vu="sft", dung_lora=True, che_do_offline=True
            )

        cong_cu_list = list(danh_sach_cong_cu or [])
        if self.rag is not None:
            rag_ref = self.rag

            def _tra_cuu_rag(truy_van: str = "") -> str:
                kq = rag_ref.hoi(truy_van)
                return str(kq.get("tra_loi", ""))

            cong_cu_list.append(
                CongCu(
                    ten="tra_cuu_tri_thuc",
                    mo_ta="Tra cứu thông tin từ cơ sở tri thức RAG",
                    ham_thuc_thi=_tra_cuu_rag,
                )
            )

        self.tac_tu = TacTu(
            llm=self.mo_hinh_hien_tai,
            danh_sach_cong_cu=cong_cu_list,
            max_iterations=max_iterations,
        )
        return self.tac_tu

    def hoi(self, cau_hoi: str, dung_rag: bool = True, dung_agent: bool = False) -> Dict[str, Any]:
        """
        Bước 5: Truy vấn đầu-cuối an toàn (Tường lửa AI + Guardrails đầu vào -> RAG / Fine-Tuned Model / Agent -> DLP đầu ra -> Metrics).
        """
        bat_dau = time.time()
        if self.tuong_lua is not None:
            kq_fw = self.tuong_lua.kiem_tra_dau_vao(str(cau_hoi))
            if not kq_fw["an_toan"]:
                return {
                    "cau_hoi": cau_hoi,
                    "tra_loi": f"[Bị chặn bởi Tường lửa AI]: {kq_fw['ly_do']}",
                    "an_toan": False,
                    "ly_do": kq_fw["ly_do"],
                }

        kt_vao = self.luong_an_toan.kiem_tra(cau_hoi)
        if not kt_vao["an_toan"]:
            return {
                "cau_hoi": cau_hoi,
                "tra_loi": "Yêu cầu bị từ chối bởi hệ thống kiểm duyệt an toàn (Guardrails).",
                "an_toan": False,
                "ly_do": kt_vao["loi"],
            }

        nguon_rag = []
        if dung_rag and self.rag is not None:
            kq_rag = self.rag.hoi(cau_hoi)
            nguon_rag = kq_rag.get("nguon", [])

        if dung_agent and self.tac_tu is not None:
            tra_loi = self.tac_tu.chay(cau_hoi)
        elif self.mo_hinh_hien_tai is not None:
            if getattr(self.mo_hinh_hien_tai, "nhiem_vu", "") == "text-classification":
                du_doan_cls = self.mo_hinh_hien_tai.du_doan(cau_hoi)
                tra_loi = (
                    str(du_doan_cls.get("nhan", du_doan_cls))
                    if isinstance(du_doan_cls, dict)
                    else str(du_doan_cls)
                )
            else:
                tra_loi = self.mo_hinh_hien_tai.sinh_van_ban(cau_hoi)
        elif self.rag is not None:
            tra_loi = str(self.rag.hoi(cau_hoi).get("tra_loi", ""))
        else:
            tra_loi = cau_hoi

        if self.tuong_lua is not None:
            tra_loi = self.tuong_lua.loc_dau_ra(str(tra_loi))

        kt_ra = self.luong_an_toan.kiem_tra(str(tra_loi))
        if not kt_ra["an_toan"]:
            tra_loi = self.luong_an_toan.kiem_tra_va_sua(str(tra_loi))

        latency_ms = (time.time() - bat_dau) * 1000.0
        if hasattr(self.metrics, "ghi_nhan_request"):
            self.metrics.ghi_nhan_request(latency_ms=latency_ms, thanh_cong=True)

        return {
            "cau_hoi": cau_hoi,
            "tra_loi": tra_loi,
            "nguon_rag": nguon_rag,
            "an_toan": kt_ra["an_toan"],
            "latency_ms": round(latency_ms, 2),
        }

    def xuat_mo_hinh(
        self,
        dinh_dang: str = "gguf",
        duong_dan: Optional[str] = None,
        quantization: str = "q8_0",
    ) -> Dict[str, Any]:
        """
        Bước 6a: Xuất mô hình đã tinh chỉnh ra định dạng `.gguf` hoặc `.onnx`.
        """
        if self.mo_hinh_hien_tai is None:
            raise RuntimeError("Chưa có mô hình để xuất. Hãy gọi huan_luyen_mo_hinh() trước.")

        export_dir = self.thu_muc / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        if dinh_dang.lower() == "gguf":
            out_path = duong_dan or str(export_dir / f"{self.ten_du_an}.gguf")
            return self.mo_hinh_hien_tai.xuat_gguf(out_path, quantization=quantization)
        if dinh_dang.lower() == "onnx":
            out_path = duong_dan or str(export_dir / f"{self.ten_du_an}.onnx")
            return self.mo_hinh_hien_tai.xuat_onnx(out_path)
        raise ValueError(f"Định dạng xuất '{dinh_dang}' không hỗ trợ. Chọn 'gguf' hoặc 'onnx'.")

    def khoi_dong_serving(self, kich_thuoc_batch: int = 8) -> MayChuBatch:
        """
        Bước 6b: Khởi động `MayChuBatch` phục vụ truy vấn đồng thời kèm kiểm tra sức khỏe (`KiemTraSucKhoe`).
        """
        if self.mo_hinh_hien_tai is None:
            raise RuntimeError("Chưa có mô hình để phục vụ.")

        self.may_chu_batch = MayChuBatch(
            mo_hinh=self.mo_hinh_hien_tai,
            kich_thuoc_batch=kich_thuoc_batch,
            ham_du_doan=lambda batch: [self.hoi(str(x))["tra_loi"] for x in batch],
        )
        self.may_chu_batch.bat_dau()
        return self.may_chu_batch

    def dung_serving(self) -> None:
        """Dừng máy chủ batch serving nếu đang chạy."""
        if self.may_chu_batch is not None:
            self.may_chu_batch.dung()

    def tao_pipeline(self) -> Pipeline:
        """Đóng gói toàn bộ luồng (Guardrail -> Mô hình) thành một đối tượng `Pipeline` chuẩn."""
        pipe = Pipeline(ten=f"{self.ten_du_an}_Pipeline")
        pipe.them_buoc("kiem_duyet", self.luong_an_toan)
        if self.mo_hinh_hien_tai is not None:
            pipe.them_buoc("mo_hinh", self.mo_hinh_hien_tai)
        pipe.da_fit = True
        return pipe

    def chay_dau_cuoi(
        self,
        du_lieu: List[Dict[str, Any]],
        mo_hinh: Any = "vgpt2",
        nhiem_vu: str = "sft",
        dung_lora: bool = True,
        so_vong: int = 3,
        cau_hoi_kiem_thu: Optional[str] = None,
        xuat_gguf: bool = True,
    ) -> Dict[str, Any]:
        """
        Thực thi trọn vẹn chu trình Đầu-Cuối chỉ trong 1 lệnh gọi:
        DataLake DVC -> RAG Indexing -> Fine-Tuning (LoRA) -> Experiment Tracking ->
        Model Registry (Production) -> Agent + Guardrails -> GGUF Export.
        """
        b1 = self.nap_du_lieu(du_lieu, ten_dataset="e2e_dataset.jsonl")
        b2 = self.xay_dung_rag(ten_dataset="e2e_dataset.jsonl")
        b3 = self.huan_luyen_mo_hinh(
            mo_hinh=mo_hinh,
            nhiem_vu=nhiem_vu,
            dung_lora=dung_lora,
            so_vong=so_vong,
        )
        self.tao_tac_tu()

        ket_qua_hoi = None
        if cau_hoi_kiem_thu:
            ket_qua_hoi = self.hoi(cau_hoi_kiem_thu, dung_rag=True)

        ket_qua_xuat = None
        if xuat_gguf:
            ket_qua_xuat = self.xuat_mo_hinh(dinh_dang="gguf")

        return {
            "datalake": b1,
            "rag": b2,
            "huan_luyen": b3,
            "registry_models": self.quan_ly_mo_hinh.danh_sach(),
            "kiem_thu": ket_qua_hoi,
            "xuat_gguf": ket_qua_xuat,
        }


HeSinhThaiAI = QuyTrinhDauCuoi
EndToEndWorkflow = QuyTrinhDauCuoi
