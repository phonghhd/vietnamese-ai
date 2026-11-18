"""QuyTrinhDauCuoi (HeSinhThaiAI / EndToEndWorkflow) - Cầu nối đầu-cuối hợp nhất toàn bộ Vietnamese AI Framework."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.state_graph import END, BoLuuDiemKiemTra, DoThiTrangThai
from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.core.engine import Engine
from vietnamese_ai.core.pipeline import Pipeline
from vietnamese_ai.datalake.dataset_manager import HoDuLieu
from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem
from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh
from vietnamese_ai.production.circuit_breaker import MachCat
from vietnamese_ai.production.health import KiemTraSucKhoe
from vietnamese_ai.production.metrics import QuanLyMetrics
from vietnamese_ai.prompts.guardrails import LuongAnToan
from vietnamese_ai.rag.rag_pipeline import RAGPipeline
from vietnamese_ai.registry.model_registry import QuanLyMoHinh
from vietnamese_ai.security.red_team import RedTeamSimulator
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
        self.mach_cat = MachCat(
            so_loi_toi_da=3,
            timeout_phuc_hoi=10.0,
            ten=f"{self.ten_du_an}_CircuitBreaker",
            on_circuit_open=self._xu_ly_su_co_circuit_breaker,
        )
        self.suc_khoe.dang_ky_checks_mac_dinh(mach_cat=self.mach_cat)
        self.suc_khoe.dang_ky_on_unhealthy(self._xu_ly_su_co_suc_khoe)

        self.rag: Optional[RAGPipeline] = None
        self.mo_hinh_hien_tai: Optional[Any] = None
        self.tac_tu: Optional[TacTu] = None
        self.may_chu_batch: Optional[MayChuBatch] = None
        self._dataset_hien_tai: Optional[str] = None
        self._ten_mo_hinh_dang_ky: Optional[str] = None
        self.dieu_phoi_swarm: Optional[Any] = None
        self._duong_dan_gguf_moi_nhat: Optional[str] = None
        self._duong_dan_onnx_moi_nhat: Optional[str] = None

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
        self._ten_mo_hinh_dang_ky = ten_model_reg
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

        if self.dieu_phoi_swarm is not None:
            swarm_ref = self.dieu_phoi_swarm

            def _suy_luan_swarm_fn(truy_van: str = "") -> str:
                kq_s = swarm_ref.suy_luan_swarm(truy_van, so_token_sinh=16)
                return str(kq_s.get("cau_tra_loi", ""))

            cong_cu_list.append(
                CongCu(
                    ten="suy_luan_edge_swarm",
                    mo_ta="Gửi prompt tính toán phân tán qua cụm Edge CPU Swarm",
                    ham_thuc_thi=_suy_luan_swarm_fn,
                )
            )

        self.tac_tu = TacTu(
            llm=self.mo_hinh_hien_tai,
            danh_sach_cong_cu=cong_cu_list,
            max_iterations=max_iterations,
        )
        return self.tac_tu

    def hoi(
        self,
        cau_hoi: str,
        dung_rag: bool = True,
        dung_agent: bool = False,
        dung_edge: bool = False,
    ) -> Dict[str, Any]:
        """
        Bước 5: Truy vấn đầu-cuối an toàn (Tường lửa AI + Guardrails đầu vào -> RAG / Fine-Tuned Model / Agent / Edge Swarm -> DLP đầu ra -> Metrics).
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

        def _thuc_hien_suy_luan() -> Any:
            if dung_edge and self.dieu_phoi_swarm is not None:
                kq_edge = self.suy_luan_edge(cau_hoi)
                return str(kq_edge.get("cau_tra_loi", kq_edge))
            elif dung_agent and self.tac_tu is not None:
                so_refine_truoc = len(getattr(self.tac_tu, "lich_su_refine", []))
                ket_qua_agent = self.tac_tu.chay(cau_hoi)
                so_refine_sau = len(getattr(self.tac_tu, "lich_su_refine", []))
                if so_refine_sau > so_refine_truoc:
                    kq_rf = self.tac_tu.lich_su_refine[-1]
                    out_cuoi = kq_rf.get("output_cuoi")
                    out_dau = kq_rf.get("output_ban_dau")
                    if out_cuoi and out_dau and out_cuoi != out_dau:
                        self.ghi_nhan_phan_hoi(
                            cau_hoi=cau_hoi,
                            tra_loi_tot=str(out_cuoi),
                            tra_loi_xau=str(out_dau),
                            diem=1.0,
                            metadata={"nguon": "self_refine_tu_dong"},
                        )
                return ket_qua_agent
            elif self.mo_hinh_hien_tai is not None:
                if getattr(self.mo_hinh_hien_tai, "nhiem_vu", "") == "text-classification":
                    du_doan_cls = self.mo_hinh_hien_tai.du_doan(cau_hoi)
                    return (
                        str(du_doan_cls.get("nhan", du_doan_cls))
                        if isinstance(du_doan_cls, dict)
                        else str(du_doan_cls)
                    )
                else:
                    return self.mo_hinh_hien_tai.sinh_van_ban(cau_hoi)
            elif self.rag is not None:
                return str(self.rag.hoi(cau_hoi).get("tra_loi", ""))
            else:
                return cau_hoi

        if self.mach_cat is not None:
            try:
                tra_loi = self.mach_cat.thuc_hien(_thuc_hien_suy_luan)
            except Exception as e:
                self.logger.error(f"Sự cố suy luận: {e}. Circuit Breaker can thiệp.")
                if self.rag is not None:
                    try:
                        tra_loi = str(self.rag.hoi(cau_hoi).get("tra_loi", ""))
                    except Exception:
                        tra_loi = f"[Hệ thống đang tự phục hồi]: {e}"
                else:
                    tra_loi = f"[Hệ thống đang tự phục hồi]: {e}"
        else:
            tra_loi = _thuc_hien_suy_luan()

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
            "edge": dung_edge,
        }

    def ghi_nhan_phan_hoi(
        self,
        cau_hoi: str,
        tra_loi_tot: str,
        tra_loi_xau: str = "",
        diem: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        ten_dataset: str = "feedback_dpo.jsonl",
    ) -> Dict[str, Any]:
        """
        Khép kín vòng lặp (Feedback Ingestion): Ghi nhận phản hồi người dùng hoặc Tác tử Self-Refine
        vào Hồ Dữ Liệu (DataLake) để chuẩn bị cho DPO/RLHF tự động hóa.
        """
        ban_ghi = {
            "prompt": cau_hoi,
            "chosen": tra_loi_tot,
            "rejected": tra_loi_xau,
            "diem": diem,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        ma_bam = self.ho_du_lieu.them_ban_ghi(ban_ghi, ten_dataset=ten_dataset)
        self.logger.info(f"Đã ghi nhận phản hồi vào DataLake ({ten_dataset})")
        return {
            "trang_thai": "thanh_cong",
            "ma_bam_sha256": ma_bam,
            "ban_ghi": ban_ghi,
        }

    def tinh_chinh_tu_phan_hoi(
        self,
        ten_dataset: str = "feedback_dpo.jsonl",
        so_vong: int = 1,
        beta: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Khép kín bánh đà học liên tục: Đọc dữ liệu phản hồi (DPO preference data) từ DataLake
        và tự động chạy quá trình tối ưu hóa trực tiếp (DPOTrainer) để cải tiến mô hình.
        """
        du_lieu_dpo: List[Dict[str, Any]] = []
        for lo in self.ho_du_lieu.doc_luong_du_lieu(ten_dataset):
            for item in lo:
                if isinstance(item, dict) and item.get("prompt") and item.get("chosen"):
                    du_lieu_dpo.append(
                        {
                            "prompt": item["prompt"],
                            "chosen": item["chosen"],
                            "rejected": item.get("rejected", ""),
                        }
                    )

        if not du_lieu_dpo:
            return {
                "trang_thai": "that_bai",
                "ly_do": f"Không tìm thấy mẫu phản hồi hợp lệ trong {ten_dataset}",
                "so_mau": 0,
            }

        self.logger.info(f"Bắt đầu tinh chỉnh DPO từ {len(du_lieu_dpo)} mẫu phản hồi...")
        try:
            from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

            trainer = DPOTrainer(so_vong=so_vong, beta=beta)
            kq = trainer.huan_luyen(
                model=self.mo_hinh_hien_tai,
                ref_model=self.mo_hinh_hien_tai,
                preference_data=du_lieu_dpo,
            )
            phien_ban_moi = f"dpo_{int(time.time())}"
            ten_dpo_reg = f"{self.ten_du_an}_DPO"
            self.quan_ly_mo_hinh.dang_ky(
                self.mo_hinh_hien_tai,
                ten=ten_dpo_reg,
                version=phien_ban_moi,
                metrics=kq if isinstance(kq, dict) else {},
                ghi_chu=f"Tự động tinh chỉnh từ {len(du_lieu_dpo)} mẫu feedback DPO",
            )
            self.quan_ly_mo_hinh.promote(ten_dpo_reg, phien_ban_moi, "production")
            self._ten_mo_hinh_dang_ky = ten_dpo_reg
            return {
                "trang_thai": "thanh_cong",
                "so_mau_dpo": len(du_lieu_dpo),
                "phien_ban_moi": phien_ban_moi,
                "ket_qua": kq,
            }
        except Exception as e:
            self.logger.warning(f"Lỗi khi tinh chỉnh DPO: {e}")
            return {
                "trang_thai": "that_bai",
                "so_mau_dpo": len(du_lieu_dpo),
                "loi": str(e),
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
            if hasattr(self.mo_hinh_hien_tai, "xuat_gguf"):
                res = self.mo_hinh_hien_tai.xuat_gguf(out_path, quantization=quantization)
            else:
                from vietnamese_ai.export.gguf import XuatGGUF

                xuat_tool = XuatGGUF()
                res_path = xuat_tool.xuat_tu_mo_hinh(
                    self.mo_hinh_hien_tai, out_path, quantization=quantization
                )
                res = {
                    "duong_dan": res_path,
                    "dinh_dang": "gguf",
                    "quantization": quantization,
                    "kich_thuoc_bytes": Path(res_path).stat().st_size if Path(res_path).exists() else 0,
                }
            self._duong_dan_gguf_moi_nhat = out_path
            return res
        if dinh_dang.lower() == "onnx":
            out_path = duong_dan or str(export_dir / f"{self.ten_du_an}.onnx")
            if hasattr(self.mo_hinh_hien_tai, "xuat_onnx"):
                res = self.mo_hinh_hien_tai.xuat_onnx(out_path)
            else:
                from vietnamese_ai.export.onnx_export import XuatONNX

                xuat_tool = XuatONNX()
                res_path = xuat_tool.xuat_sklearn(self.mo_hinh_hien_tai, out_path)
                res = {
                    "duong_dan": res_path,
                    "dinh_dang": "onnx",
                    "kich_thuoc_bytes": Path(res_path).stat().st_size if Path(res_path).exists() else 0,
                }
            self._duong_dan_onnx_moi_nhat = out_path
            return res
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

    def dong_goi_va_phan_phoi_edge(
        self,
        so_node: int = 3,
        duong_dan_gguf: Optional[str] = None,
        quantization: str = "q8_0",
        nen_fp16: bool = False,
        xac_thuc_mat_ma: bool = True,
    ) -> Dict[str, Any]:
        """
        Bước 6c: Tự động đóng gói mô hình sang GGUF và phân phối tới cụm Edge Swarm CPU
        (Pipeline Parallelism đa nút kèm chứng thực mật mã học Proof of Inference).
        """
        if duong_dan_gguf is None:
            if self._duong_dan_gguf_moi_nhat and Path(self._duong_dan_gguf_moi_nhat).exists():
                duong_dan_gguf = self._duong_dan_gguf_moi_nhat
            else:
                if self.mo_hinh_hien_tai is None:
                    self.mo_hinh_hien_tai = BoTinhChinhMoHinh(
                        mo_hinh="vgpt2", nhiem_vu="sft", dung_lora=True, che_do_offline=True
                    )
                kq_xuat = self.xuat_mo_hinh(dinh_dang="gguf", quantization=quantization)
                duong_dan_gguf = kq_xuat.get(
                    "duong_dan", str(self.thu_muc / "exports" / f"{self.ten_du_an}.gguf")
                )

        from vietnamese_ai.edge.swarm_inference import DieuPhoiSwarmCPU

        tong_so_layer = max(4, so_node * 2)
        self.dieu_phoi_swarm = DieuPhoiSwarmCPU(
            d_model=128,
            tong_so_layer=tong_so_layer,
            bat_xac_thuc_mat_ma=xac_thuc_mat_ma,
            nen_fp16=nen_fp16,
        )
        self.dieu_phoi_swarm.khoi_tao_cum_gia_lap(so_node=so_node, so_core_moi_node=4)
        kq_phan_manh = self.dieu_phoi_swarm.phan_manh_mo_hinh_gguf(duong_dan_gguf)

        if self.tac_tu is not None:
            swarm_ref = self.dieu_phoi_swarm

            def _suy_luan_swarm_tool(truy_van: str = "") -> str:
                kq_s = swarm_ref.suy_luan_swarm(truy_van, so_token_sinh=16)
                return str(kq_s.get("cau_tra_loi", ""))

            tool_sw = CongCu(
                ten="suy_luan_edge_swarm",
                mo_ta="Gửi prompt tính toán phân tán qua cụm Edge CPU Swarm",
                ham_thuc_thi=_suy_luan_swarm_tool,
            )
            self.tac_tu.them_cong_cu(tool_sw)

        self.logger.info(
            f"Đã đóng gói và phân mảnh mô hình GGUF thành công tới {so_node} node Edge Swarm."
        )
        return {
            "trang_thai": "thanh_cong",
            "duong_dan_gguf": duong_dan_gguf,
            "so_node": so_node,
            "tong_so_layer": tong_so_layer,
            "phan_manh": kq_phan_manh,
            "cum_swarm": self.dieu_phoi_swarm.lay_trang_thai_cum(),
        }

    def suy_luan_edge(
        self,
        prompt: str,
        so_token: int = 16,
        xac_minh_proof: bool = True,
    ) -> Dict[str, Any]:
        """
        Thực thi suy luận phân tán (Pipeline Parallelism) trên cụm Edge CPU Swarm
        kèm xác thực chứng thực mật mã học Verifiable AI.
        """
        if self.dieu_phoi_swarm is None:
            self.dong_goi_va_phan_phoi_edge()

        assert self.dieu_phoi_swarm is not None
        return self.dieu_phoi_swarm.suy_luan_swarm(
            prompt=prompt,
            so_token_sinh=so_token,
            xac_minh_proof=xac_minh_proof,
        )

    def tao_cau_noi_orchestrator(
        self,
        port_http: int = 8085,
        host: str = "127.0.0.1",
        ma_bi_mat_cum: Optional[str] = None,
    ) -> Any:
        """Tạo Cầu nối HTTP Swarm Worker để liên kết cụm Swarm vào NutChinh (V-Orchestrator)."""
        if self.dieu_phoi_swarm is None:
            self.dong_goi_va_phan_phoi_edge()
        assert self.dieu_phoi_swarm is not None
        return self.dieu_phoi_swarm.tao_cau_noi_orchestrator(
            port_http=port_http,
            host=host,
            ma_bi_mat_cum=ma_bi_mat_cum,
        )

    def _tim_ten_mo_hinh_registry(self) -> str:
        """Tìm tên mô hình chính xác đã đăng ký trong QuanLyMoHinh."""
        if self._ten_mo_hinh_dang_ky and self._ten_mo_hinh_dang_ky in self.quan_ly_mo_hinh._metadata:
            return self._ten_mo_hinh_dang_ky
        for k in self.quan_ly_mo_hinh._metadata:
            if self.ten_du_an in k:
                return k
        return self._ten_mo_hinh_dang_ky or self.ten_du_an

    def _xu_ly_su_co_circuit_breaker(self, circuit: Any) -> None:
        """
        Tự phục hồi (Self-Healing Loop): Khi Circuit Breaker bị ngắt mạch do lỗi dồn dập
        ở mô hình hiện tại, tự động rollback Model Registry về phiên bản ổn định gần nhất.
        """
        ten_mo_hinh = self._tim_ten_mo_hinh_registry()
        self.logger.warning(
            f"Phát hiện sự cố: Circuit Breaker '{getattr(circuit, 'ten', 'CircuitBreaker')}' đã ngắt mạch! "
            f"Đang tự động rollback mô hình '{ten_mo_hinh}'..."
        )
        try:
            ver_moi, mo_hinh_moi = self.quan_ly_mo_hinh.rollback(
                ten_mo_hinh, alias_muc_tieu="production"
            )
            self.mo_hinh_hien_tai = mo_hinh_moi
            if self.tac_tu is not None:
                self.tac_tu.mo_hinh = mo_hinh_moi
            if hasattr(circuit, "reset"):
                circuit.reset()
            self.logger.warning(
                f"Tự phục hồi thành công: Đã khôi phục mô hình về v{ver_moi} và đóng lại Circuit Breaker."
            )
        except Exception as e:
            self.logger.error(f"Tự phục hồi thất bại: {e}")

    def _xu_ly_su_co_suc_khoe(self, health_checker: Any, ket_qua: Dict[str, Any]) -> None:
        """
        Tự phục hồi khi Health Check phát hiện hệ sinh thái chuyển sang trạng thái 'unhealthy'.
        """
        if getattr(self, "_dang_tu_phuc_hoi", False):
            return
        self._dang_tu_phuc_hoi = True
        try:
            ten_mo_hinh = self._tim_ten_mo_hinh_registry()
            self.logger.warning(
                f"Health check báo cáo sự cố (unhealthy). Kích hoạt tự phục hồi cho '{ten_mo_hinh}'..."
            )
            ver_moi, mo_hinh_moi = self.quan_ly_mo_hinh.rollback(
                ten_mo_hinh, alias_muc_tieu="production"
            )
            self.mo_hinh_hien_tai = mo_hinh_moi
            if self.tac_tu is not None:
                self.tac_tu.mo_hinh = mo_hinh_moi
            if self.mach_cat is not None:
                self.mach_cat.reset()
            self.logger.warning(
                f"Tự phục hồi qua Health Check thành công: Đã chuyển sang mô hình v{ver_moi}."
            )
        except Exception as e:
            self.logger.error(f"Tự phục hồi từ health check thất bại: {e}")
        finally:
            self._dang_tu_phuc_hoi = False

    def kiem_tra_suc_khoe(self) -> Dict[str, Any]:
        """Kiểm tra sức khỏe toàn diện của hệ sinh thái."""
        return self.suc_khoe.kiem_tra()

    def tu_dong_phuc_hoi(self) -> Dict[str, Any]:
        """
        Chủ động kiểm tra và thực hiện tự phục hồi nếu phát hiện bất thường trong hệ sinh thái.
        """
        mo_hinh_truoc = getattr(self.mo_hinh_hien_tai, "ten", str(self.mo_hinh_hien_tai))

        kt = self.kiem_tra_suc_khoe()
        trang_thai_suc_khoe = kt.get("trang_thai", "healthy")
        trang_thai_mach = self.mach_cat.trang_thai if self.mach_cat else "dong"

        da_phuc_hoi = False
        thong_tin_rollback = None

        mo_hinh_sau = getattr(self.mo_hinh_hien_tai, "ten", str(self.mo_hinh_hien_tai))
        if mo_hinh_sau != mo_hinh_truoc:
            da_phuc_hoi = True
            current_ver = (
                self.quan_ly_mo_hinh._metadata.get(self._tim_ten_mo_hinh_registry(), {})
                .get("aliases", {})
                .get("production", "rollback")
            )
            thong_tin_rollback = {"version": current_ver}
        elif trang_thai_suc_khoe == "unhealthy" or trang_thai_mach == "mo":
            ten_mo_hinh = self._tim_ten_mo_hinh_registry()
            self.logger.warning(
                f"Phát hiện hệ sinh thái không khỏe mạnh. Đang tự phục hồi cho '{ten_mo_hinh}'..."
            )
            try:
                ver, mo_hinh_moi = self.quan_ly_mo_hinh.rollback(ten_mo_hinh)
                self.mo_hinh_hien_tai = mo_hinh_moi
                if self.tac_tu is not None:
                    self.tac_tu.mo_hinh = mo_hinh_moi
                if self.mach_cat is not None:
                    self.mach_cat.reset()
                da_phuc_hoi = True
                thong_tin_rollback = {"version": ver}
            except Exception as e:
                thong_tin_rollback = {"loi": str(e)}

        return {
            "trang_thai_truoc": trang_thai_suc_khoe,
            "mach_cat": trang_thai_mach,
            "da_phuc_hoi": da_phuc_hoi,
            "chi_tiet_phuc_hoi": thong_tin_rollback,
            "trang_thai_suc_khoe": trang_thai_suc_khoe,
        }

    def tap_kich_an_ninh_va_tu_luyen(
        self,
        so_mau: int = 10,
        danh_sach_prompt_red_team: Optional[List[str]] = None,
        tu_dong_dpo: bool = True,
        ten_dataset: str = "adversarial_dpo.jsonl",
        so_vong_dpo: int = 1,
    ) -> Dict[str, Any]:
        """
        Bước 7a (Đồng tiến hóa đối kháng - Adversarial Co-Evolution Loop):
        Kích hoạt Red Team Simulator tấn công thử nghiệm hệ sinh thái (Prompt Injection,
        Jailbreak, System Leak, Payload độc hại), tự động đóng gói các cặp phản hồi đối kháng
        (DPO pairs: prompt, chosen=từ chối an toàn, rejected=vượt rào nguy hại) vào Hồ Dữ Liệu
        (DataLake), và tự động huấn luyện DPO nhằm tôi luyện các trọng số của mô hình.
        """
        sim = RedTeamSimulator(muc_tieu=self)
        prompts = danh_sach_prompt_red_team or sim.sinh_du_lieu_tan_cong(so_luong=so_mau)
        kq_test = sim.chay_kiem_thu_doi_khang(muc_tieu=self, danh_sach_prompt=prompts)

        du_lieu_dpo = kq_test.get("du_lieu_dpo", [])
        for cap in du_lieu_dpo:
            self.ho_du_lieu.them_ban_ghi(cap, ten_dataset=ten_dataset)

        self.logger.info(
            f"Đã lưu {len(du_lieu_dpo)} mẫu đối kháng vào DataLake ({ten_dataset}). "
            f"Tỷ lệ bảo vệ hiện tại: {kq_test.get('ty_le_bao_ve', 0):.1f}%"
        )

        kq_dpo = None
        if tu_dong_dpo and self.mo_hinh_hien_tai is not None:
            kq_dpo = self.tinh_chinh_tu_phan_hoi(
                ten_dataset=ten_dataset, so_vong=so_vong_dpo
            )
            ten_reg = self._tim_ten_mo_hinh_registry()
            ver_moi = kq_dpo.get("phien_ban_moi")
            if ver_moi:
                try:
                    self.quan_ly_mo_hinh.promote(ten_reg, ver_moi, "hardened")
                except Exception:
                    pass

        return {
            "trang_thai": "thanh_cong",
            "tong_so_mau": kq_test.get("tong_so_mau", len(prompts)),
            "so_lan_chan": kq_test.get("so_lan_chan_thanh_cong", 0),
            "so_lan_that_bai": kq_test.get("so_lan_that_bai", 0),
            "ty_le_bao_ve": kq_test.get("ty_le_bao_ve", 100.0),
            "so_cap_dpo_da_luu": len(du_lieu_dpo),
            "ten_dataset_dpo": ten_dataset,
            "dpo_hardening": kq_dpo,
        }

    def tao_do_thi_trang_thai(
        self,
        che_do_suy_luan: str = "auto",
        bo_luu_diem: bool = True,
    ) -> DoThiTrangThai:
        """
        Bước 7b (Hợp nhất Đồ thị Trạng thái Đa Agent & Swarm / Cụm - Distributed StateGraph):
        Xây dựng một Đồ thị Trạng thái (StateGraph / LangGraph Tiếng Việt) hoàn chỉnh, khép kín mọi mắt xích:
        [START] -> kiem_tra_an_ninh -> (an toàn?)
                     ├─ False -> tu_choi_an_toan -> [END]
                     └─ True  -> truy_xuat_rag -> suy_luan_he_thong -> tu_sua_chat_luong -> loc_dau_ra -> [END]
        """
        cp = BoLuuDiemKiemTra() if bo_luu_diem else None
        graph = DoThiTrangThai(checkpointer=cp)

        def _kiem_tra_an_ninh(state: Dict[str, Any]) -> Dict[str, Any]:
            cau_hoi = str(state.get("cau_hoi", state.get("prompt", "")))
            an_toan = True
            ly_do = "An toàn."
            if self.tuong_lua is not None:
                kt_fw = self.tuong_lua.kiem_tra_dau_vao(cau_hoi)
                if not kt_fw.get("an_toan", True):
                    an_toan = False
                    ly_do = kt_fw.get("ly_do", "Bị chặn bởi Tường lửa AI")
            if an_toan and self.luong_an_toan is not None:
                kt_g = self.luong_an_toan.kiem_tra(cau_hoi)
                if not kt_g.get("an_toan", True):
                    an_toan = False
                    ly_do = f"Vi phạm luồng an toàn: {kt_g.get('ly_do', '')}"
            return {"an_toan": an_toan, "ly_do_an_ninh": ly_do}

        def _tu_choi_an_toan(state: Dict[str, Any]) -> Dict[str, Any]:
            ly_do = state.get("ly_do_an_ninh", "Vi phạm chính sách bảo mật.")
            return {
                "tra_loi": f"[Bị từ chối an toàn]: {ly_do}",
                "hoan_tat": True,
                "trang_thai_dieu_huong": "tu_choi_an_toan",
            }

        def _truy_xuat_rag(state: Dict[str, Any]) -> Dict[str, Any]:
            cau_hoi = str(state.get("cau_hoi", state.get("prompt", "")))
            nguon_rag = []
            ngu_canh = ""
            if self.rag is not None:
                try:
                    res_rag = self.rag.tim_kiem(cau_hoi, top_k=3)
                    nguon_rag = res_rag
                    ngu_canh = "\n".join(str(doc.get("van_ban", doc)) for doc in res_rag)
                except Exception:
                    pass
            return {"nguon_rag": nguon_rag, "ngu_canh": ngu_canh}

        def _suy_luan_he_thong(state: Dict[str, Any]) -> Dict[str, Any]:
            cau_hoi = str(state.get("cau_hoi", state.get("prompt", "")))
            ngu_canh = state.get("ngu_canh", "")
            prompt_cuoi = f"Ngữ cảnh:\n{ngu_canh}\n\nCâu hỏi: {cau_hoi}" if ngu_canh else cau_hoi

            thuc_hien_swarm = (
                che_do_suy_luan == "swarm"
                or (che_do_suy_luan == "auto" and self.dieu_phoi_swarm is not None)
            )

            if thuc_hien_swarm and self.dieu_phoi_swarm is not None:
                kq_s = self.suy_luan_edge(prompt_cuoi, so_token=16)
                proof = kq_s.get("proof") or kq_s.get("cam_ket_chi_tiet")
                return {
                    "tra_loi": str(kq_s.get("cau_tra_loi", "")),
                    "swarm_proof": proof,
                    "che_do": "swarm",
                }
            elif self.tac_tu is not None:
                res_ag = self.tac_tu.chay(prompt_cuoi)
                return {"tra_loi": str(res_ag), "che_do": "agent"}
            elif self.mo_hinh_hien_tai is not None:
                if getattr(self.mo_hinh_hien_tai, "nhiem_vu", "") == "text-classification":
                    res_m = self.mo_hinh_hien_tai.du_doan(prompt_cuoi)
                else:
                    res_m = self.mo_hinh_hien_tai.sinh_van_ban(prompt_cuoi)
                return {"tra_loi": str(res_m), "che_do": "model"}
            elif self.rag is not None and ngu_canh:
                return {"tra_loi": ngu_canh, "che_do": "rag"}
            else:
                return {"tra_loi": cau_hoi, "che_do": "echo"}

        def _tu_sua_chat_luong(state: Dict[str, Any]) -> Dict[str, Any]:
            tra_loi = str(state.get("tra_loi", ""))
            ngu_canh = str(state.get("ngu_canh", ""))
            da_tu_sua = False
            if not tra_loi.strip() and ngu_canh:
                tra_loi = f"Dựa trên tri thức: {ngu_canh[:200]}"
                da_tu_sua = True
            return {"tra_loi": tra_loi, "da_tu_sua": da_tu_sua}

        def _loc_dau_ra(state: Dict[str, Any]) -> Dict[str, Any]:
            tra_loi = str(state.get("tra_loi", ""))
            if self.tuong_lua is not None:
                tra_loi = self.tuong_lua.loc_dau_ra(tra_loi)
            if self.luong_an_toan is not None:
                tra_loi = self.luong_an_toan.kiem_tra_va_sua(tra_loi)
            return {"tra_loi": tra_loi, "hoan_tat": True}

        graph.them_nut("kiem_tra_an_ninh", _kiem_tra_an_ninh)
        graph.them_nut("tu_choi_an_toan", _tu_choi_an_toan)
        graph.them_nut("truy_xuat_rag", _truy_xuat_rag)
        graph.them_nut("suy_luan_he_thong", _suy_luan_he_thong)
        graph.them_nut("tu_sua_chat_luong", _tu_sua_chat_luong)
        graph.them_nut("loc_dau_ra", _loc_dau_ra)

        graph.dat_diem_bat_dau("kiem_tra_an_ninh")
        graph.them_canh_dieu_kien(
            "kiem_tra_an_ninh",
            lambda st: "truy_xuat_rag" if st.get("an_toan", True) else "tu_choi_an_toan",
        )
        graph.them_canh("truy_xuat_rag", "suy_luan_he_thong")
        graph.them_canh("suy_luan_he_thong", "tu_sua_chat_luong")
        graph.them_canh("tu_sua_chat_luong", "loc_dau_ra")
        graph.them_canh("loc_dau_ra", END)
        graph.them_canh("tu_choi_an_toan", END)

        return graph

    def chay_do_thi_trang_thai(
        self,
        cau_hoi: str,
        che_do_suy_luan: str = "auto",
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Thực thi quy trình khép kín thông qua Đồ thị Trạng thái (Distributed StateGraph).
        """
        do_thi = self.tao_do_thi_trang_thai(che_do_suy_luan=che_do_suy_luan)
        bien_dich = do_thi.bien_dich()
        init_state = {"cau_hoi": cau_hoi, "an_toan": True}
        return bien_dich.thuc_thi(init_state, session_id=session_id, thread_id=thread_id)

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
        phan_phoi_edge: bool = False,
        so_node_edge: int = 3,
        dong_tien_hoa_doi_khang: bool = False,
        dung_do_thi_trang_thai: bool = False,
    ) -> Dict[str, Any]:
        """
        Thực thi trọn vẹn chu trình Đầu-Cuối chỉ trong 1 lệnh gọi:
        DataLake DVC -> RAG Indexing -> Fine-Tuning (LoRA) -> Experiment Tracking ->
        Model Registry (Production) -> Agent + Guardrails -> GGUF Export -> Edge Swarm Sharding
        -> Adversarial Co-Evolution Hardening -> Unified StateGraph Execution.
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

        ket_qua_co_evolution = None
        if dong_tien_hoa_doi_khang:
            ket_qua_co_evolution = self.tap_kich_an_ninh_va_tu_luyen(so_mau=5)

        ket_qua_xuat = None
        if xuat_gguf:
            ket_qua_xuat = self.xuat_mo_hinh(dinh_dang="gguf")

        ket_qua_edge = None
        if phan_phoi_edge:
            ket_qua_edge = self.dong_goi_va_phan_phoi_edge(so_node=so_node_edge)

        ket_qua_hoi = None
        if cau_hoi_kiem_thu:
            if dung_do_thi_trang_thai:
                ket_qua_hoi = self.chay_do_thi_trang_thai(
                    cau_hoi_kiem_thu,
                    che_do_suy_luan="swarm" if phan_phoi_edge else "auto",
                )
            else:
                ket_qua_hoi = self.hoi(cau_hoi_kiem_thu, dung_rag=True, dung_edge=phan_phoi_edge)

        return {
            "datalake": b1,
            "rag": b2,
            "huan_luyen": b3,
            "registry_models": self.quan_ly_mo_hinh.danh_sach(),
            "kiem_thu": ket_qua_hoi,
            "xuat_gguf": ket_qua_xuat,
            "edge_swarm": ket_qua_edge,
            "dong_tien_hoa_doi_khang": ket_qua_co_evolution,
        }


HeSinhThaiAI = QuyTrinhDauCuoi
EndToEndWorkflow = QuyTrinhDauCuoi
