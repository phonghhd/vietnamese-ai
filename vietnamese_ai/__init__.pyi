"""PEP 561 Static Type Stubs cho Vietnamese AI Framework."""

from vietnamese_ai.agents.agent import TacTu as TacTu
from vietnamese_ai.agents.code_healing import CodeHealingAgent as CodeHealingAgent
from vietnamese_ai.agents.code_healing import TacTuLapTrinhTuSua as TacTuLapTrinhTuSua
from vietnamese_ai.agents.deep_research import DeepResearchAgent as DeepResearchAgent
from vietnamese_ai.agents.deep_research import TacTuNghienCuuSau as TacTuNghienCuuSau
from vietnamese_ai.agents.long_term_memory import HethongNhoMemGPT as HethongNhoMemGPT
from vietnamese_ai.agents.mcp import MCPBridge as MCPBridge
from vietnamese_ai.agents.mcp import MCPClient as MCPClient
from vietnamese_ai.agents.mcp import MCPServer as MCPServer
from vietnamese_ai.agents.mcts_planning import LapKeHoachMCTS as LapKeHoachMCTS
from vietnamese_ai.agents.memory import BoNhoTacTu as BoNhoTacTu
from vietnamese_ai.agents.moa import MoA as MoA
from vietnamese_ai.agents.state_graph import (
    END as END,
)
from vietnamese_ai.agents.state_graph import (
    START as START,
)
from vietnamese_ai.agents.state_graph import (
    BoLuuDiemKiemTra as BoLuuDiemKiemTra,
)
from vietnamese_ai.agents.state_graph import (
    Checkpointer as Checkpointer,
)
from vietnamese_ai.agents.state_graph import (
    DoThiTrangThai as DoThiTrangThai,
)
from vietnamese_ai.agents.state_graph import (
    MemorySaver as MemorySaver,
)
from vietnamese_ai.agents.state_graph import (
    StateGraph as StateGraph,
)
from vietnamese_ai.agents.swarm import HeThongSwarm as HeThongSwarm
from vietnamese_ai.agents.swarm import TacTuSwarm as TacTuSwarm
from vietnamese_ai.agents.tools import ComputerUseTool as ComputerUseTool
from vietnamese_ai.agents.tools import CongCu as CongCu
from vietnamese_ai.agents.tools import CongCuManHinh as CongCuManHinh
from vietnamese_ai.agents.tools import cong_cu as cong_cu
from vietnamese_ai.agents.tools import cong_cu_man_hinh as cong_cu_man_hinh
from vietnamese_ai.api.server import MayChuOpenAI as MayChuOpenAI
from vietnamese_ai.api.server import OpenAICompatibleServer as OpenAICompatibleServer
from vietnamese_ai.api.server import ServerDonGian as ServerDonGian
from vietnamese_ai.audio.speech import NhanDienGiongNoi as NhanDienGiongNoi
from vietnamese_ai.audio.speech import TongHopGiongNoi as TongHopGiongNoi
from vietnamese_ai.automl.auto_ml import AutoML as AutoML
from vietnamese_ai.automl.nas import TimKiemKienTruc as TimKiemKienTruc
from vietnamese_ai.benchmark.comparative_benchmark import BoDoLuongDoiDau as BoDoLuongDoiDau
from vietnamese_ai.benchmark.comparative_benchmark import (
    ComparativeBenchmarkSuite as ComparativeBenchmarkSuite,
)
from vietnamese_ai.compression.distillation import HocRutGon as HocRutGon
from vietnamese_ai.compression.extreme import BitLinear as BitLinear
from vietnamese_ai.compression.extreme import BitNetTransformerBlock as BitNetTransformerBlock
from vietnamese_ai.compression.extreme import RMSNorm as RMSNorm
from vietnamese_ai.compression.pruning import CatTiaMoHinh as CatTiaMoHinh
from vietnamese_ai.core.cross_validation import KiemDinhCheo as KiemDinhCheo
from vietnamese_ai.core.end_to_end import EndToEndWorkflow as EndToEndWorkflow
from vietnamese_ai.core.end_to_end import HeSinhThaiAI as HeSinhThaiAI
from vietnamese_ai.core.end_to_end import QuyTrinhDauCuoi as QuyTrinhDauCuoi
from vietnamese_ai.core.engine import Engine as Engine
from vietnamese_ai.core.hyperparameter import TimKiemThamSo as TimKiemThamSo
from vietnamese_ai.core.pipeline import Pipeline as Pipeline
from vietnamese_ai.datalake.dataset_manager import HoDuLieu as HoDuLieu
from vietnamese_ai.distributed.distributed import PhanTanHuanLuyen as PhanTanHuanLuyen
from vietnamese_ai.distributed.multi_gpu import MultiGPUTrainer as MultiGPUTrainer
from vietnamese_ai.edge.c_bridge import CauNoiCtypesNative as CauNoiCtypesNative
from vietnamese_ai.edge.c_bridge import CtypesLlamaBridge as CtypesLlamaBridge
from vietnamese_ai.edge.c_bridge import NativeLlamaBridge as NativeLlamaBridge
from vietnamese_ai.edge.intelligent_router import EdgeRouter as EdgeRouter
from vietnamese_ai.edge.native_inference import BoGiaiLuongTuQ4 as BoGiaiLuongTuQ4
from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge as DongCoSuyLuanEdge
from vietnamese_ai.edge.native_inference import EdgeInferenceEngine as EdgeInferenceEngine
from vietnamese_ai.edge.native_inference import GGUFMemoryMappedStreamer as GGUFMemoryMappedStreamer
from vietnamese_ai.edge.native_inference import ZeroGPUInferenceEngine as ZeroGPUInferenceEngine
from vietnamese_ai.edge.node_llama import NodeLlamaEngine as NodeLlamaEngine
from vietnamese_ai.edge.swarm_inference import (
    DieuPhoiSwarmCPU as DieuPhoiSwarmCPU,
)
from vietnamese_ai.edge.swarm_inference import (
    NutSwarmCPU as NutSwarmCPU,
)
from vietnamese_ai.edge.swarm_inference import (
    SwarmCPUNode as SwarmCPUNode,
)
from vietnamese_ai.edge.swarm_inference import (
    SwarmInferenceOrchestrator as SwarmInferenceOrchestrator,
)
from vietnamese_ai.embeddings.fasttext import FastTextTiengViet as FastTextTiengViet
from vietnamese_ai.embeddings.word2vec import Word2VecTiengViet as Word2VecTiengViet
from vietnamese_ai.enterprise.audit import NhatKyHoatDong as NhatKyHoatDong
from vietnamese_ai.enterprise.auth import HeThongXacThuc as HeThongXacThuc
from vietnamese_ai.enterprise.deployment import CloudDeployment as CloudDeployment
from vietnamese_ai.enterprise.marketplace import Marketplace as Marketplace
from vietnamese_ai.enterprise.platform import NenTangDichVu as NenTangDichVu
from vietnamese_ai.experiment_tracking.tracker import TheoDoiThiNghiem as TheoDoiThiNghiem
from vietnamese_ai.export.gguf import XuatGGUF as XuatGGUF
from vietnamese_ai.export.onnx_export import XuatONNX as XuatONNX
from vietnamese_ai.federated.learning import HocLienKet as HocLienKet
from vietnamese_ai.federated.rlhf import FederatedRLHF as FederatedRLHF
from vietnamese_ai.fine_tuning.auto_finetuner import AutoFineTuner as AutoFineTuner
from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh as BoTinhChinhMoHinh
from vietnamese_ai.fine_tuning.dataset import InstructionDataset as InstructionDataset
from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer as DPOTrainer
from vietnamese_ai.fine_tuning.grpo_trainer import GRPOTrainer as GRPOTrainer
from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper as HuggingFaceWrapper
from vietnamese_ai.fine_tuning.instruction_trainer import (
    InstructionTuningTrainer as InstructionTuningTrainer,
)
from vietnamese_ai.fine_tuning.lora import DoRALayer as DoRALayer
from vietnamese_ai.fine_tuning.lora import LoRAAdapter as LoRAAdapter
from vietnamese_ai.fine_tuning.lora import QLoRAAdapter as QLoRAAdapter
from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft as LoRAPeft
from vietnamese_ai.fine_tuning.peft_config import PEFTConfig as PEFTConfig
from vietnamese_ai.fine_tuning.pytorch_trainer import HuanLuyenPyTorch as HuanLuyenPyTorch
from vietnamese_ai.fine_tuning.reward_model import RewardModel as RewardModel
from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline as RLHFPipeline
from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer as SFTTrainer
from vietnamese_ai.fine_tuning.training_accelerator import BenchmarkTangToc as BenchmarkTangToc
from vietnamese_ai.fine_tuning.training_accelerator import BoGopChuoi as BoGopChuoi
from vietnamese_ai.fine_tuning.training_accelerator import BoTangTocHuanLuyen as BoTangTocHuanLuyen
from vietnamese_ai.fine_tuning.training_accelerator import BoToiUuGaLore as BoToiUuGaLore
from vietnamese_ai.fine_tuning.training_accelerator import (
    ChunkedCrossEntropy as ChunkedCrossEntropy,
)
from vietnamese_ai.fine_tuning.training_accelerator import GaLoreOptimizer as GaLoreOptimizer
from vietnamese_ai.fine_tuning.training_accelerator import HamMatMatPhanManh as HamMatMatPhanManh
from vietnamese_ai.fine_tuning.training_accelerator import SequencePacker as SequencePacker
from vietnamese_ai.fine_tuning.training_accelerator import SpeedBenchmark as SpeedBenchmark
from vietnamese_ai.fine_tuning.training_accelerator import (
    TrainingAccelerator as TrainingAccelerator,
)
from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper as UnslothWrapper
from vietnamese_ai.interpretability.explainer import GiaiThichMoHinh as GiaiThichMoHinh
from vietnamese_ai.llm.benchmark_runner import BenchmarkRunner as BenchmarkRunner
from vietnamese_ai.llm.lm_eval import LMEvalHarness as LMEvalHarness
from vietnamese_ai.llm.model_configs import ModelConfig as ModelConfig
from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM as VietnameseLLM
from vietnamese_ai.llm.wrappers import LlamaCppWrapper as LlamaCppWrapper
from vietnamese_ai.llm.wrappers import OllamaWrapper as OllamaWrapper
from vietnamese_ai.llm.wrappers import VLLMWrapper as VLLMWrapper
from vietnamese_ai.mobile.deployment import TriKhaiDiDong as TriKhaiDiDong
from vietnamese_ai.models.base import BaseModel as BaseModel
from vietnamese_ai.models.classifier import PhanLoai as PhanLoai
from vietnamese_ai.models.clustering import PhanCum as PhanCum
from vietnamese_ai.models.ensemble import MoHinhTapHop as MoHinhTapHop
from vietnamese_ai.models.mang_sau import MangSau as MangSau
from vietnamese_ai.models.neural_net import MangNron as MangNron
from vietnamese_ai.models.regression import HoiQuy as HoiQuy
from vietnamese_ai.monitor.cost_engine import MayTinhTien as MayTinhTien
from vietnamese_ai.nlp.native_tokenizer import BoNhungVanBanSieuToc as BoNhungVanBanSieuToc
from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc as BoTachTuTiengVietSieuToc
from vietnamese_ai.nlp.native_tokenizer import FastVietnameseEmbedder as FastVietnameseEmbedder
from vietnamese_ai.nlp.native_tokenizer import FastVietnameseTokenizer as FastVietnameseTokenizer
from vietnamese_ai.nlp.native_tokenizer import (
    NativeVietnameseTokenizer as NativeVietnameseTokenizer,
)
from vietnamese_ai.nlp.ner import NhanDienThucThe as NhanDienThucThe
from vietnamese_ai.nlp.pretrained.phobert import PhoBERTWrapper as PhoBERTWrapper
from vietnamese_ai.nlp.qa import HoiDapTiengViet as HoiDapTiengViet
from vietnamese_ai.nlp.sentiment import PhanTichCamXuc as PhanTichCamXuc
from vietnamese_ai.nlp.spelling import KiemTraChinhTa as KiemTraChinhTa
from vietnamese_ai.nlp.summarization import TomTatVanBan as TomTatVanBan
from vietnamese_ai.nlp.translation import DichThuat as DichThuat
from vietnamese_ai.orchestrator.cluster_manager import DieuPhoiCum as DieuPhoiCum
from vietnamese_ai.orchestrator.master import NutChinh as NutChinh
from vietnamese_ai.orchestrator.worker import NutPhu as NutPhu
from vietnamese_ai.plugin.legacy_manager import PluginManager as PluginManager
from vietnamese_ai.plugin.plugin_manager import QuanLyPlugin as QuanLyPlugin
from vietnamese_ai.portable.desktop_packager import DesktopPackager as DesktopPackager
from vietnamese_ai.portable.desktop_packager import GoiDocLapDesktop as GoiDocLapDesktop
from vietnamese_ai.portable.desktop_packager import StandalonePackager as StandalonePackager
from vietnamese_ai.preprocessing.feature_engineering import TaoDacTrung as TaoDacTrung
from vietnamese_ai.preprocessing.numerical import XuLySo as XuLySo
from vietnamese_ai.preprocessing.text import XuLyVanBan as XuLyVanBan
from vietnamese_ai.preprocessing.text_augmenter import TangCuongVanBan as TangCuongVanBan
from vietnamese_ai.production.circuit_breaker import MachCat as MachCat
from vietnamese_ai.production.health import KiemTraSucKhoe as KiemTraSucKhoe
from vietnamese_ai.production.logging import LoggerCauTruc as LoggerCauTruc
from vietnamese_ai.production.metrics import QuanLyMetrics as QuanLyMetrics
from vietnamese_ai.production.warmup import LamNongModel as LamNongModel
from vietnamese_ai.prompts.chains import ChuoiPrompt as ChuoiPrompt
from vietnamese_ai.prompts.guardrails import LuongAnToan as LuongAnToan
from vietnamese_ai.prompts.parser import PhanTichDauRa as PhanTichDauRa
from vietnamese_ai.prompts.parser import SinhCoCauTruc as SinhCoCauTruc
from vietnamese_ai.prompts.templates import MauPrompt as MauPrompt
from vietnamese_ai.rag.chunker import CatVanBan as CatVanBan
from vietnamese_ai.rag.context_optimization import BoNhoDemNguNghia as BoNhoDemNguNghia
from vietnamese_ai.rag.context_optimization import NenNguCanh as NenNguCanh
from vietnamese_ai.rag.context_optimization import PromptCompressor as PromptCompressor
from vietnamese_ai.rag.context_optimization import SemanticCache as SemanticCache
from vietnamese_ai.rag.evaluation import DanhGiaRAG as DanhGiaRAG
from vietnamese_ai.rag.graph import GraphExtractor as GraphExtractor
from vietnamese_ai.rag.graph import GraphRetriever as GraphRetriever
from vietnamese_ai.rag.graph import NetworkXStore as NetworkXStore
from vietnamese_ai.rag.multimodal import ImageEmbedder as ImageEmbedder
from vietnamese_ai.rag.multimodal import MultimodalStore as MultimodalStore
from vietnamese_ai.rag.rag_pipeline import RAGPipeline as RAGPipeline
from vietnamese_ai.rag.reranker import SapXepLai as SapXepLai
from vietnamese_ai.rag.retriever import TrichXuat as TrichXuat
from vietnamese_ai.rag.vector_store import CSDLVector as CSDLVector
from vietnamese_ai.rag.vietnamese_context import BoToiUuNguCanhHanhChinh as BoToiUuNguCanhHanhChinh
from vietnamese_ai.rag.vietnamese_context import (
    VietnameseAdministrativeRAG as VietnameseAdministrativeRAG,
)
from vietnamese_ai.realtime.pipeline import PipelineThoiGianThuc as PipelineThoiGianThuc
from vietnamese_ai.realtime.processor import XuLyStream as XuLyStream
from vietnamese_ai.registry.model_hub import ModelHub as ModelHub
from vietnamese_ai.registry.model_registry import QuanLyMoHinh as QuanLyMoHinh
from vietnamese_ai.salm.adaptive_lora import AdaptiveLoRA as AdaptiveLoRA
from vietnamese_ai.salm.self_consistency import SelfConsistency as SelfConsistency
from vietnamese_ai.salm.self_data import SinhDuLieuTuDong as SinhDuLieuTuDong
from vietnamese_ai.salm.self_refine import SelfRefine as SelfRefine
from vietnamese_ai.salm.test_time_training import TestTimeTraining as TestTimeTraining
from vietnamese_ai.sandbox.executor import ThucThiDocLap as ThucThiDocLap
from vietnamese_ai.security.agent_sandbox import MoiTruongCachLy as MoiTruongCachLy
from vietnamese_ai.security.data_defense import BoGiamSatTichTuDuLieu as BoGiamSatTichTuDuLieu
from vietnamese_ai.security.data_defense import CongAnToanDauRa as CongAnToanDauRa
from vietnamese_ai.security.data_defense import CongChatLuongDauVao as CongChatLuongDauVao
from vietnamese_ai.security.data_defense import DatasetDefenseShield as DatasetDefenseShield
from vietnamese_ai.security.data_defense import ImageSecurityShield as ImageSecurityShield
from vietnamese_ai.security.data_defense import InputQualityGate as InputQualityGate
from vietnamese_ai.security.data_defense import KhienBaoVeHinhAnh as KhienBaoVeHinhAnh
from vietnamese_ai.security.data_defense import OutputSafetyQualityGate as OutputSafetyQualityGate
from vietnamese_ai.security.data_defense import PhongThuDuLieu as PhongThuDuLieu
from vietnamese_ai.security.data_defense import (
    StreamingPoisoningMonitor as StreamingPoisoningMonitor,
)
from vietnamese_ai.security.data_sanitizer import DataSanitizer as DataSanitizer
from vietnamese_ai.security.llm_firewall import AIFirewallShield as AIFirewallShield
from vietnamese_ai.security.llm_firewall import KhienBaoVeAI as KhienBaoVeAI
from vietnamese_ai.security.llm_firewall import TuongLuaAI as TuongLuaAI
from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework as TuongLuaBaoVeFramework
from vietnamese_ai.security.verifiable_inference import (
    CayMerkleSuyLuan as CayMerkleSuyLuan,
)
from vietnamese_ai.security.verifiable_inference import (
    InferenceMerkleTree as InferenceMerkleTree,
)
from vietnamese_ai.security.verifiable_inference import (
    VerifiableInference as VerifiableInference,
)
from vietnamese_ai.security.verifiable_inference import (
    XacThucSuyLuanMaoHiem as XacThucSuyLuanMaoHiem,
)
from vietnamese_ai.serving.batch_server import MayChuBatch as MayChuBatch
from vietnamese_ai.serving.continuous_batching import BoPhanLoLienTuc as BoPhanLoLienTuc
from vietnamese_ai.serving.continuous_batching import ContinuousBatcher as ContinuousBatcher
from vietnamese_ai.serving.paged_attention import BlockManager as BlockManager
from vietnamese_ai.serving.paged_attention import (
    PagedAttentionSimulation as PagedAttentionSimulation,
)
from vietnamese_ai.serving.rate_limiter import BoGioiHanTocDo as BoGioiHanTocDo
from vietnamese_ai.serving.speculative import SpeculativeEngine as SpeculativeEngine
from vietnamese_ai.serving.streaming import MayChuStream as MayChuStream
from vietnamese_ai.solutions.cskh import CustomerSupportBot as CustomerSupportBot
from vietnamese_ai.solutions.cskh import TongDaiCSKH as TongDaiCSKH
from vietnamese_ai.solutions.phap_ly import LegalAssistant as LegalAssistant
from vietnamese_ai.solutions.phap_ly import TroLyPhapLy as TroLyPhapLy
from vietnamese_ai.solutions.trich_xuat import SmartExtractor as SmartExtractor
from vietnamese_ai.solutions.trich_xuat import TrichXuatThongMinh as TrichXuatThongMinh
from vietnamese_ai.ssm.jamba_block import JambaBlock as JambaBlock
from vietnamese_ai.ssm.jamba_block import SparseMoE as SparseMoE
from vietnamese_ai.ssm.mamba_block import MambaBlock as MambaBlock
from vietnamese_ai.studio.builder import StudioKeoTha as StudioKeoTha
from vietnamese_ai.timeseries.forecaster import DuDoanChuoiThoiGian as DuDoanChuoiThoiGian
from vietnamese_ai.transformer.attention import MultiHeadAttention as MultiHeadAttention
from vietnamese_ai.transformer.gpt_model import GPTModel as GPTModel
from vietnamese_ai.transformer.model import TransformerModel as TransformerModel
from vietnamese_ai.transformer.moe_rope import GroupedQueryAttention as GroupedQueryAttention
from vietnamese_ai.transformer.moe_rope import RotaryPositionEmbedding as RotaryPositionEmbedding
from vietnamese_ai.transformer.moe_rope import SparseMoELayer as SparseMoELayer
from vietnamese_ai.transformer.pretrainer import PreTrainer as PreTrainer
from vietnamese_ai.transformer.pretrainer import TextDataset as TextDataset
from vietnamese_ai.transformer.tokenizer import VietnameseTokenizer as VietnameseTokenizer
from vietnamese_ai.ui.studio_dashboard import MayChuStudioDashboard as MayChuStudioDashboard
from vietnamese_ai.ui.studio_dashboard import StudioDashboard as StudioDashboard
from vietnamese_ai.ui.web_app import UngDungWeb as UngDungWeb
from vietnamese_ai.utils.io_utils import LuuTai as LuuTai
from vietnamese_ai.utils.lazy_loader import DependencyManager as DependencyManager
from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc as QuanLyPhuThuoc
from vietnamese_ai.utils.logger import Logger as Logger
from vietnamese_ai.utils.metrics import Metrics as Metrics
from vietnamese_ai.utils.validators import Validator as Validator
from vietnamese_ai.vision.image_classifier import PhanLoaiHinhAnh as PhanLoaiHinhAnh
from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh as TrichXuatVanBanAnh
from vietnamese_ai.vision.vlm import MoHinhThiGiacNgonNgu as MoHinhThiGiacNgonNgu
from vietnamese_ai.vision.vlm import VietnameseVLM as VietnameseVLM
from vietnamese_ai.visualization.plots import BieuDo as BieuDo
from vietnamese_ai.workflow.dag_engine import DongChayDAG as DongChayDAG

__version__: str
__author__: str
__all__: list[str]
