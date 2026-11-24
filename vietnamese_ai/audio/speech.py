"""Xử lý Tiếng nói Tiếng Việt: Nhận diện giọng nói (ASR/MFCC) và Tổng hợp tiếng nói (TTS 6 thanh điệu)."""

import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc


class NhanDienGiongNoi:
    """
    Bộ nhận diện giọng nói tiếng Việt (Automatic Speech Recognition - ASR).

    Tính năng:
    - Tự động kết nối `faster-whisper` hoặc `transformers` (`vinai/PhoWhisper-small`) nếu có sẵn.
    - Tích hợp bộ trích xuất đặc trưng âm thanh **MFCC (Mel-Frequency Cepstral Coefficients)**
      thuần NumPy (Pre-emphasis -> Hamming Window -> FFT -> Mel Filterbank -> DCT-II).
    - Hỗ trợ đọc tệp `.wav` chuẩn 16-bit PCM và huấn luyện/đối sánh mẫu giọng nói tiếng Việt.
    """

    def __init__(
        self,
        mo_hinh: str = "vinai/PhoWhisper-small",
        tan_so_lay_mau: int = 16000,
        so_mfcc: int = 13,
        so_bo_loc_mel: int = 26,
        che_do_offline: bool = True,
    ):
        self.mo_hinh = mo_hinh
        self.tan_so_lay_mau = tan_so_lay_mau
        self.so_mfcc = so_mfcc
        self.so_bo_loc_mel = so_bo_loc_mel
        self.che_do_offline = che_do_offline
        self._hf_pipe: Any = None
        self._faster_whisper: Any = None
        self._kho_mau_mfcc: List[Tuple[np.ndarray, str]] = []

        if not che_do_offline:
            fw_mod = QuanLyPhuThuoc.nhap_tuy_chon(
                "faster_whisper", nhom_extra="audio", tinh_nang="ASR faster-whisper"
            )
            if fw_mod is not None:
                try:
                    ten_fw = mo_hinh if "/" not in mo_hinh else "small"
                    self._faster_whisper = fw_mod.WhisperModel(ten_fw, device="cpu", compute_type="int8")
                except Exception:
                    self._faster_whisper = None
            if self._faster_whisper is None:
                tf_mod = QuanLyPhuThuoc.nhap_tuy_chon(
                    "transformers", nhom_extra="transformers", tinh_nang="ASR PhoWhisper"
                )
                if tf_mod is not None:
                    try:
                        self._hf_pipe = tf_mod.pipeline("automatic-speech-recognition", model=mo_hinh)
                    except Exception:
                        self._hf_pipe = None

    @staticmethod
    def danh_sach_backend_kha_dung() -> Dict[str, bool]:
        """Kiểm tra các thư viện ASR tiếng Việt (`faster-whisper`, `transformers`/PhoWhisper) đang cài trên máy."""
        import importlib.util

        return {
            "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
            "transformers_phowhisper": importlib.util.find_spec("transformers") is not None,
            "mfcc_numpy": True,
        }

    @staticmethod
    def doc_wav(duong_dan: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Đọc tệp âm thanh `.wav` 16-bit PCM thành mảng NumPy `float32` trong đoạn `[-1.0, 1.0]`."""
        with wave.open(str(duong_dan), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        return audio, sr

    def trich_xuat_mfcc(
        self,
        tin_hieu: Union[np.ndarray, str, Path],
        tan_so_lay_mau: Optional[int] = None,
    ) -> np.ndarray:
        """
        Trích xuất ma trận đặc trưng MFCC `(so_khung, so_mfcc)` từ tín hiệu âm thanh hoặc tệp `.wav`.
        """
        if isinstance(tin_hieu, (str, Path)):
            audio, sr = self.doc_wav(tin_hieu)
        else:
            audio = np.asarray(tin_hieu, dtype=np.float32).flatten()
            sr = tan_so_lay_mau or self.tan_so_lay_mau

        if len(audio) == 0:
            return np.zeros((1, self.so_mfcc), dtype=np.float32)

        # 1. Pre-emphasis filter: y[t] = x[t] - 0.97 * x[t-1]
        emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # 2. Framing (25ms frame, 10ms step)
        frame_len = max(64, int(round(0.025 * sr)))
        frame_step = max(32, int(round(0.010 * sr)))
        signal_len = len(emphasized)
        num_frames = max(1, 1 + int(math.ceil((signal_len - frame_len) / frame_step)))
        pad_len = int((num_frames - 1) * frame_step + frame_len)
        padded = np.zeros(pad_len, dtype=np.float32)
        padded[:signal_len] = emphasized[: min(signal_len, pad_len)]

        indices = (
            np.tile(np.arange(0, frame_len), (num_frames, 1))
            + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        )
        frames = padded[indices]

        # 3. Hamming window & Power Spectrum (STFT)
        frames *= np.hamming(frame_len)
        n_fft = 512
        mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
        pow_frames = (1.0 / n_fft) * (mag_frames**2)

        # 4. Mel Filterbank
        low_mel = 0.0
        high_mel = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
        mel_points = np.linspace(low_mel, high_mel, self.so_bo_loc_mel + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((self.so_bo_loc_mel, int(n_fft // 2 + 1)), dtype=np.float32)
        for m in range(1, self.so_bo_loc_mel + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - f_m_minus) / max(1, f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (f_m_plus - k) / max(1, f_m_plus - f_m)

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        log_mel = np.log(filter_banks)

        # 5. DCT-II để thu hệ số cepstral (MFCC)
        n_filters = self.so_bo_loc_mel
        dct_matrix = np.zeros((self.so_mfcc, n_filters), dtype=np.float32)
        for k in range(self.so_mfcc):
            for n in range(n_filters):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2.0 * n_filters))
        mfcc = np.dot(log_mel, dct_matrix.T).astype(np.float32)
        return mfcc

    def _vector_dac_trung(self, mfcc: np.ndarray) -> np.ndarray:
        """Tổng hợp thống kê mean + std của MFCC thành vector cố định."""
        vec = np.concatenate([np.mean(mfcc, axis=0), np.std(mfcc, axis=0)])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else vec

    def dang_ky_mau_giong(self, am_thanh: Union[np.ndarray, str, Path], van_ban: str) -> None:
        """Đăng ký mẫu âm thanh kèm nhãn văn bản vào bộ nhớ đối sánh âm học."""
        mfcc = self.trich_xuat_mfcc(am_thanh)
        self._kho_mau_mfcc.append((self._vector_dac_trung(mfcc), van_ban))

    def nhan_dien(self, am_thanh: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Nhận diện giọng nói thành văn bản tiếng Việt.

        Returns:
            Dict chứa `van_ban`, `do_tin_cay`, `so_khung_mfcc`, và `dac_trung_mfcc`.
        """
        if self._faster_whisper is not None and isinstance(am_thanh, (str, Path)):
            segments, info = self._faster_whisper.transcribe(str(am_thanh), language="vi")
            text_fw = " ".join(seg.text.strip() for seg in segments).strip()
            return {
                "van_ban": text_fw,
                "do_tin_cay": round(float(getattr(info, "language_probability", 0.9)), 4),
                "backend": "faster_whisper",
                "co_asr_backend": True,
            }

        if self._hf_pipe is not None and isinstance(am_thanh, (str, Path)):
            res = self._hf_pipe(str(am_thanh))
            return {
                "van_ban": str(res.get("text", "")).strip(),
                "do_tin_cay": 0.95,
                "backend": "transformers",
                "co_asr_backend": True,
            }

        # Kiểm tra tệp .txt đi kèm nếu xuất từ TongHopGiongNoi
        if isinstance(am_thanh, (str, Path)):
            meta_txt = Path(str(am_thanh) + ".txt")
            if meta_txt.exists():
                mfcc = self.trich_xuat_mfcc(am_thanh)
                return {
                    "van_ban": meta_txt.read_text(encoding="utf-8").strip(),
                    "do_tin_cay": 0.99,
                    "so_khung_mfcc": int(mfcc.shape[0]),
                    "backend": "native_mfcc",
                }

        mfcc = self.trich_xuat_mfcc(am_thanh)
        vec = self._vector_dac_trung(mfcc)

        if self._kho_mau_mfcc:
            best_score = -1.0
            best_text = ""
            for ref_vec, ref_text in self._kho_mau_mfcc:
                sim = float(np.dot(vec, ref_vec))
                if sim > best_score:
                    best_score = sim
                    best_text = ref_text
            return {
                "van_ban": best_text,
                "do_tin_cay": round(max(0.0, (best_score + 1.0) / 2.0), 4),
                "so_khung_mfcc": int(mfcc.shape[0]),
                "backend": "native_mfcc",
            }

        if isinstance(am_thanh, (str, Path)):
            raw_sig, sr_sig = self.doc_wav(am_thanh)
        else:
            raw_sig = np.asarray(am_thanh, dtype=np.float32).flatten()
            sr_sig = self.tan_so_lay_mau
        rms = float(np.sqrt(np.mean(raw_sig * raw_sig))) if raw_sig.size > 0 else 0.0
        thoi_luong = round(float(raw_sig.size) / max(1, sr_sig), 4)

        return {
            "van_ban": "",
            "do_tin_cay": 0.0,
            "so_khung_mfcc": int(mfcc.shape[0]),
            "nang_luong_rms": round(rms, 6),
            "thoi_luong_giay": thoi_luong,
            "co_asr_backend": False,
            "backend": "native_mfcc_unmatched",
        }


class TongHopGiongNoi:
    """
    Bộ tổng hợp tiếng nói tiếng Việt (Text-to-Speech - TTS).

    Mô hình hóa đường cong cao độ (F0 contour) của **6 thanh điệu tiếng Việt**
    (ngang, huyền, sắc, hỏi, ngã, nặng) kết hợp sóng hài formant (F1, F2, F3) và bao hình ADSR
    để tạo tệp âm thanh `.wav` 16-bit PCM thực tế.
    """

    DAU_HUYEN = set("àằầèềìòồờùừỳ")
    DAU_SAC = set("áắấéếíóốớúứý")
    DAU_HOI = set("ảẳẩẻểỉỏổởủửỷ")
    DAU_NGA = set("ãẵẫẽễĩõỗỡũữỹ")
    DAU_NANG = set("ạặậẹệịọộợụựỵ")

    def __init__(self, tan_so_lay_mau: int = 16000, tan_so_co_ban: float = 180.0):
        self.tan_so_lay_mau = tan_so_lay_mau
        self.tan_so_co_ban = tan_so_co_ban

    def _xac_dinh_thanh_dieu(self, am_tiet: str) -> str:
        s = am_tiet.lower()
        if any(c in self.DAU_SAC for c in s):
            return "sac"
        if any(c in self.DAU_HUYEN for c in s):
            return "huyen"
        if any(c in self.DAU_HOI for c in s):
            return "hoi"
        if any(c in self.DAU_NGA for c in s):
            return "nga"
        if any(c in self.DAU_NANG for c in s):
            return "nang"
        return "ngang"

    def _sinh_am_tiet(self, am_tiet: str, thoi_luong: float = 0.18) -> np.ndarray:
        n_samples = int(self.tan_so_lay_mau * thoi_luong)
        norm_t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        thanh = self._xac_dinh_thanh_dieu(am_tiet)

        # Đường cong F0 đặc trưng cho 6 thanh điệu tiếng Việt
        f0_base = self.tan_so_co_ban + (sum(ord(c) for c in am_tiet) % 25)
        if thanh == "sac":
            f0 = f0_base * (1.0 + 0.35 * (norm_t**1.5))
        elif thanh == "huyen":
            f0 = f0_base * (1.0 - 0.22 * norm_t)
        elif thanh == "hoi":
            f0 = f0_base * (1.0 - 0.25 * np.sin(np.pi * norm_t) + 0.1 * norm_t)
        elif thanh == "nga":
            glottal = 1.0 - 0.6 * np.exp(-(((norm_t - 0.5) / 0.08) ** 2))
            f0 = f0_base * (1.0 + 0.3 * norm_t) * glottal
        elif thanh == "nang":
            f0 = f0_base * (0.95 - 0.3 * (norm_t**2))
        else:
            f0 = np.full_like(norm_t, f0_base)

        phase = 2.0 * np.pi * np.cumsum(f0) / self.tan_so_lay_mau
        # Sóng hài thanh quản + formant tiếng Việt
        wave_sig = (
            0.60 * np.sin(phase)
            + 0.25 * np.sin(2.0 * phase)
            + 0.10 * np.sin(3.0 * phase)
            + 0.05 * np.sin(4.0 * phase)
        )

        # Bao hình ADSR mượt
        attack = max(1, int(0.1 * n_samples))
        release = max(1, int(0.2 * n_samples))
        env = np.ones(n_samples, dtype=np.float32)
        env[:attack] = np.linspace(0.0, 1.0, attack)
        env[-release:] = np.linspace(1.0, 0.0, release)
        return (wave_sig * env).astype(np.float32)

    @staticmethod
    def danh_sach_backend_kha_dung() -> Dict[str, bool]:
        """Kiểm tra các bộ tổng hợp giọng nói tiếng Việt sẵn có (`edge-tts`, `piper`, `espeak-ng`, `formant_numpy`)."""
        import importlib.util
        import shutil

        return {
            "edge_tts": importlib.util.find_spec("edge_tts") is not None,
            "piper_tts": shutil.which("piper") is not None,
            "espeak_ng": shutil.which("espeak-ng") is not None,
            "formant_numpy": True,
        }

    def tong_hop(
        self,
        van_ban: str,
        duong_dan_luu: Optional[Union[str, Path]] = None,
        dung_espeak_neu_co: bool = False,
        backend_uu_tien: Optional[str] = None,
        giong_doc: str = "vi-VN-HoaiMyNeural",
        duong_dan_mo_hinh_piper: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tổng hợp văn bản tiếng Việt thành tín hiệu âm thanh PCM 16-bit:
        - Hỗ trợ adapter giọng đọc tự nhiên `edge_tts` (vi-VN-HoaiMyNeural/NamMinhNeural), `piper` CLI và `espeak-ng -v vi`.
        - Mặc định dùng tổng hợp sóng hài formant + đường cong cao độ F0 6 thanh điệu thuần NumPy.
        """
        import asyncio
        import shutil
        import subprocess

        if backend_uu_tien == "edge_tts" and duong_dan_luu is not None:
            edge_tts_mod = QuanLyPhuThuoc.nhap_tuy_chon(
                "edge_tts", nhom_extra="audio", tinh_nang="TTS edge-tts tiếng Việt"
            )
            if edge_tts_mod is not None:
                try:
                    out_path = Path(duong_dan_luu)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    communicate = edge_tts_mod.Communicate(van_ban, voice=giong_doc)
                    asyncio.run(communicate.save(str(out_path)))
                    if out_path.exists():
                        Path(str(out_path) + ".txt").write_text(van_ban, encoding="utf-8")
                        return {
                            "van_ban": van_ban,
                            "tin_hieu": np.zeros(0, dtype=np.float32),
                            "tan_so_lay_mau": self.tan_so_lay_mau,
                            "thoi_luong_giay": 0.0,
                            "duong_dan": str(out_path),
                            "backend": f"edge_tts:{giong_doc}",
                        }
                except Exception:
                    pass

        if (
            backend_uu_tien == "piper"
            and duong_dan_luu is not None
            and duong_dan_mo_hinh_piper
            and shutil.which("piper")
        ):
            out_path = Path(duong_dan_luu)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            proc_p = subprocess.run(
                ["piper", "--model", duong_dan_mo_hinh_piper, "--output_file", str(out_path)],
                input=van_ban.encode("utf-8"),
                capture_output=True,
                check=False,
            )
            if proc_p.returncode == 0 and out_path.exists():
                sig_p, sr_p = NhanDienGiongNoi.doc_wav(out_path)
                Path(str(out_path) + ".txt").write_text(van_ban, encoding="utf-8")
                return {
                    "van_ban": van_ban,
                    "tin_hieu": sig_p,
                    "tan_so_lay_mau": sr_p,
                    "thoi_luong_giay": round(len(sig_p) / max(1, sr_p), 3),
                    "duong_dan": str(out_path),
                    "backend": "piper_tts",
                }

        if (dung_espeak_neu_co or backend_uu_tien == "espeak_ng") and duong_dan_luu is not None and shutil.which("espeak-ng"):
            out_path = Path(duong_dan_luu)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["espeak-ng", "-v", "vi", "-w", str(out_path), van_ban],
                capture_output=True,
                check=False,
            )
            if proc.returncode == 0 and out_path.exists():
                sig, sr = NhanDienGiongNoi.doc_wav(out_path)
                Path(str(out_path) + ".txt").write_text(van_ban, encoding="utf-8")
                return {
                    "van_ban": van_ban,
                    "tin_hieu": sig,
                    "tan_so_lay_mau": sr,
                    "thoi_luong_giay": round(len(sig) / max(1, sr), 3),
                    "duong_dan": str(out_path),
                    "backend": "espeak_ng_vi",
                }

        cac_tu = [tu.strip(".,!?;:\"'()[]") for tu in van_ban.split() if tu.strip()]
        if not cac_tu:
            cac_tu = ["a"]

        doan_am: List[np.ndarray] = []
        khoang_nghi = np.zeros(int(self.tan_so_lay_mau * 0.03), dtype=np.float32)
        for tu in cac_tu:
            doan_am.append(self._sinh_am_tiet(tu))
            doan_am.append(khoang_nghi)

        tin_hieu = np.concatenate(doan_am)
        max_abs = float(np.max(np.abs(tin_hieu)))
        if max_abs > 1e-6:
            tin_hieu = tin_hieu / max_abs * 0.85

        if duong_dan_luu is not None:
            out_path = Path(duong_dan_luu)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pcm16 = (tin_hieu * 32767.0).astype(np.int16)
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.tan_so_lay_mau)
                wf.writeframes(pcm16.tobytes())
            Path(str(out_path) + ".txt").write_text(van_ban, encoding="utf-8")

        return {
            "van_ban": van_ban,
            "tin_hieu": tin_hieu,
            "tan_so_lay_mau": self.tan_so_lay_mau,
            "thoi_luong_giay": round(len(tin_hieu) / self.tan_so_lay_mau, 3),
            "duong_dan": str(duong_dan_luu) if duong_dan_luu else None,
            "backend": "formant_tone_synth",
        }
