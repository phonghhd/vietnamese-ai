import hashlib
import os
import re
from typing import List, Union

import numpy as np


class ImageEmbedder:
    """
    Trích xuất embedding ảnh/văn bản cho Multi-modal RAG.

    Khi đường dẫn ảnh tồn tại trên đĩa, trích xuất đặc trưng thị giác thực tế
    512 chiều bằng Pillow + NumPy (RGB histogram, 4x4 spatial pyramid, HOG-lite
    Sobel gradient orientations, aHash/dHash và moment màu).
    Khi có `sentence-transformers` / CLIP, có thể bật `dung_clip=True`.
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        dung_clip: bool = False,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.dung_clip = dung_clip
        self.seed = seed
        self._da_tai_model = False
        self._clip_model = None

    def _tai_model(self) -> None:
        """Tải CLIP thật nếu `dung_clip=True` và dependency có sẵn."""
        if not self.dung_clip:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._clip_model = SentenceTransformer(self.model_name)
        except Exception:
            self._clip_model = None

    @staticmethod
    def _vector_tu_khoa(khoa: str, dim: int = 512) -> List[float]:
        """Feature hashing đa n-gram xác định trên chuỗi văn bản/khóa."""
        vec = np.zeros(dim, dtype=np.float64)
        text = khoa.strip().lower()
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        features: List[str] = [f"raw:{text}"]
        for w in words:
            features.append(f"w1:{w}")
        for i in range(len(words) - 1):
            features.append(f"w2:{words[i]}_{words[i + 1]}")
        compact = re.sub(r"\s+", " ", text)
        for n in (3, 4, 5):
            for i in range(max(0, len(compact) - n + 1)):
                features.append(f"c{n}:{compact[i : i + n]}")

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=16).digest()
            h1 = int.from_bytes(digest[:8], "little")
            h2 = int.from_bytes(digest[8:], "little")
            idx1 = h1 % dim
            sign1 = 1.0 if (h1 & 1) == 0 else -1.0
            idx2 = h2 % dim
            sign2 = 1.0 if (h2 & 1) == 0 else -1.0
            vec[idx1] += sign1
            vec[idx2] += 0.5 * sign2

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @classmethod
    def _trich_xuat_dac_trung_anh(cls, duong_dan: str, dim: int = 512) -> List[float]:
        """Trích xuất vector đặc trưng thị giác 512 chiều từ file ảnh thực tế."""
        if not os.path.isfile(duong_dan):
            return cls._vector_tu_khoa(f"img:{duong_dan}", dim=dim)

        from vietnamese_ai.security.data_defense import KhienBaoVeHinhAnh

        khien = KhienBaoVeHinhAnh()
        bc_pre = khien.quet_anh(duong_dan, chan_khi_nguy_hiem=False)
        for de_doa in bc_pre.get("moi_de_doa", []):
            if any(
                tk in de_doa
                for tk in ("Decompression Bomb", "ZIP/JAR", "ELF", "PE/EXE", "Shell script", "JavaScript", "XXE", "Pickle")
            ):
                raise ValueError(f"Bảo mật ImageEmbedder chặn tệp ảnh nguy hiểm: {de_doa}")

        try:
            from PIL import Image

            arr_cdr, _ = khien.khu_doc_anh_cdr(duong_dan)
            rgb_img = Image.fromarray(arr_cdr).convert("RGB").resize((64, 64))
            arr = np.asarray(rgb_img, dtype=np.float64) / 255.0
        except ValueError:
            raise
        except Exception:
            return cls._vector_tu_khoa(f"img:{duong_dan}", dim=dim)

        # 1. RGB Histograms: 3 kênh x 32 bins = 96 chiều
        hist_feats = []
        for c in range(3):
            h, _ = np.histogram(arr[:, :, c], bins=32, range=(0.0, 1.0), density=False)
            hist_feats.append(h.astype(np.float64) / (64.0 * 64.0))
        f_hist = np.concatenate(hist_feats)  # 96

        # 2. Spatial Pyramid 4x4: mean và std mỗi ô (16 ô x 3 kênh x 2 = 96 chiều)
        spatial_feats = []
        for r in range(4):
            for c in range(4):
                patch = arr[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16, :]
                spatial_feats.append(np.mean(patch, axis=(0, 1)))
                spatial_feats.append(np.std(patch, axis=(0, 1)))
        f_spatial = np.concatenate(spatial_feats)  # 96

        # 3. HOG-lite (Sobel gradient 4x4 cells x 8 orientation bins = 128 chiều)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.hypot(gx, gy)
        ori = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # [0, 1]

        hog_feats = []
        for r in range(4):
            for c in range(4):
                m_patch = mag[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16].ravel()
                o_patch = ori[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16].ravel()
                hist_hog, _ = np.histogram(
                    o_patch, bins=8, range=(0.0, 1.0), weights=m_patch
                )
                hog_feats.append(hist_hog / (np.linalg.norm(hist_hog) + 1e-6))
        f_hog = np.concatenate(hog_feats)  # 128

        # 4. Perceptual aHash (64 chiều) + dHash (64 chiều) = 128 chiều
        small8 = gray.reshape(8, 8, 8, 8).mean(axis=(1, 3))
        ahash = (small8 > np.mean(small8)).astype(np.float64).ravel() * 2.0 - 1.0
        dhash = (np.roll(small8, -1, axis=1) > small8).astype(np.float64).ravel() * 2.0 - 1.0
        f_hash = np.concatenate([ahash, dhash]) * 0.25  # 128

        # 5. Color moments + Radial frequency profile = 64 chiều
        yy, xx = np.mgrid[0:64, 0:64]
        rad = np.hypot(yy - 31.5, xx - 31.5)
        max_rad = np.max(rad) + 1e-6
        rad_norm = rad / max_rad
        rad_hist, _ = np.histogram(rad_norm.ravel(), bins=32, range=(0.0, 1.0), weights=gray.ravel())
        rad_edge, _ = np.histogram(rad_norm.ravel(), bins=32, range=(0.0, 1.0), weights=mag.ravel())
        f_rad = np.concatenate([
            rad_hist / (np.linalg.norm(rad_hist) + 1e-6),
            rad_edge / (np.linalg.norm(rad_edge) + 1e-6),
        ])  # 64

        vec = np.concatenate([f_hist, f_spatial, f_hog, f_hash, f_rad])
        if vec.shape[0] < dim:
            vec = np.pad(vec, (0, dim - vec.shape[0]))
        elif vec.shape[0] > dim:
            vec = vec[:dim]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def nhung_hinh_anh(self, image_paths: Union[str, List[str]]) -> List[List[float]]:
        """Nhúng ảnh thành vector 512 chiều từ pixel thật hoặc CLIP."""
        if not self._da_tai_model:
            self._tai_model()
            self._da_tai_model = True

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if self._clip_model is not None:
            try:
                return self._clip_model.encode(image_paths).tolist()
            except Exception:
                pass

        return [self._trich_xuat_dac_trung_anh(p) for p in image_paths]

    def nhung_van_ban(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Nhúng văn bản cùng không gian 512 chiều với ảnh."""
        if not self._da_tai_model:
            self._tai_model()
            self._da_tai_model = True

        if isinstance(texts, str):
            texts = [texts]

        if self._clip_model is not None:
            try:
                return self._clip_model.encode(texts).tolist()
            except Exception:
                pass

        return [self._vector_tu_khoa(f"txt:{t}") for t in texts]
