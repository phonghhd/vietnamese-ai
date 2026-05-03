"""VietnameseTokenizer - Tokenizer tiếng Việt (BPE, WordPiece)."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vietnamese_ai.utils.logger import Logger


class VietnameseTokenizer:
    """
    Tokenizer tiếng Việt hỗ trợ BPE và WordPiece.

    Tính năng:
    - Byte-Pair Encoding (BPE)
    - WordPiece (BERT-style)
    - Tách từ tiếng Việt cơ bản
    - Special tokens ([CLS], [SEP], [PAD], [UNK], [MASK])
    - Encode/Decode
    - Save/Load

    Sử dụng:
        >>> tok = VietnameseTokenizer(che_do="bpe", kich_thuoc_vocab=5000)
        >>> tok.huan_luyen(cac_van_ban)
        >>> ids = tok.ma_hoa("Học máy rất hay")
        >>> text = tok.giai_ma(ids)
    """

    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }

    def __init__(
        self,
        che_do: str = "bpe",
        kich_thuoc_vocab: int = 5000,
        do_dai_toi_da: int = 512,
    ):
        if che_do not in ("bpe", "wordpiece"):
            raise ValueError(f"che_do phải là 'bpe' hoặc 'wordpiece', nhận: '{che_do}'")
        if kich_thuoc_vocab < 100:
            raise ValueError("kich_thuoc_vocab phải >= 100")

        self.che_do = che_do
        self.kich_thuoc_vocab = kich_thuoc_vocab
        self.do_dai_toi_da = do_dai_toi_da
        self.logger = Logger("VietnameseTokenizer")

        self._vocab: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self._vocab_nguoc: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self._merges: List[Tuple[str, str]] = []
        self._da_huan_luyen = False

    def _tach_tu_co_ban(self, text: str) -> List[str]:
        """Tách văn bản thành các từ/tokens cơ bản."""
        text = text.lower().strip()
        text = re.sub(r'([.,!?;:(){}[\]])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.split()

    def _tach_ky_tu(self, tu: str) -> List[str]:
        """Tách từ thành các ký tự (cho BPE init)."""
        if len(tu) <= 1:
            return [tu]
        return list(tu) + ["</w>"]

    def _dem_cap(self, cac_tokens: List[List[str]]) -> Counter:
        """Đếm tần suất các cặp token liên tiếp."""
        cap = Counter()
        for tokens in cac_tokens:
            for i in range(len(tokens) - 1):
                cap[(tokens[i], tokens[i + 1])] += 1
        return cap

    def _tron_cap(self, tokens: List[str], cap: Tuple[str, str]) -> List[str]:
        """Gộp cặp token thành token mới."""
        ket_qua = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == cap[0] and tokens[i + 1] == cap[1]:
                ket_qua.append(cap[0] + cap[1])
                i += 2
            else:
                ket_qua.append(tokens[i])
                i += 1
        return ket_qua

    def huan_luyen(self, cac_van_ban: List[str]) -> Dict[str, int]:
        """
        Huấn luyện tokenizer trên corpus.

        Args:
            cac_van_ban: Danh sách văn bản

        Returns:
            Dict thống kê
        """
        self.logger.info(f"Huấn luyện {self.che_do.upper()} tokenizer ({len(cac_van_ban)} văn bản)")

        cac_tu = []
        for vb in cac_van_ban:
            cac_tu.extend(self._tach_tu_co_ban(vb))

        tu_dem = Counter(cac_tu)

        if self.che_do == "bpe":
            self._huan_luyen_bpe(cac_tu, tu_dem)
        else:
            self._huan_luyen_wordpiece(cac_tu, tu_dem)

        self._da_huan_luyen = True

        self.logger.info(f"Hoàn tất: vocab_size={len(self._vocab)}")
        return {
            "che_do": self.che_do,
            "vocab_size": len(self._vocab),
            "so_merges": len(self._merges),
        }

    def _huan_luyen_bpe(self, cac_tu: List[str], tu_dem: Counter) -> None:
        """Huấn luyện BPE."""
        cac_tokens = []
        for tu, dem in tu_dem.items():
            cac_tokens.extend([self._tach_ky_tu(tu)] * min(dem, 100))

        so_merges = self.kich_thuoc_vocab - len(self.SPECIAL_TOKENS)

        for _ in range(so_merges):
            if len(self._vocab) >= self.kich_thuoc_vocab:
                break

            cap_dem = self._dem_cap(cac_tokens)
            if not cap_dem:
                break

            cap_tot_nhat = cap_dem.most_common(1)[0][0]
            self._merges.append(cap_tot_nhat)

            token_moi = cap_tot_nhat[0] + cap_tot_nhat[1]
            if token_moi not in self._vocab:
                self._vocab[token_moi] = len(self._vocab)
                self._vocab_nguoc[len(self._vocab) - 1] = token_moi

            cac_tokens = [self._tron_cap(tokens, cap_tot_nhat) for tokens in cac_tokens]

    def _huan_luyen_wordpiece(self, cac_tu: List[str], tu_dem: Counter) -> None:
        """Huấn luyện WordPiece."""
        tu_sap_xep = sorted(tu_dem.items(), key=lambda x: -x[1])

        for tu, dem in tu_sap_xep:
            if len(self._vocab) >= self.kich_thuoc_vocab:
                break

            ky_tu = list(tu)
            for kc in ky_tu:
                if kc not in self._vocab and len(self._vocab) < self.kich_thuoc_vocab:
                    self._vocab[kc] = len(self._vocab)
                    self._vocab_nguoc[len(self._vocab) - 1] = kc

            if tu not in self._vocab and len(self._vocab) < self.kich_thuoc_vocab:
                self._vocab[tu] = len(self._vocab)
                self._vocab_nguoc[len(self._vocab) - 1] = tu

    def ma_hoa(self, text: str, them_cls_sep: bool = True) -> List[int]:
        """
        Mã hóa văn bản thành token IDs.

        Args:
            text: Văn bản đầu vào
            them_cls_sep: Thêm [CLS] và [SEP]

        Returns:
            Danh sách token IDs
        """
        if not self._da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện. Gọi huan_luyen() trước.")

        tu_list = self._tach_tu_co_ban(text)
        ids = []

        if them_cls_sep:
            ids.append(self._vocab["[CLS]"])

        for tu in tu_list:
            if tu in self._vocab:
                ids.append(self._vocab[tu])
            elif self.che_do == "bpe":
                ids.extend(self._bpe_encode_tu(tu))
            else:
                ids.extend(self._wordpiece_encode_tu(tu))

        if them_cls_sep:
            ids.append(self._vocab["[SEP]"])

        ids = ids[:self.do_dai_toi_da]
        return ids

    def _bpe_encode_tu(self, tu: str) -> List[int]:
        """BPE encode một từ."""
        tokens = self._tach_ky_tu(tu)
        for cap in self._merges:
            tokens = self._tron_cap(tokens, cap)

        ids = []
        for t in tokens:
            if t in self._vocab:
                ids.append(self._vocab[t])
            else:
                for c in t:
                    ids.append(self._vocab.get(c, self._vocab["[UNK]"]))
        return ids

    def _wordpiece_encode_tu(self, tu: str) -> List[int]:
        """WordPiece encode một từ."""
        if tu in self._vocab:
            return [self._vocab[tu]]

        ids = []
        for c in tu:
            ids.append(self._vocab.get(c, self._vocab["[UNK]"]))
        return ids

    def giai_ma(self, ids: List[int], bo_special: bool = True) -> str:
        """Giải mã token IDs thành văn bản."""
        tu_list = []
        for idx in ids:
            if idx in self._vocab_nguoc:
                token = self._vocab_nguoc[idx]
                if bo_special and token in self.SPECIAL_TOKENS:
                    continue
                tu_list.append(token.replace("</w>", ""))
        return " ".join(tu_list)

    def pad(self, ids: List[int], do_dai: Optional[int] = None) -> List[int]:
        """Padding hoặc truncate đến độ dài cố định."""
        do_dai = do_dai or self.do_dai_toi_da
        if len(ids) >= do_dai:
            return ids[:do_dai]
        return ids + [self._vocab["[PAD]"]] * (do_dai - len(ids))

    def luu(self, duong_dan: str) -> str:
        """Lưu tokenizer."""
        data = {
            "che_do": self.che_do,
            "kich_thuoc_vocab": self.kich_thuoc_vocab,
            "do_dai_toi_da": self.do_dai_toi_da,
            "vocab": self._vocab,
            "merges": self._merges,
        }
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Đã lưu tokenizer: {duong_dan}")
        return str(duong_dan_path)

    @classmethod
    def tai(cls, duong_dan: str) -> "VietnameseTokenizer":
        """Tải tokenizer."""
        with open(duong_dan, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(
            che_do=data["che_do"],
            kich_thuoc_vocab=data["kich_thuoc_vocab"],
            do_dai_toi_da=data["do_dai_toi_da"],
        )
        tok._vocab = data["vocab"]
        tok._vocab_nguoc = {int(v): k for k, v in data["vocab"].items()}
        tok._merges = [tuple(m) for m in data.get("merges", [])]
        tok._da_huan_luyen = True
        Logger("VietnameseTokenizer").info(f"Đã tải tokenizer: {duong_dan}")
        return tok

    def thong_ke(self) -> Dict:
        """Thống kê tokenizer."""
        return {
            "che_do": self.che_do,
            "da_huan_luyen": self._da_huan_luyen,
            "vocab_size": len(self._vocab),
            "so_merges": len(self._merges),
            "do_dai_toi_da": self.do_dai_toi_da,
        }
