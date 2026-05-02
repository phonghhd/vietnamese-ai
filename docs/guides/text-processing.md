# Xử lý văn bản tiếng Việt

## Tách từ (underthesea)

```python
from vietnamese_ai import XuLyVanBan

xl = XuLyVanBan()

# Tách từ chuẩn - nhận diện từ ghép
xl.tach_tu("Trí tuệ nhân tạo rất hay")
# ['trí_tuệ_nhân_tạo', 'rất', 'hay']

# Chuẩn hóa Unicode
xl.chuan_hoa("  Đây là VD!  ")
# "đây là vd"
```

## Sentiment Analysis

```python
from vietnamese_ai import PhanTichCamXuc

ptx = PhanTichCamXuc(che_do="underthesea")
ptx.phan_tich("Sản phẩm rất tốt, tôi rất hài lòng")
# {'nhan': 'positive', 'xac_suat': {'positive': 1.0}, 'nguon': 'underthesea'}

# Tự huấn luyện
ptx = PhanTichCamXuc(che_do="tu_huan")
ptx.huan_luyen(van_ban_list, nhan_list)
```

## Word2Vec / FastText

```python
from vietnamese_ai import Word2VecTiengViet, FastTextTiengViet

# Word2Vec
w2v = Word2VecTiengViet(kich_thuoc=100)
w2v.huan_luyen(cac_van_ban, so_vong=5)
w2v.tim_tu_giong("học", top_n=5)

# FastText (xử lý từ mới)
ft = FastTextTiengViet(kich_thuoc=100)
ft.huan_luyen(cac_van_ban)
ft.lay_vector("từ_mới_chưa_từng_thấy")  # Vẫn có vector!
```

## TF-IDF

```python
xl = XuLyVanBan()
tfidf = xl.ma_hoa_tfidf(["văn bản 1", "văn bản 2"])
# Shape: (2, n_words)
```

## Tăng cường dữ liệu

```python
from vietnamese_ai import TangCuongVanBan

tc = TangCuongVanBan(seed=42)
van_ban_moi = tc.tang_cuong("Sản phẩm rất tốt", so_mau=5)
# ['Sản phẩm rất tốt', 'Sản phẩm rất hay', 'phẩm rất tốt', ...]
```
