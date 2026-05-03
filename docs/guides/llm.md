# Vietnamese LLM

## Mô hình ngôn ngữ tiếng Việt

`VietnameseLLM` là n-gram language model cho text generation và completion.

## Huấn luyện

```python
from vietnamese_ai import VietnameseLLM

llm = VietnameseLLM(bac=3)  # trigram
ket_qua = llm.huan_luyen(cac_van_ban, so_vong=5)
print(f"Vocab: {ket_qua['vocab_size']}")
```

## Sinh văn bản

```python
van_ban = llm.sinh_van_ban("học máy là", do_dai=50, nhiet_do=0.8)
print(van_ban)
```

## Hoàn thành câu

```python
lua_chon = llm.hoan_thanh_cau("trí tuệ nhân tạo", so_lua_chon=5)
for lc in lua_chon:
    print(f"{lc['van_ban']} (perplexity: {lc['perplexity']:.2f})")
```

## Gợi ý từ tiếp theo

```python
goi_y = llm.lay_tu_ke_tiep("học máy", top_n=5)
for gy in goi_y:
    print(f"{gy['tu']}: {gy['xac_suat']:.4f}")
```

## Templates

```python
van_ban = llm.sinh_theo_template("tin_tuc", {
    "chu_de": "AI đang phát triển",
    "nhan_dinh": "đây là xu hướng tất yếu"
})
```

## Perplexity

```python
ppl = llm.tinh_perplexity("học máy rất thú vị")
# Perplexity thấp = mô hình dự đoán tốt hơn
```

## Lưu/Tải

```python
llm.luu("llm.json")
llm2 = VietnameseLLM.tai("llm.json")
```
