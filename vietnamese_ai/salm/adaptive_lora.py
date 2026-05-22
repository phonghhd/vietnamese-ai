```python
class AdaptiveLoRA:
    def some_method(self, diem_map, top_k):
        for ten in diem_map:
            diem_map[ten] *= self._adapters[ten]["trong_so"]

        # Sắp xếp từ cao xuống thấp
        sorted_adapters = sorted(diem_map.items(), key=lambda x: x[1], reverse=True)

        # [VÁ BUG TẠI ĐÂY] 🛡️ Lọc bỏ các chuyên gia bị 0 điểm
        danh_sach_hop_le = [(ten, diem) for ten, diem in sorted_adapters if diem > 0]

        # Nếu không có ai qua bài test (hoặc người dùng hỏi vu vơ) -> Dùng Base Model
        if not danh_sach_hop_le:
            return []

        ket_qua = []
        # Chỉ lấy top_k từ danh sách ĐÃ HỢP LỆ
        for ten, diem in danh_sach_hop_le[:top_k]:
            self._usage_stats[ten] += 1
            ket_qua.append({
                "ten": ten,
                "diem": diem
            })
        return ket_qua
```