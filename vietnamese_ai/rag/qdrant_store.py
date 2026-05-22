"""QdrantVectorStore - Lưu trữ và tìm kiếm vector bằng Qdrant."""

import uuid
from typing import Any, Dict, List, Optional
import numpy as np

class QdrantVectorStore:
    """
    Cơ sở dữ liệu vector sử dụng Qdrant.
    Thích hợp cho Production, tốc độ cao.
    
    Yêu cầu: pip install qdrant-client
    """

    def __init__(
        self,
        url: Optional[str] = None,
        path: Optional[str] = "./qdrant_db",
        ten_collection: str = "vietnamese_ai",
        kich_thuoc: int = 128,
        khoang_cach: str = "cosine"
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError:
            raise ImportError("Vui lòng cài đặt qdrant-client: pip install qdrant-client")

        # Map distance metric string to Qdrant Distance
        dist_map = {
            "cosine": Distance.COSINE,
            "l2": Distance.EUCLID,
            "inner_product": Distance.DOT
        }
        self.distance = dist_map.get(khoang_cach, Distance.COSINE)
        self.kich_thuoc = kich_thuoc
        self.ten_collection = ten_collection
        
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
            
        # Create collection if not exists
        collections = self.client.get_collections().collections
        if not any(c.name == self.ten_collection for c in collections):
            self.client.create_collection(
                collection_name=self.ten_collection,
                vectors_config=VectorParams(size=self.kich_thuoc, distance=self.distance)
            )

    def chen(
        self,
        ma: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Chen một vector vào CSDL."""
        from qdrant_client.http.models import PointStruct
        
        vec = np.asarray(vector, dtype=float).flatten().tolist()
        if len(vec) != self.kich_thuoc:
            raise ValueError(f"Vector kích thước {len(vec)}, mong đợi {self.kich_thuoc}")
            
        if ma.isdigit():
            point_id = int(ma)
        else:
            try:
                point_id = str(uuid.UUID(ma))
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ma))
                
        meta = metadata.copy() if metadata else {}
        meta["_original_id"] = ma
        
        point = PointStruct(
            id=point_id, 
            vector=vec,
            payload=meta
        )
        
        self.client.upsert(
            collection_name=self.ten_collection,
            points=[point]
        )

    def tim_kiem(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        nguong: Optional[float] = None,
        bo_loc: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm vector gần nhất."""
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        
        vec = np.asarray(query_vector, dtype=float).flatten().tolist()
        
        qdrant_filter = None
        if bo_loc:
            conditions = []
            for k, v in bo_loc.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            qdrant_filter = Filter(must=conditions)
            
        results = self.client.query_points(
            collection_name=self.ten_collection,
            query=vec,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=nguong
        )
        
        ket_qua = []
        for r in results.points:
            payload = r.payload or {}
            ma = payload.pop("_original_id", str(r.id))
            ket_qua.append({
                "ma": ma,
                "diem": r.score,
                "metadata": payload
            })
            
        return ket_qua

    def so_luong(self) -> int:
        """Số lượng vector trong CSDL."""
        return self.client.count(collection_name=self.ten_collection).count
