import sys
import asyncio
from pathlib import Path

# Đảm bảo đầu ra console hỗ trợ tiếng Việt trên Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.services.rag_service import RAGService

async def test():
    print("Khởi tạo RAGService...")
    try:
        rag = RAGService()
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    # In ra số lượng tài liệu hiện có trong ChromaDB
    try:
        count = rag.collection.count()
        print(f"Số lượng tài liệu trong ChromaDB: {count}")
    except Exception as e:
        print(f"Lỗi đếm số lượng tài liệu: {e}")
        return

    # Trích xuất toàn bộ metadata để đếm nguồn và category
    try:
        all_data = rag.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        sources = {}
        categories = {}
        for meta in metadatas:
            if not meta:
                continue
            src = meta.get("source", "Unknown")
            cat = meta.get("category", "Unknown")
            sources[src] = sources.get(src, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
            
        print("\n--- Phân phối theo Nguồn (Source) ---")
        for src, val in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {src}: {val}")
            
        print("\n--- Phân phối theo Phân loại (Category) ---")
        for cat, val in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {val}")
            
    except Exception as e:
        print(f"Lỗi phân tích phân phối metadata: {e}")


    # Danh sách truy vấn kiểm thử
    queries = [
        "vắc xin covid chứa vi chíp theo dõi",
        "UBND thành phố hỗ trợ 5 triệu đồng",
        "Ngân hàng nhà nước bác bỏ tin đồn phá sản",
        "Sơn Tùng M-TP giải nghệ",
        "giá xăng tăng lên 40.000 đồng một lít",
        "Việt Nam xuất khẩu gạo chất lượng cao",
        "hoãn kỳ thi tốt nghiệp THPT quốc gia"
    ]

    print("\n--- Chạy thử nghiệm truy vấn ngữ nghĩa (Top 3, threshold=0.30) ---")
    for q in queries:
        print(f"\nTruy vấn: '{q}'")
        results = await rag.query_semantic_async(q, top_k=3, threshold=0.30)
        print(f"Tìm thấy: {len(results)} kết quả")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Tương đồng: {r['similarity_score']:.4f}")
            print(f"      Tiêu đề: {r['metadata'].get('title')}")
            print(f"      Nguồn: {r['metadata'].get('source')} | Loại (Category): {r['metadata'].get('category')}")
            print(f"      Nội dung: {r['document'][:100]}...")

if __name__ == "__main__":
    asyncio.run(test())
