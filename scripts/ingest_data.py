import os
import sys
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import Set, List, Dict, Any

# Cấu hình logging
logger = logging.getLogger("fake_news_detector.ingestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Cấu hình sys.path để import được app module từ thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import RAGService một cách loose-coupled
try:
    from app.services.rag_service import RAGService
    HAS_RAG = True
except ImportError as e:
    logger.warning(
        f"Could not import RAGService: {e}.\n"
        "Vector database ingestion will be skipped. Only domain list extraction (whitelist/blacklist) will run."
    )
    HAS_RAG = False


class DataIngester:
    def __init__(self, csv_path: Path, data_dir: Path) -> None:
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.whitelist_path = data_dir / "whitelist_domains.json"
        self.blacklist_path = data_dir / "blacklist_domains.json"

    def clean_domain(self, domain: str) -> str:
        """Chuẩn hóa domain trước khi ghi nhận."""
        if not domain or not isinstance(domain, str):
            return ""
        domain = domain.strip().lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def update_domain_files(self) -> None:
        """Bước 1: Trích xuất domain từ CSV và cập nhật whitelist/blacklist JSON."""
        logger.info(f"Step 1: Reading domains from dataset '{self.csv_path.name}'...")
        
        # Đọc chỉ các cột cần thiết để tiết kiệm RAM
        df = pd.read_csv(self.csv_path, usecols=["domain", "type"])
        
        # 1. Xử lý Whitelist (loại 'reliable')
        reliable_df = df[df["type"] == "reliable"]
        reliable_domains = {self.clean_domain(d) for d in reliable_df["domain"].dropna()}
        reliable_domains.discard("") # Loại bỏ chuỗi rỗng
        
        # 2. Xử lý Blacklist (loại 'fake' và 'unreliable')
        unreliable_df = df[df["type"].isin(["fake", "unreliable"])]
        unreliable_domains = {self.clean_domain(d) for d in unreliable_df["domain"].dropna()}
        unreliable_domains.discard("")

        logger.info(f"Extracted {len(reliable_domains)} reliable domains and {len(unreliable_domains)} suspicious domains from CSV.")

        # --- Cập nhật Whitelist File ---
        existing_whitelist: Set[str] = set()
        if self.whitelist_path.exists():
            try:
                with open(self.whitelist_path, "r", encoding="utf-8") as f:
                    existing_whitelist = set(json.load(f))
            except Exception as e:
                logger.error(f"Error reading existing whitelist: {e}")
                
        new_whitelist = existing_whitelist.union(reliable_domains)
        try:
            with open(self.whitelist_path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(new_whitelist)), f, indent=2, ensure_ascii=False)
            added_w = len(new_whitelist) - len(existing_whitelist)
            logger.info(f"Whitelist updated: Added {added_w} new domains. Total now: {len(new_whitelist)}")
        except Exception as e:
            logger.error(f"Failed to write whitelist file: {e}")

        # --- Cập nhật Blacklist File ---
        existing_blacklist: Set[str] = set()
        if self.blacklist_path.exists():
            try:
                with open(self.blacklist_path, "r", encoding="utf-8") as f:
                    existing_blacklist = set(json.load(f))
            except Exception as e:
                logger.error(f"Error reading existing blacklist: {e}")
                
        new_blacklist = existing_blacklist.union(unreliable_domains)
        try:
            with open(self.blacklist_path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(new_blacklist)), f, indent=2, ensure_ascii=False)
            added_b = len(new_blacklist) - len(existing_blacklist)
            logger.info(f"Blacklist updated: Added {added_b} new domains. Total now: {len(new_blacklist)}")
        except Exception as e:
            logger.error(f"Failed to write blacklist file: {e}")

    def ingest_to_chromadb(self, max_docs: int = 1500) -> None:
        """Bước 2: Nạp các bài viết sự thật và đối chứng từ CSV vào ChromaDB."""
        if not HAS_RAG:
            logger.warning("Skipping step 2: ChromaDB or RAG dependencies are not installed.")
            return

        logger.info(f"Step 2: Ingesting articles to ChromaDB (Limit: {max_docs} documents)...")
        
        # Khởi tạo RAG Service
        try:
            rag_service = RAGService()
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")
            return

        # Đọc dữ liệu bài viết
        logger.info("Reading articles from CSV...")
        df = pd.read_csv(self.csv_path, usecols=["title", "content", "domain", "type", "url"])
        
        # Ưu tiên lấy toàn bộ bài viết 'reliable' làm dữ liệu chuẩn
        reliable_articles = df[df["type"] == "reliable"]
        
        # Lấy thêm các bài viết 'fake' hoặc 'rumor' để làm đối chứng tin đồn đã được xác minh
        other_articles = df[df["type"].isin(["fake", "rumor"])].sample(
            n=min(max_docs - len(reliable_articles), len(df[df["type"].isin(["fake", "rumor"])])) if max_docs > len(reliable_articles) else 0,
            random_state=42
        )
        
        # Gộp dữ liệu
        final_df = pd.concat([reliable_articles, other_articles]).drop_duplicates(subset=["content"]).dropna(subset=["content"])
        
        # Giới hạn số lượng cuối cùng nếu vượt quá max_docs
        if len(final_df) > max_docs:
            final_df = final_df.head(max_docs)

        logger.info(f"Preparing to ingest {len(final_df)} articles into ChromaDB...")

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for _, row in final_df.iterrows():
            title = str(row["title"]) if pd.notna(row["title"]) else "No Title"
            content = str(row["content"])
            domain = str(row["domain"]) if pd.notna(row["domain"]) else "unknown"
            art_type = str(row["type"]) if pd.notna(row["type"]) else "unknown"
            url = str(row["url"]) if pd.notna(row["url"]) else ""

            # Nhúng thông tin tiêu đề vào nội dung để tìm kiếm ngữ nghĩa tốt hơn
            formatted_document = f"Tiêu đề: {title}\nNội dung: {content}"
            
            documents.append(formatted_document)
            metadatas.append({
                "title": title,
                "source": domain,
                "category": art_type,
                "url": url
            })

        # Nạp dữ liệu theo từng Batch để tránh lỗi tài nguyên và theo dõi tiến độ
        batch_size = 100
        total_ingested = 0
        
        # Sử dụng asyncio event loop để gọi phương thức async của RAGService
        loop = asyncio.get_event_loop()
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            
            logger.info(f"Ingesting batch {i // batch_size + 1} ({len(batch_docs)} articles)...")
            try:
                loop.run_until_complete(rag_service.add_documents_async(batch_docs, batch_meta))
                total_ingested += len(batch_docs)
            except Exception as e:
                logger.error(f"Error ingesting batch {i // batch_size + 1}: {e}")

        logger.info(f"Successfully finished data ingestion. Total ingested to ChromaDB: {total_ingested} articles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data from CSV to Fake News Detector System.")
    parser.add_argument(
        "--csv",
        type=str,
        default="opensources_fake_news_cleaned_100k.csv",
        help="Path to the source CSV file (default: opensources_fake_news_cleaned_100k.csv)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=1500,
        help="Maximum number of documents to ingest into Vector DB (default: 1500)"
    )
    
    args = parser.parse_args()
    
    import asyncio
    
    csv_file = BASE_DIR / args.csv
    data_directory = BASE_DIR / "data"

    if not csv_file.exists():
        logger.error(f"CSV file not found at: {csv_file}. Please check the path and try again.")
        sys.exit(1)

    ingester = DataIngester(csv_file, data_directory)
    
    # Chạy cập nhật whitelist/blacklist
    ingester.update_domain_files()
    
    # Chạy nạp ChromaDB
    ingester.ingest_to_chromadb(max_docs=args.max_docs)
    
    logger.info("Data Ingestion Script finished execution.")
