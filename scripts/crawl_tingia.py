import os
import sys
import logging
import asyncio
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

# Cấu hình logging
logger = logging.getLogger("fake_news_detector.crawl_tingia")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Cấu hình sys.path để truy cập thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Đảm bảo UTF-8 cho console output trên Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False
    logger.warning("Thư viện 'newspaper3k' chưa được cài đặt. Việc cào nội dung chi tiết bài viết sẽ bị bỏ qua.")

try:
    from app.services.rag_service import RAGService
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    logger.warning("RAG Service hoặc ChromaDB chưa được cài đặt. Việc nạp dữ liệu vào Vector DB sẽ bị bỏ qua.")


class TinGiaCrawler:
    def __init__(self) -> None:
        self.rag_service: Optional[RAGService] = None
        if HAS_RAG:
            try:
                self.rag_service = RAGService()
            except Exception as e:
                logger.error(f"Không thể khởi tạo RAGService: {e}")

    def fetch_article_links(self) -> Dict[str, str]:
        """Thu thập danh sách các liên kết bài báo cảnh báo tin giả từ tingia.gov.vn."""
        target_url = "http://tingia.gov.vn/"
        unique_links = {}
        
        try:
            logger.info(f"Đang tải trang chủ {target_url} để tìm kiếm liên kết cảnh báo...")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(target_url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"Không thể tải trang chủ. HTTP Status: {response.status_code}")
                return unique_links

            soup = BeautifulSoup(response.text, 'html.parser')
            
            for a in soup.find_all('a'):
                href = a.get('href')
                if not href:
                    continue
                
                # Chuyển đổi thành URL đầy đủ
                full_url = urllib.parse.urljoin(target_url, href)
                parsed = urllib.parse.urlparse(full_url)
                
                # Chỉ lọc các bài viết cụ thể kết thúc bằng .html và thuộc tingia.gov.vn
                if parsed.netloc == 'tingia.gov.vn' and parsed.path.endswith('.html') and parsed.path != '/':
                    title_text = a.text.strip()
                    if len(title_text) > 10:  # Bỏ qua các nhãn liên kết quá ngắn
                        unique_links[full_url] = title_text
                        
            logger.info(f"Đã tìm thấy {len(unique_links)} liên kết cảnh báo tin giả độc duy nhất.")
            
        except Exception as e:
            logger.error(f"Lỗi khi thu thập liên kết từ tingia.gov.vn: {e}")
            
        return unique_links

    def crawl_article_content(self, url: str) -> Optional[str]:
        """Sử dụng newspaper3k để cào nội dung chi tiết bài viết từ tingia.gov.vn."""
        if not HAS_NEWSPAPER:
            return None
        try:
            article = Article(url, language='vi')
            article.download()
            article.parse()
            if article.text and len(article.text.strip()) > 50:
                return article.text.strip()
        except Exception as e:
            logger.warning(f"Không thể tải nội dung chi tiết tại '{url}': {e}")
        return None

    async def run(self, max_articles: int = 15) -> None:
        """Thực hiện quy trình cào dữ liệu tin giả từ tingia.gov.vn và nạp vào ChromaDB."""
        logger.info("Bắt đầu quy trình cào dữ liệu từ tingia.gov.vn...")
        
        article_links = self.fetch_article_links()
        if not article_links:
            logger.warning("Không tìm thấy liên kết nào để cào dữ liệu.")
            return

        crawled_count = 0
        documents = []
        metadatas = []

        # Giới hạn số lượng bài viết cào để tránh spam/timeout
        for url, title in list(article_links.items())[:max_articles]:
            logger.info(f"Đang cào bài viết ({crawled_count + 1}/{max_articles}): '{title}'...")
            
            content = self.crawl_article_content(url)
            
            if content:
                # Định dạng tài liệu đối chứng
                formatted_doc = f"Tiêu đề: {title}\nNội dung: {content}"
                documents.append(formatted_doc)
                
                # Metadata ghi nhận nguồn tingia.gov.vn dưới nhãn "fake" để ChromaDB nhận diện
                metadatas.append({
                    "title": title,
                    "source": "tingia.gov.vn",
                    "category": "fake",  # Đánh dấu là nguồn tin giả đã xác minh để gán nhãn đúng ở Trạm 4
                    "url": url
                })
                
                logger.info(f"-> Cào thành công chi tiết bài viết. Độ dài: {len(content)} ký tự.")
                crawled_count += 1
            else:
                logger.warning(f"-> Bỏ qua bài viết: '{title}' (không thể đọc nội dung)")

        # Nạp dữ liệu vào database đối chứng (ChromaDB)
        if HAS_RAG and self.rag_service and documents:
            logger.info(f"Đang tiến hành nạp {len(documents)} cảnh báo tin giả từ VAFC vào ChromaDB...")
            try:
                await self.rag_service.add_documents_async(documents, metadatas)
                logger.info("Nạp dữ liệu từ tingia.gov.vn vào ChromaDB thành công!")
            except Exception as e:
                logger.error(f"Lỗi khi nạp dữ liệu vào ChromaDB: {e}")
        else:
            if not HAS_RAG:
                logger.warning("Không nạp được ChromaDB vì thiếu thư viện.")
            elif not documents:
                logger.warning("Không có bài viết tin giả nào được cào thành công.")


if __name__ == "__main__":
    crawler = TinGiaCrawler()
    asyncio.run(crawler.run(max_articles=20))
