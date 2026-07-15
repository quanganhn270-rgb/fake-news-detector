import os
import sys
import xml.etree.ElementTree as ET
import logging
import asyncio
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

# Cấu hình logging
logger = logging.getLogger("fake_news_detector.crawl_rss")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Cấu hình sys.path để truy cập thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import các thư viện tùy chọn với cơ chế fallback thông minh
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning("Thư viện 'feedparser' chưa được cài đặt. Hệ thống sẽ tự động dùng bộ phân tích XML mặc định.")

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

# Danh sách các nguồn RSS kiểm chứng (Bao gồm Tin thật và Tin giả/Tin đồn)
RSS_FEEDS = [
    # Tin thật chính thống Việt Nam
    {"source": "VnExpress - Tin mới nhất", "url": "https://vnexpress.net/rss/tin-moi-nhat.rss", "category": "reliable_rss"},
    {"source": "VnExpress - Thời sự", "url": "https://vnexpress.net/rss/thoi-su.rss", "category": "reliable_rss"},
    {"source": "Tuổi Trẻ - Tin mới nhất", "url": "https://tuoitre.vn/rss/tin-moi-nhat.rss", "category": "reliable_rss"},
    {"source": "Tuổi Trẻ - Thời sự", "url": "https://tuoitre.vn/rss/thoi-su.rss", "category": "reliable_rss"},
    {"source": "Thanh Niên - Tin mới nhất", "url": "https://thanhnien.vn/rss/home.rss", "category": "reliable_rss"},
    {"source": "Dân Trí - Tin mới nhất", "url": "https://dantri.com.vn/rss/tin-moi-nhat.rss", "category": "reliable_rss"},
    {"source": "VOV - Tin mới nhất", "url": "https://vov.vn/rss/tin-moi-nhat.rss", "category": "reliable_rss"},
    {"source": "VTV - Tin mới nhất", "url": "https://vtv.vn/tin-moi-nhat.rss", "category": "reliable_rss"},
    {"source": "Báo Chính phủ - Tin nổi bật", "url": "https://baochinhphu.vn/rss/tin-noi-bat.rss", "category": "reliable_rss"},
    {"source": "VietnamNet - Tin nổi bật", "url": "https://vietnamnet.vn/rss/tin-noi-bat.rss", "category": "reliable_rss"},
    {"source": "Lao Động - Trang chủ", "url": "https://laodong.vn/rss/home.rss", "category": "reliable_rss"},
    
    # Nguồn kiểm chứng tin giả / tin đồn quốc tế
    {"source": "PolitiFact Fact-Checks", "url": "https://www.politifact.com/rss/factchecks/", "category": "fake_rss"},
    {"source": "FactCheck.org", "url": "https://www.factcheck.org/feed/", "category": "fake_rss"},
    {"source": "Snopes Fact-Checks", "url": "https://www.snopes.com/feed/", "category": "fake_rss"}
]



class RSSCrawler:
    def __init__(self) -> None:
        self.rag_service: Optional[RAGService] = None
        if HAS_RAG:
            try:
                self.rag_service = RAGService()
            except Exception as e:
                logger.error(f"Không thể khởi tạo RAGService: {e}")
                
    def parse_rss_fallback(self, xml_content: str) -> List[Dict[str, str]]:
        """Phân tích cú pháp RSS XML thủ công bằng ElementTree khi thiếu feedparser."""
        articles = []
        try:
            root = ET.fromstring(xml_content)
            # RSS thường nằm trong cấu trúc /rss/channel/item
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_date_elem = item.find("pubDate")
                
                title = title_elem.text if title_elem is not None else ""
                link = link_elem.text if link_elem is not None else ""
                pub_date = pub_date_elem.text if pub_date_elem is not None else ""
                
                if title and link:
                    articles.append({
                        "title": title.strip(),
                        "link": link.strip(),
                        "published": pub_date.strip()
                    })
        except Exception as e:
            logger.error(f"Lỗi phân tích XML thủ công: {e}")
        return articles

    def fetch_rss_articles(self, feed_url: str) -> List[Dict[str, str]]:
        """Lấy danh sách các bài viết mới nhất từ 1 kênh RSS."""
        import requests
        
        articles = []
        try:
            response = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if response.status_code != 200:
                logger.error(f"Không thể tải RSS Feed. HTTP Status: {response.status_code}")
                return articles

            # Nếu có feedparser, dùng feedparser
            if HAS_FEEDPARSER:
                feed = feedparser.parse(response.content)
                for entry in feed.entries:
                    articles.append({
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", "")
                    })
            else:
                # Sử dụng bộ fallback ElementTree
                articles = self.parse_rss_fallback(response.text)
                
        except Exception as e:
            logger.error(f"Lỗi khi đọc kênh RSS {feed_url}: {e}")
        return articles

    def crawl_article_content(self, url: str) -> Optional[str]:
        """Tải và cào nội dung chi tiết bài viết từ URL bằng newspaper3k."""
        if not HAS_NEWSPAPER:
            return None
        try:
            article = Article(url, language='vi')
            article.download()
            article.parse()
            if article.text and len(article.text.strip()) > 100:
                return article.text.strip()
        except Exception as e:
            logger.warning(f"Không thể cào nội dung bài viết tại '{url}': {e}")
        return None

    def get_domain(self, url: str) -> str:
        """Trích xuất domain từ URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"

    async def run(self, max_articles_per_feed: int = 5) -> None:
        """Thực thi quét toàn bộ RSS Feeds và nạp dữ liệu vào ChromaDB."""
        logger.info("Bắt đầu tiến trình quét kênh RSS chính thống...")
        
        all_crawled_articles = []
        
        for feed in RSS_FEEDS:
            logger.info(f"Đang quét nguồn: {feed['source']}...")
            entries = self.fetch_rss_articles(feed["url"])
            
            # Giới hạn số lượng bài viết mỗi feed để tránh quá tải
            selected_entries = entries[:max_articles_per_feed]
            logger.info(f"Tìm thấy {len(entries)} bài viết. Tiến hành cào {len(selected_entries)} bài mới nhất...")
            
            for entry in selected_entries:
                title = entry["title"]
                link = entry["link"]
                
                # Cào nội dung bài viết chi tiết
                content = self.crawl_article_content(link)
                
                if content:
                    domain = self.get_domain(link)
                    all_crawled_articles.append({
                        "title": title,
                        "content": content,
                        "url": link,
                        "source": domain,
                        "category": feed.get("category", "reliable_rss"),
                        "published": entry["published"]
                    })
                    logger.info(f"-> Cào thành công: '{title}' ({domain}) [{feed.get('category')}]")
                else:
                    if not HAS_NEWSPAPER:
                        logger.info(f"-> Phát hiện liên kết: '{title}' (newspaper3k chưa được cài đặt)")
                    else:
                        logger.warning(f"-> Bỏ qua (nội dung trống hoặc lỗi tải): '{title}'")

        # Nạp dữ liệu vào database đối chứng (ChromaDB)
        if HAS_RAG and self.rag_service and all_crawled_articles:
            logger.info(f"Đang nạp {len(all_crawled_articles)} bài viết mới cào vào ChromaDB...")
            
            documents = []
            metadatas = []
            
            for art in all_crawled_articles:
                formatted_doc = f"Tiêu đề: {art['title']}\nNội dung: {art['content']}"
                documents.append(formatted_doc)
                metadatas.append({
                    "title": art["title"],
                    "source": art["source"],
                    "category": art["category"],
                    "url": art["url"],
                    "published": art["published"]
                })
                
            try:
                await self.rag_service.add_documents_async(documents, metadatas)
                logger.info("Nạp dữ liệu RSS vào ChromaDB hoàn tất!")
            except Exception as e:
                logger.error(f"Lỗi nạp ChromaDB: {e}")
        else:
            if not HAS_RAG:
                logger.warning("Bỏ qua bước nạp ChromaDB vì thiếu thư viện.")
            elif not all_crawled_articles:
                logger.warning("Không có bài viết nào được cào thành công để nạp database.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crawl RSS news into ChromaDB")
    parser.add_argument("--max-articles", type=int, default=25, help="Số bài viết tối đa mỗi feed (mặc định: 25)")
    args = parser.parse_args()
    
    crawler = RSSCrawler()
    asyncio.run(crawler.run(max_articles_per_feed=args.max_articles))

