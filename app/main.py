import logging
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.services.gatekeeper import GatekeeperService, GatekeeperVerdict

# Thử import newspaper để tự động cào nội dung URL
try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

# Import RAG với cơ chế loose-coupling
try:
    from app.services.rag_service import RAGService
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False

# Import Style Classifier với cơ chế loose-coupling
try:
    from app.services.style_classifier import StyleClassifierService
    HAS_STYLE_DEPS = True
except ImportError:
    HAS_STYLE_DEPS = False

# Import Dap1 Classifier với cơ chế loose-coupling
try:
    from app.services.dap1_classifier import Dap1ClassifierService
    HAS_DAP1_DEPS = True
except ImportError:
    HAS_DAP1_DEPS = False


# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fake_news_detector.api")

app = FastAPI(
    title="Hệ thống Phát hiện Tin giả - Fake News Detection System",
    description="API phát hiện tin giả tiếng Việt/Anh dựa trên bộ lọc URL Gatekeeper và RAG Vector DB đối chứng.",
    version="1.0.0"
)

# Phục vụ file tĩnh (HTML/CSS/JS) từ thư mục static
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Khởi tạo các services
gatekeeper_service: Optional[GatekeeperService] = None
rag_service = None
style_classifier_service: Optional[StyleClassifierService] = None
dap1_classifier_service: Optional[Dap1ClassifierService] = None


try:
    gatekeeper_service = GatekeeperService()
except Exception as e:
    logger.error(f"Lỗi khởi tạo GatekeeperService: {e}")

if HAS_RAG_DEPS:
    try:
        rag_service = RAGService()
    except Exception as e:
        logger.error(f"Lỗi khởi tạo RAGService: {e}. Vui lòng cài đặt đầy đủ các dependency.")

if HAS_STYLE_DEPS:
    try:
        style_classifier_service = StyleClassifierService()
    except Exception as e:
        logger.error(f"Lỗi khởi tạo StyleClassifierService: {e}")

if HAS_DAP1_DEPS:
    try:
        dap1_classifier_service = Dap1ClassifierService()
    except Exception as e:
        logger.error(f"Lỗi khởi tạo Dap1ClassifierService: {e}")



# ==========================================
# SCHEMAS
# ==========================================
class PredictRequest(BaseModel):
    text: str  # URL bài báo hoặc đoạn văn bản cần xác thực


class PredictResponse(BaseModel):
    status: str
    label: str
    confidence_score: float
    reasoning: str
    evidence_used: List[Dict[str, Any]]
    style_analysis: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    document: str
    source: str
    title: str
    url: Optional[str] = ""


# ==========================================
# ROUTES
# ==========================================
@app.get("/")
def serve_frontend():
    """Phục vụ giao diện web chính."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Chào mừng đến với API Hệ thống Phát hiện Tin giả",
        "status": "active",
        "stations": {
            "station_1_gatekeeper": "Active" if gatekeeper_service else "Error",
            "station_3b_rag_vectordb": "Active" if rag_service else "Missing Dependencies"
        }
    }


@app.get("/api/v1/status")
def get_status():
    """Trả về trạng thái hoạt động hiện tại của các trạm xử lý."""
    return {
        "message": "Chào mừng đến với API Hệ thống Phát hiện Tin giả",
        "status": "active",
        "stations": {
            "station_1_gatekeeper": "Active" if gatekeeper_service else "Error",
            "station_3b_rag_vectordb": "Active" if rag_service else "Missing Dependencies"
        }
    }


@app.get("/api/v1/stats")
def get_stats():
    """Trả về thống kê dữ liệu hiện tại của hệ thống."""
    whitelist_count = len(gatekeeper_service.whitelist_domains) if gatekeeper_service else 0
    blacklist_count = len(gatekeeper_service.blacklist_domains) if gatekeeper_service else 0
    rag_doc_count = 0
    if rag_service:
        try:
            rag_doc_count = rag_service.collection.count()
        except Exception:
            pass

    return {
        "whitelist_count": whitelist_count,
        "blacklist_count": blacklist_count,
        "rag_documents_count": rag_doc_count
    }


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict_fake_news(request: PredictRequest):
    """
    Điểm cuối (Endpoint) chính của luồng phát hiện tin giả (4 Trạm chính).
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Nội dung kiểm tra không được để trống.")

    # ==========================================
    # TRẠM 1: GATEKEEPER (Fail-Fast URL Check)
    # ==========================================
    gatekeeper_result = None
    if gatekeeper_service:
        gatekeeper_result = gatekeeper_service.evaluate(request.text)

        if gatekeeper_result["verdict"] == GatekeeperVerdict.TIN_THAT:
            return PredictResponse(
                status="success",
                label="TIN THẬT",
                confidence_score=1.0,
                reasoning=gatekeeper_result["reason"],
                evidence_used=[]
            )
        elif gatekeeper_result["verdict"] == GatekeeperVerdict.TIN_GIA_MAO:
            return PredictResponse(
                status="success",
                label="TIN GIẢ MẠO",
                confidence_score=1.0,
                reasoning=gatekeeper_result["reason"],
                evidence_used=[]
            )

    # ==========================================
    # TRẠM 2: TIỀN XỬ LÝ (NLP Preprocessing)
    # ==========================================
    cleaned_text = request.text.strip()
    
    # Kiểm tra xem đầu vào có phải là URL hợp lệ để cào nội dung không
    is_url = False
    if not any(c.isspace() for c in cleaned_text) and ("." in cleaned_text):
        is_url = True
        
    extracted_via_url = False
    crawled_title = ""
    if is_url:
        url_to_crawl = cleaned_text
        if not url_to_crawl.lower().startswith(("http://", "https://")):
            url_to_crawl = "http://" + url_to_crawl
            
        if HAS_NEWSPAPER:
            logger.info(f"Đầu vào là URL. Đang tiến hành cào nội dung tự động từ: {url_to_crawl}...")
            try:
                def fetch_article(url):
                    art = Article(url, language='vi')
                    art.download()
                    art.parse()
                    return art.title, art.text
                
                title, text = await asyncio.to_thread(fetch_article, url_to_crawl)
                if text and len(text.strip()) > 50:
                    crawled_title = title
                    cleaned_text = f"Tiêu đề: {title}\nNội dung: {text.strip()}"
                    extracted_via_url = True
                    logger.info(f"Cào thành công bài viết. Tiêu đề: '{title}', Độ dài: {len(text)} ký tự.")
                else:
                    logger.warning("Nội dung bài viết cào được quá ngắn hoặc trống.")
            except Exception as e:
                logger.error(f"Lỗi khi cào nội dung từ URL {url_to_crawl}: {e}")
        else:
            logger.warning("Thư viện 'newspaper3k' chưa được cài đặt, không thể cào nội dung URL.")
            
    # Tinh chỉnh logic Trạm 2: Nếu là URL nhưng cào nội dung thất bại (spoofing hoặc 404)
    if is_url and not extracted_via_url:
        if gatekeeper_service and gatekeeper_result and gatekeeper_result.get("matched_domain"):
            matched_domain = gatekeeper_result["matched_domain"]
            return PredictResponse(
                status="success",
                label="TIN NGHI VẤN",
                confidence_score=0.7,
                reasoning=f"Đường dẫn thuộc tên miền Whitelist chính thống '{matched_domain}' nhưng không thể tải được nội dung bài viết (Lỗi tải trang hoặc trang 404). Vui lòng kiểm tra lại tính chính xác của đường dẫn.",
                evidence_used=[]
            )
        else:
            return PredictResponse(
                status="success",
                label="CHƯA ĐỦ BẰNG CHỨNG",
                confidence_score=0.5,
                reasoning="Không thể tải hoặc trích xuất nội dung từ đường dẫn này. URL không thuộc Whitelist và không thể phân tích tri thức.",
                evidence_used=[]
            )

    logger.info("Chuyển sang kiểm tra tri thức ngữ nghĩa đối chứng...")

    # ==========================================
    # TRẠM 3A: PHÂN TÍCH PHONG CÁCH VĂN PHONG
    # ==========================================
    style_analysis = None
    style_score = 0.0
    style_label = "Reliable Style"
    if style_classifier_service:
        style_analysis = style_classifier_service.predict_style(cleaned_text)
        style_score = style_analysis["style_score"]
        style_label = style_analysis["label"]
        logger.info(f"Phân tích văn phong hoàn tất. Nhãn style: {style_label}, Score: {style_score:.4f}")

    # ==========================================
    # TRẠM 3B: TRUY VẤN RAG VECTOR DB (ChromaDB)
    # ==========================================
    # Hạ ngưỡng truy vấn RAG xuống 0.45 để phù hợp với việc tìm kiếm ngữ nghĩa Tiếng Việt
    RAG_THRESHOLD = 0.45
    evidence_docs = []
    if rag_service:
        evidence_docs = await rag_service.query_semantic_async(cleaned_text, top_k=3, threshold=RAG_THRESHOLD)

    # ==========================================
    # TRẠM 4: SUY LUẬN AI (Nguyên tắc Precision-First + Style Boost)
    # ==========================================
    if not evidence_docs:
        # Nếu không có tài liệu đối chứng, chạy mô hình phân loại nội dung dap 1
        dap1_result = None
        if dap1_classifier_service:
            dap1_result = dap1_classifier_service.predict_content(cleaned_text)
            
        if dap1_result and dap1_result["label"] != "Unknown" and dap1_result["confidence_score"] >= 0.70:
            confidence_score = dap1_result["confidence_score"]
            if dap1_result["is_fake"]:
                label = "TIN GIẢ MẠO"
                reasoning = f"Mô hình phân tích nội dung (dap 1) phát hiện văn bản này có khả năng là Tin giả với độ tin cậy {confidence_score * 100:.1f}%."
                if style_score >= 0.70:
                    reasoning += f" Văn phong trình bày cũng thể hiện tính chất giật gân, thiếu tin cậy ({style_score * 100:.1f}%)."
            else:
                label = "TIN THẬT"
                reasoning = f"Mô hình phân tích nội dung (dap 1) xác nhận văn bản này có khả năng là Tin thật với độ tin cậy {confidence_score * 100:.1f}%."
                if style_score >= 0.70:
                    reasoning += f" Lưu ý: Mặc dù nội dung có vẻ thật, nhưng văn phong trình bày có tính chất giật gân nhiều cảm xúc ({style_score * 100:.1f}%)."
        else:
            if style_score >= 0.70:
                # Nguy cơ tin giả dựa trên phong cách viết là rất cao (Clickbait/Sensational)
                label = "TIN NGHI VẤN"
                confidence_score = style_score
                reasoning = (
                    f"Hệ thống không tìm thấy bài viết đối chứng trực tiếp nào từ nguồn chính thống. "
                    f"Tuy nhiên, phân tích phong cách viết cho thấy nguy cơ giật tít/tin giả rất cao ({style_score * 100:.1f}%) "
                    f"với các thuộc tính đáng ngờ (số viết hoa: {style_analysis['details'].get('caps_count')}, "
                    f"dấu chấm cảm: {style_analysis['details'].get('exclamation_count')}). Khuyến cáo người đọc cẩn trọng."
                )
            else:
                label = "CHƯA ĐỦ BẰNG CHỨNG"
                confidence_score = 0.5
                reasoning = f"Hệ thống không tìm thấy bài báo đối chứng chính thống nào trùng khớp ngữ nghĩa trên {RAG_THRESHOLD*100:.0f}% để xác thực. Phong cách viết có vẻ đáng tin cậy ({style_score * 100:.1f}% nguy cơ). Theo nguyên tắc suy đoán vô tội, không gán nhãn tin giả khi thiếu bằng chứng."
    else:
        best_match = evidence_docs[0]
        similarity = best_match["similarity_score"]
        category = best_match["metadata"].get("category", "reliable")
        source = best_match["metadata"].get("source", "N/A")
        matched_title = best_match["metadata"].get("title", "N/A")

        # Căn cứ nhãn dựa trên category của bài viết đối chứng tìm được
        # Nếu bài viết đối chứng là tin giả/tin đồn đã được xác minh:
        if category in ["fake", "rumor", "unreliable", "fake_rss"]:
            if similarity >= 0.70:
                label = "TIN GIẢ MẠO"
                confidence_score = max(similarity, style_score)
                reasoning = f"Xác nhận trùng khớp {similarity * 100:.1f}% với tin giả/tin đồn đã được kiểm chứng từ nguồn '{source}': '{matched_title}'."
                if style_score >= 0.70:
                    reasoning += f" Phong cách bài viết cũng thể hiện tính chất giật gân, thiếu tin cậy cao ({style_score * 100:.1f}%)."
            elif similarity >= 0.50:
                label = "TIN NGHI VẤN"
                confidence_score = similarity
                reasoning = f"Có sự tương đồng {similarity * 100:.1f}% với tin đồn từ nguồn '{source}' ('{matched_title}'), cần thận trọng kiểm chứng thêm."
            else:
                label = "CHƯA ĐỦ BẰNG CHỨNG"
                confidence_score = similarity
                reasoning = "Tìm thấy bài viết liên quan nhưng độ tương đồng ngữ nghĩa quá thấp, chưa đủ cơ sở kết luận."
        # Nếu bài viết đối chứng là tin tức chính thống/tin thật:
        else:
            if similarity >= 0.70:
                label = "TIN THẬT"
                confidence_score = similarity
                reasoning = f"Xác nhận trùng khớp {similarity * 100:.1f}% với bài báo chính thống từ nguồn '{source}': '{matched_title}'."
                if style_score >= 0.70:
                    reasoning += f" Lưu ý: Mặc dù thông tin cốt lõi là thật, nhưng văn phong trình bày có tính chất giật gân, giật tít nhiều cảm xúc ({style_score * 100:.1f}%)."
            elif similarity >= 0.50:
                label = "TIN NGHI VẤN"
                confidence_score = similarity
                reasoning = f"Tìm thấy bài viết liên quan từ nguồn '{source}' ('{matched_title}') với độ tương đồng {similarity * 100:.1f}%, cần kiểm chứng thêm do có sự sai khác chi tiết."
            else:
                label = "CHƯA ĐỦ BẰNG CHỨNG"
                confidence_score = similarity
                reasoning = "Tìm thấy bài viết liên quan nhưng độ tương đồng ngữ nghĩa quá thấp, chưa đủ cơ sở kết luận."

    return PredictResponse(
        status="success",
        label=label,
        confidence_score=confidence_score,
        reasoning=reasoning,
        evidence_used=evidence_docs,
        style_analysis=style_analysis
    )


@app.post("/api/v1/ingest")
async def ingest_document(request: IngestRequest):
    """API nạp thêm bài báo sự thật trực tiếp vào ChromaDB."""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG Service chưa được khởi tạo thành công. Vui lòng cài đặt chromadb và sentence-transformers.")

    try:
        formatted_doc = f"Tiêu đề: {request.title}\nNội dung: {request.document}"
        metadata = {
            "title": request.title,
            "source": request.source,
            "category": "manual_ingest",
            "url": request.url
        }
        await rag_service.add_documents_async([formatted_doc], [metadata])
        return {"status": "success", "message": "Nạp bài viết vào ChromaDB thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
