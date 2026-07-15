import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import ChromaDB
try:
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
except ImportError as e:
    raise ImportError(
        "ChromaDB is not installed. Please run `pip install chromadb` to install the dependency."
    ) from e

# Thử import các thư viện embedding của LangChain và SentenceTransformers để hỗ trợ cơ chế fallback linh hoạt
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HAS_LANGCHAIN_HF = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HAS_LANGCHAIN_HF = True
    except ImportError:
        HAS_LANGCHAIN_HF = False

# Cấu hình Logger
logger = logging.getLogger("fake_news_detector.rag_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class RAGService:
    """
    Trạm 3B: RAG Service sử dụng ChromaDB làm Vector Store.
    Thực hiện mã hóa nội dung tin tức đối chứng và truy vấn bất đồng bộ (async query)
    để tìm kiếm các bài báo sự thật có độ tương đồng ngữ nghĩa >= 0.75.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "fact_checking_news"
    ) -> None:
        """
        Khởi tạo dịch vụ RAG với ChromaDB và Embedding Model.
        
        Args:
            persist_directory: Thư mục lưu trữ database cục bộ.
            model_name: Tên mô hình HuggingFace Embeddings sử dụng.
            collection_name: Tên bộ sưu tập dữ liệu đối chứng trong ChromaDB.
        """
        # Xác định đường dẫn lưu trữ database mặc định tại thư mục gốc của dự án
        if persist_directory is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.persist_directory = str(base_dir / "data" / "chroma_db")
        else:
            self.persist_directory = persist_directory

        logger.info(f"Initializing ChromaDB Client with storage at: {self.persist_directory}")
        
        # Khởi tạo Persistent Client của ChromaDB
        self.chroma_client: ClientAPI = chromadb.PersistentClient(path=self.persist_directory)
        
        self.model_name = model_name
        self.collection_name = collection_name
        self.embedding_model = None
        self.raw_transformer = None
        
        # Khởi tạo mô hình sinh Vector
        self._init_embeddings()
        
        # Khởi tạo bộ sưu tập trong ChromaDB
        # Sử dụng metric khoảng cách 'cosine' để đo độ tương đồng ngữ nghĩa
        self.collection: Collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to ChromaDB collection: '{self.collection_name}'")

    def _init_embeddings(self) -> None:
        """Khởi tạo mô hình Embedding dựa trên các thư viện có sẵn (Ưu tiên LangChain -> SentenceTransformers)."""
        logger.info(f"Loading embedding model: '{self.model_name}'...")
        
        if HAS_LANGCHAIN_HF:
            try:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'} # Đảm bảo chạy ổn định trên CPU/các máy chủ phổ thông
                )
                logger.info("Successfully initialized HuggingFaceEmbeddings using LangChain wrapper.")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFaceEmbeddings via LangChain: {e}. Trying raw SentenceTransformers...")
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.raw_transformer = SentenceTransformer(self.model_name)
                logger.info("Successfully initialized raw SentenceTransformer model directly.")
                return
            except Exception as e:
                logger.error(f"Failed to load raw SentenceTransformer: {e}")
        
        # Nếu cả 2 đều không hoạt động nhưng cần chạy được
        raise RuntimeError(
            f"Unable to load embedding model. Please ensure 'sentence-transformers' or 'langchain-huggingface' is installed."
        )

    def _generate_vector(self, text: str) -> List[float]:
        """Tạo vector nhúng cho một văn bản (Đồng bộ)."""
        if self.embedding_model is not None:
            # LangChain HuggingFaceEmbeddings sử dụng embed_query
            return self.embedding_model.embed_query(text)
        elif self.raw_transformer is not None:
            # Sinh vector trực tiếp từ SentenceTransformer
            embeddings = self.raw_transformer.encode(text, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            raise RuntimeError("No active embedding model initialized.")

    def _generate_vectors_batch(self, texts: List[str]) -> List[List[float]]:
        """Tạo vector nhúng cho một danh sách văn bản (Đồng bộ)."""
        if self.embedding_model is not None:
            return self.embedding_model.embed_documents(texts)
        elif self.raw_transformer is not None:
            embeddings = self.raw_transformer.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            raise RuntimeError("No active embedding model initialized.")

    @staticmethod
    def _generate_id(content: str) -> str:
        """Tạo mã định danh (ID) duy nhất cho văn bản dựa trên thuật toán hash SHA-256."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _add_documents_sync(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Hàm đồng bộ để thêm tài liệu đối chứng vào database."""
        if not documents:
            return
        
        if metadatas is None:
            metadatas = [{"source": "imported_fact_news"} for _ in documents]
            
        ids = [self._generate_id(doc) for doc in documents]
        embeddings = self._generate_vectors_batch(documents)
        
        # Loại bỏ tài liệu trùng lặp trong cùng một lô nạp để tránh lỗi ChromaDB
        seen_ids = set()
        unique_ids = []
        unique_embeddings = []
        unique_docs = []
        unique_metadatas = []
        
        for doc_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_ids.append(doc_id)
                unique_embeddings.append(emb)
                unique_docs.append(doc)
                unique_metadatas.append(meta)
                
        self.collection.upsert(
            ids=unique_ids,
            embeddings=unique_embeddings,
            documents=unique_docs,
            metadatas=unique_metadatas
        )
        logger.info(f"Successfully upserted {len(unique_docs)} documents to ChromaDB.")


    async def add_documents_async(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Thêm tài liệu đối chứng vào database bất đồng bộ.
        Chạy trên một thread pool riêng thông qua asyncio.to_thread để tránh chặn event loop.
        """
        await asyncio.to_thread(self._add_documents_sync, documents, metadatas)

    def _query_semantic_sync(self, query_text: str, top_k: int = 3, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Hàm đồng bộ thực hiện truy vấn ngữ nghĩa và lọc kết quả dựa trên độ tương đồng.
        
        Cách tính Similarity Score đối với Cosine Distance của ChromaDB:
        ChromaDB định nghĩa Cosine Distance d = 1 - Cosine_Similarity(A, B).
        Do đó, điểm tương đồng Cosine Similarity được tính bằng: similarity = 1.0 - distance.
        Ngưỡng threshold >= 0.75 có nghĩa là khoảng cách distance phải <= 0.25.
        """
        query_vector = self._generate_vector(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        matched_docs = []
        
        # Kiểm tra kết quả trả về từ ChromaDB
        if not results or "documents" not in results or not results["documents"][0]:
            return matched_docs

        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(documents)
        distances = results["distances"][0] if "distances" in results and results["distances"] else [1.0] * len(documents)
        ids = results["ids"][0] if "ids" in results and results["ids"] else [""] * len(documents)

        for i in range(len(documents)):
            distance = distances[i]
            # Tính toán Cosine Similarity tương ứng từ Distance
            similarity_score = 1.0 - distance
            
            logger.info(
                f"Candidate document {i+1} (ID: {ids[i][:8]}): "
                f"Distance = {distance:.4f} -> Similarity = {similarity_score:.4f}"
            )
            
            # Lọc theo ngưỡng Similarity Threshold (yêu cầu >= 0.75)
            if similarity_score >= threshold:
                matched_docs.append({
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "similarity_score": round(similarity_score, 4)
                })
            else:
                logger.info(
                    f"Candidate document {i+1} filtered out because similarity "
                    f"{similarity_score:.4f} < threshold {threshold}"
                )
                
        return matched_docs

    async def query_semantic_async(
        self,
        query_text: str,
        top_k: int = 3,
        threshold: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Truy vấn ngữ nghĩa bất đồng bộ để lấy ra Top K bài báo đối chứng có độ tương đồng >= threshold.
        Sử dụng asyncio.to_thread để thực hiện song song không chặn Event Loop.
        
        Args:
            query_text: Nội dung bài báo cần truy vấn/đối chứng.
            top_k: Số lượng kết quả đối chứng tối đa cần lấy.
            threshold: Ngưỡng tương đồng tối thiểu (mặc định 0.75).
            
        Returns:
            Danh sách các tài liệu khớp cùng điểm similarity_score và metadata.
        """
        try:
            return await asyncio.to_thread(self._query_semantic_sync, query_text, top_k, threshold)
        except Exception as e:
            logger.error(f"Error querying semantic database for text: '{query_text[:50]}...': {e}", exc_info=True)
            # Trả về danh sách rỗng để đảm bảo hệ thống tiếp tục chạy sang trạm sau mà không bị sập (Fail-Safe)
            return []

    def seed_sample_facts(self) -> None:
        """Nạp dữ liệu sự thật đối chứng mẫu để phục vụ việc kiểm thử hệ thống."""
        logger.info("Seeding sample facts into ChromaDB...")
        sample_documents = [
            "Bộ Y tế Việt Nam khẳng định chưa có bất kỳ nghiên cứu khoa học nào chứng minh nước chanh nóng có thể tiêu diệt virus Corona. Người dân cần tin tưởng thông tin từ các cơ quan y tế chính thống.",
            "Ủy ban Nhân dân Thành phố Hồ Chí Minh bác bỏ tin đồn thất thiệt về việc phong tỏa toàn thành phố từ ngày mai. Mọi hoạt động sản xuất, kinh doanh thiết yếu vẫn diễn ra bình thường theo chỉ thị của Nhà nước.",
            "Ngân hàng Nhà nước Việt Nam tuyên bố thông tin đổi tiền là hoàn toàn bịa đặt, nhằm gây hoang mang dư luận. Giá trị đồng Việt Nam vẫn ổn định, hoạt động ngân hàng diễn ra bình thường."
        ]
        
        sample_metadatas = [
            {"source": "Bộ Y Tế", "category": "Y tế - COVID19", "date": "2021-05-10"},
            {"source": "UBND TP.HCM", "category": "Chính sự - Xã hội", "date": "2021-07-15"},
            {"source": "Ngân hàng Nhà nước", "category": "Kinh tế - Tài chính", "date": "2022-11-20"}
        ]
        
        self._add_documents_sync(sample_documents, sample_metadatas)
        logger.info("Successfully seeded sample facts.")
