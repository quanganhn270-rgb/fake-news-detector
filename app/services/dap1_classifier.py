import os
import re
import string
import pickle
import logging
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("fake_news_detector.dap1_classifier")

def wordopt(text: str) -> str:
    """Hàm dọn dẹp văn bản chuẩn từ notebook dap 1."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Loại bỏ ngoặc vuông [...]
    text = re.sub(r'\[.*?\]', '', text)
    # Loại bỏ ngoặc đơn ()
    text = re.sub(r'[()]', '', text)
    # Thay thế ký tự không phải chữ/số bằng khoảng trắng
    text = re.sub(r'\W', ' ', text)
    # Loại bỏ URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Loại bỏ tags HTML
    text = re.sub(r'<.*?>+', '', text)
    # Loại bỏ dấu câu
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Loại bỏ ký tự xuống dòng
    text = re.sub(r'\n', '', text)
    # Loại bỏ từ chứa chữ số
    text = re.sub(r'\w*\d\w*', '', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class Dap1TextCleaner(BaseEstimator, TransformerMixin):
    """Transformer bọc quanh hàm clean wordopt của dap 1 để đưa vào sklearn Pipeline."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [wordopt(t) for t in X]


VIETNAMESE_STOPWORDS = {"và", "của", "được", "có", "là", "trong", "cho", "nhưng", "với", "như", "các", "những", "một", "này", "đã", "đang", "sẽ", "đó", "ở", "tại", "không"}

class Dap1ClassifierService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.model_path = base_dir / "data" / "dap1_model.pkl"
        else:
            self.model_path = Path(model_path)
            
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not self.model_path.exists():
            logger.warning(f"Dap1 model file not found at {self.model_path}. Content prediction layer will not run.")
            return
            
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Successfully loaded Dap1 Content Classifier model.")
        except Exception as e:
            logger.error(f"Error loading Dap1 model: {e}")
            self.model = None

    def predict_content(self, text: str) -> dict:
        """
        Dự đoán nội dung tin tức dựa trên mô hình của dap 1.
        Trả về dict kết quả dự đoán (nhãn và độ tin cậy).
        """
        if self.model is None:
            return {
                "label": "Unknown",
                "confidence_score": 0.5,
                "is_fake": False
            }
            
        if not text or not text.strip():
            return {
                "label": "True News",
                "confidence_score": 1.0,
                "is_fake": False
            }
            
        # Kiểm tra ngôn ngữ hoặc độ phủ của từ vựng (OOD/Tiếng Việt check)
        words = set(wordopt(text).split())
        is_vietnamese = len(words.intersection(VIETNAMESE_STOPWORDS)) > 0
        
        # Kiểm tra số lượng từ khớp trong từ điển TF-IDF
        has_low_overlap = False
        try:
            tfidf = self.model.named_steps['tfidf']
            vector = tfidf.transform([text])
            if vector.nnz < 3: # Ít hơn 3 từ trùng khớp với từ điển
                has_low_overlap = True
        except Exception as e:
            logger.warning(f"Error checking TF-IDF overlap: {e}")

        if is_vietnamese or has_low_overlap:
            logger.info(f"OOD or Vietnamese text detected (is_vietnamese={is_vietnamese}, low_overlap={has_low_overlap}). Skipping Dap1 content model prediction.")
            return {
                "label": "Unknown",
                "confidence_score": 0.5,
                "is_fake": False
            }

        try:
            # Dự đoán xác suất bằng pipeline
            # lớp 0: Fake News, lớp 1: True News (Not A Fake News)
            probs = self.model.predict_proba([text])[0]
            fake_prob = float(probs[0])
            true_prob = float(probs[1])
            
            if fake_prob >= 0.5:
                label = "Fake News"
                confidence = fake_prob
                is_fake = True
            else:
                label = "True News"
                confidence = true_prob
                is_fake = False
                
            return {
                "label": label,
                "confidence_score": confidence,
                "is_fake": is_fake
            }
        except Exception as e:
            logger.error(f"Error predicting with Dap1 model: {e}")
            return {
                "label": "Unknown",
                "confidence_score": 0.5,
                "is_fake": False
            }
