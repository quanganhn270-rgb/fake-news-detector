import os
import re
import string
import pickle
import logging
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

logger = logging.getLogger("fake_news_detector.style_classifier")

LEAKAGE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", 
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept", "oct", "nov", "dec",
    "reuters", "via", "getty", "images", "image", "pic", "photo", "caption",
    "watch", "featured", "read", "url", "link", "subscribe"
}

def wordopt(text: str) -> str:
    """Hàm dọn dẹp văn bản Unicode-safe kế thừa từ dự án alo 123."""
    if not isinstance(text, str):
        return ""
    # Xóa URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Xóa HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Xóa nội dung trong ngoặc vuông (e.g. [image], [video])
    text = re.sub(r'\[.*?\]', '', text)
    # Xóa số và từ chứa số
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    # Xóa dấu câu nhưng GIỮ LẠI chữ Unicode (tiếng Việt, tiếng Anh...)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = text.replace('_', ' ')
    # Chuyển thường
    text = text.lower()
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    # Lọc từ gây rò rỉ thông tin
    words = text.split()
    cleaned_words = [w for w in words if w not in LEAKAGE_WORDS]
    return " ".join(cleaned_words)


class StylisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Trích xuất các đặc trưng dấu câu, chữ viết hoa và từ giật gân (Lexical & Style)."""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for text in X:
            if not isinstance(text, str):
                text = ""
            
            total_chars = len(text) if len(text) > 0 else 1
            total_words = len(text.split()) if len(text.split()) > 0 else 1
            
            # 1. Tỷ lệ dấu chấm cảm (!)
            exclamation_ratio = text.count('!') / total_chars
            # 2. Tỷ lệ dấu hỏi chấm (?)
            question_ratio = text.count('?') / total_chars
            # 3. Tỷ lệ dấu ba chấm (...)
            ellipsis_ratio = len(re.findall(r'\.\.\.', text)) / total_words
            # 4. Tỷ lệ viết hoa toàn bộ từ (ALL CAPS)
            words = text.split()
            caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / total_words
            
            # 5. Từ giật gân/clickbait (Anh & Việt)
            clickbait_words = [
                'shocking', 'unbelievable', 'secret', 'alert', 'warning', 'break', 'magic', 'exposed', 'critical',
                'kinh hoàng', 'khẩn cấp', 'chấn động', 'vạch trần', 'sốc', 'tin nóng', 'nguy hiểm', 'bất ngờ'
            ]
            clickbait_count = sum(1 for w in clickbait_words if w in text.lower()) / total_words
            
            features.append([
                exclamation_ratio, 
                question_ratio, 
                ellipsis_ratio, 
                caps_ratio, 
                clickbait_count
            ])
            
        return np.array(features)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Transformer bọc quanh hàm clean wordopt."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [wordopt(t) for t in X]


class StyleClassifierService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.model_path = base_dir / "data" / "style_classifier.pkl"
        else:
            self.model_path = Path(model_path)
            
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not self.model_path.exists():
            logger.warning(f"Style Classifier model file not found at {self.model_path}. Inference will run on heuristics fallback.")
            return
            
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Successfully loaded Style Classifier model.")
        except Exception as e:
            logger.error(f"Error loading Style Classifier model: {e}")
            self.model = None

    def predict_style(self, text: str) -> dict:
        """
        Dự đoán phong cách văn bản (Thật / Giả mạo/Giật gân).
        Trả về dict chi tiết kết quả.
        """
        if not text or not text.strip():
            return {
                "label": "Reliable Style",
                "confidence_score": 1.0,
                "style_score": 0.0,
                "details": {}
            }
            
        # Trích xuất một số chỉ số định lượng thô phục vụ phần mô tả trực quan của UI
        total_chars = len(text)
        total_words = len(text.split()) if len(text.split()) > 0 else 1
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_count = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        
        details = {
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "caps_count": caps_count,
            "word_count": total_words
        }

        # Kiểm tra xem văn bản có khớp từ vựng tiếng Anh của TF-IDF không.
        # Nếu là tiếng Việt hoặc ngôn ngữ khác không có từ nào trong từ điển của Vectorizer,
        # TF-IDF sẽ trả về một vector bằng 0. Khi đó ta cần fallback sang Heuristics của đặc trưng phong cách.
        has_tfidf = True
        if self.model is not None:
            try:
                # Trích xuất vectorizer từ pipeline
                features_union = self.model.named_steps['features']
                tfidf_transformer = None
                for name, trans in features_union.transformer_list:
                    if name == 'tfidf_pipeline':
                        tfidf_transformer = trans.named_steps['tfidf']
                        break
                
                if tfidf_transformer is not None:
                    vector = tfidf_transformer.transform([text])
                    if vector.nnz == 0:
                        has_tfidf = False
            except Exception as e:
                logger.warning(f"Error checking TF-IDF vocabulary overlap: {e}")

        # Nếu không có mô hình đã train HOẶC văn bản không có từ vựng tiếng Anh khớp -> Dùng Heuristics phong cách viết
        if self.model is None or not has_tfidf:
            style_score = 0.05
            if exclamation_count > 0:
                style_score += min(0.4, (exclamation_count * 0.15))
            if question_count > 2:
                style_score += min(0.2, (question_count - 2) * 0.08)
            
            caps_rate = caps_count / total_words
            if caps_rate > 0.1:
                style_score += min(0.3, caps_rate * 1.5)
                
            # Phạt từ giật gân tiếng Việt/Anh
            clickbait_words = [
                'shocking', 'unbelievable', 'secret', 'alert', 'warning', 'break', 'magic', 'exposed', 'critical',
                'kinh hoàng', 'khẩn cấp', 'chấn động', 'vạch trần', 'sốc', 'tin nóng', 'nguy hiểm', 'bất ngờ'
            ]
            clickbait_found = sum(1 for w in clickbait_words if w in text.lower())
            if clickbait_found > 0:
                style_score += min(0.4, clickbait_found * 0.20)
                
            style_score = min(style_score, 0.99)
            label = "Unreliable Style" if style_score >= 0.5 else "Reliable Style"
            confidence = style_score if label == "Unreliable Style" else (1.0 - style_score)
            
            if not has_tfidf and self.model is not None:
                logger.info(f"OOD text (zero TF-IDF overlap). Applying stylistic heuristics fallback. Style Score: {style_score:.4f}")
                
            return {
                "label": label,
                "confidence_score": confidence,
                "style_score": style_score,
                "details": details
            }

        try:
            # Chạy mô hình dự đoán (pipeline mong đợi 1 kết quả dạng danh sách)
            probs = self.model.predict_proba([text])[0]
            # lớp 0: Reliable, lớp 1: Unreliable
            style_score = float(probs[1])
            
            if style_score >= 0.5:
                label = "Unreliable Style"
                confidence = style_score
            else:
                label = "Reliable Style"
                confidence = 1.0 - style_score
                
            return {
                "label": label,
                "confidence_score": confidence,
                "style_score": style_score,
                "details": details
            }
        except Exception as e:
            logger.error(f"Error predicting style: {e}")
            return {
                "label": "Reliable Style",
                "confidence_score": 0.5,
                "style_score": 0.5,
                "details": details
            }
