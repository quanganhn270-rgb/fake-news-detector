import os
import sys
import re
import string
import time
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report

# Thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Cấu hình đường dẫn
CSV_PATH = BASE_DIR / "opensources_fake_news_cleaned_100k.csv"
MODEL_PATH = BASE_DIR / "data" / "style_classifier.pkl"

# Import các transformer tùy chỉnh để đảm bảo đồng bộ hóa Pickle
from app.services.style_classifier import TextCleaner, StylisticFeatureExtractor


def train():
    print("=" * 60)
    print("  HUẤN LUYỆN MÔ HÌNH PHÂN TÍCH VĂN PHONG (STYLE CLASSIFIER)")
    print("=" * 60)

    if not CSV_PATH.exists():
        print(f"❌ Không tìm thấy file dữ liệu tại {CSV_PATH}")
        sys.exit(1)

    print(f"⌛ Đang đọc dữ liệu từ {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH, usecols=["content", "type"])
    
    # Loại bỏ các dòng rỗng
    df = df.dropna(subset=["content", "type"])
    
    # Phân loại nhãn: reliable -> 0 (Tin thật), các loại khác -> 1 (Tin giả/không đáng tin cậy)
    df["label"] = df["type"].apply(lambda t: 0 if t == "reliable" else 1)
    
    # Cân bằng dữ liệu (đặc biệt là lấy toàn bộ lớp reliable)
    df_reliable = df[df["label"] == 0]
    df_unreliable = df[df["label"] == 1]
    
    print(f"📊 Dữ liệu gốc: {len(df_reliable)} dòng reliable (tin thật) và {len(df_unreliable)} dòng tin giả.")
    
    # Lấy mẫu tin giả cân đối (ví dụ: lấy 3000 dòng để cân bằng hơn với 560 dòng tin thật)
    df_unreliable_sample = df_unreliable.sample(n=min(3000, len(df_unreliable)), random_state=42)
    df_balanced = pd.concat([df_reliable, df_unreliable_sample]).reset_index(drop=True)
    
    print(f"📊 Tập huấn luyện cân bằng: {len(df_balanced)} dòng (Tin thật: {len(df_reliable)}, Tin giả: {len(df_unreliable_sample)})")
    
    X = df_balanced["content"]
    y = df_balanced["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Tập Train: {len(X_train)} dòng | Tập Test: {len(X_test)} dòng")
    
    # Xây dựng Pipeline xử lý song song các đặc trưng ngữ nghĩa (TF-IDF) và phong cách (Style)
    print("⚙️ Đang xây dựng pipeline xử lý đặc trưng...")
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf_pipeline', Pipeline([
                ('cleaner', TextCleaner()),
                ('tfidf', TfidfVectorizer(max_features=5000, min_df=2))
            ])),
            ('style_extractor', StylisticFeatureExtractor())
        ])),
        ('classifier', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42))
    ])
    
    print("⚡ Bắt đầu huấn luyện mô hình Logistic Regression (khoảng 15-30 giây)...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"✅ Hoàn thành huấn luyện trong {elapsed:.2f} giây!")
    
    # Đánh giá mô hình
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 Độ chính xác trên tập Test: {acc * 100:.2f}%")
    print("\nChi tiết báo cáo phân loại:")
    print(classification_report(y_test, y_pred, target_names=["Reliable Style", "Unreliable Style"]))
    
    # Tạo thư mục lưu nếu chưa có
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu mô hình hoàn chỉnh (.pkl)
    print(f"💾 Đang lưu mô hình vào: {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
        
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"🎉 Hoàn tất! Kích thước file mô hình: {size_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    train()
