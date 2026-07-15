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
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Đảm bảo UTF-8 cho console output trên Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Cấu hình đường dẫn dữ liệu dap 1 và mô hình đầu ra
DAP1_DIR = Path("C:/Users/yey81/Downloads/dap 1")

FAKE_CSV_PATH = DAP1_DIR / "Fake.csv"
TRUE_CSV_PATH = DAP1_DIR / "True.csv"
MODEL_PATH = BASE_DIR / "data" / "dap1_model.pkl"

# Import các transformer tùy chỉnh để đảm bảo Pickle có thể load được
from app.services.dap1_classifier import Dap1TextCleaner

def train():
    print("=" * 60)
    print("  HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI NỘI DUNG DAP 1 (CONTENT CLASSIFIER)")
    print("=" * 60)

    if not FAKE_CSV_PATH.exists() or not TRUE_CSV_PATH.exists():
        print(f"❌ Không tìm thấy các file dữ liệu tại {DAP1_DIR}")
        print("Vui lòng đảm bảo các file Fake.csv và True.csv tồn tại trong thư mục dap 1.")
        sys.exit(1)

    print("⌛ Đang đọc dữ liệu Fake.csv và True.csv...")
    t_start = time.time()
    
    df_fake = pd.read_csv(FAKE_CSV_PATH)
    df_true = pd.read_csv(TRUE_CSV_PATH)
    
    print(f"📊 Đã đọc {len(df_fake)} dòng tin giả và {len(df_true)} dòng tin thật.")
    
    # Tiền xử lý: loại bỏ Reuters để tránh rò rỉ dữ liệu
    print("🧹 Loại bỏ chuỗi '(Reuters)' trong tập tin thật để tránh rò rỉ dữ liệu...")
    df_true["text"] = df_true["text"].replace(r"\(Reuters\)", "", regex=True)
    
    # Gán nhãn: 0 cho Fake, 1 cho True
    df_fake["target"] = 0
    df_true["target"] = 1
    
    # Loại bỏ các cột không cần thiết
    df_fake = df_fake.drop(["title", "subject", "date"], axis=1)
    df_true = df_true.drop(["title", "subject", "date"], axis=1)
    
    # Gộp và trộn dữ liệu
    df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
    df = df.dropna(subset=["text"])
    
    # Trộn ngẫu nhiên
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df["text"]
    y = df["target"]
    
    # Chia tập train/test (75% / 25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"   Tập Train: {len(X_train)} dòng | Tập Test: {len(X_test)} dòng")
    
    # Xây dựng Pipeline xử lý đặc trưng văn bản và phân loại
    print("⚙️ Đang xây dựng pipeline xử lý đặc trưng...")
    pipeline = Pipeline([
        ('cleaner', Dap1TextCleaner()),
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    
    print("⚡ Bắt đầu huấn luyện mô hình Logistic Regression...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"✅ Hoàn thành huấn luyện trong {elapsed:.2f} giây!")
    
    # Đánh giá mô hình trên tập Test
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 Độ chính xác trên tập Test: {acc * 100:.2f}%")
    print("\nChi tiết báo cáo phân loại:")
    print(classification_report(y_test, y_pred, target_names=["Fake News", "True News"]))
    
    # Đảm bảo thư mục lưu trữ tồn tại
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu mô hình hoàn chỉnh (.pkl)
    print(f"💾 Đang lưu mô hình vào: {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
        
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"🎉 Hoàn tất! Kích thước file mô hình: {size_mb:.2f} MB")
    print(f"⏱️ Tổng thời gian chạy: {time.time() - t_start:.2f} giây")
    print("=" * 60)

if __name__ == "__main__":
    train()
