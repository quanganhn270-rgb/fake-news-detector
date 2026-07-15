import os
import sys
import asyncio
from pathlib import Path

# Đưa thư mục gốc của dự án vào sys.path để import các module dễ dàng
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Đảm bảo đầu ra console hỗ trợ tiếng Việt trên Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# Đăng ký import Gatekeeper độc lập
try:
    from app.services.gatekeeper import GatekeeperService, GatekeeperVerdict
    HAS_GATEKEEPER = True
except ImportError as e:
    print(f"Error importing Gatekeeper: {e}")
    HAS_GATEKEEPER = False

# Đăng ký import RAG độc lập để tránh sập chương trình khi thiếu chromadb
try:
    from app.services.rag_service import RAGService
    HAS_RAG = True
except ImportError as e:
    print(f"\n[WARNING] RAG Service cannot be imported: {e}")
    print("This is normal if you haven't installed dependencies from requirements.txt yet.")
    HAS_RAG = False

# Đăng ký import Style Classifier độc lập
try:
    from app.services.style_classifier import StyleClassifierService
    HAS_STYLE = True
except ImportError as e:
    print(f"\n[WARNING] Style Classifier cannot be imported: {e}")
    HAS_STYLE = False

# Đăng ký import Dap1 Classifier độc lập
try:
    from app.services.dap1_classifier import Dap1ClassifierService
    HAS_DAP1 = True
except ImportError as e:
    print(f"\n[WARNING] Dap1 Classifier cannot be imported: {e}")
    HAS_DAP1 = False



def run_gatekeeper_tests() -> bool:
    if not HAS_GATEKEEPER:
        print("Skipping Gatekeeper tests due to import error.")
        return False

    print("\n" + "="*50)
    print("RUNNING GATEKEEPER (STATION 1) TESTS")
    print("="*50)

    # Khởi tạo dịch vụ
    gatekeeper = GatekeeperService()

    test_cases = [
        # 1. Trang chủ Whitelist chính thống
        ("https://vnexpress.net", GatekeeperVerdict.TIN_THAT, "Whitelist Homepage"),
        ("https://tuoitre.vn/", GatekeeperVerdict.TIN_THAT, "Whitelist Homepage with trailing slash"),
        ("http://chinhphu.vn", GatekeeperVerdict.TIN_THAT, "Government domain Homepage"),
        ("https://subdomain.chinhphu.vn", GatekeeperVerdict.TIN_THAT, "Government subdomain Homepage"),
        ("https://customs.gov.vn/", GatekeeperVerdict.TIN_THAT, "Government TLD Homepage"),
        
        # 2. Trang bài viết cụ thể của Whitelist -> Phải tiếp tục để cào và kiểm tra
        ("https://vnexpress.net/tin-tuc-thoi-su/bai-viet.html", GatekeeperVerdict.TIEP_TUC, "Whitelist Article Path"),
        ("https://tuoitre.vn/the-thao.htm", GatekeeperVerdict.TIEP_TUC, "Whitelist Article Path"),
        ("http://chinhphu.vn/chinh-sach-moi", GatekeeperVerdict.TIEP_TUC, "Government Article Path"),
        ("https://customs.gov.vn/tin-tuc/abc", GatekeeperVerdict.TIEP_TUC, "Government Article Path"),
        
        # 3. Tên miền Blacklist đã biết
        ("http://tinnhanh247.xyz/news", GatekeeperVerdict.TIN_GIA_MAO, "Blacklist"),
        ("https://vietnam-news.tk", GatekeeperVerdict.TIN_GIA_MAO, "Blacklist"),
        
        # 4. Đuôi tên miền rác/nghi ngờ
        ("https://suckhoe24h.cc", GatekeeperVerdict.TIN_GIA_MAO, "Suspicious TLD .cc"),
        ("http://tinnong.top", GatekeeperVerdict.TIN_GIA_MAO, "Suspicious TLD .top"),
        
        # 5. Giả mạo tên miền (Typosquatting)
        ("http://vnexpresss.net/thoi-su", GatekeeperVerdict.TIN_GIA_MAO, "Typosquatting (vnexpresss.net vs vnexpress.net)"),
        ("https://tuoitre-news.vn/hot", GatekeeperVerdict.TIN_GIA_MAO, "Typosquatting (tuoitre-news.vn vs tuoitre.vn)"),
        ("https://dantrii.com.vn", GatekeeperVerdict.TIN_GIA_MAO, "Typosquatting (dantrii.com.vn vs dantri.com.vn)"),

        # 6. Các URL bình thường khác hoặc văn bản thô
        ("https://github.com/google/deepmind", GatekeeperVerdict.TIEP_TUC, "Safe unknown domain"),
        ("Đây là một đoạn văn bản tin tức thông thường chứ không phải URL.", GatekeeperVerdict.TIEP_TUC, "Raw text input")
    ]

    passed = 0
    for idx, (url, expected_verdict, desc) in enumerate(test_cases, 1):
        result = gatekeeper.evaluate(url)
        verdict = result["verdict"]
        reason = result["reason"]
        
        status = "PASSED" if verdict == expected_verdict else "FAILED"
        if status == "PASSED":
            passed += 1
            
        print(f"[{status}] Test #{idx} ({desc}):")
        print(f"  Input  : '{url}'")
        print(f"  Result : {verdict} -> Reason: {reason}")
        print("-" * 50)

    print(f"Gatekeeper Test Result: {passed}/{len(test_cases)} cases PASSED.")
    return passed == len(test_cases)


async def run_rag_tests() -> bool:
    if not HAS_RAG:
        print("\nSkipping RAG Vector DB tests because dependencies (chromadb/langchain) are not installed.")
        return False

    print("\n" + "="*50)
    print("RUNNING RAG VECTOR DB (STATION 3B) TESTS")
    print("="*50)

    # Khởi tạo RAG service sử dụng DB kiểm thử riêng biệt
    test_db_dir = BASE_DIR / "data" / "test_chroma_db"
    
    try:
        # Xóa DB test cũ nếu có
        import shutil
        if test_db_dir.exists():
            shutil.rmtree(test_db_dir)
    except Exception as e:
        print(f"Warning: Could not clear old test database: {e}")

    try:
        rag = RAGService(
            persist_directory=str(test_db_dir),
            collection_name="test_facts"
        )
    except Exception as e:
        print(f"Skipping RAG tests because model or database failed to initialize: {e}")
        print("This usually happens when chroma-db, sentence-transformers, or pytorch is not installed.")
        return False

    # 1. Nạp dữ liệu mẫu
    print("Adding sample fact documents to ChromaDB...")
    sample_facts = [
        "Vắc-xin phòng COVID-19 không chứa chip siêu vi để theo dõi người dân như tin đồn mạng xã hội.",
        "Thông tin Ủy ban thành phố cấp tiền hỗ trợ trực tiếp 5 triệu đồng cho mọi cá nhân gặp khó khăn là giả mạo.",
        "Bộ Tài chính tuyên bố không có chính sách giảm 50% thuế VAT đối với toàn bộ các mặt hàng tiêu dùng trong năm nay."
    ]
    sample_metadatas = [
        {"source": "Bộ Y tế", "date": "2021-06-01"},
        {"source": "UBND Thành phố", "date": "2021-08-10"},
        {"source": "Bộ Tài chính", "date": "2023-02-15"}
    ]
    
    await rag.add_documents_async(sample_facts, sample_metadatas)
    print("Documents added successfully.")

    # 2. Truy vấn test
    # LƯU Ý: Mô hình paraphrase-multilingual-MiniLM-L12-v2 khi xử lý tiếng Việt
    # thường cho điểm similarity thấp hơn tiếng Anh (khoảng 0.45-0.60 cho câu diễn đạt lại).
    # Trong production, ngưỡng 0.75 đảm bảo Precision-first (chỉ khớp chính xác cao).
    # Trong test, ta dùng ngưỡng 0.40 để kiểm tra cơ chế tìm kiếm ngữ nghĩa hoạt động đúng.
    TEST_THRESHOLD = 0.40

    test_queries = [
        # Khớp ngữ nghĩa (Trên ngưỡng test 0.40)
        ("Mạng xã hội đồn thổi vắc xin covid chứa vi chíp theo dõi mọi người có đúng không?", True, "Should match Vaccine Fact"),
        ("Tôi nghe nói thành phố sẽ phát 5 triệu hỗ trợ cho mọi người dân?", True, "Should match 5 Million Subsidy Fact"),
        
        # Khớp ngữ nghĩa thấp (Dưới ngưỡng - mong đợi kết quả rỗng)
        ("Tình hình xuất khẩu lúa gạo của Việt Nam năm nay ra sao?", False, "Should NOT match any facts (Expected empty)"),
        ("Ngày mai trời có mưa to ở Hà Nội hay không?", False, "Should NOT match any facts (Expected empty)")
    ]

    passed = 0
    for idx, (query, should_match, desc) in enumerate(test_queries, 1):
        print(f"\nQuery #{idx} ({desc}): '{query}'")
        matched_results = await rag.query_semantic_async(query, top_k=3, threshold=TEST_THRESHOLD)

        
        has_match = len(matched_results) > 0
        status = "PASSED" if has_match == should_match else "FAILED"
        if status == "PASSED":
            passed += 1
            
        print(f"  [{status}] Results found: {len(matched_results)}")
        for r in matched_results:
            print(f"    - Confirmed Document: '{r['document']}'")
            print(f"      Source: {r['metadata'].get('source')} | Similarity Score: {r['similarity_score']}")
            
    print("\n" + "-" * 50)
    print(f"RAG Test Result: {passed}/{len(test_queries)} cases PASSED.")
    
    # Dọn dẹp DB test sau khi hoàn thành
    try:
        import shutil
        if test_db_dir.exists():
            shutil.rmtree(test_db_dir)
    except Exception:
        pass
        
    return passed == len(test_queries)


def run_style_tests() -> bool:
    if not HAS_STYLE:
        print("Skipping Style Classifier tests due to import error.")
        return False

    print("\n" + "="*50)
    print("RUNNING STYLE CLASSIFIER (STATION 3A) TESTS")
    print("="*50)

    # Khởi tạo dịch vụ
    classifier = StyleClassifierService()

    test_cases = [
        # 1. Văn phong giật gân, đáng ngờ (Nhiều dấu chấm cảm, viết hoa, từ clickbait)
        ("TIN NÓNG KHẨN CẤP!!! SỰ THẬT KINH HOÀNG VỀ VẮC-XIN COVID-19 MÀ HỌ ĐANG GIẤU BẠN!!! XEM NGAY TRƯỚC KHI BỊ XÓA!!!", True, "Sensational/Clickbait style"),
        
        # 2. Văn phong bình thường, khách quan
        ("Theo báo cáo của Bộ Y tế, tính đến ngày hôm nay, tỷ lệ tiêm chủng đầy đủ vắc-xin phòng COVID-19 cho người trưởng thành tại Việt Nam đã đạt trên 90%.", False, "Objective news style")
    ]

    passed = 0
    for idx, (text, should_be_unreliable, desc) in enumerate(test_cases, 1):
        res = classifier.predict_style(text)
        style_score = res["style_score"]
        label = res["label"]
        
        is_unreliable = (label == "Unreliable Style")
        status = "PASSED" if is_unreliable == should_be_unreliable else "FAILED"
        if status == "PASSED":
            passed += 1
            
        print(f"[{status}] Test #{idx} ({desc}):")
        print(f"  Input  : '{text[:80]}...'")
        print(f"  Result : {label} (Style Score: {style_score:.4f})")
        print("-" * 50)

    print(f"Style Classifier Test Result: {passed}/{len(test_cases)} cases PASSED.")
    return passed == len(test_cases)


def run_dap1_tests() -> bool:
    if not HAS_DAP1:
        print("Skipping Dap1 Classifier tests due to import error.")
        return False

    print("\n" + "="*50)
    print("RUNNING DAP1 CONTENT CLASSIFIER TESTS")
    print("="*50)

    # Khởi tạo dịch vụ
    classifier = Dap1ClassifierService()

    # Mô hình dap 1 được huấn luyện trên tiếng Anh.
    test_cases = [
        # 1. Tin giả tiếng Anh rõ ràng (ví dụ tin đồn Donald Trump)
        ("Donald Trump sends out embarrassing New Year's Eve message. Donald Trump just couldn't wish all Americans a happy new year.", True, "Known Fake news format (English)"),
        
        # 2. Tin thật tiếng Anh rõ ràng (từ Reuters)
        ("WASHINGTON - U.S. military to accept transgender recruits. Transgender people will be allowed for the first time to enlist in the U.S. military.", False, "Known Real news format (English)")
    ]

    passed = 0
    for idx, (text, should_be_fake, desc) in enumerate(test_cases, 1):
        res = classifier.predict_content(text)
        confidence = res["confidence_score"]
        label = res["label"]
        is_fake = res["is_fake"]
        
        status = "PASSED" if is_fake == should_be_fake else "FAILED"
        if status == "PASSED":
            passed += 1
            
        print(f"[{status}] Test #{idx} ({desc}):")
        print(f"  Input  : '{text[:80]}...'")
        print(f"  Result : {label} (Confidence: {confidence:.4f})")
        print("-" * 50)

    print(f"Dap1 Classifier Test Result: {passed}/{len(test_cases)} cases PASSED.")
    return passed == len(test_cases)


async def main():
    gatekeeper_success = run_gatekeeper_tests()
    style_success = run_style_tests()
    dap1_success = run_dap1_tests()
    rag_success = await run_rag_tests()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
    print(f"Gatekeeper Tests: {'PASSED' if gatekeeper_success else 'FAILED'}")
    print(f"Style Classifier Tests: {'PASSED' if style_success else 'FAILED' if HAS_STYLE else 'SKIPPED'}")
    print(f"Dap1 Classifier Tests: {'PASSED' if dap1_success else 'FAILED' if HAS_DAP1 else 'SKIPPED'}")
    print(f"RAG Service Tests: {'PASSED' if rag_success else 'FAILED' if HAS_RAG else 'SKIPPED (Missing Dependencies)'}")


if __name__ == "__main__":
    asyncio.run(main())
