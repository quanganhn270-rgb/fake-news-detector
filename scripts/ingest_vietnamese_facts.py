import os
import sys
from pathlib import Path

# Đưa thư mục gốc vào PATH để import app module
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from app.services.rag_service import RAGService
except ImportError as e:
    print(f"❌ Không thể import RAGService: {e}")
    sys.exit(1)

# Đảm bảo đầu ra console hỗ trợ tiếng Việt trên Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    print("=" * 60)
    print("   NẠP CƠ SỞ DỮ LIỆU ĐỐI CHỨNG TIẾNG VIỆT (CHROMADB)")
    print("=" * 60)
    
    # Khởi tạo dịch vụ RAG
    try:
        rag = RAGService()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo RAGService: {e}")
        sys.exit(1)
        
    # Danh sách các tài liệu đối chứng
    documents = [
        # 1. Tin vắc xin chứa chip
        "Tiêu đề: Vắc-xin phòng COVID-19 không chứa chip siêu vi để theo dõi người dân.\n"
        "Nội dung: Bộ Y tế và các chuyên gia y tế khẳng định vắc-xin phòng COVID-19 hoàn toàn không chứa chip siêu vi hay bất kỳ thiết bị theo dõi nào. Đây là tin đồn nhảm nhí lan truyền trên mạng xã hội để gây hoang mang dư luận.",
        
        # 2. Tin thành phố phát 5 triệu đồng hỗ trợ
        "Tiêu đề: UBND Thành phố không có chính sách hỗ trợ tiền mặt 5 triệu đồng cho mọi người dân.\n"
        "Nội dung: Đại diện Ủy ban nhân dân thành phố khẳng định thông tin thành phố cấp tiền hỗ trợ trực tiếp 5 triệu đồng tiền mặt cho mọi cá nhân gặp khó khăn là tin giả mạo. Mọi chính sách hỗ trợ đều được công bố chính thức qua Cổng thông tin điện tử của UBND.",
        
        # 3. Tin đồn ngân hàng phá sản
        "Tiêu đề: Ngân hàng Nhà nước khẳng định hệ thống ngân hàng hoạt động an toàn, không phá sản.\n"
        "Nội dung: Ngân hàng Nhà nước khuyến cáo người dân không nên tin vào các tin đồn thất thiệt lan truyền trên mạng xã hội về việc một số ngân hàng thương mại phá sản dẫn tới đi rút tiền hàng loạt. Tiền gửi của người dân tại ngân hàng được Nhà nước bảo đảm trong mọi trường hợp.",
        
        # 4. Tin đồn Sơn Tùng M-TP giải nghệ
        "Tiêu đề: M-TP Entertainment phủ nhận tin đồn Sơn Tùng giải nghệ.\n"
        "Nội dung: Trên các trang mạng xã hội xuất hiện tin đồn ca sĩ Sơn Tùng M-TP sẽ chính thức giải nghệ vào cuối năm nay để tập trung làm nhà sản xuất. Đại diện truyền thông công ty M-TP Entertainment lên tiếng phủ nhận tin đồn và khẳng định nam ca sĩ đang chuẩn bị cho dự án âm nhạc mới.",
        
        # 5. Tin đồn giá xăng tăng đột biến
        "Tiêu đề: Bộ Công Thương bác bỏ tin đồn giá xăng dầu tăng lên 40.000 đồng/lít.\n"
        "Nội dung: Bộ Công Thương và liên Bộ Tài chính khẳng định thông tin giá xăng dầu trong nước sắp được điều chỉnh tăng lên mức 40.000 đồng một lít vào ngày mai là hoàn toàn bịa đặt, sai sự thật. Việc điều chỉnh giá được thực hiện theo chu kỳ và công bố công khai.",
        
        # 6. Tin thật về xuất khẩu gạo
        "Tiêu đề: Việt Nam duy trì vị thế xuất khẩu gạo hàng đầu thế giới.\n"
        "Nội dung: Theo báo cáo tổng kết của Bộ Nông nghiệp và Phát triển nông thôn, xuất khẩu gạo của Việt Nam tiếp tục tăng trưởng mạnh mẽ, đóng góp lớn vào kim ngạch xuất khẩu và giữ vững vị thế là nước xuất khẩu gạo chất lượng cao hàng đầu trên thị trường quốc tế.",
        
        # 7. Tin đồn hoãn kỳ thi THPT quốc gia
        "Tiêu đề: Bộ Giáo dục và Đào tạo bác bỏ thông tin hoãn kỳ thi tốt nghiệp THPT.\n"
        "Nội dung: Bộ Giáo dục và Đào tạo khẳng định văn bản lan truyền trên mạng xã hội về việc hoãn kỳ thi tốt nghiệp THPT quốc gia do diễn biến thời tiết phức tạp là giả mạo. Lịch thi vẫn diễn ra bình thường theo đúng kế hoạch đã công bố."
    ]
    
    # Metadata tương ứng
    metadatas = [
        {"title": "Vắc-xin COVID-19 không chứa chip siêu vi", "source": "Bộ Y tế", "category": "fake"},
        {"title": "UBND Thành phố bác tin hỗ trợ 5 triệu đồng", "source": "UBND Thành phố", "category": "fake"},
        {"title": "Ngân hàng Nhà nước bác tin đồn ngân hàng phá sản", "source": "Ngân hàng Nhà nước", "category": "fake"},
        {"title": "Đại diện phủ nhận tin Sơn Tùng M-TP giải nghệ", "source": "M-TP Entertainment", "category": "rumor"},
        {"title": "Bộ Công Thương bác tin đồn giá xăng dầu tăng lên 40.000đ/lít", "source": "Bộ Công Thương", "category": "fake"},
        {"title": "Việt Nam duy trì xuất khẩu gạo hàng đầu thế giới", "source": "Bộ Nông nghiệp và Phát triển nông thôn", "category": "reliable"},
        {"title": "Bộ Giáo dục bác tin hoãn kỳ thi tốt nghiệp THPT", "source": "Bộ Giáo dục và Đào tạo", "category": "fake"}
    ]
    
    print(f"📊 Đang nạp {len(documents)} tài liệu đối chứng sự thật tiếng Việt vào ChromaDB...")
    try:
        rag._add_documents_sync(documents, metadatas)
        print("✅ Hoàn tất nạp cơ sở dữ liệu đối chứng tiếng Việt thành công!")
    except Exception as e:
        print(f"❌ Gặp lỗi khi nạp dữ liệu: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
