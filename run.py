"""
=============================================================
  HỆ THỐNG PHÁT HIỆN TIN GIẢ - FAKE NEWS DETECTION SYSTEM
  Script khởi chạy toàn bộ hệ thống (All-in-One Launcher)
=============================================================
Chạy file này để tự động:
  1. Cài đặt thư viện còn thiếu
  2. Đồng bộ danh sách đen (Blacklist) từ hostsVN
  3. Nạp dữ liệu đối chứng từ CSV vào ChromaDB
  4. Cào tin tức mới nhất từ RSS báo chính thống
  5. Chạy kiểm thử tự động (Unit Tests)
  6. Khởi động API Server + Giao diện Web

Cách dùng:
  python run.py
  python run.py --skip-install      (Bỏ qua cài thư viện)
  python run.py --skip-data         (Bỏ qua nạp dữ liệu)
  python run.py --skip-tests        (Bỏ qua kiểm thử)
  python run.py --port 3000         (Đổi cổng server)
  python run.py --max-docs 5000     (Nạp nhiều dữ liệu hơn)
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent


def print_banner():
    print()
    print("=" * 60)
    print("  🛡️  HỆ THỐNG PHÁT HIỆN TIN GIẢ  🛡️")
    print("  Fake News Detection System - All-in-One Launcher")
    print("=" * 60)
    print()


def print_step(step_num: int, title: str):
    print()
    print(f"{'─' * 60}")
    print(f"  Bước {step_num}: {title}")
    print(f"{'─' * 60}")


def run_cmd(cmd: list, cwd: str = None, check: bool = True) -> bool:
    """Chạy lệnh và in output theo thời gian thực."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(BASE_DIR),
            check=check,
            encoding="utf-8",
            errors="replace"
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Lệnh thất bại (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"  ⚠️  Lỗi: {e}")
        return False


def step_install_deps():
    """Bước 1: Cài đặt thư viện từ requirements.txt."""
    print_step(1, "Cài đặt thư viện phụ thuộc")

    req_file = BASE_DIR / "requirements.txt"
    if not req_file.exists():
        print("  ❌ Không tìm thấy file requirements.txt!")
        return False

    print("  📦 Đang cài đặt từ requirements.txt...")
    success = run_cmd([
        sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"
    ])

    if success:
        print("  ✅ Cài đặt thư viện hoàn tất!")
    else:
        print("  ⚠️  Một số thư viện có thể chưa cài được. Tiếp tục...")
    return True


def step_sync_blacklist():
    """Bước 2: Đồng bộ danh sách đen từ hostsVN."""
    print_step(2, "Đồng bộ danh sách tên miền độc hại (Blacklist)")

    script = BASE_DIR / "scripts" / "sync_blacklist.py"
    if not script.exists():
        print("  ⚠️  Không tìm thấy scripts/sync_blacklist.py. Bỏ qua.")
        return True

    print("  🌐 Đang tải danh sách đen từ hostsVN...")
    success = run_cmd([sys.executable, str(script)], check=False)

    if success:
        print("  ✅ Đồng bộ Blacklist hoàn tất!")
    else:
        print("  ⚠️  Đồng bộ Blacklist có lỗi nhưng không ảnh hưởng hệ thống chính.")
    return True


def step_ingest_csv(max_docs: int):
    """Bước 3: Nạp dữ liệu từ CSV vào ChromaDB."""
    print_step(3, f"Nạp dữ liệu đối chứng vào ChromaDB (tối đa {max_docs} tài liệu)")

    script = BASE_DIR / "scripts" / "ingest_data.py"
    csv_file = BASE_DIR / "opensources_fake_news_cleaned_100k.csv"

    if not script.exists():
        print("  ⚠️  Không tìm thấy scripts/ingest_data.py. Bỏ qua.")
        return True
    if not csv_file.exists():
        print("  ⚠️  Không tìm thấy file CSV dữ liệu. Bỏ qua nạp ChromaDB.")
        return True

    print(f"  📊 Đang xử lý file CSV ({csv_file.stat().st_size // (1024*1024)} MB)...")
    success = run_cmd([
        sys.executable, str(script), "--max-docs", str(max_docs)
    ], check=False)

    if success:
        print("  ✅ Nạp dữ liệu đối chứng hoàn tất!")
    else:
        print("  ⚠️  Nạp dữ liệu có lỗi. Hệ thống vẫn hoạt động với dữ liệu hiện có.")
    return True


def step_crawl_rss():
    """Bước 4: Cào tin tức mới nhất từ RSS."""
    print_step(4, "Cào tin tức mới nhất từ báo chính thống (RSS)")

    script = BASE_DIR / "scripts" / "crawl_rss.py"
    if not script.exists():
        print("  ⚠️  Không tìm thấy scripts/crawl_rss.py. Bỏ qua.")
        return True

    print("  📰 Đang quét VnExpress, Tuổi Trẻ, Thanh Niên, VietnamNet...")
    success = run_cmd([sys.executable, str(script), "--max-articles", "25"], check=False)

    if success:
        print("  ✅ Cào tin tức RSS hoàn tất!")
    else:
        print("  ⚠️  Cào RSS có lỗi nhưng không ảnh hưởng hệ thống chính.")
    return True


def step_crawl_tingia():
    """Bước 4b: Cào cảnh báo tin giả từ VAFC (tingia.gov.vn)."""
    print_step(4, "Cào cảnh báo tin giả từ VAFC (tingia.gov.vn)")

    script = BASE_DIR / "scripts" / "crawl_tingia.py"
    if not script.exists():
        print("  ⚠️  Không tìm thấy scripts/crawl_tingia.py. Bỏ qua.")
        return True

    print("  🛡️  Đang quét các cảnh báo tin giả mới nhất từ tingia.gov.vn...")
    success = run_cmd([sys.executable, str(script)], check=False)

    if success:
        print("  ✅ Cào cảnh báo tin giả VAFC hoàn tất!")
    else:
        print("  ⚠️  Cào VAFC có lỗi nhưng không ảnh hưởng hệ thống chính.")
    return True



def step_run_tests():
    """Bước 5: Chạy kiểm thử tự động."""
    print_step(5, "Chạy kiểm thử tự động (Unit Tests)")

    test_file = BASE_DIR / "tests" / "test_components.py"
    if not test_file.exists():
        print("  ⚠️  Không tìm thấy tests/test_components.py. Bỏ qua.")
        return True

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print("  🧪 Đang chạy kiểm thử...")
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(BASE_DIR),
            env=env,
            encoding="utf-8",
            errors="replace",
            check=False
        )
        if result.returncode == 0:
            print("  ✅ Kiểm thử hoàn tất thành công!")
            return True
        else:
            print("  ⚠️  Một số bài kiểm thử chưa đạt. Hệ thống vẫn có thể chạy.")
            return True
    except Exception as e:
        print(f"  ⚠️  Lỗi khi chạy test: {e}")
        return True


def step_start_server(port: int):
    """Bước 6: Khởi động FastAPI server."""
    print_step(6, "Khởi động API Server + Giao diện Web")

    print(f"  🚀 Server đang khởi động tại:")
    print()
    print(f"     ┌─────────────────────────────────────────┐")
    print(f"     │  🌐 Giao diện Web: http://127.0.0.1:{port}  │")
    print(f"     │  📄 API Docs:  http://127.0.0.1:{port}/docs │")
    print(f"     └─────────────────────────────────────────┘")
    print()
    print(f"  💡 Mở trình duyệt và truy cập địa chỉ trên để sử dụng.")
    print(f"  💡 Nhấn Ctrl+C để dừng server.")
    print()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", str(port)],
            cwd=str(BASE_DIR),
            env=env
        )
    except KeyboardInterrupt:
        print("\n  🛑 Server đã dừng.")


def main():
    parser = argparse.ArgumentParser(
        description="Khởi chạy toàn bộ Hệ thống Phát hiện Tin giả"
    )
    parser.add_argument("--skip-install", action="store_true", help="Bỏ qua cài đặt thư viện")
    parser.add_argument("--skip-data", action="store_true", help="Bỏ qua tất cả bước nạp dữ liệu")
    parser.add_argument("--skip-tests", action="store_true", help="Bỏ qua kiểm thử")
    parser.add_argument("--port", type=int, default=8000, help="Cổng server (mặc định: 8000)")
    parser.add_argument("--max-docs", type=int, default=3000, help="Số tài liệu tối đa nạp vào ChromaDB (mặc định: 3000)")

    args = parser.parse_args()

    print_banner()

    start_time = time.time()

    # Bước 1: Cài thư viện
    if not args.skip_install:
        step_install_deps()
    else:
        print("  ⏭️  Bỏ qua bước cài đặt thư viện (--skip-install)")

    # Bước 2-4: Nạp dữ liệu
    if not args.skip_data:
        step_sync_blacklist()
        step_ingest_csv(args.max_docs)
        step_crawl_rss()
        step_crawl_tingia()
    else:
        print("  ⏭️  Bỏ qua bước nạp dữ liệu (--skip-data)")

    # Bước 5: Kiểm thử
    if not args.skip_tests:
        step_run_tests()
    else:
        print("  ⏭️  Bỏ qua bước kiểm thử (--skip-tests)")

    elapsed = time.time() - start_time
    print()
    print(f"  ⏱️  Hoàn tất thiết lập trong {elapsed:.1f} giây")

    # Bước 6: Chạy server
    step_start_server(args.port)


if __name__ == "__main__":
    main()
