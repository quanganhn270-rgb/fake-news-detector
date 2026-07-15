import os
import sys
import json
import logging
import requests
from urllib.parse import urlparse
from pathlib import Path
from typing import Set, Optional

# Cấu hình logging
logger = logging.getLogger("fake_news_detector.sync_blacklist")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Cấu hình sys.path để truy cập thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Nguồn dữ liệu hostsVN - Dự án cộng đồng chặn tên miền độc hại, lừa đảo, cờ bạc và rác phổ biến nhất Việt Nam
HOSTSVN_URL = "https://raw.githubusercontent.com/bigdargon/hostsVN/master/hosts"


class BlacklistSynchronizer:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.blacklist_path = data_dir / "blacklist_domains.json"

    def clean_domain(self, domain: str) -> Optional[str]:
        """Chuẩn hóa domain (chuyển chữ thường, bỏ www., kiểm tra định dạng)."""
        if not domain or not isinstance(domain, str):
            return None
        domain = domain.strip().lower()
        if domain.startswith("www."):
            domain = domain[4:]
        
        # Bỏ qua các domain nội bộ hoặc không hợp lệ
        if domain in {"localhost", "broadcasthost", "local"} or domain.startswith("255.255.255"):
            return None
            
        # Kiểm tra xem có cấu trúc domain hợp lệ (chứa ít nhất một dấu chấm và không chứa ký tự đặc biệt)
        if "." in domain and not any(c in domain for c in ("/", "?", "#", "@", ":")):
            return domain
        return None

    def fetch_hostsvn_blacklist(self) -> Set[str]:
        """
        Tải và phân tích file hosts của dự án hostsVN.
        Định dạng file hosts:
        # Comment
        0.0.0.0 scam-domain.com
        127.0.0.1 malware-domain.net
        """
        logger.info(f"Downloading blacklist database from hostsVN project: {HOSTSVN_URL}...")
        downloaded_domains: Set[str] = set()

        try:
            response = requests.get(HOSTSVN_URL, timeout=20)
            if response.status_code == 200:
                lines = response.text.splitlines()
                logger.info(f"Successfully downloaded {len(lines)} lines from hostsVN.")
                
                for line in lines:
                    line = line.strip()
                    # Bỏ qua dòng trống, dòng comment hoặc dòng cấu hình hệ thống
                    if not line or line.startswith("#"):
                        continue
                    
                    # File hosts thường phân tách bằng khoảng trắng: IP <domain>
                    parts = line.split()
                    if len(parts) >= 2:
                        ip = parts[0]
                        # Chỉ lấy các dòng trỏ về IP chặn (0.0.0.0 hoặc 127.0.0.1)
                        if ip in {"0.0.0.0", "127.0.0.1"}:
                            domain = parts[1]
                            cleaned = self.clean_domain(domain)
                            if cleaned:
                                downloaded_domains.add(cleaned)
                
                logger.info(f"Parsed and extracted {len(downloaded_domains)} unique clean domains.")
            else:
                logger.error(f"Failed to fetch data. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching hostsVN blacklist: {e}", exc_info=True)
            
        return downloaded_domains

    def sync(self) -> None:
        """Đồng bộ danh sách đen cục bộ với dữ liệu cộng đồng mới nhất."""
        # 1. Tải blacklist từ hostsVN
        external_blacklist = self.fetch_hostsvn_blacklist()
        if not external_blacklist:
            logger.warning("No data retrieved from external sources. Synchronization aborted.")
            return

        # 2. Đọc blacklist cục bộ hiện tại
        existing_blacklist: Set[str] = set()
        if self.blacklist_path.exists():
            try:
                with open(self.blacklist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    existing_blacklist = {d.strip().lower() for d in data if d.strip()}
                logger.info(f"Loaded {len(existing_blacklist)} existing local blacklist domains.")
            except Exception as e:
                logger.error(f"Error reading existing local blacklist: {e}")
        else:
            logger.info("Local blacklist file does not exist. A new file will be created.")

        # 3. Gộp danh sách và lọc trùng lặp
        merged_blacklist = existing_blacklist.union(external_blacklist)
        new_domains_count = len(merged_blacklist) - len(existing_blacklist)

        # 4. Ghi lại dữ liệu đã gộp xuống file JSON
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.blacklist_path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(merged_blacklist)), f, indent=2, ensure_ascii=False)
            logger.info(
                f"Sync Completed! Added {new_domains_count} new scam domains. "
                f"Total blacklist domains now: {len(merged_blacklist)}"
            )
        except Exception as e:
            logger.error(f"Failed to save synchronized blacklist to {self.blacklist_path}: {e}")


if __name__ == "__main__":
    data_dir_path = BASE_DIR / "data"
    synchronizer = BlacklistSynchronizer(data_dir_path)
    synchronizer.sync()
