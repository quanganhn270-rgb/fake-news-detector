import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from typing import Set, Tuple, Optional

# Cấu hình Logger
logger = logging.getLogger("fake_news_detector.gatekeeper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class GatekeeperVerdict(str, Enum):
    TIN_THAT = "TIN THẬT"
    TIN_GIA_MAO = "TIN GIẢ MẠO"
    TIEP_TUC = "TIẾP TỤC"

class GatekeeperService:
    """
    Trạm 1: Gatekeeper (Fail-Fast URL Check).
    Nhận diện URL, đối chiếu Whitelist/Blacklist và phát hiện Typosquatting (nhái tên miền).
    """

    def __init__(
        self,
        whitelist_path: Optional[Path] = None,
        blacklist_path: Optional[Path] = None,
        typosquatting_threshold: float = 0.85
    ) -> None:
        # Xác định thư mục gốc của dự án để tải file dữ liệu
        base_dir = Path(__file__).resolve().parent.parent.parent
        
        self.whitelist_path = whitelist_path or (base_dir / "data" / "whitelist_domains.json")
        self.blacklist_path = blacklist_path or (base_dir / "data" / "blacklist_domains.json")
        self.typosquatting_threshold = typosquatting_threshold
        
        # Danh sách đuôi mở rộng tên miền độc hại phổ biến
        self.suspicious_tlds: Set[str] = {
            "xyz", "tk", "cc", "top", "ga", "cf", "gq", "ml", "work", "click", "fit", "buzz"
        }
        
        # Các từ khóa lừa đảo ghép thêm để nhái thương hiệu lớn
        self.typo_keywords: Set[str] = {
            "news", "tinnhanh", "tinmoinhat", "24h", "tin24h", "hot", "live", "online", 
            "today", "press", "daily", "portal", "tinnong", "suckhoe", "soha", "kenh14"
        }
        
        self.whitelist_domains: Set[str] = set()
        self.blacklist_domains: Set[str] = set()
        
        self._load_data()

    def _load_data(self) -> None:
        """Tải danh sách whitelist và blacklist từ file JSON."""
        # Tải Whitelist
        try:
            if self.whitelist_path.exists():
                with open(self.whitelist_path, "r", encoding="utf-8") as f:
                    domains = json.load(f)
                    self.whitelist_domains = {d.strip().lower() for d in domains if d.strip()}
                logger.info(f"Loaded {len(self.whitelist_domains)} whitelist domains from {self.whitelist_path}")
            else:
                logger.warning(f"Whitelist file not found at {self.whitelist_path}. Using empty whitelist.")
        except Exception as e:
            logger.error(f"Error loading whitelist from {self.whitelist_path}: {e}", exc_info=True)

        # Tải Blacklist
        try:
            if self.blacklist_path.exists():
                with open(self.blacklist_path, "r", encoding="utf-8") as f:
                    domains = json.load(f)
                    self.blacklist_domains = {d.strip().lower() for d in domains if d.strip()}
                logger.info(f"Loaded {len(self.blacklist_domains)} blacklist domains from {self.blacklist_path}")
            else:
                logger.warning(f"Blacklist file not found at {self.blacklist_path}. Using empty blacklist.")
        except Exception as e:
            logger.error(f"Error loading blacklist from {self.blacklist_path}: {e}", exc_info=True)

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Tính khoảng cách Levenshtein giữa hai chuỗi s1 và s2.
        Sử dụng Dynamic Programming với tối ưu hóa bộ nhớ O(min(len(s1), len(s2))).
        """
        if len(s1) < len(s2):
            return GatekeeperService.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    @classmethod
    def calculate_similarity(cls, s1: str, s2: str) -> float:
        """Tính toán tỷ lệ tương đồng giữa 2 chuỗi dựa trên khoảng cách Levenshtein."""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = cls.levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)

    def extract_domain(self, text: str) -> Optional[str]:
        """
        Trích xuất và chuẩn hóa domain từ chuỗi đầu vào.
        Trả về None nếu không trích xuất được.
        """
        text = text.strip()
        if not text:
            return None
        
        # Nếu chuỗi chứa khoảng trắng hoặc ký tự đặc biệt xuống dòng, có thể là văn bản thô, không phải URL
        if any(c.isspace() for c in text):
            return None

        # Tự động thêm scheme nếu thiếu để urlparse phân tích chính xác netloc
        if not re.match(r'^(?:http|ftp)s?://', text, re.IGNORECASE):
            # Nếu chứa dấu gạch chéo hoặc dấu chấm, ta coi như định dạng URL/Domain
            if "." in text:
                parsed = urlparse("http://" + text)
            else:
                return None
        else:
            parsed = urlparse(text)
            
        domain = parsed.netloc or parsed.path
        if not domain:
            return None
            
        # Loại bỏ thông tin port (ví dụ: localhost:8000 -> localhost)
        if ":" in domain:
            domain = domain.split(":")[0]
            
        # Loại bỏ tiền tố 'www.' và chuyển thành chữ thường
        domain = domain.lower()
        if domain.startswith("www."):
            domain = domain[4:]
            
        return domain

    @staticmethod
    def get_sld(domain: str) -> str:
        """
        Trích xuất Second Level Domain (SLD) - thương hiệu chính của trang báo.
        Ví dụ: 'vnexpress.net' -> 'vnexpress', 'dantri.com.vn' -> 'dantri'
        """
        parts = domain.split(".")
        if len(parts) >= 3 and parts[-2] in {"com", "gov", "org", "edu", "net", "com", "org"}:
            return parts[-3]
        elif len(parts) >= 2:
            return parts[-2]
        return domain

    def check_typosquatting(self, domain: str) -> Tuple[bool, Optional[str], float]:
        """
        Kiểm tra xem domain có dấu hiệu typosquatting (nhái tên miền) so với whitelist hay không.
        Áp dụng cả khoảng cách Levenshtein và quy tắc trích xuất thương hiệu để tăng tỷ lệ phát hiện.
        
        Trả về: (is_typosquatting, matched_whitelist_domain, similarity_score)
        """
        best_match = None
        highest_score = 0.0

        # Lấy thương hiệu chính (SLD) của tên miền cần kiểm tra
        input_sld = self.get_sld(domain)

        for whitelist_domain in self.whitelist_domains:
            # Nếu trùng khớp hoàn toàn, bỏ qua kiểm tra typosquatting
            if domain == whitelist_domain:
                continue

            # 1. So sánh khoảng cách Levenshtein của toàn bộ domain
            similarity = self.calculate_similarity(domain, whitelist_domain)
            if similarity > highest_score:
                highest_score = similarity
                best_match = whitelist_domain

            # 2. Phát hiện typosquatting tinh vi (ví dụ: tuoitre-news.vn vs tuoitre.vn)
            # Trích xuất thương hiệu chính của whitelist
            whitelist_sld = self.get_sld(whitelist_domain)

            # Quy luật 2.1: Tên miền đầu vào chứa thương hiệu chính và kết nối với các từ khóa lừa đảo
            # Ví dụ: tuoitre-news, tuoitre24h, vnexpress-tinnhanh
            is_suspicious_combo = False
            if whitelist_sld in input_sld:
                # Kiểm tra xem có chứa từ khóa nhái đi kèm hoặc ký tự nối không
                for keyword in self.typo_keywords:
                    pattern = rf"({whitelist_sld}[-_]?{keyword})|({keyword}[-_]?{whitelist_sld})"
                    if re.search(pattern, input_sld):
                        is_suspicious_combo = True
                        break
            
            # Quy luật 2.2: So sánh Levenshtein riêng biệt của phần thương hiệu (SLD) để phát hiện viết sai chính tả nhẹ
            # Ví dụ: vnexpresss vs vnexpress, dantrii vs dantri
            sld_similarity = self.calculate_similarity(input_sld, whitelist_sld)
            
            if is_suspicious_combo or sld_similarity >= self.typosquatting_threshold:
                # Nếu là combo lừa đảo tinh vi, gán điểm tương đồng cực cao để kích hoạt cảnh báo
                effective_score = max(similarity, sld_similarity, 0.90 if is_suspicious_combo else 0.0)
                if effective_score > highest_score:
                    highest_score = effective_score
                    best_match = whitelist_domain

        # Nếu độ tương đồng vượt ngưỡng threshold, xác định là hành vi typosquatting giả mạo thương hiệu
        if highest_score >= self.typosquatting_threshold:
            logger.warning(
                f"Typosquatting detected! Domain '{domain}' is highly similar to Whitelist '{best_match}' "
                f"(Score: {highest_score:.4f} >= Threshold: {self.typosquatting_threshold})"
            )
            return True, best_match, highest_score

        return False, None, highest_score

    def is_whitelist_domain(self, domain: str) -> bool:
        """
        Kiểm tra xem domain có nằm trong Whitelist hoặc là subdomain của Whitelist không.
        Ví dụ: 'sub.chinhphu.vn' là subdomain hợp lệ của 'chinhphu.vn'.
        """
        if domain in self.whitelist_domains:
            return True
            
        # Kiểm tra xem có phải subdomain của một whitelist domain nào không
        for whitelist_domain in self.whitelist_domains:
            if domain.endswith("." + whitelist_domain):
                return True
                
        return False

    def evaluate(self, input_text: str) -> dict:
        """
        Đánh giá đầu vào qua Trạm 1 (Gatekeeper).
        Trả về dict kết quả chuẩn hóa bao gồm: verdict, reason, matched_domain, similarity_score.
        """
        try:
            domain = self.extract_domain(input_text)
            
            if not domain:
                # Không phải là URL, chuyển tiếp sang Trạm 2 xử lý văn bản thô
                return {
                    "verdict": GatekeeperVerdict.TIEP_TUC,
                    "reason": "Đầu vào không phải là URL hợp lệ. Chuyển sang trích xuất và phân tích văn bản.",
                    "matched_domain": None,
                    "similarity_score": None
                }

            logger.info(f"Processing URL/Domain check for: '{domain}'")

            # 1. Kiểm tra khớp Whitelist trực tiếp (bao gồm cả subdomain chính thống)
            if self.is_whitelist_domain(domain):
                url_to_parse = input_text.strip()
                if not re.match(r'^(?:http|ftp)s?://', url_to_parse, re.IGNORECASE):
                    url_to_parse = "http://" + url_to_parse
                parsed_url = urlparse(url_to_parse)
                path = parsed_url.path.strip('/')
                is_homepage = not path or path.lower() in {"index.html", "index.php", "index.htm", "default.aspx", "default.html"}

                if is_homepage:
                    return {
                        "verdict": GatekeeperVerdict.TIN_THAT,
                        "reason": f"Domain '{domain}' nằm trong Whitelist chính thống hoặc là tên miền con được bảo trợ.",
                        "matched_domain": domain,
                        "similarity_score": 1.0
                    }
                else:
                    logger.info(f"Whitelisted domain '{domain}' has article path. Continuing to content check.")
                    return {
                        "verdict": GatekeeperVerdict.TIEP_TUC,
                        "reason": f"Tên miền '{domain}' thuộc Whitelist chính thống nhưng có đường dẫn con cụ thể. Chuyển sang cào và phân tích nội dung bài viết.",
                        "matched_domain": domain,
                        "similarity_score": 0.8
                    }

            # 2. Kiểm tra tên miền chính phủ Việt Nam (.gov.vn)
            if domain.endswith(".gov.vn"):
                url_to_parse = input_text.strip()
                if not re.match(r'^(?:http|ftp)s?://', url_to_parse, re.IGNORECASE):
                    url_to_parse = "http://" + url_to_parse
                parsed_url = urlparse(url_to_parse)
                path = parsed_url.path.strip('/')
                is_homepage = not path or path.lower() in {"index.html", "index.php", "index.htm", "default.aspx", "default.html"}

                if is_homepage:
                    return {
                        "verdict": GatekeeperVerdict.TIN_THAT,
                        "reason": f"Domain '{domain}' là trang thông tin điện tử của cơ quan chính quyền (.gov.vn).",
                        "matched_domain": domain,
                        "similarity_score": 1.0
                    }
                else:
                    logger.info(f"Government domain '{domain}' has article path. Continuing to content check.")
                    return {
                        "verdict": GatekeeperVerdict.TIEP_TUC,
                        "reason": f"Tên miền chính phủ '{domain}' có đường dẫn con cụ thể. Chuyển sang cào và phân tích nội dung bài viết.",
                        "matched_domain": domain,
                        "similarity_score": 0.9
                    }

            # 3. Kiểm tra khớp Blacklist trực tiếp
            if domain in self.blacklist_domains:
                return {
                    "verdict": GatekeeperVerdict.TIN_GIA_MAO,
                    "reason": f"Domain '{domain}' nằm trong Blacklist lừa đảo/tin giả đã được định danh.",
                    "matched_domain": domain,
                    "similarity_score": 1.0
                }

            # 4. Kiểm tra đuôi tên miền độc hại/nghi ngờ (.xyz, .tk, .cc...)
            suffix = domain.split(".")[-1]
            if suffix in self.suspicious_tlds:
                return {
                    "verdict": GatekeeperVerdict.TIN_GIA_MAO,
                    "reason": f"Domain '{domain}' sử dụng đuôi tên miền rác/nghi ngờ '.{suffix}' chuyên dùng cho các trang lừa đảo.",
                    "matched_domain": None,
                    "similarity_score": None
                }

            # 5. Kiểm tra Typosquatting (Giả mạo tên miền của Whitelist)
            is_typo, matched_domain, score = self.check_typosquatting(domain)
            if is_typo:
                return {
                    "verdict": GatekeeperVerdict.TIN_GIA_MAO,
                    "reason": f"Phát hiện dấu hiệu giả mạo tên miền (Typosquatting). Domain '{domain}' cực kỳ tương đồng với báo chính thống '{matched_domain}' (Tỷ lệ giống: {score * 100:.2f}%).",
                    "matched_domain": matched_domain,
                    "similarity_score": score
                }

            # 6. Mặc định: Không khớp bất kỳ quy luật nào -> Chuyển sang Trạm tiếp theo
            return {
                "verdict": GatekeeperVerdict.TIEP_TUC,
                "reason": f"Domain '{domain}' an toàn nhưng chưa được phân loại tin tức trực tiếp. Cần chuyển sang phân tích nội dung.",
                "matched_domain": None,
                "similarity_score": None
            }

        except Exception as e:
            logger.error(f"Error executing Gatekeeper evaluation for input '{input_text}': {e}", exc_info=True)
            # Trong trường hợp có lỗi hệ thống ngoài ý muốn, để bảo toàn tính ổn định, chuyển sang Trạm 2
            return {
                "verdict": GatekeeperVerdict.TIEP_TUC,
                "reason": f"Lỗi hệ thống khi kiểm tra Gatekeeper: {str(e)}. Chuyển tiếp luồng phân tích nội dung.",
                "matched_domain": None,
                "similarity_score": None
            }
