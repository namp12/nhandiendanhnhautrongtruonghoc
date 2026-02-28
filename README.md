<div align="center">
    <h2>Hệ thống AI Phát hiện và Cảnh báo Hành vi Bạo lực trong Trường học</h2>
    <p><i>Violence Detection System</i></p>
    <img src="https://img.shields.io/badge/Made%20by-Phuong%20Nam-blue?style=for-the-badge" alt="Made by Phuong Nam">
    <img src="https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python" alt="Python">
    <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" alt="Flask">
    <img src="https://img.shields.io/badge/MySQL-Database-orange?style=for-the-badge&logo=mysql" alt="MySQL">
    <br><br>
</div>

---

### Giới thiệu
**Violence Detection System** là một dự án nghiên cứu và ứng dụng Trí tuệ nhân tạo (AI) kết hợp Thị giác máy tính (Computer Vision) nhằm tự động nhận diện các hành vi bất thường như **Đánh nhau (Fighting)** và **Khụy ngã (Falling)** thông qua luồng video trích xuất từ Camera giám sát (CCTV/Webcam). 

Hệ thống giúp cảnh báo kịp thời các sự cố bạo lực học đường, tự động lưu trữ bằng chứng video thành thư mục bảo mật và ghi lại sự cố (Log) vào cơ sở dữ liệu MySQL để phục vụ công tác quản lý của giám thị/bảo vệ.

### Thành viên tham gia
| STT | Tên sinh viên | Mã sinh viên | Nhóm | Lớp |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Nguyễn Thế Phương Nam** | `[1871070011]` | `[Nhóm 2]` | `[HTTT18-01]` |
| 1 | **Lê Duy An** | `[1871070001]` | `[Nhóm 2]` | `[HTTT18-01]` |
| 1 | **Phạm Đăng Quốc Dũng** | `[1871070011]` | `[Nhóm 2]` | `[KHMT18-01]` |

*Tài liệu này hướng dẫn chi tiết cách cài đặt, cấu hình và khởi chạy toàn bộ hệ thống từ A đến Z.*

---

### Mô hình hoạt động
*(Sơ đồ luồng hoạt động của Hệ thống)*
`![System Architecture](./uploads/image.png)`

---

### 💡 Công nghệ sử dụng:
* **Deep Learning Model (PyTorch)**: Xử lý và phân tích chuỗi khung hình (Spatio-temporal) để phân loại hành vi phức tạp của con người theo thời gian.
* **OpenCV**: Trích xuất luồng video stream từ Webcam, xử lý tiền khung hình (Resize, Chuyển hệ màu) để nạp vào AI.
* **Flask (Python)**: Xây dựng nền tảng Web Backend để điều phối API và render giao diện.
* **MySQL**: Hệ quản trị cơ sở dữ liệu (DBMS) lưu trữ lịch sử báo động sự cố có tổ chức.
* **FFmpeg**: Chuyển đổi chuẩn mã hóa video tự động sang `H.264` để phát lại video mượt mà trực tiếp trên mọi trình duyệt Web.
* **HTML5/CSS3/Bootstrap 5**: Xây dựng giao diện người dùng trực quan, có chế độ Dark Theme cực ngầu cho giám sát viên.

---

### Yêu cầu hệ thống
* **Hệ điều hành:** Windows 10/11 hoặc Linux.
* **Python:** Phiên bản `3.8` trở lên.
* **Máy chủ CSDL:** `XAMPP` (Bao gồm MySQL Server) hoặc cài đặt độc lập.
* **Phần cứng:** Ưu tiên máy có Card rời (NVIDIA GPU) để chạy mô hình mượt mà, tuy nhiên vẫn chạy tốt trên CPU với tốc độ FPS được điều tiết.
* **Các thư viện Python (Xem file `requirements.txt`)**: `opencv-python`, `flask`, `mysql-connector-python`, `numpy`, `torch`, `imageio[ffmpeg]`...

---

### Hướng dẫn cài đặt

#### 1. Cài đặt các thư viện lõi
Mở Terminal/Command Prompt (có quyền Admin) tại thư mục dự án và chạy lệnh sau để tải các thư viện AI:
```bash
pip install -r requirements.txt
```

#### 2. Thiết lập cơ sở dữ liệu MySQL
**2.1. Cài đặt MySQL Server**
* Mở ứng dụng **XAMPP Control Panel**.
* Nhấn nút **Start** ở ứng dụng `Apache` và `MySQL`. Đảm bảo module chuyển xanh báo hiệu đang chạy thành công.

**2.2. Khởi tạo Database và Bảng Hệ thống**
Dùng trình quản trị phpMyAdmin (`http://localhost/phpmyadmin`) và dán dòng code sau vào mục SQL để tạo Bảng sự cố. Mã SQL này cũng có tại file `database/schema.sql`:
```sql
CREATE DATABASE IF NOT EXISTS violence_db;
USE violence_db;

CREATE TABLE IF NOT EXISTS EVENT_LOGS (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    video_filename VARCHAR(255) NOT NULL,
    video_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**2.3. Cấu hình Mật khẩu kết nối**
Mở file `app_web/app.py` và sửa biến `DB_CONFIG` (Khoảng dòng 18).
Đảm bảo biến password khớp với Mật khẩu `root` trong máy bạn. Nếu XAMPP mặc định không có Mật khẩu thì để biến thành rỗng (`''`).
```python
DB_CONFIG = {
    'host': 'localhost',      
    'user': 'root',           
    'password': 'MẬT_KHẨU_MYSQL_CỦA_BẠN',
    'database': 'violence_db' 
}
```

#### 3. Chạy hệ thống & Truy cập trải nghiệm
Mở cửa sổ dòng lệnh tại hệ thống thư mục của mã nguồn (`Violence_Detection_System`) và chạy:
```bash
python app_web/app.py
```
*Giao diện Server sẽ hiển thị chữ `Loading Model...` Khi Load xong, hãy mở Chrome/Edge và gõ tên miền Localhost:* **http://localhost:5000**

---

### Các API Endpoint / Tuyến đường (Routes) của Hệ thống

| Phương thức | Endpoint | Mô tả chức năng |
| :---: | :--- | :--- |
| **GET** | `/` | Trang chủ - Bảng Điều khiển: Màn hình giám sát trực tiếp từ Webcam. |
| **GET/POST**| `/upload` | Trang Tải File: Kéo thả một Video bất kỳ để AI phân tích bằng chứng. |
| **GET** | `/history` | Bảng điều tra Sự cố: Xem lại Lịch sử các clip/khung giờ đã đánh nhau/ngã. |
| **GET** | `/testing` | Trang Thống kê kỹ thuật: Màn hình vẽ biểu đồ FPS và % tự tin của AI thời gian thực. |
| **GET** | `/video_feed` | (Băng thông riêng) API Stream luồng hình ảnh MJPEG từ OpenCV tới Trình duyệt. |
| **POST**| `/api/upload` | API Ngầm. Xử lý lưu File Upload, giải nén và chấm điểm cảnh báo trên thẻ Card UI. |
| **GET** | `/results/<filename>`| API Tĩnh. Cho phép frontend chiếu lại bất kỳ clip `.mp4` sự cố nào trong file hệ thống. |
| **GET** | `/api/logs` | Cấp phát dữ liệu danh sách vi phạm từ Database cho React/JS UI Frontend. |

---

### Ghi chú quan trọng
* ✅ **Môi trường Trực tiếp**: Nếu bạn dùng nhiều Camera mạng hoặc OBS Studio, hệ thống AI đã được code lệnh `cv2.CAP_DSHOW` để thông minh tự chọn Cam vật lý thật. Nếu thấy Không có ảnh trên UI, nhớ rà soát quyền (Permissions) Camera trên Windows Setting nhé.
* ✅ **Bảo mật không gian đĩa cứng (SSD/HDD)**: Nhằm tránh làm phình Database và làm chậm hệ điều hành, AI của Phương Nam đã thiết lập logic **chỉ trích xuất, encode (FFmpeg) và lưu Video Nguy Hiểm/Bạo Lực thực sự**. Các vận động đi bộ hoặc chạy thể thao thông thường sẽ được tự động phân tích và tự động hủy bỏ trong RAM để tối ưu tài nguyên máy.
* ✅ **Cơ chế cửa cổng báo động giả (False-Positive Gate)**: Ở các nơi công cộng đông người, nhiều người đi lại sẽ thỉnh thoảng làm tăng độ nhiễu. Nếu AI nghi ngờ hành động với mức tự tin (Confidence) đo được dưới 85%, nó sẽ dùng logic lọc nhiễu Fall-back tự bỏ qua vụ án đó.

---

### Tóm tắt Luồng hoạt động Tự động của AI (AI Pipeline Architecture)

**1. Khảo sát Thời Gian Thực (Live Surveillance & Inference)**
* Camera thu hình với tốc độ cao đệm vào mảng liên tục (RAM deque). Mỗi 16 Frame hợp nhất lại tạo thành một chuỗi (Sequence) chuyển động không có độ trễ.
* Neural Network chấm điểm xác suất (Softmax Probs) của tất cả 8 phân lớp `danh_nhau`, `nga`, `normal`,...
* Ngay trong tích tắc nếu `danh_nhau` vọt lên top, hệ thống tạo Thread ghi file `.mp4`, nhãn trạng thái đổi thành "Cảnh Báo 🔴".
* Ghi hình kết thúc -> Lưu xuống máy tính -> Chèn truy vấn INSERT vào MySQL.

**2. Điều Tra Pháp Y (Nguội) qua Chức năng Tải lên (Upload Forensics)**
* Giám thị kéo thả 1 file Camera ghi chép ngày hôm qua lên Trình duyệt.
* Backend dùng VideoCapture tua nhanh từng mili-giây, nếu có Bạo Lực, nó cắt video thành bản sao lưu chứng cứ riêng rẽ, nhúng FFmpeg đổi chuẩn nén sang Web-Ready Data.
* Trả kết quả Màn hình hiển thị Thẻ vụ án màu Đỏ kèm trình xem lại hình ảnh lập tức.

<br>
<div align="center">
  <p>🔥 <b>Chúc bạn trải nghiệm Hệ thống Nhận diện Bạo lực thành công!</b> 🔥</p>
</div>