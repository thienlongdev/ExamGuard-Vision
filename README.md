# ExamGuard Vision

**Hệ thống AI hỗ trợ giám sát phòng thi theo thời gian thực và rà soát sự kiện dành cho giám thị**

[![Python 3.13](https://img.shields.io/badge/Python-3.13.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.14+cu126](https://img.shields.io/badge/PyTorch-2.14.1%2Bcu126-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Git LFS](https://img.shields.io/badge/Checkpoints-Git%20LFS-F05032.svg?logo=git-lfs&logoColor=white)](https://git-lfs.github.com/)
[![Status](https://img.shields.io/badge/School%20Demo-Certified-brightgreen.svg)]()

---

> [!IMPORTANT]
> ### NGUYÊN LÝ THIẾT KẾ VÀ GIỚI HẠN PHÁP LÝ CỐT LÕI
> **Hệ thống này TUYỆT ĐỐI KHÔNG tự động kết luận hay gắn nhãn thí sinh "gian lận" từ bất kỳ khung hình đơn lẻ hay chuỗi suy luận AI nào.**
> 
> - **AI chỉ phát hiện các hành vi và đối tượng vật lý quan sát được** trong không gian phòng thi (ví dụ: gục đầu xuống bàn/ngủ, quay đầu sang hai bên kéo dài, điện thoại di động gắn với khu vực ngồi của thí sinh, đứng dậy rời chỗ).
> - Thuật toán theo dõi đa đối tượng (`ByteTrack`) kết hợp phân tích chuỗi thời gian cửa sổ trượt (sliding temporal buffer), hợp nhất đa tín hiệu `V4D` và bộ quy tắc tính điểm rủi ro bằng chứng (Evidence Risk Scorer) sẽ tổng hợp thành các **"sự kiện hành vi quan sát được cần chú ý"** (Observable / Suspicious Events) kèm các mức độ bằng chứng (`LOW`, `MEDIUM`, `HIGH`).
> - **Giám thị phòng thi là người duy nhất có thẩm quyền đưa ra kết luận cuối cùng** (Xác nhận sự kiện - Confirm Event hoặc Bác bỏ - Dismiss) thông qua Bảng điều khiển rà soát bằng chứng trực quan.
> - Điểm số đơn khung hình bị giới hạn cứng không vượt quá 25 điểm (`max_single_frame_score: 25.0`), đảm bảo không có cảnh báo rủi ro cao nào bị kích hoạt chỉ từ một frame ngẫu nhiên.

---

## Mục lục

1. [Bắt đầu nhanh trong 5 phút (Dành cho người mới clone repo)](#1-bắt-đầu-nhanh-trong-5-phút-dành-cho-người-mới-clone-repo)
2. [Xử lý sự cố thường gặp lần đầu chạy](#2-xử-lý-sự-cố-thường-gặp-lần-đầu-chạy)
3. [Tổng quan hệ thống](#3-tổng-quan-hệ-thống)
4. [Kiến trúc luồng xử lý (Pipeline Architecture)](#4-kiến-trúc-luồng-xử-lý-pipeline-architecture)
5. [Danh mục hành vi và tư thế quan sát](#5-danh-mục-hành-vi-và-tư-thế-quan-sát)
6. [Nguồn video hỗ trợ và tính trung thực nguồn gốc](#6-nguồn-video-hỗ-trợ-và-tính-trung-thực-nguồn-gốc)
7. [Bảng điều khiển ExamGuard Vision Control Center](#7-bảng-điều-khiển-examguard-vision-control-center)
8. [Hệ thống API và WebSocket](#8-hệ-thống-api-và-websocket)
9. [Danh mục Checkpoint và Model Registry](#9-danh-mục-checkpoint-và-model-registry)
10. [Quản lý bằng chứng và An toàn bảo mật](#10-quản-lý-bằng-chứng-và-an-toàn-bảo-mật)
11. [Cấu trúc thư mục repository](#11-cấu-trúc-thư-mục-repository)
12. [Kiểm thử và Đánh giá tính toàn vẹn](#12-kiểm-thử-và-đánh-giá-tính-toàn-vẹn)
13. [Kết quả xác thực trên cấu hình ASUS TUF Gaming A17](#13-kết-quả-xác-thực-trên-cấu-hình-asus-tuf-gaming-a17)
14. [Đánh giá tính sẵn sàng: Demo trường học vs Sản xuất](#14-đánh-giá-tính-sẵn-sàng-demo-trường-học-vs-sản-xuất)
15. [Nghiên cứu và Huấn luyện ngoại tuyến](#15-nghiên-cứu-và-huấn-luyện-ngoại-tuyến)

---

## 1. Bắt đầu nhanh trong 5 phút (Dành cho người mới clone repo)

Phần này hướng dẫn người dùng mới, kỹ sư đánh giá hoặc reviewer triển khai chạy sản phẩm ngay từ con số không sau khi clone mã nguồn từ GitHub.

### Bước 1: Cài đặt Git LFS và Clone mã nguồn

Repository này sử dụng **Git LFS (Large File Storage)** để lưu trữ các model checkpoint kích thước lớn (như `v4_headpose_yaw_best.pt` 284 MB). Hãy đảm bảo bạn đã cài đặt Git LFS trước khi clone:

```powershell
# 1. Cài đặt Git LFS vào môi trường git của bạn (chỉ cần chạy một lần)
git lfs install

# 2. Clone mã nguồn dự án
git clone https://github.com/thienlongdev/ExamGuard-Vision.git
cd ExamGuard-Vision

# 3. Kéo đầy đủ các file checkpoint nặng về máy (bắt buộc)
git lfs pull
```

> [!TIP]
> **Kiểm tra file LFS:** Nếu file `models/trained/v4_headpose_yaw_best.pt` có dung lượng ~284 MB thì đã tải thành công. Nếu chỉ có vài trăm bytes (chứa text `version https://git-lfs...`), bạn hãy chạy lại lệnh `git lfs pull`.

### Bước 2: Tạo và kích hoạt môi trường ảo Python

Khuyến nghị sử dụng **Python 3.10 đến Python 3.13** (Dự án được kiểm thử chuẩn mực trên Python 3.13.11 64-bit trên Windows):

```powershell
# Tạo môi trường ảo .venv
python -m venv .venv
```

Kích hoạt môi trường ảo:
- Trên **PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Nếu gặp lỗi `ExecutionPolicy`, bạn có thể chạy bằng cách gọi trực tiếp `.\.venv\Scripts\python.exe` mà không cần kích hoạt).*
- Trên **Command Prompt (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Bước 3: Cài đặt PyTorch và Dependencies

Để tránh việc cài đặt đè phiên bản PyTorch CPU hoặc xung đột CUDA, bạn nên cài đặt PyTorch theo cấu hình phần cứng của mình trước, sau đó mới cài đặt các thư viện bổ trợ:

```powershell
# Nâng cấp pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# LỰA CHỌN A: Máy có NVIDIA GPU (Cấu hình chuẩn CUDA 12.6, tương thích dòng RTX 3050/3060/40xx)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# LỰA CHỌN B: Máy không có GPU rời (Chạy trên CPU)
# .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các gói phụ thuộc runtime của dự án
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

### Bước 4: Kiểm tra nhanh môi trường phần cứng và CUDA

Chạy lệnh 1 dòng sau để xác nhận PyTorch đã nhận diện GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
*Kết quả kỳ vọng trên máy có card NVIDIA: `CUDA Available: True` và tên GPU hiển thị rõ ràng.*

### Bước 5: Chạy công cụ tiền kiểm định tự động (Laptop Preflight)

Dự án cung cấp sẵn kịch bản kiểm tra toàn diện phần cứng, GPU VRAM, 7 file checkpoint cryptographic hash, camera và cổng mạng:

```powershell
.\.venv\Scripts\python.exe scripts/laptop_preflight.py
```
*Kịch bản sẽ thông báo `LAPTOP_PREFLIGHT_COMPLETE` và in báo cáo tình trạng sẵn sàng.*

### Bước 6: Khởi chạy ExamGuard Vision Demo

Chạy launcher một lệnh chuẩn dành cho máy demo:

```powershell
& ".\.venv\Scripts\python.exe" scripts/run_asus_a17_demo.py
```

Tuỳ chọn dòng lệnh bổ sung (nếu cần):
- `--camera-index 1`: Đổi sang camera gắn ngoài (mặc định là `0`).
- `--port 8000`: Đổi cổng dịch vụ dashboard (mặc định là `8000`).
- `--headless`: Tắt cửa sổ hiển thị OpenCV cục bộ (chỉ theo dõi qua trình duyệt web).
- `--burn-in-web-hud`: Bật ghi đè thông số debug HUD vào luồng video web (mặc định tắt để giữ video web sạch chuẩn thương mại).

### Bước 7: Mở Bảng điều khiển Giám thị (Dashboard)

Mở trình duyệt web bất kỳ (Chrome, Edge, Firefox) và truy cập:
- **Trang chủ Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Đường dẫn phụ (Alias)**: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
- **Kênh dữ liệu WebSocket**: `ws://127.0.0.1:8000/ws/events`

### Bước 8: Dừng hệ thống an toàn

Khi muốn kết thúc phiên giám sát:
- Nhấp chuột vào cửa sổ OpenCV HUD và nhấn phím **`q`** hoặc **`ESC`**.
- Hoặc nhấn tổ hợp phím **`Ctrl + C`** trên cửa sổ terminal.
- Tiến trình sẽ tự động giải phóng tài nguyên phần cứng camera, ngắt kết nối WebSocket và tắt máy chủ HTTP sạch sẽ.

---

## 2. Xử lý sự cố thường gặp lần đầu chạy

| Hiện tượng lỗi | Nguyên nhân khả dĩ | Cách khắc phục |
|---|---|---|
| **Lỗi PyTorch khi tải checkpoint** (`_pickle.UnpicklingError` hoặc báo file quá nhỏ) | Checkpoint chưa được Git LFS tải thực sự, hiện chỉ là file con trỏ text vài trăm bytes | Chạy lệnh `git lfs pull` tại thư mục gốc của repository. Kiểm tra lại kích thước `models/trained/v4_headpose_yaw_best.pt` phải đạt ~284 MB. |
| **`torch.cuda.is_available() == False`** | Chưa cài đặt bản build PyTorch CUDA phù hợp hoặc driver NVIDIA chưa tương thích | Kiểm tra `nvidia-smi`. Gỡ `torch` hiện tại và cài đặt lại bằng lệnh: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`. Hệ thống vẫn có thể chạy trên CPU nhưng tốc độ khung hình sẽ thấp hơn. |
| **Không mở được Webcam** (`Could not open physical webcam`) | Camera đang bị phần mềm khác chiếm dụng (Zoom, Teams, Camera app của Windows) hoặc sai chỉ mục | Đóng tất cả ứng dụng đang dùng webcam. Kiểm tra quyền riêng tư camera trong Windows Settings. Chạy lệnh: `.\.venv\Scripts\python.exe scripts/run_local_live_validation.py --check-camera` để dò tìm camera khả dụng. |
| **Lỗi cổng 8000 bị chiếm dụng** (`Address already in use`) | Có một tiến trình khác đang mở cổng 8000 trên máy | Chạy launcher với tham số đổi cổng: `.\.venv\Scripts\python.exe scripts/run_asus_a17_demo.py --port 8080`, sau đó truy cập [http://127.0.0.1:8080/](http://127.0.0.1:8080/). |
| **Trang web mở được nhưng không thấy video camera** | Trình duyệt chưa tải được luồng MJPEG hoặc pipeline khởi động trễ | Kiểm tra thanh trạng thái `Health Strip` ở đầu trang xem camera có báo `STREAMING` không. Kiểm tra endpoint `/api/cameras`. Thử F5 làm mới trình duyệt. |

---

## 3. Tổng quan hệ thống

**ExamGuard Vision** là nền tảng thị giác máy tính và phân tích hành vi chuỗi thời gian nhiều tầng, được thiết kế chuyên biệt để hỗ trợ giám thị trong công tác giám sát phòng thi, phát hiện sớm các bất thường thể hiện bằng chứng rõ ràng và rà soát khách quan.

### Điểm nổi bật về mặt công nghệ:
1. **Phân tách rành mạch giữa Nhận diện và Kết luận**: AI phát hiện cử chỉ, vật thể, góc quay đầu; động cơ suy luận tổng hợp độ nghi vấn theo thời gian; giám thị là người quyết định.
2. **Khắc phục nhược điểm Single-frame**: Giảm cảnh báo do chuyển động thoáng qua thông qua temporal fusion, debounce và hysteresis, kết hợp máy trạng thái thời gian và bộ đệm trượt 30 giây.
3. **Luồng video độc lập (Single-Camera Ownership)**: Camera vật lý chỉ mở một lần duy nhất trong toàn hệ thống bởi pipeline chính. Luồng video hiển thị trên Web tiêu thụ frame đã qua xử lý downstream, không gây xung đột cổng USB/driver webcam.
4. **Giao diện thương mại sạch (Clean Product Stream)**: Luồng video trên web hoàn toàn sạch sẽ, không bị vẽ đè chữ debug khó nhìn; khung bounding box và thông số được vẽ mượt mà ở phía client (trình duyệt).
5. **Vận hành hoàn toàn Offline**: Hệ thống có thể vận hành offline và không yêu cầu CDN/cloud trong luồng runtime chính, không gửi dữ liệu ra ngoài môi trường mạng nội bộ.

---

## 4. Kiến trúc luồng xử lý (Pipeline Architecture)

Hệ thống được tổ chức theo kiến trúc đường ống xử lý theo tầng (Multi-Stage Pipeline) hoàn chỉnh:

```
[ Nguồn Video (Webcam / File / RTSP CCTV) ]
                    │
                    ▼
[ Ingestion & Đệm hàng đợi giải mã độc lập (Bounded Ingestion Queue) ]
                    │
                    ▼
[ General Object Detector: yolo26m.pt (640x640: person, cell phone) ]
                    │
                    ▼
[ ByteTrack Multi-Object Tracker (Gán và duy trì ID thí sinh liên tục) ]
                    │
                    ▼
[ Crop Scheduler & Scale Gating (Cắt vùng đối tượng & lọc độ phân giải) ]
                    │
    ┌───────────────┼───────────────┬─────────────────────────┐
    ▼               ▼               ▼                         ▼
[ Posture Model ] [ Headpose Yaw ] [ Macro Behavior ] [ Phone Association ]
 (MobileNetV3)     (HopeNet-Yaw)     (Stage 1.5)       (Ràng buộc không gian)
  224x224, 4 lớp   224x224, yaw deg   768x768           với bounding box người
    └───────────────┬───────────────┴─────────────────────────┘
                    │
                    ▼
[ Đóng gói UnifiedTrackUpdate (Kèm định danh nguồn gốc SourceOrigin) ]
                    │
                    ▼
[ Động cơ Hợp nhất Đa tín hiệu V4D (Multi-Cue Temporal Fusion Engine) ]
   - Sliding Temporal Buffer (Cửa sổ trượt lưu lịch sử 30s)
   - Bộ lọc nhất quán & trừ hao tín hiệu trùng lặp (Anti Double-Counting)
                    │
                    ▼
[ Observable Event Engine (Máy trạng thái sự kiện: CANDIDATE -> ACTIVE -> COOLDOWN) ]
                    │
                    ▼
[ Bộ tính điểm rủi ro bằng chứng (Evidence Risk Scorer: LOW, MEDIUM, HIGH) ]
                    │
                    ▼
[ Quản lý bằng chứng (Evidence Manager: Chụp snapshot JPEG tại thời điểm mở sự kiện) ]
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
[ FastAPI REST API ]   [ WebSocket Broadcaster ]
 (Lưu trữ & Truy vấn)   (Phát tán sự kiện tức thời)
         │                     │
         └──────────┬──────────┘
                    ▼
[ Bảng điều khiển ExamGuard Vision Control Center ]
 (Giao diện web dành cho giám thị: MONITOR, REVIEW, SYSTEM)
                    │
                    ▼
[ Giám thị rà soát & Đưa ra quyết định cuối cùng (Confirm / Dismiss) ]
```

---

## 5. Danh mục hành vi và tư thế quan sát

Hệ thống tuân thủ nghiêm ngặt hệ thống danh pháp hành vi quan sát được (Observable Fact-based Ontology).

### 5.1 Bốn lớp tư thế chuẩn (V4 Posture Classes)

Model chuyên biệt MobileNetV3-Small (`v4_posture_best.pt`) phân loại vùng thân trên thí sinh thành đúng 4 trạng thái hình học:

| Class ID | Mã định danh tư thế | Giải thích ngữ nghĩa |
|---|---|---|
| `0` | `NORMAL_UPRIGHT` | Thí sinh ngồi thẳng, mắt hướng về phía trước hoặc làm bài bình thường. |
| `1` | `NORMAL_READ_WRITE` | Thí sinh hơi cúi đầu xuống mặt bàn để đọc đề hoặc viết bài (hành vi hợp lệ). |
| `2` | `HEAD_REST_SLEEP` | Thí sinh gục hẳn đầu xuống bàn hoặc có dấu hiệu ngủ trong giờ thi. |
| `3` | `TURN_HEAD_CLEAR` | Thí sinh quay đầu rõ rệt sang trái hoặc sang phải khỏi vị trí bài thi. |

> [!CAUTION]
> **Điện thoại di động không phải là một lớp tư thế!** Điện thoại di động được phát hiện độc lập qua detector vật thể và sau đó được liên kết không gian (Spatial Binding) với vị trí ngồi của thí sinh.

### 5.2 Các họ sự kiện quan sát được (Observable Event Families)

Dựa trên dữ liệu chuỗi thời gian, động cơ `EventEngine` sẽ kích hoạt các sự kiện sau khi hành vi kéo dài vượt ngưỡng định trước:

1. **`SUSTAINED_HEAD_REST` (Gục đầu kéo dài)**:
   - Kích hoạt khi thí sinh duy trì tư thế `HEAD_REST_SLEEP` liên tục trên **2,0 giây** (với xác suất $\ge 0.55$).
   - Tự động bị triệt tiêu nếu điểm tư thế đọc/viết `NORMAL_READ_WRITE` chiếm ưu thế.
2. **`SUSTAINED_LATERAL_HEAD_ORIENTATION` (Quay đầu kéo dài)**:
   - Kích hoạt khi thí sinh duy trì tư thế `TURN_HEAD_CLEAR` hoặc góc quay đầu lệch $\ge 25^\circ$ liên tục trên **1,5 giây**.
   - Hợp nhất giữa phân loại tư thế thân người và góc quay đầu liên tục (HopeNet-Yaw).
3. **`PHONE_ASSOCIATED` (Điện thoại liên kết với thí sinh)**:
   - Kích hoạt khi phát hiện điện thoại di động nằm trong hoặc rất gần vùng bounding box của thí sinh liên tục trên **1,0 giây**.
4. **`DISCUSSION_CANDIDATE` (Dấu hiệu trao đổi bài)**:
   - Kích hoạt khi có sự phối hợp giữa quay đầu hướng về phía thí sinh khác hoặc dấu hiệu tương tác kéo dài trên **2,0 giây**.
5. **`STANDING` (Đứng dậy khỏi chỗ)**:
   - Kích hoạt khi thí sinh rời vị trí ngồi hoặc đứng lên trong phòng thi kéo dài trên **1,5 giây**.
6. **`MULTI_CUE_ATTENTION_SHIFT` (Chuyển hướng chú ý đa tín hiệu)**:
   - Sự kết hợp đồng thời giữa nhiều tín hiệu bất thường trong cùng một khoảng thời gian.

### 5.3 Thang đo rủi ro bằng chứng (Evidence Risk Scoring)

Mức rủi ro trong hệ thống đại diện cho **mức độ đầy đủ và sự bền bỉ của bằng chứng quan sát được**, hoàn toàn không mang ý nghĩa phán xét tội lỗi:

- **`LOW` (Rủi ro thấp, 0 – 39 điểm)**: Tín hiệu xuất hiện thoáng qua hoặc ở ngưỡng biên của sự kiện.
- **`MEDIUM` (Rủi ro trung bình, 40 – 74 điểm)**: Hành vi diễn ra rõ ràng trong một khoảng thời gian, cần giám thị lưu tâm quan sát.
- **`HIGH` (Rủi ro cao, 75 – 100 điểm)**: Có sự kết hợp của nhiều bằng chứng mạnh mẽ (ví dụ: phát hiện điện thoại di động hoặc quay đầu liên tục kèm góc lệch lớn).

---

## 6. Nguồn video hỗ trợ và tính trung thực nguồn gốc

Hệ thống thiết kế trừu tượng hoá nguồn video qua lớp `VideoSource`, cho phép vận hành linh hoạt trên nhiều môi trường khác nhau:

| Nguồn video | Lớp xử lý | Mục đích sử dụng | Cơ chế đặc trưng |
|---|---|---|---|
| **Webcam tích hợp / USB** | `WebcamSource` | Trình diễn trực tiếp trên laptop demo / phòng thí nghiệm | Sử dụng DirectShow (`CAP_DSHOW`) trên Windows, hỗ trợ 1280x720 và 640x480 @ 30 FPS. |
| **Tệp tin video ghi sẵn** | `VideoFileSource` | Kiểm thử hồi quy ngoại tuyến, benchmark thuật toán lặp lại | Đọc từng khung hình tuần tự theo timestamp chuẩn xác, cho phép tái hiện các ca kiểm thử phức tạp. |
| **Camera IP / CCTV trường học** | `RTSPSource` | Triển khai thực tế trên camera cố định góc cao | Thu nhận luồng RTSP H.264 qua mạng LAN, có cơ chế tự động kết nối lại khi mất mạng (exponential backoff). |

### Tính trung thực nguồn gốc (`SourceOrigin`)
Mọi bản ghi dữ liệu, track và sự kiện đều được gán nhãn nguồn gốc cụ thể (`PHYSICAL_LIVE_CAMERA`, `SOFTWARE_VALIDATION_FIXTURE`, `REPLAY_STREAM`, `VIDEO_FILE`, `RTSP_STREAM`, `SYNTHETIC_TEST`, hoặc `UNKNOWN`). Hệ thống **tuyệt đối không bao giờ nâng cấp giả mạo** dữ liệu fixture phần mềm thành camera vật lý.

---

## 7. Bảng điều khiển ExamGuard Vision Control Center

Giao diện giám sát hiện đại, tối ưu cho giám thị theo dõi trực quan theo thời gian thực trên màn hình máy tính:

### 7.1 Màn hình theo dõi chính (MONITOR View)
- **Hero Camera Viewport**: Khung phát video tỉ lệ 16:9 với độ trễ thấp từ luồng downstream MJPEG. Overlay bounding box bo góc hiện đại hiển thị ID học sinh, tư thế hiện tại và nhãn chú ý.
- **Thanh chỉ số KPI (KPI Bar)**: Bốn thẻ đếm tức thì: Tổng sự kiện (Total Events), Chờ rà soát (Awaiting Review), Giám thị đã xác nhận (Human Confirmed), Giám thị đã bác bỏ (Dismissed).
- **Thanh đo sức khỏe Telemetry (Health Strip)**: Cung cấp thông tin camera (CAM 01 LIVE), FPS thu nhận, FPS xử lý AI, số thí sinh đang bám vết, độ sâu hàng đợi, tỷ lệ rớt khung hình (0.0%), và dung lượng VRAM GPU đã sử dụng.
- **Hàng đợi rà soát (Review Queue)**: Danh sách các thẻ sự kiện mới nhất theo thứ tự thời gian, hiển thị ảnh bằng chứng snapshot thu nhỏ, thanh đo mức độ rủi ro (0–100), mốc thời gian tương đối và các nút thao tác nhanh.
- **Dòng thời gian hoạt động (Activity Timeline)**: Biểu đồ trực quan hóa diễn biến hành vi theo thời gian, chứng minh hành vi được theo dõi theo chuỗi chứ không phán xét nhất thời.

### 7.2 Màn hình rà soát lịch sử (REVIEW View)
- Bảng nhật ký đầy đủ tất cả các sự kiện đã ghi nhận trong toàn bộ ca thi.
- Bộ lọc thông minh theo trạng thái rà soát (`ALL`, `Chờ rà soát`, `Đã xác nhận`, `Đã bỏ qua`), theo mức độ rủi ro hoặc theo loại sự kiện.
- Xem chi tiết ảnh bằng chứng và mở ngăn kéo xử lý bất kỳ lúc nào.

### 7.3 Màn hình quản trị hệ thống (SYSTEM View)
- Báo cáo chi tiết phần cứng, tình trạng kết nối camera và độ phân giải thực tế.
- **Bảng phân rã độ trễ AI (AI Latency Breakdown)**: Hiển thị thời gian xử lý chi tiết của từng module: Detector, Tracker, Posture, Headpose, Macro, và Fusion.
- **Sổ đăng ký Model (Model Registry)**: Liệt kê đầy đủ đường dẫn checkpoint, mã băm SHA-256 đã chứng thực, thiết bị thực thi (`cuda:0`), chế độ chính xác (`fp32`), và kích thước ảnh đầu vào.
- Thống kê bộ nhớ GPU: VRAM cấp phát, VRAM bảo lưu và dung lượng trống còn lại.

### 7.4 Ngăn kéo chi tiết sự kiện (Event Details Slide-in Drawer)
Khi giám thị nhấp chuột vào bất kỳ sự kiện nào trong hàng đợi hoặc bảng lịch sử:
- Mở ra ảnh chụp bằng chứng JPEG độ nét cao tại thời điểm xảy ra sự kiện.
- Bảng chẩn đoán đa chiều: Xác suất tư thế (%), góc quay đầu (độ), khoảng cách điện thoại, điểm số rủi ro chi tiết.
- Ô nhập ghi chú của giám thị (Reviewer Notes).
- **Hai nút thao tác quyết định**: **Xác nhận sự kiện (Confirm Event)** hoặc **Bác bỏ sự kiện (Dismiss)**. Thao tác được gửi lên máy chủ và đồng bộ tức thời đến toàn bộ các máy giám thị khác qua WebSocket.

---

## 8. Hệ thống API và WebSocket

FastAPI backend cung cấp hệ thống endpoint REST và WebSocket chuẩn mực, bảo mật và hiệu năng cao:

### 8.1 Các endpoint REST

| Phương thức | Đường dẫn Route | Mục đích chức năng |
|---|---|---|
| `GET` | `/` hoặc `/dashboard` | Phục vụ trang giao diện Bảng điều khiển Giám thị HTML/JS (100% offline). |
| `GET` | `/health` | Kiểm tra trạng thái hoạt động của backend service (`HealthResponse`). |
| `GET` | `/api/cameras` | Lấy danh sách camera, trạng thái kết nối, độ phân giải và FPS đo được. |
| `GET` | `/api/cameras/stream` | Cung cấp luồng video MJPEG downstream từ pipeline chính cho trình duyệt. |
| `GET` | `/api/cameras/frame` | Lấy 1 khung hình JPEG mới nhất từ pipeline. |
| `GET` | `/api/cameras/tracks` | Lấy danh sách bounding box và trạng thái bám vết của các thí sinh hiện tại. |
| `GET` | `/api/events` | Truy vấn danh sách sự kiện (hỗ trợ lọc theo `risk_level`, `status`, `limit`). |
| `GET` | `/api/events/{event_id}` | Xem thông tin chi tiết của một sự kiện cụ thể. |
| `PATCH` | `/api/events/{event_id}` | Cập nhật quyết định của giám thị (`confirmed` hoặc `dismissed`) kèm ghi chú. |
| `GET` | `/api/system/status` | Lấy toàn bộ thông số viễn trắc hệ thống, FPS, VRAM, độ sâu hàng đợi, KPI. |
| `GET` | `/api/system/models` | Trả về thông tin đăng ký của các model AI kèm mã băm SHA-256. |
| `GET` | `/api/evidence/{file_path:path}` | Tải file ảnh chụp bằng chứng an toàn (có chống Path Traversal). |

### 8.2 Giao tiếp thời gian thực (WebSocket)

- **Địa chỉ kết nối**: `ws://127.0.0.1:8000/ws/events`
- **Các gói tin phát tán (Broadcast Payloads)**:
  - `EVENT_OPEN`: Phát ra ngay khi một hành vi bất thường đủ điều kiện chuyển thành sự kiện cần chú ý.
  - `EVENT_UPDATE`: Cập nhật diễn biến điểm số và bằng chứng của sự kiện khi nó đang tiếp diễn.
  - `EVENT_CLOSE`: Đóng sự kiện khi thí sinh trở về trạng thái bình thường (chuyển sang trạng thái chờ giám thị duyệt).
  - `EVENT_STATUS_UPDATED`: Đồng bộ tức thời khi giám thị bấm Xác nhận hoặc Bác bỏ sự kiện.
- **Khả năng phục hồi kết nối**: Client JavaScript tích hợp sẵn thuật toán Reconnect tự động với thời gian chờ tăng theo cấp số nhân (bounded exponential backoff), phân biệt rõ các trạng thái `connecting`, `connected`, `reconnecting`, `disconnected`.

---

## 9. Danh mục Checkpoint và Model Registry

Tất cả các model sử dụng trong luồng suy luận đều được cố định phiên bản và xác thực bằng mã băm mật mã học SHA-256:

| Tên vai trò model | File Checkpoint | Kích thước | SHA-256 Hash được chứng thực | Kiến trúc & Đầu vào |
|---|---|---|---|---|
| **General Object Detector** | `models/trained/yolo26m.pt` | ~44.3 MB | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` | YOLOv8/v11 640x640 (person, cell phone) |
| **Macro Behavior Detector** | `models/trained/stage1_5_best.pt` | ~44.0 MB | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | Custom YOLO Stage 1.5 @ 768x768 |
| **Posture Classifier** | `models/trained/v4_posture_best.pt` | ~18.5 MB | `529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180` | MobileNetV3-Small @ 224x224 (4 lớp tư thế) |
| **Headpose Yaw Estimator** | `models/trained/v4_headpose_yaw_best.pt` *(Git LFS)* | ~284.2 MB | `5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55` | HopeNet-Yaw @ 224x224 (Góc quay $[-99^\circ, +99^\circ]$) |
| **Headpose Circular Fallback** | `models/fallback/headpose_resnet18/best_model.pt` *(Git LFS)* | ~44.7 MB | `bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9` | ResNet18 Circular Yaw (Phạm vi $[-180^\circ, +180^\circ]$) |
| **High-Res Person Crop Fallback** | `models/fallback/posture_320/best_model.pt` | ~18.5 MB | `070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf` | MobileNetV3-Small @ 320x320 |
| **Stage 1 Baseline Checkpoint** | `models/trained/stage1_best.pt` | ~44.0 MB | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | Baseline foundation checkpoint |

---

## 10. Quản lý bằng chứng và An toàn bảo mật

### 10.1 Vòng đời chụp bằng chứng (Evidence Lifecycle)
Để tối ưu hóa tài nguyên phần cứng và tránh áp lực đọc ghi đĩa khi vận hành thời gian thực, cấu hình chuẩn của hệ thống sử dụng chế độ **`snapshot-only`**:
- Ngay khi một sự kiện chuyển trạng thái sang `ACTIVE`, hệ thống sẽ trích xuất khung hình gốc tại thời điểm đó và lưu bất đồng bộ dưới dạng ảnh JPEG vào thư mục phiên (ví dụ: `evidence/asus_a17_demo/snapshots/{event_id}_open.jpg`).
- Ảnh bằng chứng được liên kết chặt chẽ với metadata sự kiện, bao gồm các chỉ số suy luận tại chính thời điểm chụp.

### 10.2 Bảo vệ an toàn đường dẫn và chống Path Traversal
Endpoint `/api/evidence/{file_path:path}` được gia cố an ninh nghiêm ngặt:
1. **Chỉ phục vụ từ các thư mục gốc hợp lệ (Allowed Roots)**: Chỉ cho phép truy xuất trong `storage/evidence`, `evidence`, và `runs/local_live`.
2. **Giải mã URL 3 lần (Multi-pass unquote)**: Vô hiệu hóa kỹ thuật vượt rào bằng mã hóa kép (như `%252e%252e`).
3. **Chặn đứng mọi ký tự nguy hiểm**: Tự động từ chối yêu cầu chứa `..`, `%2e`, ký tự ổ đĩa (`C:`, `D:`), hoặc đường dẫn mạng UNC (`\\`, `//`).
4. **Chống nhầm lẫn phiên bằng mã lỗi HTTP 409 Conflict**: Nếu truy vấn theo tên file cũ trùng lặp ở nhiều thư mục phiên khác nhau, máy chủ sẽ từ chối và yêu cầu đường dẫn phiên tường minh, tránh việc trả về nhầm ảnh của ca thi khác.
5. **Không lộ đường dẫn máy chủ**: Đường dẫn tuyệt đối nội bộ trên máy trạm không bao giờ bị trả về phía client.

---

## 11. Cấu trúc thư mục repository

Cấu trúc thư mục tinh gọn và chuẩn mực kỹ thuật của dự án:

```
ExamGuard-Vision/
├── .gitattributes             # Cấu hình Git LFS cho các checkpoint lớn (> 100 MB)
├── .gitignore                # Loại trừ virtualenv, dataset thô, video và bằng chứng cục bộ
├── pytest.ini                # Cấu hình bộ kiểm thử tự động pytest
├── README.md                 # Tài liệu kỹ thuật và hướng dẫn triển khai hệ thống
├── requirements.txt          # Danh sách toàn bộ thư viện chi tiết
├── requirements-runtime.txt  # Danh sách phụ thuộc phục vụ chạy thực thi (loại trừ cố định torch)
│
├── configs/                  # Thư mục cấu hình tham số hệ thống
│   ├── camera.yaml           # Cấu hình nguồn camera và FPS
│   ├── classes.yaml          # Danh mục taxonomy nhãn và lớp
│   ├── inference.yaml        # Ngưỡng tin cậy suy luận và thiết bị
│   ├── tracking.yaml         # Tham số ByteTrack
│   ├── v4d_fusion.yaml       # Tham số hợp nhất chuỗi thời gian V4D và ngưỡng rủi ro
│   ├── deployment/           # Metadata chứng thực và kiểm định môi trường triển khai
│   └── runtime/              # Cấu hình hồ sơ thực thi đã chứng thực
│       └── asus_a17_demo.yaml # Hồ sơ vận hành chuẩn cho ASUS TUF Gaming A17
│
├── docs/                     # Tài liệu kỹ thuật chi tiết
│   ├── architecture.md       # Thiết kế kiến trúc luồng xử lý và hợp nhất đa tín hiệu
│   ├── deployment.md         # Hướng dẫn chi tiết triển khai và vận hành hệ thống
│   ├── validation.md         # Báo cáo kiểm định phần cứng và chứng thực hệ thống
│   ├── research.md           # Nghiên cứu mô hình, bộ dữ liệu và đánh giá hợp nhất V4D
│   └── testing.md            # Kiến trúc kiểm thử, phân loại test suites và hướng dẫn QA
│
├── models/                   # Thư mục lưu trữ model
│   ├── trained/              # Các model đã huấn luyện và đóng băng
│   │   ├── yolo26m.pt              # Detector vật thể tổng quát (COCO person, phone)
│   │   ├── stage1_best.pt          # Baseline foundation checkpoint
│   │   ├── stage1_5_best.pt        # Model hành vi macro Stage 1.5
│   │   ├── v4_posture_best.pt      # Model phân loại tư thế 4 lớp (MobileNetV3)
│   │   └── v4_headpose_yaw_best.pt # Model ước lượng góc quay đầu (HopeNet - Git LFS)
│   ├── fallback/             # Checkpoint dự phòng tối ưu cho thiết bị giới hạn tài nguyên
│   │   ├── posture_320/            # Model MobileNetV3 phân giải 320x320
│   │   └── headpose_resnet18/      # Model ResNet18 HopeNet (Git LFS)
│   └── manifests/            # Manifest xác thực danh tính và kiến trúc model
│       └── v4_checkpoint_identity.json
│
├── scripts/                  # Kịch bản vận hành trực tiếp dành cho người dùng
│   ├── run_asus_a17_demo.py  # Kịch bản khởi chạy một lệnh chuẩn cho demo trường học
│   ├── laptop_preflight.py   # Công cụ tiền kiểm tra phần cứng, GPU, model và cổng mạng
│   └── run_local_live_validation.py # Bộ công cụ kiểm định sâu các thành phần
│
├── src/                      # Mã nguồn logic cốt lõi
│   ├── api/                  # Tầng Backend FastAPI & Giao tiếp
│   │   ├── main.py           # Khởi tạo REST API và WebSocket router
│   │   ├── schemas.py        # Định nghĩa cấu trúc dữ liệu Pydantic
│   │   ├── websocket.py      # Quản lý kết nối WebSocket và phát tán sự kiện
│   │   ├── static_ui.py      # Trình nạp giao diện HTML nhúng
│   │   ├── templates/        # Template HTML bảng điều khiển giám thị
│   │   └── static/           # Tài nguyên tĩnh JS/CSS phục vụ dashboard 100% offline
│   ├── behavior/             # Quản lý sự kiện và bộ tính điểm
│   │   └── event_manager.py  # Quản lý vòng đời sự kiện, debounce và cooldown
│   ├── detection/            # Adapter bộ nhận diện vật thể YOLO
│   │   ├── object_detector.py # General Object Detector adapter
│   │   └── behavior_detector.py # Macro Behavior Detector adapter
│   ├── evidence/             # Chụp và lưu trữ ảnh chụp bằng chứng
│   │   └── snapshot.py       # Module ghi ảnh JPEG bất đồng bộ
│   ├── fusion/               # Động cơ hợp nhất đa tín hiệu V4D
│   │   ├── types.py          # Enums (SourceOrigin, PostureClass, EventFamily, RiskLevel)
│   │   ├── cue_state.py      # Trạng thái quan sát từng frame của mỗi thí sinh
│   │   ├── fusion_engine.py  # Động cơ hợp nhất đa tín hiệu theo thời gian
│   │   └── event_engine.py   # Máy trạng thái phát hiện sự kiện hành vi
│   ├── models/               # Định nghĩa kiến trúc PyTorch (Posture, HeadPose)
│   │   ├── posture/          # MobileNetV3 posture classifier
│   │   └── headpose/         # HopeNet yaw estimator
│   ├── orchestration/        # Tầng điều phối quy trình xử lý
│   │   ├── stage2_pipeline.py# Đường ống xử lý chính (Camera -> Model -> Fusion)
│   │   └── model_registry.py # Quản lý và kiểm tra mã băm SHA-256 của các model
│   ├── pilot/                # Module tiền kiểm tra và cấu hình camera
│   ├── tracking/             # Module bám vết đa đối tượng ByteTrack
│   └── video/                # Lớp trừu tượng hoá nguồn video
│       ├── base.py           # VideoSource ABC
│       ├── webcam.py         # WebcamSource (DirectShow / MSMF)
│       ├── video_file.py     # VideoFileSource
│       └── rtsp.py           # RTSPSource
│
├── tests/                    # Bộ kiểm thử tự động (Unit & Integration tests)
│   └── fixtures/             # Dữ liệu mẫu kiểm thử và video giả lập
│
└── tools/                    # Công cụ nội bộ phục vụ nghiên cứu và phát triển
    ├── benchmark/            # Công cụ đo lường hiệu năng suy luận và thông lượng
    ├── dataset/              # Công cụ thu thập, tiền xử lý và kiểm định tập dữ liệu
    ├── dev/                  # Tiện ích hỗ trợ phát triển và quét mã bí mật
    ├── research/             # Kịch bản huấn luyện, phân tích góc quay và tái tạo báo cáo
    └── validation/           # Bộ kịch bản kiểm tra toàn vẹn checkpoint và hiệu chuẩn camera
```

### Tài liệu kỹ thuật chuyên sâu (Technical Documentation)
- **[Kiến trúc Hệ thống & Hợp nhất Đa tín hiệu (docs/architecture.md)](docs/architecture.md)**: Sơ đồ chi tiết quy trình xử lý luồng, máy trạng thái sự kiện Observable Event Engine và cơ chế chống đếm trùng tín hiệu.
- **[Hướng dẫn Triển khai & Vận hành (docs/deployment.md)](docs/deployment.md)**: Hướng dẫn cài đặt từ đầu, tham số dòng lệnh launcher và danh mục xử lý sự cố thường gặp.
- **[Báo cáo Kiểm định Phần cứng & Chứng thực (docs/validation.md)](docs/validation.md)**: Toàn văn kết quả kiểm định soak test 10 phút, đo lường an toàn bộ nhớ VRAM, độ trễ từng module và ma trận sẵn sàng phần cứng.
- **[Nghiên cứu Mô hình & Tập Dữ liệu (docs/research.md)](docs/research.md)**: Căn cứ khoa học, phân tích tập dữ liệu SCB-Dataset5, AFLW, chuẩn hóa góc quay và đánh giá abalation V4D.
- **[Kiểm thử & Đảm bảo Chất lượng (docs/testing.md)](docs/testing.md)**: Phân loại 219 tests, hướng dẫn chạy kiểm thử tự động và giải thích chi tiết các bài test phụ thuộc dữ liệu offline.

---

## 12. Kiểm thử và Đánh giá tính toàn vẹn

### 12.1 Kết quả kiểm định Pytest thực tế

Báo cáo kiểm định độc lập được ghi nhận trung thực và minh bạch:

- **Bộ kiểm thử phục vụ Runtime / Demo**: **100% ĐẠT** (Toàn bộ 39 tests cốt lõi tập trung và 196 tests kiểm tra logic phần mềm đều vượt qua).
- **Tổng số ca kiểm thử thu thập (`pytest -ra`)**: **219 tests**
  - **196 tests ĐẠT (Passed)**: Bao gồm toàn bộ quy trình nhận diện, bám vết ByteTrack, phân loại tư thế MobileNetV3, ước lượng góc quay đầu HopeNet, suy luận hợp nhất V4D, máy trạng thái sự kiện, tính điểm rủi ro, API FastAPI, WebSocket và Dashboard UI.
  - **21 tests KHÔNG ĐẠT (Failed)**: Xuất phát hoàn toàn từ việc **các tập dữ liệu nghiên cứu huấn luyện ngoại tuyến** (`datasets/processed_v3`, `datasets/stage1_5`, `datasets/v4_crop`, `datasets/v4_head_pose`) nằm trong `.gitignore` và không được phân phối trên máy triển khai thực tế. Không có bất kỳ lỗi nào liên quan đến logic thực thi runtime của sản phẩm.
  - **2 tests ĐƯỢC BỎ QUA (Skipped)**: Bỏ qua do thiếu thư mục dữ liệu ảnh khuôn mặt gốc AFLW.

### 12.2 Các lệnh kiểm thử khuyến nghị

Chạy bộ kiểm thử tập trung cho runtime và demo:
```powershell
.\.venv\Scripts\pytest tests/test_final_certification_audit.py tests/test_dashboard_ui.py tests/test_api.py tests/test_stage2_api_contract.py tests/test_v4d_event_engine.py tests/test_v4d_fusion_engine.py tests/test_laptop_preflight.py -v
```

Chạy toàn bộ pytest của repository:
```powershell
.\.venv\Scripts\pytest -ra
```

---

## 13. Kết quả xác thực trên cấu hình ASUS TUF Gaming A17

Hệ thống đã trải qua quá trình đo đạc thực nghiệm vật lý khắt khe trên cấu hình laptop tham chiếu đại diện cho môi trường demo di động:

### 13.1 Thông số phần cứng máy thử nghiệm
- **Thiết bị**: Laptop ASUS TUF Gaming A17 (Mã model: `FA707RC_FA707RC`)
- **Vi xử lý (CPU)**: AMD Ryzen 7 6800H with Radeon Graphics (8 nhân vật lý, 16 luồng xử lý)
- **Bộ nhớ RAM**: 16 GB DDR5 (Tốc độ đo được: 4800 MT/s)
- **Card đồ họa (GPU)**: NVIDIA GeForce RTX 3050 Laptop GPU (4.096 MB VRAM)
- **Phiên bản Driver NVIDIA**: 566.07 (Hỗ trợ CUDA 12.6)
- **Camera thử nghiệm**: Integrated UVC HD WebCam (DirectShow `CAP_DSHOW`, 1280x720 @ 30 FPS)
- **Hệ điều hành**: Microsoft Windows 11 Home 64-bit

### 13.2 Kết quả đo đạc thực nghiệm trong phiên chạy trực tiếp (65 giây)
- **Thời gian chạy kiểm thử liên tục**: 65,0 giây
- **Số khung hình camera thu nhận**: 646 khung hình
- **Số khung hình xử lý qua AI**: 646 khung hình
- **Số khung hình bị rớt**: **0 khung hình (Tỷ lệ rớt 0.0%)** nhờ cơ chế điều tiết hàng đợi giải mã.
- **Tốc độ suy luận AI thực tế**: ~10,0 FPS (Phù hợp định thời cadence tiết kiệm năng lượng).
- **Độ trễ trung vị xử lý (p50 Latency)**: Ghi nhận p50 khoảng 93–96.8 ms trong phiên kiểm thử trên ASUS A17.
- **Bộ nhớ VRAM GPU cấp phát**: **289,3 MB** trên tổng dung lượng 4.096 MB (dư thừa hơn 3.800 MB VRAM an toàn, không ghi nhận CUDA OOM trong phiên kiểm thử).
- **Bộ nhớ RAM hệ thống chiếm dụng**: 1.745,8 MB RSS (Không ghi nhận xu hướng tăng bộ nhớ bất thường trong phiên smoke test).
- **Tính ổn định của WebSocket**: Kết nối liên tục, đã kiểm thử ngắt kết nối và tự động kết nối lại thành công mà không phát sinh sự kiện rác hay trùng lặp.

---

## 14. Đánh giá tính sẵn sàng: Demo trường học vs Sản xuất

Để đảm bảo tính trung thực về mặt kỹ thuật, dự án phân định rõ ràng ranh giới giữa bản thử nghiệm và hệ thống thương mại:

| Tiêu chí đánh giá | Trạng thái | Giải thích chi tiết |
|---|:---:|---|
| **Sẵn sàng cho Demo trường học (`READY_FOR_SCHOOL_DEMO`)** | **YES** | Hệ thống đã được đóng gói hoàn chỉnh, hoạt động ổn định trên một máy tính cá nhân, có kịch bản khởi chạy 1 lệnh, giao diện đẹp mắt, bằng chứng trực quan và đầy đủ tài liệu phục vụ báo cáo. |
| **Sẵn sàng cho Sản xuất thực tế (`READY_FOR_PRODUCTION`)** | **NO** | Chưa sẵn sàng để triển khai quy mô toàn trường hoặc cho các kỳ thi thật mang tính pháp lý cao. |

### Các hạng mục cần hoàn thiện trước khi đưa vào sản xuất (Production Roadmap):
1. **Quản trị đa camera và đồng bộ phòng thi**: Hỗ trợ quản lý đồng thời hàng chục camera RTSP trên nhiều phòng thi với máy chủ phân tán.
2. **Xác thực và phân quyền (RBAC)**: Tích hợp hệ thống đăng nhập tài khoản giám thị, trưởng điểm thi, bảo vệ bằng JWT token và quyền hạn tương ứng.
3. **Cơ sở dữ liệu kiểm toán lâu dài**: Thay thế kho lưu trữ sự kiện trong bộ nhớ bằng hệ cơ sở dữ liệu chuyên dụng (PostgreSQL/TimescaleDB) có lưu vết kiểm toán (audit logs) bất biến.
4. **Bảo vệ quyền riêng tư (Privacy & GDPR)**: Cơ chế tự động che mờ mặt (face blur) thí sinh không liên quan trong các bức ảnh chụp bằng chứng.
5. **Cụm giám sát tính sẵn sàng cao (High Availability)**: Tự động cân bằng tải, dự phòng nóng khi máy chủ gặp sự cố phần cứng.

---

## 15. Nghiên cứu và Huấn luyện ngoại tuyến

Dành cho các nhà nghiên cứu và kỹ sư muốn tìm hiểu quy trình xây dựng dữ liệu và huấn luyện các mô hình trong hệ thống:

```
Tập dữ liệu thô (datasets/raw/)
    │
    ▼
Kiểm định & Lọc dữ liệu (training/inspect_dataset.py, training/validate_dataset.py)
    │
    ▼
Trực quan hóa mẫu nhãn (training/visual_sample_inspector.py)
    │
    ▼
Ánh xạ nhãn chuẩn hóa (configs/dataset_mapping.yaml, training/class_mapping.py)
    │
    ▼
Phân chia tập train/val/test theo nhóm (training/split_dataset.py, training/build_manifest.py)
    │
    ▼
Huấn luyện nền tảng Stage 1 (training/train_stage1.py, configs/train_stage1.yaml)
    │
    ▼
Huấn luyện thích nghi miền Stage 2 (training/train_stage2.py, configs/train_stage2.yaml)
    │
    ▼
Đánh giá chuyên sâu (training/evaluate.py)
```

### Ràng buộc bất biến về tăng cường dữ liệu định hướng:
Trong tất cả các cấu hình huấn luyện hành vi tư thế, bắt buộc phải thiết lập:
```yaml
fliplr: 0.0
flipud: 0.0
```
*Nguyên do: Hành vi thi cử có tính định hướng không gian nghiêm ngặt (ví dụ: quay đầu nhìn bài thí sinh bên trái khác với bên phải). Việc lật ngang ảnh ngẫu nhiên (horizontal flip) sẽ làm đảo lộn ngữ nghĩa góc quay và phá hủy khả năng học góc đầu của mô hình.*

---

## Giấy phép và Tác giả

- **Dự án**: ExamGuard Vision
- **Tác giả / Nhóm phát triển**: Nguyễn Tú Thiên Long - Thái Văn Thái - Nguyễn Minh Đức
- **Mục đích**: Nghiên cứu khoa học, chuyển đổi số giáo dục và hỗ trợ nâng cao tính trung thực trong thi cử.
- **Giấy phép**: Đề tài nghiên cứu ứng dụng — Chưa công bố giấy phép mã nguồn mở chính thức. Vui lòng ghi rõ nguồn khi tham khảo hoặc tái sử dụng.
