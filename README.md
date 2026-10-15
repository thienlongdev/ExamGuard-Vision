# ExamGuard Vision

Hệ thống thị giác máy tính hỗ trợ **giám sát phòng thi và rà soát sự kiện**, phục vụ học tập, trình diễn và nghiên cứu học thuật. ExamGuard Vision phát hiện đối tượng, theo dõi người theo thời gian và cung cấp bằng chứng để giám thị xem xét. **Kết quả AI không phải bằng chứng kết luận gian lận; quyết định cuối cùng thuộc về con người.**

## Bắt đầu nhanh trên Windows

**Luồng khuyến nghị: clone GitHub → chạy setup một lần → mở ứng dụng bằng một cú nhấp.**

### 1. Chuẩn bị công cụ

Cài [Git cho Windows](https://git-scm.com/install/windows), [Git LFS](https://git-lfs.com/) và [Python 3.13 x64 bản thường](https://www.python.org/downloads/windows/), kèm Python Launcher (`py`). Không dùng bản Python free-threaded. Nếu máy có Windows Package Manager, có thể chạy các lệnh sau trong PowerShell hoặc CMD:

```powershell
winget install --exact --id Git.Git --source winget
winget install --exact --id GitHub.GitLFS --source winget
winget install --exact --id Python.Python.3.13 --source winget
```

Mở lại terminal sau khi cài để cập nhật `PATH`.

### 2. Clone mã nguồn và tải model

```powershell
git lfs install
git clone https://github.com/thienlongdev/ExamGuard-Vision.git
cd ExamGuard-Vision
git lfs pull
```

### 3. Cài đặt một lần

Nhấp đúp **Setup ExamGuard Vision.bat**, hoặc chạy trong PowerShell:

```powershell
& ".\Setup ExamGuard Vision.bat"
```

Setup tạo hoặc dùng lại `.venv`, tải checkpoint qua Git LFS, cài PyTorch và dependencies, kiểm tra tương thích, tạo shortcut rồi khởi động Auto. Lần đầu có thể mất vài phút. Nếu muốn chủ động dùng CPU:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/setup_examguard.ps1 -Backend CPU
```

### 4. Khởi chạy và bắt đầu giám sát

Mỗi lần sử dụng, nhấp đúp shortcut **ExamGuard Vision** trên Desktop, hoặc chạy:

```powershell
& ".\Start ExamGuard Vision.bat"
```

Sau khi các model sẵn sàng, trình duyệt tự mở tại **http://127.0.0.1:8765/**. Lần đầu, tạo tài khoản quản trị riêng ở `/setup`, đăng nhập rồi bấm **Bắt đầu phiên giám sát**. Không có tài khoản hay mật khẩu quản trị mặc định. Vòng lặp capture/AI bắt đầu theo phiên; bước cấu hình phần cứng ban đầu có thể mở camera trong thời gian ngắn để kiểm tra.

### 5. Kết thúc và dừng an toàn

Kết thúc phiên trong dashboard để đóng phiên bình thường, sau đó dùng shortcut **Stop ExamGuard Vision**, hoặc chạy:

```powershell
& ".\Stop ExamGuard Vision.bat"
```

Trong **CMD**, gọi tên file batch trong dấu ngoặc kép và bỏ ký hiệu `&` của PowerShell, ví dụ `"Start ExamGuard Vision.bat"`. Script dừng kiểm tra đúng tiến trình của dự án và yêu cầu tắt an toàn, không tắt các tiến trình Python khác. Sau khi khởi động lại ứng dụng, cần đăng nhập lại.

Xem [hướng dẫn Windows và xử lý lỗi](docs/universal-windows-setup.md) nếu thiếu model, camera bận, cổng bị chiếm hoặc CUDA không hoạt động.

## Chức năng chính

- **Phát hiện và tracking:** YOLO26m nhận diện người/điện thoại; ByteTrack duy trì ID theo dõi theo từng camera.
- **Liên kết điện thoại:** PhoneAssociator xử lý theo tỷ lệ khung hình, giữ các trường hợp chưa liên kết hoặc mơ hồ; không loại điện thoại hợp lệ chỉ vì bounding box lớn trong ảnh 4K.
- **Phân tích hành vi:** kết hợp tư thế, góc quay đầu, hành vi tổng quát và temporal fusion để tạo sự kiện theo thời gian.
- **Bounding box và overlay:** hiển thị người, điện thoại và ứng viên ở khung hiện tại; quy đổi tọa độ từ ảnh nguồn sang preview.
- **Nguồn video:** webcam có thương lượng MJPEG, video file để phát lại và RTSP tùy chọn; giữ thông tin nguồn gốc khi xử lý.
- **Adaptive Runtime:** chọn cấu hình theo phần cứng, xác minh CUDA hoặc dùng CPU, giữ hàng đợi giới hạn và tách nhịp capture/AI/preview.
- **Dashboard:** FastAPI có xác thực, cập nhật qua WebSocket, lịch sử phiên, xem bằng chứng và thao tác Xác nhận/Bỏ qua của giám thị.
- **Vận hành Windows:** setup một lần, shortcut Desktop, ngăn khởi động trùng và dừng an toàn; có mã nguồn mã hóa evidence và backup.

## Kiến trúc tổng quát

```text
Webcam / video file / RTSP
  → YOLO phát hiện → ByteTrack tracking + liên kết điện thoại
  → tư thế / headpose / hành vi tổng quát → temporal fusion
  → sự kiện + bằng chứng + lưu trữ → FastAPI / WebSocket
  → dashboard → giám thị rà soát
```

AI cung cấp quan sát và mức cảnh báo để rà soát. Hệ thống không tự xác nhận hành vi gian lận từ một ảnh hoặc tự thay giám thị đưa ra kết luận.

## Yêu cầu phần cứng và phần mềm

| Thành phần | Môi trường đã kiểm chứng / luồng cài đặt hiện hành |
|---|---|
| Hệ điều hành | Windows 11 x64; launcher dùng Windows PowerShell, khóa cục bộ dùng DPAPI |
| Python | Python 3.13 x64 bản thường, có lệnh `py` |
| Inference | PyTorch 2.11.0 / torchvision 0.26.0; setup chọn wheel CUDA 12.8 hoặc CPU chính thức |
| GPU NVIDIA | Không bắt buộc; muốn tăng tốc phải có driver tương thích và CUDA kernel chạy được |
| Camera | Webcam Windows truy cập được; camera USB đã thử hỗ trợ DirectShow MJPEG |
| Mạng và ổ đĩa | Cần Internet khi clone, tải LFS và cài thư viện; thư mục dự án phải ghi được, đủ chỗ cho môi trường, khoảng 561 MiB model và evidence cục bộ |
| Trình duyệt | Trình duyệt desktop hiện hành; dashboard sử dụng tiếng Việt |

CPU vẫn chạy pipeline đã triển khai nhưng throughput thấp hơn. Chưa xác minh tăng tốc GPU AMD/Intel hay hiệu năng trên mọi laptop. FFmpeg trong `PATH` hỗ trợ liệt kê định danh/chế độ DirectShow và cần thiết khi chọn an toàn giữa nhiều webcam. Với một webcam, có thể dùng cơ chế PnP/OpenCV được nêu trong hướng dẫn Windows.

## Cấu hình camera và Adaptive Runtime

Auto kiểm tra phần cứng, chạy CUDA kernel thật, xác minh hash checkpoint, thương lượng chế độ camera và chạy forward cả bốn model đang sử dụng trước khi công bố sẵn sàng. Cấu hình tương thích đã đo có thể được dùng lại ở lần khởi động sau.

- **Detector:** chọn 960×960 khi CUDA dùng được, VRAM ≥ 8192 MiB và RAM khả dụng ≥ 4096 MiB. Các trường hợp còn lại dùng 640×640, bao gồm cấu hình ASUS 4 GiB tham chiếu.
- **Model khác:** macro detector dùng 768×768; model tư thế/headpose đang sử dụng dùng 224×224.
- **Webcam:** phải xác nhận resolution/FourCC được chấp nhận và đo tốc độ thực tế. Cắm webcam 4K không bảo đảm capture 4K30 hay xử lý AI ở 30 FPS.
- **Preview và evidence:** preview giới hạn chiều rộng 1280 pixel, bounding box được scale tương ứng; evidence dùng khung hình nguồn.

| Khái niệm | Ý nghĩa |
|---|---|
| Độ phân giải camera | Kích thước ảnh thực sự thu được, ví dụ 3840×2160 |
| Kích thước đầu vào YOLO | Kích thước yêu cầu khi model inference, ví dụ 960×960; khác ảnh camera 4K |
| Capture FPS | Số khung hình camera thu được mỗi giây |
| AI FPS | Số khung hình pipeline AI xử lý mỗi giây; có thể thấp hơn Capture FPS vì chủ động bỏ khung cũ để hạn chế độ trễ |

Cấu hình gốc nằm ở `configs/stage2_pipeline.yaml`; cấu hình phần cứng/nguồn được sinh trong `.runtime/` và không đưa lên Git. Các profile ASUS và công cụ nâng cao được giữ làm tham chiếu. Xem [thiết kế Adaptive Runtime](docs/adaptive-runtime.md) để hiểu cache, lựa chọn cấu hình và xử lý lỗi.

**Đổi nguồn:** dừng ứng dụng rồi mở **Configure ExamGuard Source.bat** để chọn webcam, video file hoặc RTSP. Camera đã lưu bị mất sẽ chặn startup thay vì tự chuyển thiết bị. Credentials RTSP tùy chọn được Windows DPAPI bảo vệ; không ghi mật khẩu hay đường dẫn video riêng vào cấu hình được track.

## Model, dataset và khả năng tái lập

Repository giữ nguyên **bảy checkpoint cố định** ở đường dẫn chuẩn: năm file qua Git thường, hai file headpose lớn qua Git LFS. Clone/LFS tải model mà không cần dữ liệu riêng của workstation hay tìm URL model bên ngoài. File nhỏ bắt đầu bằng `version https://git-lfs...` chỉ là LFS pointer, chưa phải checkpoint đã tải.

Sau setup, kiểm tra model và dependencies:

```powershell
.\.venv\Scripts\python.exe tools/validation/verify_checkpoint_hashes.py
.\.venv\Scripts\python.exe -m pip check
```

Auto chặn khởi động nếu thiếu model, còn LFS pointer hoặc hash sai. [Bảng kiểm chứng model](docs/model-integrity.md) ghi vai trò, đường dẫn, kích thước và SHA-256. Các model chính, model fallback và baseline lịch sử đều được giữ nguyên byte.

Inference thông thường **không cần dataset huấn luyện**. Checkout này có cấu hình mapping/training và fixture metadata nhỏ, không có thư mục `datasets/` chứa ảnh huấn luyện. Công cụ acquisition, annotations metadata và định nghĩa tập dữ liệu vẫn được giữ; setup không tải dataset. Một số tiện ích nghiên cứu cũ cần package `training` bên ngoài chưa được phân phối trong repository; xem [nghiên cứu và nguồn dữ liệu](docs/research.md).

Để cập nhật một bản clone đã có, dừng ứng dụng trước rồi chạy:

```powershell
git switch main
git pull --ff-only origin main
git lfs pull
```

Giữ nguyên thay đổi cục bộ nếu Git báo xung đột hoặc lịch sử phân kỳ. Chạy lại setup khi dependencies thay đổi. Sau khi cài, runtime cục bộ không cần dịch vụ cloud.

## Kiểm thử phần mềm

Lựa chọn regression tập trung sau bảo vệ Adaptive Runtime, điện thoại/association, tracking/overlay, API, bảo mật và nguồn video:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_adaptive_runtime.py tests/test_high_recall_and_reliability.py tests/test_live_track_overlay.py tests/test_multi_track_validation.py tests/test_api.py tests/test_auth_and_rbac.py tests/test_video_source.py tests/test_async_capture_and_latest_frame.py tests/test_repo_secret_audit.py
```

Chạy test trong checkout tạm để fixture không dùng database vận hành. Không chạy physical preflight, benchmark camera hay test toàn bộ vòng đời launcher khi đang giám sát. [Hướng dẫn kiểm thử](docs/testing.md) phân biệt test phần mềm, kiểm tra thiết bị thật và yêu cầu dữ liệu nghiên cứu.

Trong đợt bảo trì ngày **16/10/2026**, đã cài dependencies vào môi trường CPU mới, `pip check` đạt, **115 test tập trung đạt**, import **78 module** và forward thật **bốn model trên CPU** đạt bằng đầu vào tổng hợp. Đây là kiểm chứng phần mềm/cài đặt; không phải phép đo FPS laptop hay lần kiểm tra GUI/camera vật lý mới.

## Kết quả đo đã có và giới hạn

[Báo cáo chất lượng phát hiện ngày 15/10/2026](docs/final-detection-quality-validation.md) ghi nhận cửa sổ quan sát 20 giây sau restart trên **Windows 11, Ryzen 9 9950X, RTX 5070 12 GB**, Python 3.13.9, PyTorch 2.11.0+cu128 và camera USB DirectShow:

| Chỉ số | Giá trị quan sát |
|---|---:|
| Camera acquisition | 3840×2160 MJPEG |
| Capture FPS trung bình | 26.93 FPS |
| AI FPS toàn pipeline trung bình | 17.28 FPS |
| Độ tuổi khung từ sau decode đến kết quả, p50 / p95 | 66.84 / 86.42 ms |
| Đầu vào detector thực tế | 960×960, CUDA |

Đây là số đo lịch sử trên một workstation, không phải benchmark mới của đợt cleanup hay cam kết cho máy khác. Độ tuổi khung bắt đầu từ timestamp phần mềm sau decode, không gồm thời gian phơi sáng sensor hoặc debounce để mở sự kiện. Độ phân giải camera không phải kích thước đầu vào neural network.

Precision/recall trên tập đại diện, false events/hour, độ chính xác trong phòng thi đông, liên kết sai người và phát hiện điện thoại xa/bị che vẫn **chưa được xác minh đầy đủ** bằng dữ liệu gắn nhãn độc lập. Regression tổng hợp và thử scale trên một ảnh không chứng minh độ chính xác thực tế của lớp học. Dự án dành cho trình diễn/nghiên cứu học thuật có giám thị, chưa phải hệ thống tự động đưa ra quyết định kỷ luật.

## Cấu trúc repository

```text
configs/       Cấu hình runtime, profile tham chiếu, mapping/training
docs/         Hướng dẫn, kiến trúc, bảo mật, nghiên cứu và báo cáo kiểm chứng
models/        Checkpoint, model fallback và manifest định danh
scripts/       Launcher Auto/ASUS, chọn nguồn và script Windows
src/           Detection, tracking, fusion, evidence, persistence, API/dashboard
tests/         Regression, bảo mật và fixture tổng hợp/metadata
tools/         Acquisition, nghiên cứu, benchmark, validation và công cụ phát triển
*.bat          Setup, khởi chạy, dừng, chọn nguồn và cài shortcut Windows
```

`requirements-runtime.txt` là danh sách dependencies dùng chung; `requirements.txt` chuyển tiếp tới file này. Setup chọn PyTorch CPU/CUDA riêng. `.venv/`, `.runtime/`, `storage/`, `evidence/`, `reports/`, `runs/`, backup, log và secrets được giữ cục bộ. Xem [quyết định bảo trì repository](docs/repository-maintenance.md).

## Bảo mật, quyền riêng tư và ghi công

Dữ liệu giám sát lưu cục bộ. Hệ thống có authentication/RBAC, CSRF, cách ly phiên, mã hóa evidence, lưu khóa DPAPI và human review. ID tracking ẩn danh không làm hình ảnh camera mất tính chất dữ liệu cá nhân. Người vận hành chịu trách nhiệm về sự đồng ý, quyền truy cập, thời hạn lưu và rà soát phù hợp. Khi backup, giữ database, evidence và khóa liên quan cùng nhau; clone mã nguồn không chuyển phiên riêng hay khóa gắn với tài khoản Windows sang máy khác.

Ghi công cho các tác giả bên thứ ba: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) với [điều khoản AGPL-3.0 / Enterprise](https://www.ultralytics.com/license), [ByteTrack](https://github.com/FoundationVision/ByteTrack) (upstream MIT; ứng dụng dùng adapter của Ultralytics) và [HopeNet](https://github.com/natanielruiz/deep-head-pose) (upstream Apache-2.0). Điều khoản dataset khác với code/model; dữ liệu công khai không mặc nhiên có quyền phân phối lại. [Tài liệu nghiên cứu](docs/research.md) dẫn nguồn SCB, AFLW/3DDFA, COCO và nêu các quyền chưa xác minh.

Nhóm phát triển: **Nguyễn Tú Thiên Long, Hoàng Văn Thái, Nguyễn Minh Đức**. Dự án chưa công bố giấy phép mã nguồn mở chung; đợt bảo trì này không thay giấy phép hay cấp quyền đối với tài sản bên thứ ba.

## Tài liệu bổ sung

- [Cài đặt Windows và xử lý lỗi](docs/universal-windows-setup.md)
- [Adaptive Runtime](docs/adaptive-runtime.md) và [kiến trúc pipeline](docs/architecture.md)
- [Bảo mật](docs/security.md) và [lưu trữ/bằng chứng](docs/persistence-and-evidence.md)
- [Tính toàn vẹn checkpoint](docs/model-integrity.md) và [kiểm thử](docs/testing.md)
- [Kiểm chứng học thuật RTX 5070](docs/final-academic-validation.md) và [chất lượng phát hiện](docs/final-detection-quality-validation.md)
- [Nghiên cứu/dataset](docs/research.md) và [bảo trì repository](docs/repository-maintenance.md)
