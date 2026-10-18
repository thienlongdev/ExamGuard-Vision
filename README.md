# ExamGuard Vision

**Hệ thống hỗ trợ giám sát phòng thi và phân tích hành vi bằng AI.**

ExamGuard Vision sử dụng Python và thị giác máy tính để phát hiện đối tượng, theo dõi người và phân tích các tín hiệu hành vi trong video phòng thi. Hệ thống tổng hợp quan sát theo thời gian, tạo sự kiện kèm bằng chứng để giám thị rà soát trên giao diện web. Dự án phục vụ học tập, nghiên cứu và trình diễn ứng dụng AI; quyết định đánh giá cuối cùng thuộc về giám thị.

## Tính năng chính

- **Nguồn hình ảnh:** webcam, tệp video và luồng RTSP tương thích với OpenCV.
- **Phát hiện và theo dõi:** nhận diện người, điện thoại; theo dõi nhiều người bằng ByteTrack và hiển thị khung nhận diện trên video.
- **Phân tích hành vi:** ước lượng hướng quay đầu, phân loại tư thế và quan sát chuyển động đầu/thân người.
- **Tổng hợp theo thời gian:** Temporal Fusion kết hợp các tín hiệu và độ tin cậy để tạo, cập nhật và kết thúc sự kiện.
- **Giám sát trực tiếp:** bảng điều khiển web hiển thị video, trạng thái hệ thống và các sự kiện cần xem xét.
- **Rà soát bằng chứng:** xem ảnh, đoạn video khi khả dụng, lịch sử phiên và ghi nhận quyết định **Xác nhận / Bỏ qua**.
- **Bảo vệ dữ liệu cục bộ:** xác thực, phân quyền, nhật ký thao tác và cơ chế mã hóa bằng chứng bằng AES-256-GCM, với khóa được Windows DPAPI bảo vệ.

## Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Xử lý video, điều phối AI và vận hành ứng dụng |
| PyTorch / TorchVision | Chạy mô hình phân tích tư thế và hướng đầu trên GPU hoặc CPU |
| Ultralytics YOLO | Phát hiện người, điện thoại và phân tích hành vi tổng quát |
| OpenCV | Đọc nguồn video, xử lý ảnh và tạo bằng chứng trực quan |
| ByteTrack qua Ultralytics | Duy trì ID theo dõi người qua các khung hình |
| FastAPI / WebSocket | Cung cấp API và cập nhật giao diện giám sát |
| SQLite | Lưu tài khoản, phiên giám sát, sự kiện và kết quả rà soát |

## Yêu cầu hệ thống

- **Windows 11 x64**, có Windows PowerShell; đây là môi trường mà bộ cài hiện tại hướng tới.
- **Python 3.13 x64 bản thường**, kèm Python Launcher (`py`) và thành phần Tcl/Tk để mở hộp thoại chọn nguồn. Không dùng bản free-threaded.
- **Git và Git LFS**, truy cập được bằng lệnh `git` và `git lfs` trong PowerShell.
- **Internet khi cài đặt** để clone mã nguồn, tải trọng số mô hình và cài thư viện; thư mục dự án cần quyền ghi và đủ dung lượng cho môi trường Python, mô hình và bằng chứng.
- **Webcam được Windows cho phép truy cập**, hoặc tệp video/địa chỉ RTSP hợp lệ; trình duyệt desktop để mở bảng điều khiển.
- **GPU NVIDIA là tùy chọn.** Cần driver tương thích với bộ PyTorch CUDA 12.8 để tăng tốc; ứng dụng có thể dùng CPU khi CUDA không khả dụng, nhưng tốc độ xử lý thấp hơn.

**Nếu máy có nhiều webcam:** cần FFmpeg trong `PATH` để định danh và chọn đúng thiết bị. Bộ cài không tự cài FFmpeg; với một webcam, ứng dụng có cơ chế dò thiết bị qua Windows và OpenCV.

## Hướng dẫn cài đặt

### 1. Clone mã nguồn và tải mô hình

Sau khi cài các công cụ ở trên, mở PowerShell tại thư mục muốn lưu dự án và chạy:

```powershell
git lfs install
git clone https://github.com/thienlongdev/ExamGuard-Vision.git
cd ExamGuard-Vision
git lfs pull
```

Git LFS dùng để tải các trọng số AI lớn. **Không nên dùng nút “Download ZIP” thông thường của GitHub:** tệp tải về có thể chỉ chứa LFS pointer thay cho trọng số thật. Nếu tệp mô hình bắt đầu bằng `version https://git-lfs.github.com/spec/v1`, mô hình chưa được tải đầy đủ.

### 2. Chạy bộ cài

Trong thư mục vừa clone, chạy:

```powershell
& ".\Setup ExamGuard Vision.bat"
```

Bộ cài tạo hoặc dùng lại `.venv`, tải mô hình qua Git LFS và xác minh SHA-256 của **8 checkpoint** được phân phối. Sau đó, script cài PyTorch/TorchVision bản CUDA 12.8 nếu tìm thấy `nvidia-smi`, hoặc bản CPU nếu không có; cài thư viện từ `requirements-runtime.txt` và kiểm tra tính tương thích bằng `pip check`. Cuối cùng, bộ cài tạo lối tắt khởi động/dừng trên Desktop và khởi chạy ứng dụng với bước kiểm tra phần cứng, nguồn hình ảnh và mô hình.

Lần cài đầu có thể mất nhiều thời gian tùy tốc độ mạng. Nếu cài hoặc khởi chạy bằng CUDA gặp lỗi, có thể chọn CPU rõ ràng:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\windows\setup_examguard.ps1" -Backend CPU
```

Trọng số được cung cấp trong repository; sử dụng ứng dụng không yêu cầu tải bộ dữ liệu huấn luyện riêng.

## Khởi chạy chương trình

Sau khi cài đặt, nhấp đúp lối tắt **ExamGuard Vision** trên Desktop hoặc chạy tại thư mục dự án:

```powershell
& ".\Start ExamGuard Vision.bat"
```

Khi ứng dụng sẵn sàng, trình duyệt tự mở tại [http://127.0.0.1:8765/](http://127.0.0.1:8765/). Có thể mở địa chỉ này thủ công nếu cần.

1. Lần đầu, trang `/setup` yêu cầu tạo tài khoản quản trị: nhập tên đăng nhập, tên hiển thị và mật khẩu tối thiểu 12 ký tự. **Không có tài khoản hay mật khẩu mặc định.**
2. Sau khi tạo tài khoản, hệ thống tự đăng nhập; ở những lần truy cập sau, đăng nhập bằng tài khoản đã tạo.
3. Kiểm tra nguồn hình ảnh. Khi cần đổi nguồn, dừng ứng dụng, chạy **Configure ExamGuard Source.bat**, chọn `webcam`, `video_file` hoặc `rtsp` rồi khởi động lại. Công cụ sẽ yêu cầu chọn thiết bị, tệp hoặc địa chỉ tương ứng.
4. Nhập thông tin phiên, chọn camera đã cấu hình và bấm **Bắt đầu phiên giám sát**.
5. Theo dõi video, các đối tượng được nhận diện và sự kiện xuất hiện trên bảng điều khiển.
6. Khi hoàn tất, bấm **Kết thúc phiên**, xác nhận và đợi lưu xong; sau đó dừng ứng dụng bằng:

```powershell
& ".\Stop ExamGuard Vision.bat"
```

Camera và vòng lặp AI hoạt động theo phiên giám sát; bước kiểm tra cấu hình khi khởi động có thể mở nguồn hình ảnh trong thời gian ngắn.

## Cách sử dụng cơ bản

- **Giám sát:** tại thẻ **GIÁM SÁT**, bắt đầu phiên và quan sát khung nhận diện cùng danh sách sự kiện. Điện thoại xuất hiện trong khung hình không đồng nghĩa đã tạo sự kiện hay xác định được người sử dụng.
- **Xem sự kiện:** mở **DUYỆT SỰ KIỆN** và chọn một dòng để xem chi tiết. Đối chiếu tín hiệu hành vi với ảnh hoặc video trong phần **Bằng chứng trực quan** khi đã sẵn sàng.
- **Ghi nhận đánh giá:** nhập ghi chú nếu cần, chọn **Xác nhận** hoặc **Bỏ qua** theo kết quả rà soát của giám thị.
- **Xem lại và kết thúc:** dùng **LỊCH SỬ** → **Xem lại** để xem sự kiện của từng phiên; kết thúc phiên bằng nút trong ứng dụng trước khi dùng launcher dừng.

## Cấu trúc dự án

```text
ExamGuard-Vision/
├── src/       # Mã nguồn AI, xử lý video, API, giao diện và lưu trữ.
├── scripts/   # Công cụ cài đặt, khởi chạy và cấu hình nguồn.
├── configs/   # Cấu hình mô hình, theo dõi và quy tắc sự kiện.
├── models/    # Trọng số mô hình và thông tin định danh checkpoint.
└── tests/     # Các bài kiểm thử chức năng và tích hợp.
```

## Xử lý lỗi thường gặp

- **Không nhận lệnh Python hoặc `py`:** cài Python 3.13 x64 bản thường, bật Python Launcher và tùy chọn thêm Python vào `PATH`. Mở lại PowerShell, kiểm tra bằng `py -3.13 --version`, rồi chạy lại bộ cài.
- **Thiếu mô hình, tệp LFS pointer hoặc sai hash:** trong bản clone Git, chạy lại `git lfs install` và `git lfs pull`, sau đó chạy lại setup. Nếu đang dùng ZIP, thực hiện lại bước clone; không bỏ qua lỗi xác minh checkpoint.
- **CUDA không khả dụng hoặc cài PyTorch thất bại:** kiểm tra driver NVIDIA; để tiếp tục trên CPU, dùng lệnh setup với `-Backend CPU` ở phần cài đặt. Việc có GPU NVIDIA chưa bảo đảm CUDA chạy được.
- **Không mở được webcam:** kiểm tra kết nối, bật quyền camera cho ứng dụng desktop trong cài đặt Windows và đóng ứng dụng khác đang dùng camera. Nếu đổi thiết bị, dừng ExamGuard rồi chạy **Configure ExamGuard Source.bat**; với nhiều webcam, kiểm tra FFmpeg trong `PATH`.
- **Cổng 8765 đang được sử dụng:** nếu ExamGuard đang chạy, mở lại địa chỉ bảng điều khiển hoặc dừng bằng launcher của dự án. Nếu cổng thuộc ứng dụng khác, đóng ứng dụng đó rồi thử lại; launcher không tự tắt tiến trình khác.
- **Ứng dụng không khởi động:** xác nhận setup đã hoàn tất, xem thông báo trong cửa sổ launcher và nhật ký tại `.runtime/logs/`, đặc biệt `examguard-runtime.log` và `startup.stderr.log`. Chạy lại setup nếu thiếu môi trường hoặc thư viện; launcher khởi động không tự cài dependencies.

## Lưu ý, giới hạn và ghi công

Đây là dự án học thuật hỗ trợ giám sát bằng AI. Cảnh báo thể hiện tín hiệu kỹ thuật quan sát được, cần được giám thị đối chiếu với ngữ cảnh và bằng chứng trước khi đánh giá; không phải kết luận gian lận tự động. Độ tin cậy phụ thuộc vào ánh sáng, góc đặt camera, độ phân giải, che khuất và số người trong khung hình; không bảo đảm phát hiện đầy đủ mọi tình huống.

Chỉ sử dụng với nguồn hình ảnh được cho phép và tôn trọng quyền riêng tư. Quản lý quyền truy cập, thời gian lưu và việc chia sẻ dữ liệu phòng thi phù hợp với mục đích sử dụng.

Các mô hình và phần mềm bên thứ ba giữ nguyên giấy phép và ghi công gốc. Xem [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) và thư mục [licenses/](licenses/) trước khi sử dụng hoặc phân phối lại.
