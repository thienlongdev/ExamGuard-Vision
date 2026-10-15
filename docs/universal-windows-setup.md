# ExamGuard Vision — Cài đặt và sử dụng Windows Auto

Áp dụng cho một ứng dụng dùng chung mã nguồn. Máy đã kiểm chứng trực tiếp: Windows 11, Ryzen 9 9950X, RTX 5070 12 GB và USB `Web Camera`. Không suy rộng kết quả này thành chứng nhận cho mọi GPU/camera.

## Cài đặt một lần

1. Cài Git kèm Git LFS và **Python 3.13 x64 bản thường**, không chọn bản free-threaded. Driver NVIDIA cần tương thích CUDA trong wheel PyTorch chính thức. Không cần cài CUDA Toolkit riêng để chạy wheel.
2. Clone repository chính thức vào thư mục mong muốn. Không đổi tên workspace đã có trên workstation. Trên laptop mới, mở PowerShell:

   ```powershell
   git lfs install
   git clone https://github.com/thienlongdev/ExamGuard-Vision.git
   cd ExamGuard-Vision
   git lfs pull
   ```

   Nếu laptop đã clone, kết thúc phiên và dừng ExamGuard bằng shortcut trước khi cập nhật. Từ thư mục repository chạy:

   ```powershell
   git switch main
   git pull --ff-only origin main
   git lfs pull
   ```

   Nếu Git báo thay đổi cục bộ hoặc lịch sử phân kỳ, giữ nguyên dữ liệu và xử lý riêng; không reset/force. Chạy lại setup sau khi requirements hoặc launcher thay đổi.
3. Double-click `Setup ExamGuard Vision.bat`. Setup tạo/reuse `.venv`, tải checkpoint bằng LFS, cài cặp PyTorch/torchvision chính thức, dependencies, tạo shortcuts và khởi động Auto.
4. Tạo tài khoản quản trị riêng trong trang `/setup`. Không có mật khẩu mặc định. Sau đó nhập thông tin phiên và bấm **Bắt đầu phiên giám sát**.

Setup dùng PyTorch `2.11.0`/torchvision `0.26.0`, CUDA 12.8 khi phát hiện công cụ NVIDIA; CPU wheel khi không có NVIDIA. Auto vẫn kiểm tra kernel và forward của cả bốn model trước readiness. GPU cũ không tương thích wheel cần một lần chọn build chính thức phù hợp hoặc chọn CPU; không tự cài lại package ở mỗi lần mở.

FFmpeg hỗ trợ tên/moniker DirectShow và danh sách chế độ camera. Khi FFmpeg chưa có, một camera có thể dùng PnP cùng probe OpenCV; nhiều camera cần cài FFmpeg vào PATH để ánh xạ chỉ mục an toàn. Không ánh xạ thứ tự PnP tùy ý sang nhiều camera.

Nguồn cài PyTorch: [hướng dẫn chính thức](https://pytorch.org/get-started/locally/). Bộ phiên bản thực tế của máy được lưu tại `reports/owner-runtime-lock.txt`; file này là bằng chứng cục bộ, không phải cam kết hỗ trợ mọi môi trường.

## Mỗi lần sử dụng

- Mở icon **ExamGuard Vision** trên Desktop. Launcher kiểm tra môi trường, checkpoint, thiết bị, model, rồi mới mở trình duyệt khi `/ready` xác nhận instance và các forward đã thành công.
- Dashboard mặc định: **http://127.0.0.1:8765/**. Cổng 8000 trên máy này thuộc Docker/WSL và được giữ nguyên.
- Đăng nhập; bắt đầu một phiên để mở camera và AI. Khởi động ứng dụng không tự bật giám sát. Sau khi restart, chính sách hiện có yêu cầu đăng nhập mới; phiên trước được ghi `INTERRUPTED`.
- Kết thúc phiên qua dashboard để đóng phiên bình thường. Dừng toàn ứng dụng bằng **Stop ExamGuard Vision**. Stop kiểm tra PID, thời điểm tạo process, command và thư mục dự án; không kill mọi Python process.

Setup tạo **ExamGuard Vision** và **Stop ExamGuard Vision** trên Desktop của tài khoản Windows hiện tại (kể cả Desktop được chuyển vào OneDrive). Đường dẫn được xác định tự động, không cần sửa tên người dùng trong script.

Giữ thư mục ứng dụng tại chỗ. Nếu di chuyển thư mục, chạy lại `Install ExamGuard Shortcuts.bat`. Runtime chạy ẩn; batch có thể xuất hiện ngắn khi mở shortcut.

## Chọn nguồn hoặc cấu hình nâng cao

Đóng ứng dụng, mở `Configure ExamGuard Source.bat` để chọn webcam, video file hoặc RTSP. Với nhiều webcam, chọn một lần bằng tên/chỉ mục và lưu định danh. Camera đã lưu bị mất sẽ chặn startup, không tự chuyển sang camera khác. USB đổi cổng có thể đổi moniker; chọn lại rõ ràng khi cần.

Video file dùng `VideoFileSource`, giữ timestamp và `VIDEO_FILE` provenance. RTSP dùng `RTSPSource` và reconnect/backoff có giới hạn. Nhập URL được phép truy cập cùng thông tin đăng nhập nếu cần. Công cụ kiểm tra lấy được frame trước khi lưu. URL RTSP được Windows DPAPI mã hóa theo tài khoản Windows hiện tại; JSON chỉ lưu `dpapi:camera-....dpapi`. Không đưa URL chứa mật khẩu vào YAML/Git. Cần cấu hình mạng, vị trí lắp đặt và quyền truy cập hợp lệ; ứng dụng không quét mạng.

Primary Auto launcher là một nguồn camera có session ownership. Kiến trúc `CameraManager` nhiều camera hiện có vẫn giữ nguyên; triển khai nhiều camera cần cấu hình riêng qua cơ chế hiện có, chưa kiểm chứng trên CCTV trường học trong lần này.

Các lệnh nâng cao, không bắt buộc khi sử dụng thông thường:

```powershell
# Chọn lại webcam hoặc bỏ cache để hiệu chỉnh lại
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/start_examguard.ps1 -Reconfigure
# Chọn cổng khác
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/start_examguard.ps1 -Port 8770
# Cấu hình tham chiếu ASUS cũ được giữ nguyên, không phải mặc định Auto
.venv\Scripts\python.exe scripts/run_asus_a17_demo.py
# CPU wheel trong setup ban đầu
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/setup_examguard.ps1 -Backend CPU
```

## Chẩn đoán

| Trạng thái | Ý nghĩa / cách xử lý |
|---|---|
| `INITIALIZING` | Đang kiểm tra và load/warm model; đợi launcher. |
| `READY` | Model thực thi thành công, backend sẵn sàng. Camera chỉ ACTIVE sau khi bắt đầu phiên. |
| `DEGRADED` | CUDA không khả dụng, CPU đã được kiểm tra; launcher/API nêu lý do. |
| `BLOCKED` | Thiếu/corrupt checkpoint, không mở camera, lỗi kernel/AI hoặc nguồn mất kết nối; xem lý do cụ thể. |
| `STOPPED` | Đã dừng và giải phóng tài nguyên. |

- Nhật ký: `.runtime/logs/examguard-runtime.log`, `startup.stderr.log`; trạng thái: `.runtime/status.json`.
- Model thiếu/LFS pointer: chạy lại setup hoặc `git lfs pull`; hash sai không được bỏ qua.
- Camera bận: đóng ứng dụng đang giữ webcam, kiểm tra quyền camera Windows, thử lại. Không mở benchmark khi phiên đang giữ camera.
- Camera mất trong phiên: runtime dừng an toàn với `BLOCKED`, không âm thầm đổi thiết bị. Kết nối lại, kiểm tra danh tính rồi khởi động lại và đăng nhập/bắt đầu phiên.
- Cổng bị chiếm: launcher báo rõ và giữ nguyên process lạ; chọn cổng khác qua tùy chọn trên.
- Driver camera bị kẹt ở native read: không bảo đảm tự hồi phục; stop không ép kill khi chưa hoàn tất cleanup, cần xử lý driver/ứng dụng giữ thiết bị.
- Lỗi model CUDA sau khi chọn device sẽ chặn readiness; không che lỗi bằng cách công bố READY. CPU chỉ được công bố khả dụng khi forward thành công.

Giữ nguyên `storage/`, `evidence/`, khóa DPAPI và database khi chuyển giao/sao lưu. Không xóa dữ liệu bằng chứng để “làm sạch” demo.

## Checkpoint và giới hạn laptop

Bảy checkpoint đã có trong repository: năm file dùng Git thường và hai file headpose lớn dùng Git LFS theo `.gitattributes`. Setup chạy `git lfs pull`; không cần tìm URL model ngoài hoặc sao chép dữ liệu của workstation. Sau khi tải, kiểm tra đủ cả bảy hash:

```powershell
.\.venv\Scripts\python.exe tools/validation/verify_checkpoint_hashes.py
```

Thiếu file, LFS pointer hoặc hash sai sẽ chặn khởi động. Không tắt kiểm tra hash để chạy tiếp. Cần Internet cho clone/LFS và cài dependencies; sau đó runtime cục bộ không yêu cầu cloud.

Auto kiểm tra CUDA bằng kernel thật và forward model. Detector dùng 960×960 khi CUDA khả dụng, VRAM ≥ 8192 MiB và RAM khả dụng ≥ 4096 MiB; các máy còn lại dùng 640×640. Camera mode, nhịp AI và preview được chọn riêng theo capability và số đo. Không có NVIDIA tương thích thì dùng CPU đã kiểm tra; không hứa 4K30, FPS hay chất lượng phát hiện tương đương RTX 5070. GPU AMD/Intel chưa có backend tăng tốc được xác minh trong bản này. Laptop cần tự kiểm tra camera, driver và hiệu năng thực tế.

Mỗi máy tạo tài khoản quản trị và cấu hình camera riêng. `.runtime/`, `.venv/`, database, evidence, khóa/credentials và báo cáo thô của workstation không được phân phối qua Git. Báo cáo học thuật trong `docs/` là số liệu lịch sử đã lược định danh, không chứng nhận laptop đã được thử. Remote AI Server là công việc riêng, chưa có trong bản đồng bộ này.
