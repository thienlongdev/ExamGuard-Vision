# Báo cáo xác nhận chất lượng phát hiện điện thoại

Ngày xác nhận: 2026-10-15
Phạm vi: bản làm việc cục bộ `C:\ExamGuard-Vsion`, không đẩy GitHub, không thay checkpoint, không thay ngữ nghĩa phán quyết của giám thị.

## Kết luận

**PARTIALLY VERIFIED — ADDITIONAL LABELED DATA REQUIRED**

Đường đi phát hiện điện thoại đã được sửa và đã hoạt động end-to-end trên webcam thật: hệ thống hiển thị người, ghi nhận nhiều sự kiện điện thoại gắn với Thí sinh #1, tạo ảnh/clip bằng chứng mã hóa và đưa sự kiện vào hàng đợi Xác nhận/Bỏ qua. Kết luận này không đồng nghĩa với độ chính xác đủ cho giám sát thi không người trực; chưa có tập dữ liệu có nhãn đủ rộng cho điện thoại xa, nhiều người, nhiều điện thoại, vật thể gây nhầm và lớp học đông.

## 1. Lỗi ban đầu và nguyên nhân gốc

Luồng dữ liệu đã được truy từ ảnh nguồn 4K đến detector, hậu xử lý, liên kết người–điện thoại, temporal fusion, API và trình duyệt.

1. `PhoneAssociator` dùng giới hạn tuyệt đối theo pixel nguồn (`w <= 350`, `h <= 420`, diện tích `<= 90000`). Trên ảnh có nhãn thật 640×426, YOLO tìm đúng điện thoại với confidence `0.9583` và IoU `0.9737`, nhưng hộp khoảng 300×416 px có diện tích khoảng 124,742 px² nên bị logic cũ loại bỏ. Đây là lỗi sau detector, không phải detector không nhìn thấy điện thoại.
2. Nhánh điện thoại không liên kết còn có một giới hạn tuyệt đối 280×320, tiếp tục làm mất điện thoại gần camera.
3. Dashboard chỉ nhận/render hộp người; phát hiện điện thoại hợp lệ không xuất hiện tức thời nếu chưa hình thành sự kiện temporal.
4. Hợp đồng overlay người và điện thoại chưa đồng bộ theo cùng một khung, tạo nguy cơ hộp cũ hoặc sai thời điểm.
5. Trang Hệ thống trước đây ghi cứng `640x640`, vì vậy nhãn giao diện không phải bằng chứng về `imgsz` thật.
6. Launcher nền cũ giữ thừa handle stdout/stderr, khiến kiểm thử vòng đời chờ vô hạn dù backend đã READY.

## 2. Thay đổi đã thực hiện

- `src/orchestration/phone_associator.py`: bỏ giới hạn kích thước pixel tuyệt đối; dùng hình học tỷ lệ theo khung hình, chỉ loại hộp không hợp lệ/gần phủ toàn khung; giữ điện thoại lớn chưa liên kết; jitter của ứng viên yếu tỷ lệ theo đường chéo người; giữ nguyên xử lý AMBIGUOUS và không gán bừa người.
- `src/detection/object_detector.py`: thêm diagnostics cho output sau NMS, lý do nhận/loại, confidence, lớp, kích thước detector và hệ tọa độ pixel nguồn. Lời gọi Ultralytics dùng trực tiếp `imgsz=self.image_size`.
- `src/orchestration/stage2_pipeline.py`, `camera_manager.py`: xuất trạng thái overlay người + điện thoại theo cùng khung hiện tại; hộp điện thoại được xuất độc lập với sự kiện; thay thế toàn bộ mỗi khung để không giữ hộp cũ; chuẩn hóa scale X/Y từ nguồn sang preview.
- `src/api/main.py`, `src/api/schemas.py`: endpoint overlay có xác thực/no-cache; telemetry lấy `image_size` trực tiếp từ detector đã khởi tạo thay vì giá trị mặc định.
- `src/api/static/js/{state.js,api.js,dashboard.js}` và `components/camera_view.js`: polling overlay đồng bộ; điện thoại xác nhận màu cyan, ứng viên/không chắc màu amber nét đứt; hiển thị confidence và trạng thái liên kết; tự xóa khi feed quá hạn.
- `components/system_view.js`: hiển thị kích thước detector thực từ backend, không ghi cứng 640.
- `src/orchestration/adaptive_runtime.py`: chính sách dựa trên capability: CUDA + VRAM ≥ 8192 MiB + RAM khả dụng ≥ 4096 MiB chọn 960; CPU/VRAM thấp và máy ASUS 4 GiB giữ 640. Không dùng tên GPU để quyết định.
- `scripts/run_examguard.py`: ghi `detector_image_size` và đường dẫn cấu hình thực sự đã nạp vào `.runtime/status.json` ở lần khởi động kế tiếp.
- `scripts/windows/start_examguard.ps1` và `launch_examguard_detached.py`: tách tiến trình đúng cách, không để launcher giữ pipe làm treo vòng đời.
- `tools/validation/benchmark_phone_detection_quality.py`: benchmark tái lập 640/960/1280 và kiểm tra scale 4K có phạm vi tuyên bố rõ ràng.
- `tools/validation/verify_checkpoint_hashes.py`: sửa root path để kiểm tra đúng checkpoint thật.

Diagnostics chi tiết chỉ được xuất khi bật `EXAMGUARD_DIAGNOSTIC_CUES=1`; endpoint vẫn yêu cầu xác thực. Không nới auth/CSRF, không đổi risk scoring, không tự động xác nhận gian lận.

## 3. Quyết định 640 / 960 / 1280

Nguồn đánh giá định lượng nhỏ là ảnh COCO train `000000052010.jpg` có annotation cell phone `(x=24.95, y=2.4, w=294.55, h=419.28)`, được ghi nguồn tại [Metaphos COCO sample](https://metaphos.dev/library/shapes/coco-train-52010) và ảnh gốc từ [COCO](http://images.cocodataset.org/train2017/000000052010.jpg). Đây chỉ là một ảnh có nhãn, không phải bộ đo precision/recall.

| Detector input | Confidence / IoU điện thoại gần | Mean forward | Peak CUDA trong lượt đo |
|---|---:|---:|---:|
| 640 | 0.9583 / 0.9737 | 6.35 ms | 166.47 MiB |
| 960 | 0.9588 / 0.9802 | 10.12 ms | 308.68 MiB |
| 1280 | 0.8548 / 0.7920 | 16.20 ms | 404.39 MiB |

Kiểm tra scale có kiểm soát trên canvas 3840×2160 (không phải ảnh lớp học thật):

- điện thoại biến đổi khoảng 55×79 px: 640 không tìm thấy; 960 tìm thấy `0.8036`; 1280 tìm thấy `0.8696`;
- khoảng 74×105 px: 640 `0.6532`, 960 `0.8680`, 1280 `0.8400`;
- khoảng 110×157 px: cả ba tìm thấy; 960 `0.8801`;
- khoảng 37×52 px: chỉ 1280 tìm thấy (`0.4091`), nhưng đây là một kiểm tra biến đổi tổng hợp và chưa đủ để chọn 1280 cho runtime.

960 được chọn cho workstation RTX 5070 vì phục hồi rõ trường hợp nhỏ mà 640 bỏ lỡ, trong khi chi phí thấp hơn đáng kể và ổn định hơn 1280. 1280 không được chọn vì tốn thêm thời gian/VRAM và làm localization/association của mẫu gần kém hơn.

Chuỗi cấu hình đã được kiểm tra: Desktop shortcut → `start_examguard.ps1` → `run_examguard.py` → `prepare_auto()` → `.runtime/auto-runtime.yaml` → `LiveValidationOrchestrator(config_path=...)` → `Stage2Pipeline` → `ModelRegistry` → `YOLOObjectDetector.image_size` → Ultralytics `imgsz`. Lần UI 640 trước sửa là telemetry giao diện ghi cứng/không có provenance detector trực tiếp. Sau safe restart bằng shortcut Desktop, tiến trình mới ghi `detector_image_size: 960` lấy trực tiếp từ detector đã khởi tạo và `loaded_config` trỏ tới `.runtime/auto-runtime.yaml`; trang Hệ thống đồng thời hiển thị `YOLO26m · cuda:0 fp32 · 960x960`. Vì lời gọi inference truyền `imgsz=self.image_size`, đây là cấu hình hiệu lực của YOLO, không chỉ là giá trị trong YAML.

## 4. Xác nhận điện thoại và overlay

- Reproducer trước sửa: detector trả phone `0.9583`; filter cũ loại chỉ vì diện tích tuyệt đối > 90,000 px².
- Sau sửa: cùng prediction được nhận, liên kết đúng track #1 và giữ nguyên tọa độ nguồn; hộp lớn chưa liên kết không bị xóa.
- Test overlay chứng minh hộp phone xuất hiện khi chưa có reviewable event và bị xóa ngay ở khung tiếp theo không có phone.
- Test nhiều phone giữ mọi phone; test ambiguous không gán một phone cho sai người; test non-phone oversized vẫn bị loại theo lớp.
- Phiên webcam thật (đã lược định danh): dashboard xác thực đã hiển thị nhiều sự kiện “Phát hiện điện thoại”, gồm mức “Cần chú ý” và “Cảnh báo cao”, gắn Thí sinh #1, có thumbnail/clip bằng chứng và nút Xác nhận/Bỏ qua. Đồng thời person track, đứng dậy và quay đầu vẫn hoạt động.
- Không có thao tác Xác nhận/Bỏ qua tự động. Quan sát một người không chứng minh độ đúng association trong cảnh đông.

## 5. Hiệu năng hiện tại

Đo 30 giây khi phiên thật đang ACTIVE, camera 3840×2160 MJPG, queue luôn 0:

| Chỉ số | Giá trị |
|---|---:|
| Capture FPS trung bình | 26.66 |
| Full-pipeline AI FPS trung bình | 20.67 |
| Khung AI xử lý trong cửa sổ | 597 |
| Khung bỏ có chủ đích | 160 |
| Latency p50 / p95 | 67.00 / 89.80 ms |
| RSS min / max | 2370.8 / 2590.9 MiB |
| Peak PyTorch CUDA allocated | 677.2 MiB |
| Max queue depth | 0 |

Một lượt probe độc lập 10 giây cho 4K MJPG đạt 20.08 FPS. Đây là phép đo camera độc lập tại thời điểm khác, không thay thế số full runtime bên trên. TF32 vẫn tắt; CUDA parity PASS.

Đo xác nhận 20 giây sau safe restart, với detector 960 đã được xác nhận từ đối tượng runtime:

| Chỉ số sau restart | Giá trị |
|---|---:|
| Capture FPS trung bình | 26.93 |
| Full-pipeline AI FPS trung bình | 17.28 |
| Khung AI xử lý / bỏ có chủ đích | 351 / 194 |
| Latency p50 / p95 | 66.84 / 86.42 ms |
| RSS min / max | 2371.0 / 2423.5 MiB |
| Peak PyTorch CUDA allocated | 677.2 MiB |
| Max queue depth | 0 |

Probe model ngay lúc startup xác nhận detector, macro, posture và headpose đều forward trên `cuda:0`; detector 960 có forward `16.19 ms` trong probe khởi động.

## 6. Kiểm thử

- Regression runtime rộng trước thay đổi telemetry cuối: **396 passed, 7 skipped, 2 deselected**.
- Regression bị ảnh hưởng sau thay đổi telemetry: **54 passed**.
- Launcher/lifecycle: **11 passed**; start, duplicate-start, stop, port release và restart đều qua.
- Checkpoint: **7/7 SHA-256 khớp**, gồm detector, macro, posture/headpose chính và fallback.
- JavaScript syntax check: các file state/API/dashboard/camera/system đã qua `node --check`.
- Một lượt gộp test API khi phiên thật đang chạy ban đầu có **22 passed, 2 failed**. Nguyên nhân là fixture dùng singleton persistence mặc định, trộn event giả với kho giám sát thật; một test phụ thuộc sai vào thứ tự event, test kia ghi review cho event giả vào DB thật nên vướng foreign key. Fixture đã chuyển sang SQLite file tạm riêng cho từng test (không dùng `:memory:` vì repository mở nhiều connection). Hai test lỗi cũ sau sửa: **2 passed**. Database giám sát không bị sửa/xóa để làm test qua; hash database sau safe-stop được giữ trong hồ sơ kiểm chứng cục bộ.

Artefact tái lập:

- `reports/detection-quality/coco-phone-resolution-after.json`
- `reports/detection-quality/phone-scale-sweep-after.json`
- `reports/detection-quality/affected-regression-final.xml`
- `reports/detection-quality/runtime-regression.xml`
- `reports/detection-quality/launcher-regression.xml`
- `reports/detection-quality/live-runtime-final.log`
- `reports/detection-quality/live-runtime-post-restart.log`

Snapshot trước thay đổi và hash kiểm chứng được giữ cục bộ trong `backups/`; không phân phối bản sao lưu qua Git.

## 7. Cách tái lập

Tải ảnh công khai từ liên kết COCO ở mục 3 và lưu vào đường dẫn `--image` bên dưới; ảnh và báo cáo thô không nằm trong Git. Dừng ExamGuard trước benchmark model để tránh tranh chấp tài nguyên GPU; lệnh `observe_live_runtime.py` chỉ dùng sau khi người dùng bắt đầu phiên.

```powershell
.\.venv\Scripts\python.exe tools\validation\verify_checkpoint_hashes.py
.\.venv\Scripts\python.exe tools\validation\benchmark_phone_detection_quality.py `
  --image reports\detection-quality\assets\coco-train-000000052010.jpg `
  --gt "24.95,2.4,294.55,419.28" --sizes "640,960,1280" --repeats 10
.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_adaptive_runtime.py `
  tests\test_high_recall_and_reliability.py `
  tests\test_live_track_overlay.py `
  tests\test_multi_track_validation.py
.\.venv\Scripts\python.exe tools\validation\observe_live_runtime.py --seconds 30
```

## 8. Chưa được xác minh và giới hạn

- Precision, recall, F1, AP/mAP và false events/hour: **UNVERIFIED** vì không có tập test điện thoại/negative đủ lớn có nhãn độc lập.
- Điện thoại thật xa, màn hình tối/sáng, nằm bàn/đùi, che tay, nhiều hướng: chỉ có một phần quan sát live; chưa có ground truth có hệ thống.
- Nhiều người/nhiều điện thoại/lớp đông, ID switch và wrong-person attribution: logic regression đã qua nhưng độ chính xác hiện trường **UNVERIFIED**.
- Các vật thể dễ nhầm như ví, máy tính cầm tay, remote và sách: chưa có bộ negative có nhãn đủ đại diện.
- Vật cực nhỏ khoảng 37×52 px vẫn nằm ngoài khả năng ổn định của cấu hình 960 trong kiểm tra biến đổi.
- Không thêm ROI tiling/SAHI hay thay model vì dữ liệu hiện có chưa chứng minh lợi ích đủ lớn so với chi phí và rủi ro duplicate/latency.

## 9. Trạng thái vận hành

Safe-stop đã giải phóng port 8765 và xóa PID file mà không xóa session/evidence/database. App sau đó được khởi động lại bằng shortcut Desktop hiện có. Trạng thái cuối của lần đo lịch sử ngày 15/10/2026 (không xác nhận trạng thái vận hành hiện tại):

- `READY`; PID và instance định danh được giữ cục bộ;
- detector thực tế `960x960`, CUDA `cuda:0`, model execution verified;
- camera `3840x2160 · MJPG · CAP_DSHOW`, luồng hiện hoạt động;
- dashboard xác thực mở tại `http://127.0.0.1:8765/`, WebSocket `ĐÃ KẾT NỐI`;
- sau restart, overlay người/candidate điện thoại cập nhật và một review event “Phát hiện điện thoại” mức “Cần chú ý” đã xuất hiện cho Thí sinh #1; không có thao tác tự động Xác nhận/Bỏ qua;
- ứng dụng và phiên giám sát hiện tại được để chạy.
