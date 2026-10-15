# ExamGuard Vision — Final Academic Validation

**Ngày:** 15/10/2026, Asia/Bangkok (UTC+7).
**Mã nguồn gốc:** `17f737786bed8e940612c6d83dd01d78ea3caded`, nhánh `main`, cộng thay đổi cục bộ tại thời điểm đo ngày 15/10/2026. Đây là báo cáo lịch sử trước lần đồng bộ GitHub; không mô tả trạng thái commit hoặc phiên hiện tại. Git LFS đã tải đủ checkpoint. Không xóa database, recording hoặc evidence của người dùng.
**Kết luận:** **READY FOR ACADEMIC DEMO** trên workstation được kiểm chứng; không phải chứng nhận triển khai diện rộng tại trường học.

## Môi trường và phương pháp

| Hạng mục | Quan sát thực tế |
|---|---|
| OS / Python | Windows 11 build 26200, AMD64; Python 3.13.9 x64 trong `.venv` |
| CPU / RAM | AMD Ryzen 9 9950X, 16 cores / 32 threads; RAM 31.1 GiB usable |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB, driver 610.88, compute capability 12.0 |
| Framework | PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA build 12.8; có `sm_120` |
| Video / AI packages | OpenCV 5.0.0.93; Ultralytics 8.4.177; NumPy 2.5.2 |
| Camera | `Web Camera`, DirectShow index 0, định danh moniker USB VID_6210/PID_EA20 |
| Dashboard | `http://127.0.0.1:8765/`, bind `127.0.0.1`; không dùng cổng 8000 đang thuộc Docker/WSL |

Không lấy số liệu ASUS/RTX 3050 làm benchmark mới. Tách ba loại chứng cứ: webcam thật; model forward/tensor tổng hợp; fixture/mock regression. Đo camera độc lập sau 15 frame warmup trong khoảng 10 giây. Forward CUDA có synchronize; xác minh device của **AutoBackend đang thực thi** thay vì object checkpoint CPU mà Ultralytics vẫn giữ. Phiên thật có capture background riêng và latest-frame slot; nhịp AI giới hạn ở startup bằng số đo, không tạo quan sát bù.

Đo latency là thời gian từ lúc **frame giải mã được timestamp bằng phần mềm** đến kết quả pipeline, bao gồm tuổi frame chờ consumer. Không đo thời gian từ phơi sáng sensor, không phải event-onset-to-human-alert. p50/p95 lấy trên cửa sổ tối đa 600 kết quả gần nhất; không bao gồm thời gian debounce cần thiết để một hành vi đủ điều kiện mở sự kiện.

## Kết quả đo

| Phép đo | Kết quả | Phạm vi |
|---|---:|---|
| Camera acquisition độc lập | **3840×2160, MJPG, CAP_DSHOW** | Webcam vật lý, không chạy AI cùng lúc |
| Capture độc lập | **297 frames / 10.0279 s = 29.6175 FPS** | Warmup 15 frame; interval p50 33.65 ms, p95 34.49 ms |
| Capture khi chạy toàn pipeline | **26.48 FPS trung bình** | 23 mẫu trong 46.01 s sau warmup |
| AI processed FPS | **16.26 FPS trung bình** | Cùng cửa sổ; Auto giới hạn nhịp, không phải full 4K neural input |
| Latency p50 / p95 | **56.68 / 73.49 ms** | Cửa sổ kết quả phần mềm cuối của phép quan sát |
| AI frame count / intentional skip | **758 / 477** trong cửa sổ | Bỏ có chủ đích ~38.6%, không phải bằng chứng camera mất 477 frame |
| Queue depth cao nhất | **0**, capacity **2** | Latest-frame capture slot riêng giữ tối đa một frame |
| RAM ứng dụng (RSS) | **2382.9–2409.8 MiB** | Cửa sổ ngắn; không suy thành bảo đảm không leak dài hạn |
| Peak GPU memory | **538.5 MiB** | `torch.cuda.max_memory_allocated`, không phải tổng VRAM driver/context |
| CPU trung bình | **1590%** theo psutil một-core basis, ~**49.7%** trên 32 logical CPUs | Gồm xử lý CPU/decode/crop/JPEG của process |

Nguồn: `reports/adaptive-workstation.json`, `reports/final-live-observation.json`. Đây là hai phép đo khác nhau, không gán 29.62 FPS camera thành AI FPS. Phiên đầu trước khi chốt cadence đạt khoảng 26 FPS AI nhưng capture toàn pipeline khoảng 26 FPS; không dùng làm số quảng cáo cho cấu hình cuối.

Sau lần restart cuối chỉ sửa telemetry UI/API, ảnh kiểm chứng lúc **16:04:40 UTC+7** hiển thị **26.1 capture FPS / 15.1 AI FPS**, p50/p95 **57.9/80.7 ms**, GPU RTX 5070 12,227 MiB, CAP_DSHOW MJPG. Cadence được reuse từ cấu hình đã kiểm chứng và quan sát phiên trước; chênh lệch theo tải/cảnh là bình thường. Ảnh cục bộ: `reports/screenshots/final-system-telemetry.png`. Không upload footage ra ngoài.

## CUDA, checkpoint và tính nhất quán

- **PASS:** SHA-256 7/7 checkpoint khớp `CERTIFIED_HASHES`, không còn LFS pointer. Checkpoint/taxonomy/threshold giữ nguyên.
- **PASS:** detector YOLO26m và macro Stage 1.5 thực thi `predict()` trên `cuda:0`; posture output `[1,4]`; headpose outputs `[1,66]` và `[1]`, hữu hạn và đúng device.
- **PASS:** cả bốn nhánh cũng forward thật trên CPU với cùng checkpoint. Đây là xác nhận backend fallback, không phải benchmark máy không GPU riêng biệt.
- **FAIL đã xử lý:** cài dependencies chồng lấn ban đầu để lại PyTorch file trộn phiên bản. Đã gỡ/cài lại cặp official CUDA sạch; import, kernel, full forwards và `pip check` PASS.
- **FAIL đã xử lý:** TF32 do Ultralytics bật gây lệch posture logits 0.074 trên tensor kiểm tra; không đổi lớp argmax trong mẫu đó. Auto tắt TF32 để giữ FP32. Kiểm tra lại: max posture error **0.00001717**, yaw difference **0.0000248°**, cùng class; `rtol=1e-3, atol=1e-3` cho logits và `atol=0.1°` cho yaw. Đây là numerical contract trên tensor tổng hợp, không thay thế đánh giá accuracy với ảnh lớp học gắn nhãn.

## Kiểm thử và phân loại

| Nhóm | Kết quả / chứng cứ |
|---|---|
| Runtime baseline theo hướng dẫn repository | **388 PASS, 9 FAIL, 7 SKIP, 19 deselected**, 69.99 s; giữ log/XML gốc |
| 6 FAIL do dữ liệu offline | 4 acquisition_v3 và 2 stage1_5 leakage checks thiếu `datasets/`; xếp **BLOCKED** do dữ liệu không phân phối, không sửa dữ liệu giả |
| 3 FAIL runtime/test assumptions | Backup MAX_PATH Windows được sửa; test preflight không được cấm tên desktop thật; severity không-event vốn là SAFE. Cả ba kiểm tra lại **PASS** |
| Focused security/integrity/lifecycle correction | **48 PASS** (`focused-final.xml`), gồm auth/RBAC, timestamp, encryption, session-bound restart, backup và adaptive tests |
| Adaptive/capture/scheduler/launcher contracts | **47 PASS** (`adaptive-final.xml`), gồm CPU/CUDA failure mocks, identity, rejected FOURCC/resolution, DPAPI, source factory, queue, batching và launcher syntax/foreign port/stale PID |
| API/dashboard sau telemetry changes | **20 PASS, 4 SKIP** (`telemetry-ui-final.xml`); skip là artifact lịch sử chưa có |
| API + adaptive supplemental | **38 PASS, 4 SKIP** (`api-final.xml`), trước bổ sung runtime_details UI cuối |
| CUDA/CPU + physical acquisition | **PASS** (`adaptive-workstation.json`) |
| Cache reuse | **PASS**, fingerprint live tương thích; không gọi capability/physical probe. Lần restart cuối thực tế ghi `reused_verified_settings: true` |

Không cộng các suite chồng lấn thành một tổng test giả. Baseline có collection error ban đầu ở `test_dataset_tooling.py` vì module `training` không phân phối; sau đó loại đúng module offline này. Các exclusion theo `-k` trong tài liệu cũ vẫn lọt 6 test dataset nói trên; lệnh tái lập bên dưới loại module rõ ràng.

7 SKIP gốc: 4 artifact `desktop_local_live` lịch sử chưa có; 2 raw AFLW/headpose dataset schema tests; 1 sample `samples/sample_exam.mp4` không tồn tại ở đường dẫn legacy. Fixture có sẵn `tests/fixtures/sample_exam.mp4` vẫn được sử dụng trong runtime tests liên quan. Không tạo artifact/dataset giả để đổi SKIP thành PASS.

Regression đã chạy bao gồm detector contracts, ByteTrack mapping/isolation, crop scheduling, timestamp, fusion/missing cues, EventEngine, anti-double-counting/single-frame scoring, evidence lifecycle, persistence, review Confirm/Dismiss, REST/WebSocket/auth và multi-camera software tests. Không thay luật sự kiện để chữa assertion lỗi thời.

## One-click, integration và an toàn

**PASS trên máy thật:** launcher chờ model readiness; browser mặc định Brave tự mở; tài khoản được chủ máy tự tạo; phiên giám sát thật mở camera, AI tạo track/event, evidence thumbnail và WebSocket updates hiển thị. Confirm/Dismiss được kiểm tra bằng regression fixture; không tự quyết định các sự kiện vật lý của người dùng.

**PASS:** duplicate start nhận đúng instance và giữ PID; safe stop qua script mà shortcut gọi, đóng session events/evidence, camera release, backend dừng và port 8765 được nhả; khởi động lại qua Windows Shell thực thi `.lnk`; đăng nhập lại và mở phiên mới. Không có quyền native desktop automation để mô phỏng chính xác thao tác chuột double-click; đã chạy chính shortcut qua Shell, không chỉ kiểm tra file shortcut tồn tại.

**PASS:** PID ownership kiểm tra command/root/create-time; không có unrestricted taskkill. Cổng 8000 Docker/WSL không bị thay đổi. Binding cuối chỉ `127.0.0.1:8765`; anonymous `/api/cameras` trả **401**; `/health` và `/ready` chỉ cung cấp diagnostics tối thiểu. DB `PRAGMA integrity_check` trả `ok`, chỉ một monitoring session ACTIVE; session đã dừng ứng dụng được ghi INTERRUPTED.

**PASS:** không có API authentication bypass, default password, camera credential ghi vào tracked YAML hoặc upload footage. URL RTSP tùy chọn được DPAPI bảo vệ theo CurrentUser; regression round-trip/traversal PASS. Evidence encryption/path protection và human review integrity được giữ.

## Hoàn tất và giới hạn

| Gate | Trạng thái |
|---|---|
| Repo, isolated runtime, 7 checkpoints | PASS |
| RTX 5070 detected, CUDA model forwards | PASS |
| Auto, capability selection, cache, ASUS reference preserved | PASS |
| Physical webcam, actual 4K MJPG, separate measured rates | PASS |
| Full pipeline, event/evidence/persistence | PASS |
| Dashboard, REST, WebSocket, authentication | PASS |
| Launcher, Desktop `.lnk`, safe stop/restart | PASS, Shell invocation thay cho thao tác chuột native |
| Targeted regression, academic docs | PASS với các SKIP/BLOCKED dataset được nêu rõ |
| Final application | READY, monitoring ACTIVE; dashboard ở chế độ GIÁM SÁT và được giữ mở |

**NOT TESTED vật lý:** rút USB giữa phiên; camera bị ứng dụng khác chiếm; GPU OOM thật; card AMD/Intel/DirectML; PC khác; nhiều camera vật lý; RTSP/CCTV trường học; mất mạng dài; camera geometry/seat calibration; long soak. Một số lỗi có coverage mock và recovery có giới hạn, không được gán thành đã thử trên thiết bị thật. CUDA model failure/OOM chưa xác minh được fallback an toàn sẽ BLOCKED thay vì công bố READY sai.

Chế độ này dành cho demo học thuật cục bộ có giám thị; tỷ lệ bỏ frame AI là chủ đích, độ chính xác hành vi ngoài operating envelope chưa được chứng nhận. CPU còn tải đáng kể do capture/preprocessing/JPEG; 4K acquisition khi chạy đầy đủ chưa giữ 30 FPS như phép đo độc lập. Chưa có căn cứ công bố production-ready, universal GPU compatibility, 4K30 full-pipeline inference hoặc so sánh mới với RTX 3050.

## Tái lập

Không chạy các test lifecycle/stale PID hoặc probe vật lý trong lúc người dùng đang có phiên hoạt động. Dừng ứng dụng trước suite rộng vì preflight cũng mở camera.

```powershell
# Runtime baseline, loại các module offline rõ ràng và các test điều khiển app đang chạy
.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_dataset_tooling.py --ignore=tests/test_acquisition_v3.py --ignore=tests/test_v4b_crop_dataset.py --ignore=tests/test_stage1_5_pipeline.py --ignore=tests/test_launcher_lifecycle.py -k "not dataset and not port_collision_safety_behavior and not stale_pid_stop_handling"
# Targeted adaptive và launcher contracts: chạy khi ứng dụng đã dừng
.venv\Scripts\python.exe -m pytest -q tests/test_adaptive_runtime.py tests/test_windows_launcher.py tests/test_video_source.py tests/test_async_capture_and_latest_frame.py
# Physical 4K acquisition + CPU/CUDA forwards: ứng dụng phải đã dừng
.venv\Scripts\python.exe tools/validation/validate_adaptive_workstation.py --seconds 10
# Sau khi người dùng đăng nhập và bắt đầu phiên: chỉ đọc telemetry, không mở thêm camera
.venv\Scripts\python.exe tools/validation/observe_live_runtime.py --seconds 45
```

Hướng dẫn vận hành: [universal-windows-setup.md](universal-windows-setup.md). Kiến trúc: [adaptive-runtime.md](adaptive-runtime.md). Logs/JSON/XML và screenshot chứa dữ liệu máy thật được giữ cục bộ trong `reports/`/`.runtime/`, đều bị Git ignore; nguồn mã đã thay đổi được liệt kê trong `reports/final-file-inventory.txt`.
