# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, nếu chỉ có ngân sách rà năm ảnh, tôi ưu tiên chọn 5 frame sau:
1. `frame_0182.jpg` (hạng 1, thời điểm t = 72.8s, điểm score = 0.9591, U = 0.9182, A = 1.0000, 18 box mập mờ): Đứng đầu toàn bộ tập pool về điểm ưu tiên; chứa mật độ xe cao với 18 box rơi vào vùng bất định (0.15 <= conf < 0.50), cung cấp tín hiệu điều chỉnh phong phú nhất.
2. `frame_0369.jpg` (hạng 2, thời điểm t = 147.6s, điểm score = 0.9324, U = 0.9315, A = 0.8889, 43 box): Khung hình có mật độ xe rất dày đặc (43 box) ở cả hai chiều di chuyển, điểm bất định U cao vượt trội.
3. `frame_0380.jpg` (hạng 3, thời điểm t = 152.0s, điểm score = 0.9170, U = 0.9340, A = 0.8333, 40 box): Cách `frame_0369.jpg` là 4.4 giây (lớn hơn MIN_GAP_S = 2.0s), đảm bảo tính đa dạng và không trùng lặp cảnh quay.
4. `frame_0312.jpg` (hạng 7, thời điểm t = 124.8s, điểm score = 0.9100, U = 0.8199, A = 1.0000, 37 box, 18 box mập mờ): Dù xếp sau hạng 4 và 5 (`frame_0326.jpg` và `frame_0331.jpg`), tôi chọn frame này để phân bổ bớt về mốc 124s, tránh tập trung quá dày đặc vào khoảng 130s–152s.
5. `frame_0099.jpg` (hạng 8, thời điểm t = 39.6s, điểm score = 0.9063, U = 0.9460, A = 0.7778, 29 box): Nằm ở giai đoạn đầu video (t = 39.6s) với độ bất định U cao nhất nhóm (0.9460), nơi mô hình bỏ sót hoàn toàn xe lớn ở cự ly gần.
- Quyết định loại trừ do trùng lặp: Tôi không chọn `frame_0187.jpg` (hạng 10, t = 74.8s, score = 0.8995) dù điểm cao vì nó chỉ cách `frame_0182.jpg` đúng 2.0 giây. Với camera tĩnh, các phương tiện chỉ dịch chuyển một đoạn ngắn, gán nhãn cả hai sẽ lãng phí công sức mà mô hình học thêm được rất ít thông tin mới.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV / contact sheet:
- `frame_0182.jpg` (hạng 1, score 0.9591): Trên contact sheet hiển thị nhiều xe tối màu chỉ thấy chấm đèn hậu ở làn bên phải, mô hình phân vân với 18 box mập mờ (A = 1.0).
- `frame_0099.jpg` (hạng 8, score 0.9063): Có điểm bất định U = 0.9460 cao nhất trong các ảnh được chọn; trên contact sheet thấy rõ các xe chạy tới gần bị quầng đèn pha lóa xuống mặt đường khiến AI bối rối.
- `frame_0107.jpg` (hạng 14, score 0.8876): Dự đoán 33 box với 15 box mập mờ; trên contact sheet có các cặp xe đi nối đuôi sát nhau khiến AI gộp nhầm thành một box lớn.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- `frame_0372.jpg` xếp hạng 6 trong danh sách với điểm score = 0.9101 (U = 0.9202, A = 0.8333), cao hơn nhiều ảnh trong lô 12 đã chọn (như `frame_0270.jpg` score 0.8878 hay `frame_0392.jpg` score 0.8874), nhưng bị thuật toán loại bỏ vì thời điểm chụp (t = 148.8s) chỉ cách `frame_0369.jpg` (t = 147.6s) là 1.2 giây, vi phạm quy tắc khoảng cách tối thiểu MIN_GAP_S = 2.0s. Việc bỏ qua này hoàn toàn hợp lý vì trong 1.2 giây, luồng giao thông trên camera tĩnh gần như giữ nguyên vị trí, gán nhãn cả hai ảnh sẽ gây trùng lặp dữ liệu huấn luyện và lãng phí thời gian của người rà nhãn.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Điểm số bất định và cơ chế chọn mẫu chỉ phản ánh mức độ thiếu tự tin hoặc phân vân của mô hình hiện tại quanh ngưỡng quyết định conf = 0.5; nó không phải là bằng chứng bảo đảm rằng việc sửa nhãn các ảnh này chắc chắn sẽ làm mô hình fine-tune tăng điểm AP50. Sự bất định có thể xuất phát từ các yếu tố gây nhiễu thực tế (vệt đèn pha quét trên mặt đường, quầng sáng chói lóa, xe quá xa bị nhòe) – những ca này ngay cả con người cũng khó gán nhãn biên chính xác. Nếu chỉ học trên một tập nhỏ 12 ảnh phức tạp này, mô hình có thể bị co cụm phân phối (over-cautious) khiến điểm số AP50 giảm sút như kết quả thực tế ở vòng 1.
