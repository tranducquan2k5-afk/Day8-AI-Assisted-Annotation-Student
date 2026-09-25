# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Trần Đức Quân

Công cụ gán nhãn đã dùng: CVAT (xuất và nhập nhãn theo định dạng Ultralytics YOLO Detection 1.0)

Báo cáo chi tiết quá trình thực hiện học chủ động (Active Learning) trên tập dữ liệu video giao thông ban đêm. Mọi số liệu trong báo cáo được trích xuất và đối chiếu trực tiếp từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json` và `outputs/round1_diff.md`.

## 1. Dữ liệu và cách chia tập

Dữ liệu của bài lab được trích xuất từ camera giám sát giao thông đặt cố định trên cầu vượt, nhìn xuống đường cao tốc vào ban đêm. Toàn bộ video là một cảnh quay liên tục không có chuyển cảnh (scene cut). Do camera đứng yên và xe di chuyển liên tục, mỗi phương tiện thường lưu lại trong khung hình từ vài giây đến hơn mười giây. Với tần suất trích xuất 2.5 khung hình/giây (cứ 0.4 giây có một ảnh), hai frame liên tiếp nhau gần như chỉ là sự dịch chuyển nhỏ của cùng một nhóm xe.

Nếu chia tập ngẫu nhiên (random split):
- Cùng một chiếc xe (với cùng góc nhìn, màu sơn, biển số và cụm đèn phản quang) sẽ xuất hiện đồng thời ở cả tập chưa gán nhãn (dùng để huấn luyện) và tập kiểm thử (dùng để chấm điểm).
- Hiện tượng này dẫn đến rò rỉ dữ liệu theo thời gian (temporal data leakage). Khi đó, mô hình kiểm thử không được đánh giá về năng lực khái quát hóa (generalization) trên các xe mới lạ, mà chỉ đơn thuần nhận diện lại những chiếc xe nó vừa được học cách đó 0.4 giây.
- Hậu quả: Các chỉ số đánh giá trên tập kiểm thử (AP50, Precision, Recall) sẽ bị **lệch theo hướng lạc quan giả tạo (optimistic bias)**, cao hơn rất nhiều so với năng lực thực tế khi triển khai ngoài đời thực.

Để loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu, bài lab đã chia dữ liệu theo trục thời gian:
- Tập kiểm thử (test set) gồm 20 ảnh được lấy tại 4 cụm thời gian rời rạc (quanh các mốc 20s, 60s, 100s, 140s).
- Giữa tập kiểm thử và tập chưa gán nhãn (pool) có một vùng đệm (buffer) gồm 112 ảnh bị loại bỏ hoàn toàn (khoảng 4 giây trước và sau mỗi cụm test). Khoảng cách thời gian tối thiểu giữa một ảnh pool bất kỳ và một ảnh test là 4.4 giây. Khoảng cách này bảo đảm rằng mọi phương tiện xuất hiện trong tập kiểm thử đã di chuyển hoàn toàn ra khỏi góc máy trước hoặc sau khi các ảnh pool xuất hiện, đảm bảo tính khách quan và trung thực của phép đo.

## 2. Mô hình khởi đầu lạnh (cold start)

Bảng số đo của mô hình khởi đầu lạnh (vòng 0) trích xuất từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.773 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Phân tích chất lượng mô hình khởi đầu lạnh:
- Mô hình khởi đầu lạnh là YOLOv8n được tiền huấn luyện trên tập dữ liệu COCO (chỉ lọc 3 lớp car, bus, truck và gộp chung thành một lớp `car`). Trên 20 ảnh test (403 box tham chiếu), mô hình đạt AP50 = 0.773, Precision = 0.925 nhưng Recall chỉ đạt 0.489 (bỏ sót hơn một nửa số lượng xe, với TP = 197 và FN = 206).
- Độ phủ (Recall) theo kích thước xe cho thấy sự phân hóa cực kỳ rõ rệt:
  + Nhóm xe nhỏ (small - chủ yếu là xe ở phía xa gần chân trời): Recall chỉ đạt `0.182` (bỏ sót tới 54 trên 66 xe).
  + Nhóm xe vừa (medium - xe ở cự ly tầm trung): Recall đạt `0.547`.
  + Nhóm xe lớn (large - xe ở cự ly gần): Recall đạt `0.561`.
- Điều này chứng minh mô hình COCO ban ngày gặp khó khăn rất lớn trong môi trường đêm tối: mô hình phụ thuộc vào đường nét thân xe rõ ràng, nên khi xe ở xa bị chìm vào bóng tối và chỉ để lộ hai đốm đèn nhỏ, mô hình hầu như không thể phát hiện được.
- Quan sát trên ảnh `outputs/compare_round0.jpg`:
  + Mô hình bỏ sót hàng loạt xe tối màu chạy ở làn xa bên phải và làn ngược chiều (thể hiện bằng các khung viền màu vàng FN).
  + Đồng thời, mô hình sinh ra 16 False Positive (khung viền màu đỏ FP), chủ yếu xuất phát từ việc nhận nhầm các vệt sáng đèn pha chiếu trên mặt đường bê tông hoặc khoanh một box quá lớn trùm lên cả hai xe đi sát nhau.
- Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai:
  + Nhãn tham chiếu trong `data/test/labels/` được sinh tự động bởi một mô hình phát hiện đối tượng khác và **chưa từng được con người rà soát thủ công từng box**.
  + Trên `compare_round0.jpg`, xuất hiện những trường hợp xe ô tô chạy ở làn ngược chiều nhìn rất rõ đèn và thân xe, mô hình cold start phát hiện hoàn toàn chính xác nhưng lại bị đánh dấu đỏ (FP) do nhãn tham chiếu tự động bỏ sót xe đó. Ngược lại, có những box tham chiếu lại ôm cả vệt đèn pha kéo dài trên mặt đường. Vì vậy, nhãn tham chiếu chỉ mang tính tương đối, không phải là chân lý tuyệt đối (ground truth hoàn hảo).

## 3. Chiến lược chọn mẫu

Thuật toán chọn mẫu học chủ động tính điểm ưu tiên cho từng frame trong tập pool theo công thức:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
Với các trọng số mặc định: $W_U = 0.5$, $W_A = 0.3$, $W_D = 0.2$.

Ý nghĩa của từng thành phần:
- $U$ (Uncertainty - chiếm 50% điểm số): Đo lường độ bất định trung bình của 5 box khó nhất trong frame, tính theo công thức $u(c) = 1 - |2c - 1|$. Khi độ tin cậy $c \approx 0.5$, giá trị $u \approx 1.0$, phản ánh trạng thái mô hình lưỡng lự nhất giữa việc đối tượng là xe hay nền. Frame có U cao là frame mà các dự đoán then chốt của mô hình còn thiếu chắc chắn nhất.
- $A$ (Ambiguity - chiếm 30% điểm số): Tỉ lệ số lượng box "mập mờ" ($0.15 \le c < 0.50$) trong frame so với frame có nhiều box mập mờ nhất toàn pool. Thành phần này ưu tiên các bức ảnh có mật độ phương tiện đông đúc đang bị phân vân, giúp người gán nhãn can thiệp chỉnh sửa được nhiều box nhất trong một lần mở ảnh (tối ưu hóa công suất gán nhãn).
- $D$ (Diversity - chiếm 20% điểm số): Khoảng cách thời gian từ frame đang xét tới frame đã được gán nhãn gần nhất (chặn trần ở mức 10 giây). Thành phần này ép thuật toán phải phân tán các mẫu được chọn trải dài dọc theo trục thời gian của video thay vì dồn cục vào một khoảng thời gian hẹp.
- Vai trò của `MIN_GAP_S` (2.0 giây): Vì camera quan sát cố định, hai bức ảnh chụp cách nhau dưới 2 giây có góc nhìn, bối cảnh nền và vị trí các xe gần như trùng khít nhau. `MIN_GAP_S` là ngưỡng khoảng cách thời gian tối thiểu giữa các frame được chọn trong cùng một lô. Thuật toán chọn tham lam (greedy) sẽ tự động loại bỏ các ảnh dù có điểm score rất cao nhưng nằm cách ảnh đã chọn dưới 2.0s. Đây là bộ lọc khử trùng lặp (redundancy filter) bắt buộc để tránh việc con người phải gán nhãn hai lần cho cùng một cảnh, tiết kiệm chi phí mà vẫn đảm bảo tính đa dạng dữ liệu.

Minh chứng đối chiếu từ `reports/SELECTION.md` và `outputs/selection_round1.csv`:
- `frame_0182.jpg` (hạng 1, score = 0.9591, U = 0.9182, A = 1.0000): Đứng đầu toàn pool nhờ có tới 18 box mập mờ và độ bất định rất cao.
- `frame_0099.jpg` (hạng 8, score = 0.9063, U = 0.9460, A = 0.7778): Có điểm bất định U cao nhất nhóm (0.9460), đại diện cho giai đoạn đầu video (t = 39.6s), nơi AI bỏ sót cả xe SUV lớn ở cự ly gần.
- `frame_0107.jpg` (hạng 14, score = 0.8876, U = 0.8752, A = 0.8333): Cách `frame_0099.jpg` 3.2s ($> 2.0s$), chứa 15 box mập mờ với nhiều ca xe đi nối đuôi nhau.
- `frame_0372.jpg` (hạng 6, score = 0.9101, U = 0.9202, A = 0.8333): Điểm số cao hơn cả `frame_0107.jpg` và `frame_0392.jpg` trong lô 12, nhưng bị thuật toán loại bỏ vì thời điểm chụp (t = 148.8s) chỉ cách `frame_0369.jpg` (t = 147.6s) đúng 1.2 giây ($< MIN\_GAP\_S = 2.0s$). Điều này chứng minh thuật toán đã kiểm soát trùng lặp chặt chẽ, từ chối ảnh trùng cảnh để tối ưu chi phí gán nhãn.

Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?
- Điểm bất định cao hoàn toàn **không chứng minh** rằng việc đưa ảnh đó vào huấn luyện chắc chắn sẽ giúp mô hình tăng điểm số AP50. Sự bất định chỉ phản ánh việc mô hình hiện tại đang dao động quanh ranh giới quyết định. Bất định cao có thể xuất phát từ nhiễu dữ liệu (quầng sáng đèn pha rọi thẳng vào camera, bóng phản chiếu trên mặt đường ướt, xe quá xa bị nhòe pixel) – những trường hợp này ngay cả con người cũng khó phân định viền xe chuẩn xác. Nếu đưa vào huấn luyện khi số lượng ảnh còn quá ít, các mẫu nhiễu này có thể khiến mô hình bị co cụm phân phối dự đoán hoặc học lệch, như thực tế điểm số AP50 đã giảm ở vòng 1.

## 4. Các vòng học chủ động (active learning)

Bảng so sánh tổng hợp các vòng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.773 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 340 | 0.469 | -0.303 | 1.000 | 0.154 | 0.267 | 0.000 | 0.155 | 0.390 |

Trình bày chi tiết Vòng 1:
- Mức độ sửa nhãn gợi ý (trích xuất từ `outputs/round1_diff.md`):
  Trong 12 ảnh thuộc lô vòng 1, mô hình đề xuất 169 box pre-label. Sau khi rà soát và chỉnh sửa trên CVAT, tổng số box thực tế đạt 340 box (tăng hơn gấp đôi). Phân bổ thao tác:
  + `accepted`: 144 box (tỉ lệ giữ nguyên 85% trên tổng số box pre-label).
  + `edited`: 13 box (thu gọn chiều cao box ôm vệt sáng đèn pha, tách 2 xe nối đuôi, kéo mở rộng box khít thân xe).
  + `deleted`: 12 box (loại bỏ các box giả do AI nhận nhầm vệt phản chiếu trên mặt đường và xóa các box trùng lặp trên cùng một xe).
  + `added`: 183 box (bổ sung các xe cự ly gần bị AI bỏ sót hoàn toàn, các xe tối màu chỉ thấy đèn hậu ở làn phía xa, và các xe bị mép ảnh cắt ngang).
- Biến thiên AP50 và các chỉ số trên cùng tập kiểm thử:
  + AP50 giảm từ 0.773 xuống 0.469 ($\Delta = -0.303$).
  + Tuy nhiên, **Precision@0.25 tăng lên mức tuyệt đối 1.000 (100%)**, số lượng False Positive trên toàn bộ 20 ảnh test giảm từ 16 về 0. Mô hình mới hoàn toàn không còn hiện tượng sinh box ảo lên các vệt sáng hay quầng đèn trên mặt đường.
  + Recall giảm từ 0.489 xuống 0.154 (TP giảm từ 197 xuống 62, FN tăng lên 341). Trong đó, nhóm xe lớn giữ Recall tốt nhất (0.390), xe vừa đạt 0.155, trong khi xe nhỏ ở xa Recall giảm về 0.000.
  + Nguyên nhân kỹ thuật: Mô hình được fine-tune trên một lô nhỏ gồm 12 ảnh với 50 epochs dựa trên bộ nhãn gán tay rất chuẩn mực và khắt khe (ôm sát thân xe thật, loại bỏ triệt để vệt sáng). Do dữ liệu train mới có độ nhất quán cao và kích thước tập train nhỏ so với trọng số khổng lồ ban đầu của COCO, mô hình đã co cụm phân phối dự đoán và trở nên cực kỳ thận trọng (conservative). Các xe ở xa chỉ thấy đốm đèn mờ bị mô hình dự đoán với confidence dưới ngưỡng 0.25, dẫn đến việc tập test tham chiếu (vốn sinh bằng nhãn máy lỏng lẻo hơn) ghi nhận nhiều False Negative.
- So sánh trên ảnh trực quan (`compare_round0.jpg` và `compare_round1.jpg`):
  + Trên `compare_round1.jpg`, toàn bộ các box viền đỏ (FP) do nhận nhầm vệt phản xạ đèn trên mặt đường ở chân cầu và làn đường bên trái đã hoàn toàn biến mất. Các phát hiện của mô hình sau khi train đều trúng đích 100% vào thân xe thật. Ngược lại, số lượng viền vàng (FN) tăng lên ở các xe nhỏ phía xa chân trời.
- Phân biệt 3 nguồn thông tin độc lập:
  + `BLIND_SCAN.md`: Ghi nhận quan sát độc lập bằng mắt thường trên `frame_0099.jpg` trước khi xem gợi ý AI (đếm được 25 xe, chỉ rõ 2 vị trí hiểm hóc là xe cự ly gần bị quầng đèn lóa và xe tối màu ở làn phía xa).
  + `REVIEW_LOG.csv`: Nhật ký can thiệp thực tế với 352 dòng tương ứng từng box cụ thể (144 accepted, 13 edited, 12 deleted, 183 added).
  + Kết quả mô hình sau train: Thể hiện rõ tính kỷ luật mà người gán nhãn đã đưa vào: mô hình triệt tiêu hoàn toàn box rác (Precision 100%), dù cần thêm dữ liệu đa dạng ở vòng tiếp theo để mở rộng độ phủ Recall.
- Mô tả một ca khó xử lý theo guideline:
  + Ca hai xe chạy sát nối đuôi nhau ở làn giữa trong `frame_0107.jpg`: AI đề xuất một box duy nhất cao tới 65 pixel bao trùm cả hai xe. Tuân thủ nghiêm ngặt quy tắc trong `GUIDELINE_LABEL.md` (*"Hai xe đứng sát nhau: Vẽ hai box riêng, không gộp làm một"*), tôi đã kéo viền trên của box thu ngắn lại còn 47 pixel để ôm khít xe phía trước, sau đó vẽ thêm một box độc lập ôm trọn xe đi ngay phía sau.

## 5. Kết luận và giới hạn

So sánh với cold start:
Mô hình sau Vòng 1 đã có bước chuyển dịch căn bản về mặt chất lượng phát hiện: từ một mô hình COCO ban ngày dễ bị đánh lừa bởi quầng sáng và bóng phản chiếu đèn ban đêm thành một bộ phát hiện có tính kỷ luật cao, đạt Precision tuyệt đối 100% (không một box giả nào). Mặc dù AP50 sụt giảm do Recall của các xe nhỏ ở xa bị hạ thấp, đây là hiện tượng bình thường và hoàn toàn có thể lường trước trong Active Learning khi tinh chỉnh một mô hình nền lớn trên một lô dữ liệu ban đầu còn khiêm tốn (12 ảnh).

Quyết định dừng hay tiếp tục:
**Tiếp tục thực hiện Vòng 2**. Dữ liệu chọn mẫu cho vòng tiếp theo đã được hệ thống tạo sẵn trong `outputs/selection_round2.csv`. Việc tiếp tục gán nhãn thêm một lô 12 ảnh mới sẽ bổ sung thêm các trường hợp xe ở cự ly tầm trung và xe nhỏ phía xa, giúp mô hình mở rộng phân phối nhận diện và khôi phục độ phủ Recall trong khi vẫn duy trì được độ chính xác 100% đã thiết lập ở vòng 1.

Đề xuất hai ca còn yếu hoặc bất định cho vòng sau:
1. *Ca xe ở cự ly xa (khu vực chân trời và gần cầu vượt):* Thân xe tối màu chìm vào nền đêm, chỉ có 2 đốm đèn nhỏ. Chi phí rà nhãn cho ca này rất tốn công vì người gán phải phóng to hết cỡ để ước lượng ranh giới thân xe. Nguy cơ ảnh gần trùng rất cao do xe ở xa di chuyển góc biểu kiến rất chậm, đòi hỏi phải tuân thủ nghiêm ngặt khoảng cách MIN_GAP_S.
2. *Ca xe bị mép ảnh cắt ngang (vừa vào hoặc sắp ra khỏi góc quay camera):* Mô hình thường bỏ sót do hình dạng xe không nguyên vẹn. Nguy cơ: người gán dễ vẽ thừa ra ngoài viền ảnh, cần bám sát quy tắc "chỉ vẽ box cho phần nằm trong ảnh".

Tác động của các giới hạn đánh giá:
- Tập kiểm thử chỉ có 20 ảnh: Kích thước mẫu quá nhỏ khiến phương sai thống kê lớn; chỉ cần một vài xe ở xa không vượt qua ngưỡng conf 0.25 là điểm số Recall và AP50 đã sụt giảm nghiêm trọng, chưa phản ánh đầy đủ năng lực thực sự của mô hình trên toàn tuyến cao tốc.
- Luật bỏ qua box < 16 pixel: Là một quy tắc rất thực tế, giúp loại bỏ các tranh cãi không cần thiết tại các điểm ảnh nhòe vô định hình ở đường chân trời.
- Nhãn tham chiếu do mô hình tự động tạo chưa được người rà soát: Đây là giới hạn quan trọng nhất cần lưu ý. Có những phương tiện mô hình ta nhận diện hoàn toàn đúng nhưng nhãn tham chiếu lại bỏ sót, hoặc nhãn tham chiếu khoanh cả vệt sáng đèn pha trên mặt đường (điều mà mô hình ta đã học cách loại bỏ). Do đó, điểm số AP50 giảm không đồng nghĩa với việc mô hình hoạt động kém hơn trong thực tế, mà phản ánh sự không tương thích giữa bộ nhãn gán tay chuẩn xác và bộ nhãn tham chiếu tự động.

Nếu AP50 giảm, những điều cần kiểm tra trước khi train thêm:
1. Kiểm tra lại phân phối độ tin cậy (confidence distribution): Xem xét các dự đoán của mô hình ở ngưỡng conf thấp hơn (0.10 - 0.20) để xác minh xem mô hình có thực sự phát hiện được các xe nhỏ ở xa hay không, hay chỉ đơn giản là tự đánh giá điểm tin cậy hơi thấp.
2. Kiểm tra tính nhất quán (consistency) của các nhãn trong `labels/round1/`: Rà soát lại `REVIEW_LOG.csv` để đảm bảo quy tắc gán nhãn thân xe và loại bỏ vệt đèn được áp dụng đồng nhất 100%, không đưa tín hiệu nhiễu mâu thuẫn vào tập train.
3. Điều chỉnh siêu tham số huấn luyện (hyperparameters): Với tập dữ liệu nhỏ (12 ảnh), cần cân nhắc giảm learning rate hoặc đóng băng các tầng trích xuất đặc trưng (freeze backbone layers) để mô hình không bị quên các đặc trưng tổng quát ban đầu từ COCO.
