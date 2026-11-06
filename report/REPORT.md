# Báo cáo bài thực hành Ngày 1: Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime:** T4 GPU Colab

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.15.0+cu132 (chạy trên CPU) / Ultralytics 8.4.145 (đúng bản ghim, assert phiên bản PASS)

**Checkpoint:** `yolo11n-cls.pt` (sha256 `c62d41bf9625…`), `yolo11n.pt` (sha256 `0ebbc80d4a76…`), `yolo11n-seg.pt` (sha256 `55ed65c56c91…`); cả ba PASS checksum

**Ảnh mẫu:** `traffic` (COCO 210273, sha256 `3ec23de63592…`), `kitchen` (COCO 397133, sha256 `09e1d25c75f7…`), `dining` (COCO 166918, sha256 `a7f8457580a2…`); cả ba PASS checksum

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (không đổi; ô so sánh của notebook chạy thêm `0.20` và `0.60`)

**Thay đổi so với notebook nguồn:** Không đổi mã nguồn, checkpoint hay threshold. Môi trường chạy chặn host `images.cocodataset.org`, nên ba ảnh được lấy từ bản lưu trữ COCO val2017 công khai của Ultralytics (`coco2017val.zip` trên GitHub Releases) rồi đặt sẵn vào `day1_lab_images/`. Ô tải ảnh của notebook vẫn kiểm tra SHA-256 và báo PASS cho cả ba ảnh, nên đầu vào trùng byte với bản gốc. Ô validation cuối báo `PASS: đủ 3 JSON, 3 PNG, attribution và instance_id duy nhất.` Một số số liệu phụ không có sẵn trong output notebook được tính thêm bằng script chỉ đọc trên cùng ảnh, checkpoint và JSON: tổng score top-5, score các box thêm ở ngưỡng 0.20, diện tích polygon, số pixel chồng giữa các mask và độ cao dải đen của ảnh `traffic`. Các script này không ghi đè hay thay đổi output.


### Từ cần nhớ

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số "tin" của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`. Ảnh `sample_id = traffic`, dòng `rank = 1`.

**1. Record hạng 1:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K (ảnh 640 x 428, `model_file`=`yolo11n-cls.pt`). Bốn hạng tiếp theo: minibus 0.164285, police_van 0.085848, recreational_vehicle 0.054110, streetcar 0.048193.

**2. Cả bức ảnh hay một vật thể?** Record nói về cả bức ảnh. Bằng chứng: record không có `bbox_xyxy` hay `polygon_xy`, chỉ có lớp, hạng và score gắn với `sample_id`; năm score top-5 cộng lại bằng 0.863351, phần còn lại (khoảng 0.137) chia cho 995 lớp khác, tức đây là một phân phối xác suất trên toàn bộ 1000 lớp cho một ảnh. Trên cùng ảnh này, model detection tìm thấy 14 `car`, 8 `bus` và 4 `person` (file `detection_predictions.json`), nhưng record phân loại không chỉ vào chiếc xe nào cả; nhãn "cab" là kết luận chung cho toàn ảnh.

**3. Class list đến từ đâu?** Đi kèm taxonomy ImageNet-1K của checkpoint, không do model tự nghĩ ra. `result.names` của `yolo11n-cls.pt` có đúng 1000 tên, phần tử số 0 là `tench` (một loài cá), tên viết thường có gạch dưới như `police_van`, `recreational_vehicle`. ImageNet-1K không có lớp "bus" chung chung, chỉ có `minibus` (654), `school_bus` (779), `trolleybus` (874), nên model này không thể trả về "bus" dù ảnh đầy xe buýt; ngược lại checkpoint COCO-80 có lớp `bus` (id 5). Model chỉ chọn được trong danh sách nó được huấn luyện.

**4. Vì sao ghi cả `class_id`, `class_name`, `taxonomy_name`?** Vì một mã số chỉ có nghĩa trong một taxonomy: id 0 là `tench` trong ImageNet-1K nhưng là `person` trong COCO-80. Cùng một khái niệm cũng có id và cách viết khác nhau: `dining_table` là id 532 ở ImageNet (hạng 2 của ảnh `kitchen`) nhưng `dining table` là id 60 ở COCO-80. Chỉ ghi tên thì dễ nhầm do khác chính tả, từ đồng nghĩa (cab và taxi) hoặc khác phiên bản danh sách; chỉ ghi id thì người review không đọc được. Ghi đủ ba trường giúp kiểm tra chéo và ánh xạ giữa các bộ dữ liệu.

**5. Guideline cần nói gì khi ảnh có nhiều vật/chủ đề?** Cần định nghĩa rõ (a) nhãn mô tả gì: cảnh tổng thể hay vật chính; (b) quy tắc chọn vật chính, ví dụ lớp chiếm diện tích nhìn thấy lớn nhất trong vùng nội dung, kèm thứ tự ưu tiên khi hòa; (c) định nghĩa và ví dụ phân biệt các lớp gần nhau (vd. `cab` chỉ khi thấy dấu hiệu taxi như đèn hoặc chữ TAXI; `minibus`, `trolleybus`, `school_bus` khác nhau thế nào); (d) bỏ qua phần đệm ảnh: ảnh `traffic` có hai dải đen 57 px ở trên và dưới (nội dung nằm từ hàng 57 đến hàng 370), dải đen không được ảnh hưởng tới nhãn; (e) khi không có lớp nào phù hợp thì escalation, không đoán. Quan sát bằng mắt: cảnh chủ yếu là xe buýt xanh và ô tô con; ở độ phân giải này tôi không xác định được xe nào chắc chắn là taxi, nên top-1 `cab` đáng nghi. Tôi ghi nhận nghi vấn này chứ không sửa prediction.

**6. Vì sao không lấy `score` làm ground truth?** Score là đầu ra của chính model, chưa được con người xác nhận theo guideline, và không đo độ đúng. Bằng chứng trong JSON: ảnh `kitchen` có top-1 là `gong` với score 0.420192, trong khi ảnh là một gian bếp (giả thuyết của tôi, chưa kiểm chứng: các chảo đồng tròn treo tường có thể làm model nhầm thành chiêng); `traffic` có top-1 `cab` chỉ 0.510915 cho một cảnh đông xe buýt. Nếu dùng prediction/score làm ground truth thì lỗi của model trở thành nhãn huấn luyện (vòng lặp tự xác nhận) và QC mất mốc so sánh độc lập. Score chỉ nên dùng để xếp thứ tự ưu tiên review.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`. Ảnh `sample_id = kitchen` (640 x 427), 11 record ở ngưỡng 0.35.

**1. Record được chọn (người đầu bếp):** `class_name`=person (`class_id`=0) ; `score`=0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68 ; `coordinate_unit`=pixel ; `bbox_format`=xyxy ; `score_threshold`=0.35.

**2. Giải thích box bằng lời:** Gốc (0, 0) ở góc trên-trái ảnh, x tăng sang phải, y tăng xuống dưới. Góc trên-trái của box ở (385.33, 69.24), tức cách mép trái khoảng 385 px và cách mép trên khoảng 69 px, ngang đỉnh tóc người đầu bếp. Góc dưới-phải ở (498.92, 348.92), ngang mũi giày, cách mép dưới ảnh khoảng 78 px. Box rộng 113.58 px và cao 279.68 px, diện tích khoảng 31.8 nghìn px², tức khoảng 11.6% ảnh. Lưu ý nhỏ: lấy hai tọa độ đã làm tròn trừ nhau ra 113.59, còn `bbox_width` được tính từ tọa độ chưa làm tròn nên là 113.58.

**3. So sánh ngưỡng:** Ngưỡng 0.20 có 17 prediction; ngưỡng 0.60 có 6 prediction (person, bowl, bowl, oven, oven, person); ngưỡng 0.35 trong JSON có 11. Notebook chỉ in tên lớp ở ngưỡng 0.20; tôi chạy thêm cùng model `yolo11n.pt` với `conf=0.20` trên ảnh `kitchen` để lấy score của 6 box được thêm so với 0.35: spoon 0.318, potted plant 0.2533, spoon 0.2399, dining table 0.2333, spoon 0.2262, bottle 0.2001. Box `bottle` chỉ vượt ngưỡng 0.20 rất sát, nên trên runtime khác (GPU, bản PyTorch khác) số đếm ở 0.20 có thể chênh 1.

**4. Hạ ngưỡng thì reviewer phải làm thêm gì?** Reviewer phải kiểm từng box được thêm và phân loại thành: (a) vật thật bị bỏ sót ở ngưỡng cao, ví dụ `dining table` 0.2333 đúng là chiếc bàn gỗ lớn ở nửa trái phía dưới ảnh, nghĩa là JSON ngưỡng 0.35 thiếu một vật có thật; (b) dương tính giả, ví dụ `potted plant` 0.2533 ở góc trên-trái thực ra là một bó lá xanh treo trên tường, không có chậu; (c) sai hoặc mơ hồ về lớp, ví dụ ba box `spoon` đều là dụng cụ treo trên tường: thìa vớt có lỗ (0.318), dao băm lưỡi cong (0.2399, trong khi COCO-80 có lớp `knife`) và một dụng cụ gỗ đầu dẹt (0.2262); `bottle` 0.2001 thực ra là lọ mứt có cắm dụng cụ; (d) box trùng hoặc chồng nhau. Số việc tăng theo số box (17 so với 6), và trong các box được thêm có cả box đúng lẫn box sai, nên không thể chấp nhận hàng loạt. Ngược lại, ngưỡng không phải quy tắc ground truth: dù model cho chiếc bàn score thấp, ground truth vẫn phải có box cho chiếc bàn.

**5. Quy tắc box cho annotator:** Vẽ đúng một box cho mỗi vật thể nhìn thấy được thuộc taxonomy COCO-80, với bốn cạnh chạm điểm pixel ngoài cùng của phần nhìn thấy (sai lệch không quá khoảng 2 px), không ước đoán phần bị che hoặc nằm ngoài khung ảnh.
- Vật bị cắt ở mép ảnh: cạnh box dừng ở mép ảnh và gắn cờ `truncated`; vật bị che một phần: gắn cờ `occluded`.
- Gán nhãn độc lập với prediction: vật model bỏ sót vẫn phải vẽ, box của model không được chép nguyên làm nhãn.

**6. Vật bị cắt mép trên ảnh `kitchen`:** Ở mép trái có một cánh tay mặc áo xanh của người thứ hai, chỉ thấy tay và bàn tay đang chỉ vào các chiếc bánh. Model ghi `person` 0.610931 với `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78] (61.44 x 47.59 px); `x_min`=0.12 cho thấy box chạm mép trái. Tôi tách làm hai phần: cách vẽ box cho vật cắt mép thuộc **guideline**, vì đây là tình huống lặp lại ở nhiều ảnh và quy tắc 5 xử lý được (box dừng ở mép, gắn `truncated`). Còn việc chỉ một cánh tay có tính là một instance `person` hay không thì **escalation** cho Lab Coach, vì đó là quyết định định nghĩa (tỉ lệ nhìn thấy tối thiểu) ảnh hưởng toàn bộ dữ liệu và annotator không có đủ thông tin để tự quyết; sau khi có câu trả lời thì bổ sung vào guideline kèm ví dụ này.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`. Ảnh `sample_id = kitchen`, 11 instance ở ngưỡng 0.35.

**1. Record được chọn:** `instance_id`=kitchen-001 ; `class_name`=person (`class_id`=0) ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]] ; `bbox_xyxy`=[385.45, 66.44, 498.02, 348.58]. Năm điểm đầu nằm ở đỉnh đầu (y khoảng 70, sát cạnh trên box 66.44) và đi dần sang trái từng 1 px theo đường viền tóc.

**2. Polygon cho thêm thông tin gì so với box?** Box chỉ cho vị trí và kích thước hình chữ nhật; polygon cho hình dạng thật và phần pixel nào thuộc vật. Tính theo công thức shoelace, polygon kitchen-001 có diện tích khoảng 18 170 px² trong box khoảng 31 760 px², tức chỉ khoảng 57% box là người, 43% còn lại là nền (chảo treo, khăn, tủ) ở các góc box và hai bên thân người. Với bàn `dining table` kitchen-005, polygon chiếm khoảng 55% box vì mặt bàn nhìn nghiêng gần như hình thang, còn box hình chữ nhật trùm cả phần gầm bàn (các thùng chứa bên dưới) và một phần bếp lò phía sau. Đường biên cũng lõm vào để chừa lọ mứt và các cốc ở mép sau bàn, điều box không thể hiện được.

**3. `instance_id` dùng để làm gì?** `instance_id` (dạng `kitchen-001` đến `kitchen-011`) chỉ là số thứ tự để phân biệt từng instance trong bộ output của lần chạy này, theo thứ tự model trả về. Nó **không phải class ID**: kitchen-001 (đầu bếp) và kitchen-009 (cánh tay ở mép trái) cùng `class_id`=0 nhưng là hai instance khác nhau; bốn instance kitchen-002, -003, -006, -010 cùng là `bowl` (id 45). Nó **không phải tracking ID**: đây là ảnh tĩnh, không có chuỗi khung hình; số này không nối cùng một vật qua thời gian, qua file detection (record đầu bếp bên detection không có `instance_id`), hay qua lần chạy khác; đổi ngưỡng hoặc model thì số thứ tự có thể thay đổi.

**4. Quy tắc vẽ biên mask:** Mask chỉ phủ các pixel nhìn thấy của chính instance đó, đi sát biên vật (sai lệch không quá khoảng 2 px), loại bỏ mọi phần bị vật khác che hoặc nằm đè lên trên, và không để mask của hai instance chung pixel nào; ở chỗ hai vật chạm nhau, pixel biên thuộc về vật ở phía trước; nếu vật bị che tách thành nhiều mảnh thì các mảnh vẫn thuộc một instance.

**5. Chỗ hai vật chạm nhau trên ảnh `kitchen`:** Ba chiếc bánh/khuôn bánh ở góc trái bàn đặt sát nhau: kitchen-006 `bowl` 0.528202 (bánh phủ vỏ bột trang trí, bị cắt ở mép trái ảnh), kitchen-003 `bowl` 0.698697 (bánh nhân rau), kitchen-002 `bowl` 0.735744 (khuôn trắng viền lượn). Khi tô lại các polygon, mask kitchen-006 chồng 77 px lên kitchen-003 và 30 px lên kitchen-002, đồng thời chồng 39 px lên cánh tay kitchen-009. Tôi tách hai vấn đề: ranh giới giữa các vật chạm nhau xử lý bằng **guideline** (quy tắc 4: không chung pixel, pixel biên thuộc vật phía trước), vì đây là tình huống lặp lại và quyết định được từ ảnh. Còn lớp của các chiếc bánh thì **escalation**: COCO-80 không có lớp pie, plate hay khuôn bánh; gán `bowl`, `cake` hay bỏ không gán là quyết định taxonomy cho cả bộ dữ liệu, annotator không được tự bịa.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Trong lab này, các prediction ở mục 1 đến 3 đóng vai trò khâu "prediction": chúng cho thấy những chỗ guideline phải nói rõ (vật cắt mép, vật ngoài taxonomy, vật chạm nhau). Khi QC so prediction với ground truth và phát hiện lỗi lặp lại (ví dụ bó lá treo trên tường bị gọi là `potted plant`), lỗi đó quay lại thành rework nhãn hoặc bổ sung quy tắc vào guideline trước vòng huấn luyện sau.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh: `class_id` + `class_name` + `taxonomy_name`, không có tọa độ (trường hợp single-label) | `traffic` top-1 `cab` 0.510915 cho cảnh chủ yếu là xe buýt; ImageNet không có lớp "bus" chung; `kitchen` top-1 `gong` 0.420192 là sai rõ | Xem toàn ảnh (bỏ dải đen đệm), chọn một lớp theo quy tắc vật chính trong guideline; không có lớp phù hợp thì escalation | Lớp có khớp định nghĩa trong guideline không, đúng taxonomy/phiên bản không, các ảnh annotator chọn khác nhau hoặc gắn cờ "không có lớp phù hợp"; không dùng score làm căn cứ đúng sai |
| Phát hiện vật thể | Một box `[x_min, y_min, x_max, y_max]` (pixel) + một lớp COCO-80 cho mỗi vật nhìn thấy, kèm cờ truncated/occluded | Bàn gỗ chỉ có score 0.2333 nên thiếu trong JSON 0.35; bó lá treo trên tường thành `potted plant` 0.2533 (chỉ xuất hiện ở ngưỡng 0.20); tủ đen bên phải có bồn rửa kim loại ở trên nhưng bị gọi `oven` 0.6343 (COCO-80 có cả `sink` lẫn `oven`, cần quyết định lớp); cánh tay cắt mép thành `person` 0.610931; bánh thành `bowl` | Vẽ box sát phần nhìn thấy cho mọi vật thuộc taxonomy, kể cả vật model bỏ sót; gắn cờ truncated/occluded; vật ngoài taxonomy thì không vẽ hoặc escalation | Vật bị bỏ sót (so ảnh, không so prediction), box lỏng hoặc vượt biên, sai lớp, box trùng; các ca escalation đã có quyết định và được ghi vào guideline chưa |
| Instance segmentation | Một polygon/mask + một lớp cho mỗi instance; hai vật cùng lớp là hai instance riêng | Mask bàn kitchen-005 phủ cả cây cán bột và bột đặt trên bàn (vùng này nằm trọn trong polygon), chưa rõ vật không thuộc taxonomy đặt trên bàn có phải loại khỏi mask không; mask bánh kitchen-006 chồng 77 px lên kitchen-003; bó lá treo trên tường thành `potted plant` 0.632236 (kitchen-004); xẻng dẹt treo tường thành `spoon` 0.359146 (kitchen-011) | Đi biên theo pixel nhìn thấy, loại vật đè lên trên, tách rõ các vật chạm nhau, không để hai mask chung pixel | Phủ mask lên ảnh để tìm chỗ tràn biên, lỗ hổng, chồng mask giữa các instance, tách/gộp instance sai, lớp sai |

## 5. An toàn dữ liệu

**1. Quy tắc bảo vệ dữ liệu:** Chỉ dùng ba ảnh COCO công khai đã được notebook cố định bằng SHA-256 (giấy phép CC BY 2.0, giữ `IMAGE_ATTRIBUTION.md`), không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ VinFast, không ghi họ tên, MSSV, email hay số điện thoại vào báo cáo và output; ảnh `traffic` có biển số và người đi đường, ảnh `kitchen` có người, nên tôi không cắt, phóng to hay tách riêng các chi tiết này ra file khác ngoài phạm vi bài lab.

**2. Gặp dữ liệu ngoài phạm vi:** Dừng ngay, không upload hay commit, và báo Lab Coach (Lab Coach chuyển GV/Lab Lead nếu cần); nếu đã lỡ push thì xóa dữ liệu đó khỏi repo và báo GV/Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
