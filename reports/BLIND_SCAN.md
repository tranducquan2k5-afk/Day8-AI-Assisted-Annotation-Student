# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 25

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe: 
Vị trí 1 (Làn xe chạy lại gần - xe ở cự ly gần):Hai chiếc xe ở góc dưới cùng (đặc biệt là chiếc xe bên phải có đèn pha rất mạnh): Ánh sáng đèn pha chiếu xuống mặt đường bê tông tạo thành một vệt sáng chói lóa kéo dài về phía trước. AI hoặc người gán nhãn rất dễ vẽ bounding box quá rộng bao luôn cả quầng sáng phản chiếu trên mặt đường thay vì chỉ bám sát phần mép cản trước và thân xe thật.   
Vị trí 2 (Làn xe di chuyển ra xa - khu vực tầm trung và phía xa bên phải):Các xe đang chạy ra xa ở làn bên phải (như xe có đèn hậu đỏ ở giữa làn và các xe phía xa): Do trời tối và xe sơn màu tối, thân xe gần như chìm hoàn toàn vào màn đêm, chỉ nhìn thấy hai đốm đèn hậu màu đỏ. Vị trí này rất dễ bị AI bỏ sót (False Negative) do kích thước nhận diện quá mờ, hoặc vẽ box bị hụt mất phần nóc và bánh xe do không thấy rõ đường bao thân xe.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
