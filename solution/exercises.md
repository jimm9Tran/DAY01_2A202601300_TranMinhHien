# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Qua 4 phản hồi, temperature tăng dần làm tính ngẫu nhiên và sáng tạo của câu trả lời tăng lên. Ở temperature 0.0 và 0.7, câu trả lời rất nhất quán, chính xác và mượt mà. Ở mức 1.2, câu từ bắt đầu bay bổng và độc lạ hơn. Khi đạt mức 1.8, phản hồi bắt đầu mất mạch lạc, lặp từ và xuất hiện các từ ngữ ghép vô nghĩa do việc chọn token ngẫu nhiên quá mức.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Tôi sẽ chọn temperature = 0.0 cho trợ lý soạn thảo hợp đồng pháp lý để đảm bảo tính chính xác tuyệt đối, nhất quán và không phát sinh từ ngữ ngẫu nhiên có thể gây sai lệch về mặt pháp lý. Ngược lại, tôi chọn temperature = 0.8 cho trợ lý viết slogan quảng cáo để khuyến khích mô hình tạo ra nhiều ý tưởng độc đáo, mới lạ và đa dạng góc nhìn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Tổng lượng token đầu ra mỗi ngày là 20.000 x 2 x 500 = 20.000.000 token (20M token). Chi phí mỗi ngày cho GPT-4o (output $0.010/1K) là (20.000K x 0.010) = $200/ngày (~$6.000/tháng). Chi phí cho GPT-4o-mini (output $0.0006/1K) là (20.000K x 0.0006) = $12/ngày (~$360/tháng). Model lớn xứng đáng khi xử lý tư vấn y khoa/pháp lý hoặc phân tích tài chính phức tạp cần độ chính xác cao. Model nhỏ là lựa chọn đúng cho tác vụ hỗ trợ FAQ tự động, phân loại tin nhắn hoặc tóm tắt ý chính đơn giản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Phản hồi của nhà thơ mang giọng văn giàu hình ảnh ví von (như khu vườn tự học lớn lên), không có thuật ngữ kỹ thuật. Trong khi đó, kỹ sư senior giải thích trực diện bản chất thuật toán, tối ưu hóa hàm mất mát và kèm ví dụ code Python cụ thể. Qua đó rút ra System Prompt giúp điều khiển chính xác: (1) Vai trò/Giọng văn (tone of voice), (2) Đối tượng độc giả mục tiêu, (3) Mức độ chuyên sâu kỹ thuật và (4) Định dạng đầu ra của phản hồi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với đoạn văn tiếng Việt 150 từ, tiktoken đếm được khoảng 220 token do đặc thù tiếng Việt có nhiều từ ghép và dấu thanh (unicode) bị tách thành nhiều sub-word token. Công thức ước lượng thô cho ra 150 / 0.75 = 200 token. Hai con số chênh nhau khoảng 10% - 20%. Nếu dùng ước lượng thô cho tiếng Việt, bạn sẽ dự toán **thiếu** ngân sách API vì tiếng Việt tốn nhiều sub-word token hơn tiếng Anh trên các bộ tokenizer hiện tại.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> (a) Chatbot văn bản và (b) trợ lý giọng nói hưởng lợi nhiều nhất từ streaming vì giúp giảm thời gian chờ đợi cảm nhận (perceived latency); người dùng đọc hoặc nghe phản hồi ngay lập tức khi token vừa tạo ra. Trái lại, (c) pipeline dịch tài liệu chạy ngầm ban đêm không cần streaming vì đây là tác vụ xử lý lô (batch processing) không có tương tác trực tiếp với người dùng theo thời gian thực, việc dùng streaming chỉ làm phức tạp code mà không đem lại giá trị UX.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff giúp dãn cách thời gian thử lại thưa dần ra, giúp hệ thống có khoảng nghỉ để phục hồi thay vì bị dồn dập request ở các mốc thời gian cố định (gây ra hiện tượng thảm họa tháo chạy - thundering herd problem). Kỹ thuật "jitter" thêm độ trễ ngẫu nhiên vào thời gian chờ giúp làm lệch pha thời điểm retry của các client xuất phát cùng lúc, tránh việc các client đồng loạt gửi request lại tại cùng một tích tắc.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." 
> Chỗ 1: Nếu xóa "ngắn gọn", trợ lý sẽ trả lời rất dài dòng, giải thích quá chi tiết gây tốn kém token và chậm trễ.
> Chỗ 2: Nếu xóa "bằng tiếng Việt", trợ lý có thể tự động trả lời bằng tiếng Anh nếu nhận câu hỏi bằng tiếng Anh hoặc theo ngôn ngữ của tài liệu huấn luyện gốc.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Tình huống: Người dùng khai báo tên biến cấu hình ở lượt 1. Sau 5 lượt hội thoại thảo luận các chủ đề khác, ở lượt 6 người dùng hỏi "Hãy dùng biến ở lượt 1 để chạy code". Vì history chỉ giữ 4 lượt cuối nên thông tin ở lượt 1 đã bị xóa, làm trợ lý không còn nhớ tên biến và trả lời sai.
> Cách khắc phục: Triển khai kỹ thuật Conversation Summarization — khi history vượt quá ngưỡng, gọi model nhỏ (như gpt-4o-mini) tóm tắt các ý chính/biến quan trọng của các lượt cũ thành một đoạn summary ngắn và gắn cố định vào System Prompt.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
