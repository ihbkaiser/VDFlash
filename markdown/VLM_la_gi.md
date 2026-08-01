## 1. VLM là gì?

**Vision–Language Model (VLM)** là mô hình có khả năng tiếp nhận và kết hợp thông tin từ:

* Thị giác: ảnh, khung hình video, vùng ảnh, đối tượng.
* Ngôn ngữ: câu hỏi, mô tả, chỉ dẫn, hội thoại.
* Đôi khi thêm âm thanh, phụ đề, tọa độ không gian hoặc tín hiệu hành động.

Có thể biểu diễn tổng quát:

$$
y = f_\theta(x_{\text{vision}}, x_{\text{text}})
$$

Trong đó:

* $x_{\text{vision}}$: ảnh hoặc video.
* $x_{\text{text}}$: câu hỏi hoặc instruction.
* $y$: câu trả lời, mô tả, tọa độ, nhãn hoặc hành động.

---

# 2. Cấu trúc cơ bản của một VLM

Một VLM hiện đại thường gồm ba thành phần chính:

$$\text{Image} \rightarrow \text{Vision Encoder} \rightarrow \text{Projector / Adapter} \rightarrow \text{LLM} \rightarrow \text{Text Output}$$

## 2.1. Vision encoder

Vision encoder chuyển ảnh thành một chuỗi vector đặc trưng.

Các kiến trúc phổ biến gồm:

* Vision Transformer, hay ViT.
* CLIP vision encoder.
* ConvNeXt.
* Swin Transformer.
* Các vision encoder được huấn luyện bằng contrastive learning hoặc self-supervised learning.

Giả sử ảnh có kích thước (H \times W), được chia thành các patch kích thước (P \times P). Số visual token gần đúng là:

$$N_{\text{vision}} = \frac{H}{P}\frac{W}{P}$$

Ví dụ, ảnh (336 \times 336) với patch (14 \times 14):

$$N_{\text{vision}} = 24 \times 24 = 576$$

Vision encoder tạo ra:

$$Z_v = E_v(I)$$

với:

$$Z_v \in \mathbb{R}^{N_v \times d_v}$$

Trong đó (N_v) là số visual token và (d_v) là chiều đặc trưng của vision encoder.

---

## 2.2. Projector hoặc visual adapter

Vision encoder và LLM thường có embedding dimension khác nhau. Vì vậy cần một module để chuyển visual embedding sang không gian mà LLM có thể xử lý.

$$ Z_l = P(Z_v) $$

Trong đó:

$$ Z_l \in \mathbb{R}^{N'_v \times d_{\text{LLM}}} $$

Projector có thể là:

* Linear layer.
* MLP hai tầng.
* Cross-attention module.
* Q-Former.
* Perceiver Resampler.
* Token pooling hoặc token compression module.

Hai nhiệm vụ chính của projector là:

1. **Alignment**: căn chỉnh biểu diễn thị giác với không gian ngôn ngữ.
2. **Compression**: giảm số lượng visual token trước khi đưa vào LLM.

Ví dụ, 576 patch token có thể được nén xuống 64 hoặc 32 token.

---

# 3. Các kiểu kiến trúc VLM

## 3.1. Dual-encoder

Ảnh và văn bản được mã hóa độc lập:

$$z_I = E_I(I), \qquad z_T = E_T(T)$$

Kiểu này phù hợp với:

* Image–text retrieval.
* Zero-shot classification.
* Tìm kiếm đa phương thức.
* Contrastive pretraining.

Ưu điểm:

* Có thể mã hóa ảnh và văn bản độc lập.
* Tìm kiếm nhanh bằng vector database.
* Phù hợp với quy mô dữ liệu lớn.

Hạn chế:

* Không mạnh về sinh câu trả lời dài.
* Khó thực hiện reasoning phức tạp.
* Tương tác giữa ảnh và văn bản diễn ra tương đối muộn.

---

## 3.2. Fusion encoder

Ảnh và văn bản được đưa vào một Transformer chung hoặc tương tác thông qua cross-attention.

Ví dụ:

$$
H^{(l+1)}_T =
\operatorname{CrossAttn}
\left(
H_T^{(l)},H_V^{(l)}
\right)
$$

Kiểu này phù hợp với:

* Visual question answering.
* Visual reasoning.
* Visual grounding.
* Image–text matching.

Ưu điểm là tương tác đa phương thức sâu. Nhược điểm là chi phí tính toán cao.

---

## 3.3. LLM-centric architecture

Đây là thiết kế phổ biến của multimodal LLM hiện nay:

* Một vision encoder có sẵn.
* Một adapter nối vision encoder với LLM.
* LLM đóng vai trò reasoning và generation engine.

Ví dụ luồng xử lý:

```text
Ảnh
  ↓
ViT / CLIP encoder
  ↓
Visual embeddings
  ↓
MLP / Q-Former / Resampler
  ↓
Visual tokens
  ↓
LLM + text prompt
  ↓
Câu trả lời
```

Ưu điểm:

* Tận dụng năng lực ngôn ngữ và reasoning của LLM.
* Dễ instruction tuning.
* Có khả năng hội thoại.
* Có thể mở rộng từ ảnh sang video.

---

# 4. VLM được huấn luyện như thế nào?

Thông thường gồm ba hoặc bốn giai đoạn.

## 4.1. Vision–language pretraining

Mục tiêu là học mối quan hệ giữa ảnh và văn bản.

### Contrastive learning

Một cặp ảnh–văn bản đúng được kéo gần nhau, các cặp sai được đẩy xa nhau.

$$
\mathcal{L}_{I\rightarrow T} = -\frac{1}{B} \sum_i \log \frac{\exp(s(z_i^I,z_i^T)/\tau)}{\sum_j \exp(s(z_i^I,z_j^T)/\tau)}
$$

Tương tự cho chiều text-to-image.

### Image–text matching

Mô hình dự đoán ảnh và văn bản có khớp nhau hay không.

$$
\mathcal{L}_{ITM} = -\sum_i [y_i\log p_i + (1-y_i)\log(1-p_i)]
$$

### Captioning hoặc language modeling

Mô hình sinh mô tả ảnh:

$$
\mathcal{L}_{LM} = -\sum_t \log p(y_t|y_{<t},I)
$$

---

## 4.2. Multimodal alignment

Trong giai đoạn này, vision encoder và LLM có thể được giữ cố định, chỉ projector được huấn luyện.

Mục tiêu là học cách biến visual features thành các embedding mà LLM hiểu được.

Dữ liệu thường là:

```text
Ảnh + mô tả ảnh
Ảnh + câu hỏi + câu trả lời
Ảnh + instruction + response
```

---

## 4.3. Multimodal instruction tuning

Mô hình được huấn luyện trên các hội thoại đa phương thức:

```text
User: Trong ảnh có những đối tượng nào?
Assistant: Có một ô tô màu đỏ và hai người đi bộ.

User: Người đi bộ có đang đi qua đường không?
Assistant: Có, họ đang đi trên vạch sang đường.
```

Loss thường vẫn là causal language modeling:

$$
\mathcal{L}_{SFT} = -\sum_{t \in \text{response}} \log p(y_t|y_{<t},I,x)
$$

Loss thường chỉ được tính trên phần trả lời của assistant, không nhất thiết tính trên toàn prompt.

---

## 4.4. Preference alignment

Có thể sử dụng:

* RLHF.
* DPO.
* GRPO.
* Reward-model-based optimization.
* AI feedback.

Mục tiêu là cải thiện:

* Độ chính xác.
* Khả năng tuân thủ instruction.
* Chất lượng giải thích.
* Giảm hallucination.
* Độ an toàn.

---

# 5. VLM “hiểu” hình ảnh như thế nào?

Cần phân biệt giữa ba mức độ.

## 5.1. Perception

Nhận biết nội dung trực tiếp trong ảnh:

* Có vật gì?
* Vật nằm ở đâu?
* Màu sắc là gì?
* Có bao nhiêu người?
* Văn bản trong ảnh ghi gì?

## 5.2. Relational understanding

Hiểu quan hệ:

* Người đang cầm vật gì?
* Chiếc xe nằm bên trái hay phải?
* Ai đang tương tác với ai?
* Vật A có nằm trong vật B không?

## 5.3. Reasoning

Suy luận vượt ra ngoài nhận dạng trực tiếp:

* Tại sao người đó cầm ô?
* Điều gì có thể xảy ra tiếp theo?
* Tình huống có nguy hiểm không?
* Biểu đồ cho thấy xu hướng nào?

Tuy nhiên, cần thận trọng: nhiều câu trả lời trông giống reasoning nhưng có thể được sinh từ pattern thống kê, prior knowledge hoặc shortcut trong dữ liệu.

---

# 6. Video VLM là gì?

**Video Vision–Language Model**, thường gọi là Video VLM hoặc Video-LLM, mở rộng VLM từ một ảnh tĩnh sang chuỗi khung hình có thứ tự thời gian.

Đầu vào là:

$$
V =
{F_1,F_2,\ldots,F_T}
$$

Trong đó ($F_t$) là frame tại thời điểm ($t$).

Mô hình cần trả lời dựa trên cả:

* Nội dung không gian trong từng frame.
* Sự thay đổi theo thời gian.
* Thứ tự xảy ra của sự kiện.
* Chuyển động.
* Tương tác kéo dài.
* Thông tin âm thanh và lời nói, nếu có.

Một mô hình ảnh có thể nhận biết:

> Người đang đứng cạnh chiếc cốc.

Nhưng một Video VLM cần hiểu:

> Người đó lấy chiếc cốc, rót nước rồi đặt nó lên bàn.

Điểm khác biệt nằm ở **temporal reasoning**.

---

# 7. Video khó hơn ảnh ở đâu?

## 7.1. Số lượng dữ liệu đầu vào rất lớn

Nếu mỗi frame tạo ra 576 visual token và video có 100 frame:

$$
N = 100 \times 576 = 57,600$$

Chưa tính text token.

Self-attention tiêu chuẩn có độ phức tạp:

$$O(N^2)$$

Với 57.600 token, attention matrix có hơn:

$$57,600^2 \approx 3.3 \times 10^9
$$

phần tử cho một attention head.

Do đó, không thể đơn giản đưa tất cả patch của tất cả frame vào LLM.

---

## 7.2. Temporal redundancy

Hai frame liên tiếp thường gần giống nhau:

$$
F_t \approx F_{t+1}$$

Nếu video 30 FPS, nhiều khung hình không cung cấp thông tin mới đáng kể.

Việc xử lý tất cả các frame gây:

* Lãng phí tính toán.
* Context quá dài.
* Tăng nhiễu.
* Làm suy yếu khả năng tập trung vào sự kiện quan trọng.

---

## 7.3. Sự kiện xảy ra ở nhiều thang thời gian

Video có thể chứa:

* Chuyển động ngắn: vài phần mười giây.
* Hành động: vài giây.
* Hoạt động: hàng chục giây.
* Cốt truyện hoặc quy trình: vài phút đến hàng giờ.

Một biểu diễn video tốt phải xử lý nhiều cấp:

$$\text{frame}
\rightarrow
\text{motion}
\rightarrow
\text{action}
\rightarrow
\text{event}
\rightarrow
\text{story}$$

---

## 7.4. Thứ tự thời gian quan trọng

Hai video có thể chứa cùng các frame nhưng thứ tự khác nhau.

Ví dụ:

```text
A: Đập trứng → đánh trứng → chiên
B: Chiên → đánh trứng → đập trứng
```

Nếu mô hình chỉ thực hiện pooling toàn cục không phụ thuộc thứ tự, hai video có thể có biểu diễn gần giống nhau, dù ý nghĩa hoàn toàn khác.

---

## 7.5. Thông tin quan trọng có thể rất thưa

Trong video dài 30 phút, câu trả lời có thể phụ thuộc vào một đoạn 3 giây.

Đây là vấn đề:

* Temporal localization.
* Moment retrieval.
* Evidence selection.
* Long-context reasoning.

---

# 8. Kiến trúc tổng quát của Video VLM

Một pipeline phổ biến:

$$\text{Video}
\rightarrow
\text{Frame Sampling}
\rightarrow
\text{Visual Encoder}
\rightarrow
\text{Temporal Modeling}
\rightarrow
\text{Token Compression}
\rightarrow
\text{LLM}$$

Chi tiết:

```text
Video
  ↓
Chọn frame hoặc clip
  ↓
Mã hóa từng frame
  ↓
Mô hình hóa quan hệ thời gian
  ↓
Giảm số visual token
  ↓
Ghép với prompt
  ↓
LLM reasoning và generation
```

---

# 9. Bước 1: Lấy mẫu frame

Với video dài $L$ giây và tốc độ $r$ FPS, tổng số frame là:

$$ T = Lr $$

Ví dụ video 10 phút ở 30 FPS:

$$ T = 600 \times 30 = 18,000 $$

Không thể xử lý toàn bộ. Vì vậy cần lấy mẫu.

## 9.1. Uniform sampling

Chọn $K$ frame cách đều:

$$ i_k = \left\lfloor \frac{k(T-1)}{K-1} \right\rfloor $$

Ưu điểm:

* Đơn giản.
* Phủ toàn bộ video.
* Chi phí cố định.

Hạn chế:

* Có thể bỏ lỡ sự kiện ngắn.
* Không chú ý đến nội dung.

---

## 9.2. Random sampling

Chọn frame ngẫu nhiên, thường dùng trong training để tăng diversity.

Hạn chế là kết quả inference không ổn định nếu sự kiện quan trọng rất ngắn.

---

## 9.3. Keyframe sampling

Chọn frame có thay đổi đáng kể theo:

* Histogram.
* Optical flow.
* Scene boundary.
* Feature distance.
* Object change.
* Motion magnitude.

Ví dụ:

$$d_t =
|E(F_t)-E(F_{t-1})|_2
$$

Chọn các frame có (d_t) lớn.

---

## 9.4. Query-aware sampling

Việc chọn frame phụ thuộc vào câu hỏi.

Ví dụ câu hỏi:

> Người đàn ông đã làm gì sau khi mở cửa?

Mô hình tìm các frame liên quan đến:

* Người đàn ông.
* Hành động mở cửa.
* Khoảng thời gian ngay sau đó.

Có thể tính:

$$s_t =
\operatorname{sim}
(E_v(F_t),E_q(q))$$

Sau đó chọn top-(K) frame có score cao.

Đây là cách hiệu quả cho video dài, nhưng có rủi ro bỏ sót evidence nếu retriever chưa tốt.

---

# 10. Bước 2: Mã hóa video

Có ba nhóm chính.

## 10.1. Frame-wise image encoder

Mỗi frame được mã hóa độc lập bằng image encoder:

$$ Z_t = E_v(F_t) $$

Toàn bộ video:

$$ Z = [Z_1, Z_2, \ldots, Z_K] $$

Ưu điểm:

* Tận dụng pretrained image VLM.
* Dễ mở rộng từ ảnh sang video.
* Có thể reuse CLIP hoặc ViT.

Hạn chế:

* Encoder không nhìn thấy chuyển động.
* Temporal modeling được đẩy sang module phía sau.
* Có thể không phân biệt được các hành động có frame tĩnh giống nhau.

---

## 10.2. Video encoder

Video được chia thành các tubelet theo cả không gian và thời gian.

Nếu patch ảnh là:

$$ P_h \times P_w $$

thì video patch có thể là:

$$ P_t \times P_h \times P_w $$

Một token biểu diễn một khối không-thời gian.

Ưu điểm:

* Học motion và temporal pattern trực tiếp.
* Phù hợp với action recognition.
* Nắm bắt chuyển động tốt hơn image encoder.

Hạn chế:

* Huấn luyện đắt.
* Khó tận dụng hoàn toàn các image encoder mạnh.
* Số token vẫn rất lớn.

---

## 10.3. Hybrid image–video encoding

Kết hợp:

* Image encoder cho semantic appearance.
* Temporal module cho motion.
* Optical flow hoặc motion encoder.
* Audio encoder.
* Subtitle encoder.

Biểu diễn có thể là:

$$ Z = \operatorname{Fuse}(Z_{\text{appearance}}, Z_{\text{motion}}, Z_{\text{audio}}) $$

---

# 11. Temporal modeling

Đây là phần phân biệt Video VLM với Image VLM.

## 11.1. Temporal positional encoding

Mỗi frame cần thông tin vị trí thời gian:

$$ \tilde{Z}_t = Z_t + P_t $$

Nếu không có temporal position, mô hình có thể không phân biệt thứ tự frame.

Temporal encoding có thể là:

* Learned embedding.
* Sinusoidal embedding.
* Rotary position embedding.
* Relative temporal bias.
* Timestamp embedding.

Ví dụ có thể gắn token:

```text
<frame_1> ... </frame_1>
<frame_2> ... </frame_2>
<frame_3> ... </frame_3>
```

hoặc timestamp:

```text
<time=00:12.5>
```

---

## 11.2. Temporal Transformer

Các frame feature được đưa qua Transformer:

$$ H = \operatorname{Transformer}(Z_1,\ldots,Z_K) $$

Cơ chế attention cho phép frame (t) tương tác với frame (s):

$$
A_{ts} = \operatorname{softmax} \left( \frac{Q_tK_s^\top}{\sqrt d} \right)
$$

Ưu điểm là mô hình hóa quan hệ dài hạn. Hạn chế là chi phí bậc hai theo số token thời gian.

---

## 11.3. Factorized spatial–temporal attention

Thay vì attention trên toàn bộ token không-thời gian, mô hình tách thành:

1. Spatial attention trong từng frame.
2. Temporal attention trên cùng vị trí hoặc giữa frame summaries.

$$
\operatorname{Attention}_{ST} \approx \operatorname{Attention}_{S} + \operatorname{Attention}_{T}
$$

Nếu có (T) frame và mỗi frame có (S) spatial token:

* Full attention:

$$ O((TS)^2) $$

* Factorized attention:

$$ O(TS^2 + ST^2) $$

Thường rẻ hơn đáng kể.

---

## 11.4. Temporal pooling

Có thể gộp feature theo thời gian:

$$
z_{\text{video}} = \frac{1}{T}\sum_{t=1}^{T}z_t
$$

hoặc attention pooling:

$$
z_{\text{video}} = \sum_t \alpha_t z_t
$$

với:

$$ \alpha_t = \frac{\exp(q^\top z_t)}{\sum_s \exp(q^\top z_s)} $$

Pooling đơn giản và rẻ, nhưng có thể làm mất:

* Thứ tự.
* Chi tiết thời gian.
* Sự kiện ngắn.
* Quan hệ trước–sau.

---

## 11.5. Recurrent hoặc state-space modeling

Có thể cập nhật trạng thái video:

$$ h_t = f(h_{t-1},z_t) $$

Phù hợp với:

* Streaming video.
* Online inference.
* Video rất dài.
* Memory-efficient processing.

Tuy nhiên, thông tin cũ có thể bị nén hoặc quên.

---

# 12. Token compression trong Video VLM

Đây là vấn đề trung tâm của Video VLM.

Giả sử:

* (T): số frame.
* (S): token mỗi frame.
* (N = TS): tổng visual token.

Ta cần ánh xạ:

$$ \mathbb{R}^{N \times d} \rightarrow \mathbb{R}^{M \times d} $$

với:

$$ M \ll N $$

## 12.1. Spatial pooling

Trong mỗi frame, gộp các patch:

$$ S \rightarrow S' $$

Ví dụ:

$$ 576 \rightarrow 144 \rightarrow 36 $$

---

## 12.2. Temporal pooling

Gộp các frame gần nhau:

$$ T \rightarrow T' $$

Ví dụ 64 frame được gộp thành 16 temporal segments.

---

## 12.3. Learnable resampler

Dùng $M$ query token học được để đọc visual features:

$$ Q' = \operatorname{CrossAttn}(Q,Z_v) $$

Với:

* $(Q \in \mathbb{R}^{M\times d})$.
* $(Z_v \in \mathbb{R}^{N\times d})$.
* $(M \ll N)$.

Cách này cho số output token cố định, bất kể video dài bao nhiêu.

Hạn chế: nén quá mạnh có thể làm mất sự kiện nhỏ.

---

## 12.4. Token merging

Gộp các token tương tự:

$$
z_{\text{merge}} = \frac{w_i z_i + w_j z_j}{w_i + w_j}
$$

Các token dư thừa giữa frame liên tiếp là ứng viên tốt để merge.

---

## 12.5. Token pruning

Loại bỏ token có score thấp:

$$ s_i = g(z_i,q) $$

Giữ:

$$ \operatorname{TopK}(s_1,\ldots,s_N) $$

Nếu (g) phụ thuộc câu hỏi, đây là query-aware token pruning.

---

## 12.6. Hierarchical compression

Một thiết kế hiệu quả cho video dài:

```text
Patch tokens
   ↓
Frame tokens
   ↓
Clip tokens
   ↓
Event tokens
   ↓
Global video tokens
```

Ví dụ:

$$ \text{patch} \rightarrow \text{frame summary} \rightarrow \text{clip summary} \rightarrow \text{video memory} $$

Cách này phản ánh cấu trúc nhiều thang thời gian của video.

---

# 13. Các cách đưa video token vào LLM

## 13.1. Token concatenation

Ghép trực tiếp:

```text
video tokens question tokens
```

$$
X = [Z_v; Z_q]
$$

Ưu điểm là đơn giản. Nhược điểm là chi phí context lớn.

---

## 13.2. Interleaved representation

Visual token và text mô tả thời gian được xen kẽ:

```text
Frame 1: visual tokens
Frame 2: visual tokens
...
Question: ...
```

Cách này giữ cấu trúc thời gian rõ hơn.

---

## 13.3. Cross-attention

LLM không cần chứa toàn bộ visual token trong self-attention. Thay vào đó, text hidden states truy vấn một visual memory:

$$ H'_T = \operatorname{CrossAttn}(H_T,Z_V) $$

Ưu điểm:

* Visual memory có thể nằm ngoài context chính.
* Có thể cache visual features.
* Phù hợp video dài.

---

## 13.4. Retrieval-augmented video reasoning

Video được chia thành nhiều clip. Một retriever chọn clip liên quan:

$$
C^* = \operatorname{TopK} \left( \operatorname{sim}(q,C_i) \right)
$$

Chỉ các clip được chọn mới đưa vào VLM hoặc LLM.

Pipeline:

```text
Video dài
  ↓
Phân đoạn và lập chỉ mục
  ↓
Câu hỏi
  ↓
Truy xuất clip liên quan
  ↓
VLM xử lý clip
  ↓
LLM tổng hợp câu trả lời
```

Cách này rất phù hợp với video dài hàng chục phút hoặc hàng giờ.

---

# 14. Audio trong Video VLM

Video không chỉ chứa hình ảnh.

Một hệ thống đầy đủ có thể xử lý:

$$
\text{Video} = \text{Frames} + \text{Audio} + \text{Speech} + \text{Subtitle} + \text{Metadata}
$$

## 14.1. Speech

Âm thanh lời nói được chuyển thành transcript bằng ASR:

$$ A \rightarrow \text{ASR} \rightarrow T_{\text{speech}} $$

Transcript sau đó được đưa vào LLM.

## 14.2. Non-speech audio

Các tín hiệu như:

* Tiếng chuông.
* Tiếng nổ.
* Tiếng động cơ.
* Tiếng vỗ tay.
* Nhạc nền.

có thể chứa thông tin không xuất hiện trong frame.

Audio encoder tạo:

$$ Z_a = E_a(A) $$

Sau đó fusion:

$$ Z = \operatorname{Fuse}(Z_v,Z_a,Z_t) $$

## 14.3. Vấn đề đồng bộ

Audio và video cần được căn chỉnh theo timestamp:

$$ (z^v_t,z^a_t,z^s_t) $$

Nếu bỏ qua đồng bộ thời gian, mô hình có thể gắn lời nói hoặc âm thanh với sai sự kiện.

---

# 15. Các tác vụ chính của Video VLM

## 15.1. Video captioning

Sinh mô tả toàn bộ video:

$$ V \rightarrow y_{\text{caption}} $$

Ví dụ:

> Một người bước vào bếp, lấy rau trong tủ lạnh và bắt đầu chuẩn bị bữa ăn.

---

## 15.2. Video question answering

$$ (V,q) \rightarrow a $$

Các dạng câu hỏi:

### Perception

> Người phụ nữ mặc áo màu gì?

### Temporal

> Người đó làm gì trước khi ngồi xuống?

### Causal

> Tại sao chiếc cốc bị rơi?

### Counting

> Người đàn ông mở cửa bao nhiêu lần?

### Prediction

> Điều gì có khả năng xảy ra tiếp theo?

---

## 15.3. Temporal grounding

Tìm khoảng thời gian tương ứng với mô tả:

$$
(V,q) \rightarrow [t_s,t_e]
$$

Ví dụ:

> Khoảnh khắc người đàn ông bắt đầu rửa xe diễn ra khi nào?

Mô hình trả:

$$[02{:}15, 03{:}02]$$

---

## 15.4. Video retrieval

Tìm video phù hợp với mô tả:

$$ q_{\text{text}} \rightarrow V^* $$

hoặc tìm đoạn video trong một video dài.

---

## 15.5. Video summarization

Tạo:

* Tóm tắt văn bản.
* Danh sách sự kiện.
* Các chương.
* Highlight.
* Keyframes.
* Timeline.

---

## 15.6. Action recognition

Dự đoán hành động:

$$ V \rightarrow c $$

Ví dụ:

* Uống nước.
* Mở cửa.
* Ném bóng.
* Lắp ráp linh kiện.

---

## 15.7. Embodied và agentic video understanding

Mô hình quan sát video từ camera hoặc robot rồi quyết định hành động:

$$ a_t = \pi(V_{\leq t},x_{\text{instruction}}) $$

Ứng dụng:

* Robot.
* Xe tự hành.
* Trợ lý giao diện.
* Camera giám sát.
* Gameplay agent.
* Process monitoring.

---

# 16. Các mức độ hiểu thời gian

Có thể chia temporal reasoning thành nhiều cấp.

## 16.1. Local motion

Nhận biết chuyển động giữa các frame gần nhau:

* Di chuyển sang trái.
* Cầm vật lên.
* Rơi xuống.
* Quay đầu.

## 16.2. Action segmentation

Phân chia video thành các hành động:

```text
Mở tủ → lấy cốc → rót nước → uống
```

## 16.3. Event ordering

Hiểu quan hệ:

* Trước.
* Sau.
* Đồng thời.
* Trong khi.
* Cho đến khi.

## 16.4. Long-range dependency

Một sự kiện đầu video ảnh hưởng đến sự kiện cuối video.

Ví dụ:

* Nhân vật cất chìa khóa ở phút 1.
* Đến phút 20, họ quay lại tìm chìa khóa.

## 16.5. Causal reasoning

Phân biệt:

$$ A \text{ xảy ra trước } B $$

với:

$$ A \text{ gây ra } B $$

Temporal precedence không đồng nghĩa với causality.

---

# 17. Video VLM có thật sự hiểu chuyển động không?

Không phải mô hình nào nhận nhiều frame cũng thật sự hiểu video.

Một mô hình có thể đạt kết quả tốt nhờ:

* Nhận dạng một vài keyframe.
* Đọc subtitle.
* Dựa vào prior knowledge.
* Đoán từ câu hỏi.
* Phát hiện đối tượng nhưng không hiểu motion.
* Chỉ pooling toàn bộ frame.

Để kiểm tra temporal understanding, có thể thực hiện các phép thử:

## 17.1. Shuffle test

Đảo thứ tự frame:

$$ (F_1,F_2,\ldots,F_T) \rightarrow (F_{\pi(1)},F_{\pi(2)},\ldots,F_{\pi(T)}) $$

Nếu kết quả gần như không đổi với câu hỏi temporal, mô hình có thể chưa thực sự dùng thứ tự.

## 17.2. Reverse test

Phát video ngược.

Các câu hỏi như “trước” và “sau” phải thay đổi đáp án.

## 17.3. Frame removal

Loại bỏ frame chứa bằng chứng chính. Nếu mô hình vẫn trả lời đúng với độ tự tin cao, có thể nó đang dựa vào language prior.

## 17.4. Counterfactual editing

Thay đổi một sự kiện trong video nhưng giữ nguyên phần còn lại.

Ví dụ:

* Bản gốc: người đổ nước vào cốc.
* Bản sửa: người đổ nước xuống sàn.

Mô hình phải thay đổi câu trả lời tương ứng.

---

# 18. Video dài và cơ chế bộ nhớ

## 18.1. Sliding window

Chia video thành các cửa sổ:

$$ W_i = \{F_i,\ldots,F_{i+L-1}\} $$

Xử lý lần lượt và duy trì summary hoặc memory.

Ưu điểm:

* Context cố định.
* Phù hợp streaming.

Nhược điểm:

* Quan hệ xuyên nhiều cửa sổ có thể bị mất.

---

## 18.2. Recurrent memory

Sau mỗi đoạn video:

$$ m_t = f(m_{t-1},C_t) $$

Trong đó (C_t) là clip hiện tại.

Bộ nhớ có thể lưu:

* Đối tượng đã xuất hiện.
* Trạng thái của đối tượng.
* Sự kiện quan trọng.
* Timeline.
* Các câu hỏi chưa được giải quyết.

---

## 18.3. External memory

Mỗi clip được lưu trong cơ sở dữ liệu:

$$
\mathcal{M} = \{(k_i,v_i,t_i)\}_{i=1}^{N}
$$

Trong đó:

* (k_i): embedding truy xuất.
* (v_i): visual hoặc textual representation.
* (t_i): timestamp.

Khi có câu hỏi, mô hình truy xuất các memory item liên quan.

---

## 18.4. Hierarchical summarization

Ví dụ video 1 giờ:

1. Tóm tắt mỗi 10 giây.
2. Gộp thành tóm tắt mỗi phút.
3. Gộp thành chương.
4. Dùng các chương để trả lời câu hỏi toàn cục.
5. Truy ngược về clip gốc để xác minh chi tiết.

Cách này tiết kiệm context nhưng có thể tích lũy lỗi tóm tắt.

---

# 19. Hallucination trong VLM và Video VLM

VLM có thể mô tả đối tượng hoặc sự kiện không tồn tại.

## 19.1. Language prior

LLM dự đoán dựa trên các mẫu quen thuộc.

Ví dụ thấy nhà bếp, mô hình có thể suy đoán có tủ lạnh dù không nhìn thấy rõ.

## 19.2. Visual information bottleneck

Projector nén quá nhiều:

$$ N_{\text{input}} \gg N_{\text{output}} $$

Thông tin nhỏ bị mất trước khi đến LLM.

## 19.3. Temporal confusion

Mô hình nhận đúng hai hành động nhưng đảo thứ tự.

Ví dụ video:

```text
Cầm cốc → rót nước
```

Mô hình trả:

```text
Rót nước → cầm cốc
```

## 19.4. Long-video forgetting

Thông tin ở đầu video bị bỏ qua do:

* Context quá dài.
* Attention dilution.
* Positional bias.
* Compression quá mạnh.
* Retrieval sai.

## 19.5. Subtitle shortcut

Mô hình trả lời dựa vào subtitle thay vì hình ảnh. Điều này có thể giúp trên benchmark nhưng không phản ánh visual understanding thực sự.

---

# 20. Đánh giá Video VLM

Không nên chỉ dùng một metric tổng quát.

## 20.1. Captioning metrics

* BLEU.
* ROUGE.
* METEOR.
* CIDEr.
* SPICE.
* Embedding-based similarity.
* LLM-as-a-judge.

Các metric lexical có thể bỏ sót hai mô tả khác từ nhưng cùng nghĩa.

---

## 20.2. Question answering

* Exact match.
* Multiple-choice accuracy.
* Soft matching.
* Semantic similarity.
* Human evaluation.

---

## 20.3. Temporal grounding

Thường dùng Intersection over Union:

$$

\operatorname{IoU}
=

\frac{
\left|[t_s,t_e]\cap[\hat{t}_s,\hat{t}_e]\right|
}{
\left|[t_s,t_e]\cup[\hat{t}_s,\hat{t}_e]\right|
}
$$

Các metric:

* Recall@K at IoU threshold $\theta$.
* Mean IoU.
* Temporal localization accuracy.

---

## 20.4. Long-video evaluation

Cần đánh giá riêng:

* Needle-in-a-haystack retrieval.
* Multi-event reasoning.
* Cross-segment dependency.
* Order sensitivity.
* Evidence localization.
* Robustness theo độ dài video.
* Performance theo số frame được cung cấp.
* Accuracy–latency trade-off.

---

# 21. Chi phí tính toán của Video VLM

Chi phí thường nằm ở ba khu vực.

## 21.1. Video decoding

Đọc và giải mã video thành frame có thể trở thành bottleneck CPU hoặc I/O.

## 21.2. Vision encoding

Nếu có $T$ frame:

$$ C_{\text{vision}} \approx T \cdot C_{\text{frame}} $$

Có thể batch các frame để tận dụng GPU.

## 21.3. LLM prefill

Visual tokens làm prompt rất dài. Prefill cost có thể chiếm phần lớn latency:

$$ C_{\text{prefill}} \propto N_{\text{input}}^2 $$

hoặc gần tuyến tính hơn với các kiến trúc attention tối ưu, nhưng vẫn tăng mạnh theo context.

## 21.4. Autoregressive decoding

Sinh $M$ output token:

$$ y_1,y_2,\ldots,y_M $$

mỗi bước cần một forward pass, dù KV cache giúp không phải tính lại toàn bộ context.

Với Video VLM, có hai nhóm acceleration khác nhau:

* Giảm chi phí hiểu video: frame sampling, token compression, retrieval, caching.
* Giảm chi phí sinh văn bản: speculative decoding, quantization, KV-cache optimization, parallel decoding.

---

# 22. Speculative decoding liên quan thế nào đến Video VLM?

Speculative decoding tăng tốc phần sinh output.

Ý tưởng:

1. Một draft model nhỏ đề xuất nhiều token.
2. Target model lớn kiểm tra các token đó song song.
3. Các token hợp lệ được chấp nhận cùng lúc.

Giả sử draft model đề xuất:

$$ \tilde y_{t+1:t+k} $$

Target model tính xác suất cho toàn bộ block bằng một lần forward, rồi chấp nhận prefix thỏa điều kiện.

Tuy nhiên, speculative decoding không trực tiếp giảm:

* Số frame.
* Chi phí vision encoder.
* Số visual token.
* Chi phí prefill của video dài.

Nếu hệ thống Video VLM có:

* Prefill 8 giây.
* Generation 2 giây.

và speculative decoding giảm generation xuống 1 giây, tổng thời gian chỉ giảm:

$$ 10\text{ s} \rightarrow 9\text{ s} $$

Vì vậy, acceleration toàn diện cần kết hợp:

$$ \text{Frame reduction} + \text{Visual token compression} + \text{Efficient prefill} + \text{Fast decoding} $$

---

# 23. Phân biệt các khái niệm liên quan

## VLM

Mô hình kết hợp hình ảnh và ngôn ngữ. Có thể chỉ xử lý ảnh.

## Multimodal LLM

LLM nhận nhiều modality như ảnh, video, audio, tài liệu hoặc sensor.

## Video VLM

VLM chuyên xử lý dữ liệu có chiều thời gian là video.

## Video encoder

Mô hình sinh representation cho video; không nhất thiết có khả năng hội thoại hoặc sinh văn bản.

## Video-language model

Có thể là mô hình retrieval, classification hoặc generation kết hợp video và text.

## Video-LLM

Thường ám chỉ hệ thống video encoder kết nối với LLM có khả năng instruction-following và generation.

---

# 24. Ví dụ pipeline hoàn chỉnh

Giả sử người dùng đưa một video 10 phút và hỏi:

> Người đàn ông đã đặt chiếc chìa khóa ở đâu trước khi rời phòng?

Pipeline có thể là:

### Bước 1: Chia video thành clip

$$ V \rightarrow \{C_1,C_2,\ldots,C_N\} $$

### Bước 2: Sinh embedding từng clip

$$ z_i = E(C_i) $$

### Bước 3: Truy xuất theo câu hỏi

$$ s_i = \operatorname{sim}(z_i,E_q(q)) $$

### Bước 4: Chọn các clip liên quan

$$ \mathcal{C}^* = \operatorname{TopK}(s_i) $$

### Bước 5: Lấy frame chi tiết trong clip

Chọn các frame chứa:

* Người đàn ông.
* Chìa khóa.
* Hành động đặt xuống.
* Hành động rời phòng.

### Bước 6: Mã hóa temporal sequence

$$ H = E_{\text{temporal}}(F_{t_1},\ldots,F_{t_k}) $$

### Bước 7: LLM reasoning

Mô hình xác định:

1. Vị trí chìa khóa.
2. Hành động xảy ra trước khi rời phòng.
3. Evidence timestamp.

### Bước 8: Trả lời có dẫn chứng

> Người đàn ông đặt chìa khóa trên chiếc bàn cạnh cửa, vào khoảng 07:42, rồi mới rời khỏi phòng.

Một hệ thống tốt nên trả cả evidence span, thay vì chỉ sinh một câu trả lời không thể kiểm chứng.

---

# 25. Các hướng nghiên cứu quan trọng

## 25.1. Long-video understanding

Xử lý video hàng giờ mà không mất sự kiện nhỏ.

## 25.2. Adaptive frame selection

Tự quyết định frame nào cần xem dựa trên câu hỏi và trạng thái reasoning.

## 25.3. Dynamic token allocation

Clip quan trọng nhận nhiều token; clip ít quan trọng nhận ít token.

$$ M_i \propto \operatorname{importance}(C_i,q) $$

với tổng ngân sách:

$$ \sum_i M_i \leq B $$

## 25.4. Temporal grounding có thể kiểm chứng

Mỗi câu trả lời gắn với timestamp hoặc frame evidence.

## 25.5. Streaming Video VLM

Xử lý video online mà không cần xem toàn bộ tương lai.

$$ p(y_t|V_{\leq t}) $$

Ứng dụng trong robot, camera và live video.

## 25.6. Multimodal memory

Lưu trạng thái đối tượng và sự kiện qua thời gian dài.

## 25.7. Active perception

Mô hình tự quyết định:

* Xem lại đoạn nào.
* Lấy thêm frame ở đâu.
* Tăng độ phân giải vùng nào.
* Có cần nghe audio hay đọc subtitle không.

## 25.8. Efficient inference

Tối ưu đồng thời:

* Video decoding.
* Frame selection.
* Vision encoding.
* Token pruning.
* Prefill.
* KV cache.
* Autoregressive decoding.

---

# 26. Cách nhìn tổng quát nhất

Có thể coi Image VLM là mô hình:

$$ \text{What is present?} $$

Trong khi Video VLM cần giải quyết thêm:

$$ \text{What changed?} $$

$$ \text{In what order?} $$

$$ \text{When did it happen?} $$

$$ \text{Why did it happen?} $$

$$ \text{What happened before or after it?} $$

$$ \text{Which frames provide the evidence?} $$

Do đó, Video VLM không đơn thuần là “VLM nhận nhiều ảnh”, mà là một hệ thống cần kết hợp:

$$ \boxed{\text{Spatial perception} + \text{Temporal modeling} + \text{Memory} + \text{Multimodal alignment} + \text{Language reasoning}} $$

Thách thức lớn nhất hiện nay không chỉ là tăng số frame đầu vào, mà là **chọn đúng thông tin, giữ đúng thứ tự thời gian, không làm mất evidence khi nén và suy luận hiệu quả dưới ngân sách token hữu hạn**.
