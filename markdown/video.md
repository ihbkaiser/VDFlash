# Nhánh speculative decoding cho Video-LLM đang làm gì?

Nhánh này hiện còn khá nhỏ nhưng đã hình thành một tiến trình kỹ thuật rõ ràng. Các công trình cốt lõi gồm **Sparse-to-Dense** tại ACL 2025, **SpecVLM** tại EMNLP 2025, **Sparrow** tại ACL 2026, **ParallelVLM** tại CVPR 2026; ngoài ra còn có các preprint đáng chú ý như **HIPPO** và **LVSpec** trong năm 2026. ([ACL Anthology][1])

Điểm chung của chúng không phải là cải tiến vision encoder, mà là làm cho vòng lặp:

$$
\text{draft nhiều token} \rightarrow \text{target verify song song}
$$

hoạt động hiệu quả khi prefix chứa hàng nghìn đến hàng chục nghìn video tokens.

---

## 1. Tại sao speculative decoding thông thường thất bại với video?

Với speculation length $\gamma$, chi phí một vòng có thể viết gần đúng:

$$
T_{\mathrm{round}} = \gamma T_d + T_v^\gamma,
$$

trong đó:

* (T_d): thời gian draft một token;
* (T_v^\gamma): target verify (\gamma) token song song;
* (\tau): số token trung bình được chấp nhận.

Thời gian trên mỗi token đầu ra là:

$$
T_{\mathrm{token}} = \frac{\gamma T_d+T_v^\gamma}{\tau}.
$$

Speculative decoding chỉ có lợi khi:

1. (T_d) đủ nhỏ;
2. (\tau) đủ lớn.

Đối với Video-LLM, cả hai điều kiện này đều khó thỏa mãn.

### Draft model vẫn bị KV-cache của video làm chậm

Dù drafter chỉ có 7B tham số so với target 72B, ở mỗi decoding step nó vẫn phải đọc KV cache của hàng chục nghìn visual tokens. Khi đó latency không còn chủ yếu phụ thuộc vào số tham số mà phụ thuộc vào lượng KV phải chuyển từ HBM vào bộ nhớ gần GPU. SpecVLM chỉ ra rằng khi context video tăng, KV-cache access trở thành bottleneck chính của drafter. ([arXiv][2])

### Visual tokens có thể làm drafter dự đoán kém hơn

Drafter nhỏ phải attention trên một chuỗi mà hơn 99% token là visual tokens. Sparrow cho thấy các phương pháp vốn tốt trên ảnh có thể giảm mạnh accepted length khi chuyển sang video dài; full-visual-input drafting thậm chí có thể đạt **negative speedup** vì latency tăng trong khi chất lượng proposal giảm do attention dilution. ([arXiv][3])

### Pruning làm drafter nhanh nhưng dễ lệch target

Nếu cắt video tokens quá mạnh, drafter có thể bỏ qua:

* hành động ngắn;
* vật thể nhỏ;
* scene ở giữa video;
* quan hệ thời gian;
* chi tiết được câu hỏi nhắc tới.

Khi đó (T_d) giảm nhưng (\tau) cũng giảm. Vì accepted prefix bị dừng tại token sai đầu tiên, giảm alignment một chút có thể làm accepted length giảm đáng kể.

### Draft và target còn phải chờ nhau

Speculative decoding cổ điển chạy tuần tự:

$$
\text{draft} \rightarrow \text{verify} \rightarrow \text{draft tiếp}.
$$

Trong thời gian drafter chạy, target rảnh; trong lúc target verify, drafter rảnh. ParallelVLM minh họa rằng với 24K video tokens, riêng target prefill có thể mất hàng chục giây, còn phần draft prefill bổ sung tạo thành một chi phí tuần tự không nhỏ. ([arXiv][4])

Do đó, các phương pháp hiện nay đang xử lý bốn nút thắt khác nhau:

$$
\boxed{\text{KV cache} + \text{visual alignment} + \text{sequential waiting} + \text{strict verification}}
$$

---

# 2. Pipeline chung của các phương pháp hiện tại

Một hệ thống speculative decoding cho Video-LLM thường thực hiện:

$$
\text{Video} \rightarrow V_{1:m} \rightarrow \begin{cases}
\text{Target: full visual context} \\
\text{Draft: compressed/sparse/implicit context}
\end{cases}
$$

Sau đó:

1. Target giữ toàn bộ video context để bảo đảm output cuối.
2. Drafter nhận một biểu diễn rẻ hơn.
3. Drafter sinh (\gamma) token.
4. Target kiểm tra (\gamma) token trong một forward pass.
5. Chấp nhận prefix hợp lệ và lặp lại.

Các phương pháp khác nhau chủ yếu ở ba câu hỏi:

* **Drafter nhìn phần nào của video?**
* **Drafter và target chạy tuần tự hay song song?**
* **Verifier yêu cầu exact match hay semantic match?**

---

# 3. Hướng 1 — Dùng sparse attention thay vì một draft model riêng

## Sparse-to-Dense

Sparse-to-Dense không dùng một model nhỏ độc lập. Nó dùng chính Video-LLM gốc dưới hai chế độ:

$$
\mathcal{M}_s: \text{sparse top-}K \text{ attention}
$$

và:

$$
\mathcal{M}: \text{dense full attention}.
$$

Hai model chia sẻ hoàn toàn tham số; khác nhau duy nhất ở lượng KV cache được đọc.

### Cách chọn sparse KV cache

Trong bước prefill, với từng layer và attention head, phương pháp tính mức attention trung bình từ text tokens đến mỗi visual token:

$$
s_i^{(l,h)} = \frac{1}{|X_t|} \sum_{x\in X_t} A_{l,h}(x,v_i).
$$

Sau đó chỉ giữ top-$K$ visual KV pairs cho mỗi layer/head:

$$
\operatorname{Cache}_s^{(l,h)} = \operatorname{TopK}_i \left(s_i^{(l,h)}\right).
$$

Việc lựa chọn chỉ được thực hiện một lần ở prefill, thay vì chọn lại ở từng decoding step. Textual KV cache vẫn được giữ đầy đủ. ([arXiv][5])

### Khi decode

* Sparse mode đọc top-(K) visual KV để sinh (\gamma) token.
* Dense mode đọc full KV một lần để verify.
* Nếu (n) token đầu khớp, chúng được chấp nhận; dense model còn cung cấp thêm một bonus token.

Chi phí I/O của drafter giảm gần từ:

$$ \gamma(m_v+m_t) $$

xuống:

$$ \gamma(K+m_t), \qquad K\ll m_v. $$

### Đặc điểm

**Ưu điểm**

* Không cần draft model thứ hai.
* Không cần training.
* Không tăng memory để lưu thêm model.
* Output giữ giống dense Video-LLM vì dense model vẫn verify đầy đủ.

**Hạn chế**

* Cùng model nên drafter vẫn phải chạy toàn bộ layer.
* Top-(K) visual cache là cố định sau prefill.
* Không thay đổi theo token đang được sinh.
* Speedup phụ thuộc mạnh vào sparse-attention kernel thực tế.

Sparse-to-Dense báo cáo tốc độ wall-clock tối đa khoảng ($1.94\times$). ([arXiv][6])

Insight của hướng này là:

> Không nhất thiết cần một model nhỏ hơn; có thể tạo drafter nhanh hơn bằng cách giảm working set của cùng một model.

---

# 4. Hướng 2 — Drafter nhỏ nhưng chỉ nhìn visual tokens đã prune

Đây là pipeline phổ biến nhất:

$$
\begin{aligned}
\text{Target: } & [V_{\mathrm{full}},X], \\
\text{Draft: }  & [V_{\mathrm{selected}},X],
\quad |V_{\mathrm{selected}}|\ll |V_{\mathrm{full}}|.
\end{aligned}
$$

Target giữ thông tin đầy đủ; chỉ drafter bị nén. Vì vậy, nếu verification vẫn dùng rejection sampling chuẩn, việc pruning drafter không thay đổi output distribution cuối cùng.

## Verifier-guided pruning trong SpecVLM

SpecVLM khai thác một quan sát thực nghiệm: accepted length của drafter khá ít nhạy với random pruning ở mức vừa phải; thậm chí bỏ một phần video tokens có thể cải thiện proposal vì giảm dư thừa. Tuy nhiên, random pruning suy giảm mạnh khi vượt khoảng 50%, nên cần lựa chọn có hướng dẫn. ([arXiv][2])

### Bước 1: Target đánh giá visual token

Target thực hiện full prefill. Từ attention matrix, phương pháp lấy phần language-query–visual-key:

$$
G = \operatorname{Attention}(Q_L,K_V).
$$

Điểm của visual token $v_j$:

$$
a_j = \frac{1}{|L|} \sum_{i\in L} G_{ij},
$$

sau đó trung bình qua layer và head.

Khác với pruning thông thường nhằm giữ accuracy của một model bị nén, mục tiêu ở đây là:

$$
\text{chọn visual tokens làm drafter giống target nhất}.
$$

([arXiv][2])

### Bước 2: Giữ phần đầu của phân phối attention

Attention trên video tokens thường có long-tail:

* một nhóm nhỏ nhận phần lớn attention;
* phần còn lại đều có điểm thấp và khó phân biệt.

SpecVLM dùng Top-$P$ retention: sắp xếp token theo $a_j$ và giữ tập nhỏ nhất có tổng attention đạt ngưỡng $\lambda_r$:

$$
V_R = \min\left\{ V': \sum_{v_j\in V'}a_j \ge \lambda_r\sum_j a_j \right\}.
$$

### Bước 3: Lấy mẫu đều từ phần tail

Nếu tiếp tục chỉ giữ token attention cao, mô hình có thể mất cấu trúc không gian. Vì vậy, trong nhóm attention thấp, SpecVLM lấy token theo khoảng cách không gian đều:

$$
V_{\mathrm{draft}} = V_R \cup V_U.
$$

Cách này bảo tồn:

* các token semantic nổi bật;
* độ phủ không gian của phần còn lại.

Target dùng full video; drafter được prefill bằng tập token có thể chỉ còn khoảng 10%. Paper báo cáo pruning tới 90%, với tốc độ decoding tối đa ($2.68\times$) trên LLaVA-OneVision-72B và ($2.11\times$) trên Qwen2.5-VL-32B. ([ACL Anthology][7])

### Vấn đề của attention-only pruning

Attention score không hoàn toàn tương đương semantic importance. Các paper sau chỉ ra hai vấn đề:

1. **Position bias:** token ở đầu/cuối frame hoặc gần ranh giới modality có thể nhận attention cao bất thường.
2. **Semantic incompleteness:** attention cao có thể giữ một vùng nổi bật nhưng mất diễn biến thời gian hoặc cấu trúc toàn cảnh.

Đây là lý do HIPPO và ParallelVLM chuyển từ raw attention sang tín hiệu semantic phong phú hơn.

---

# 5. Hướng 3 — Kết hợp relevance, temporal novelty và spatial diversity

## Holistic pruning trong HIPPO

HIPPO không xem token importance là một scalar attention đơn lẻ. Nó kết hợp ba loại tín hiệu.

### Global semantic relevance

Tính text-to-video attention nhằm đo mức liên quan với prompt:

$$
S_i^{\mathrm{global}} = \operatorname{AggregateAttention}(Q_{\mathrm{text}},K_{v_i}).
$$

Tín hiệu này giữ các vật thể hoặc vùng liên quan trực tiếp tới câu hỏi.

### Inter-frame temporal novelty

Với token ở cùng vị trí không gian giữa hai frame lân cận, HIPPO đo cosine similarity. Nếu hai token rất giống nhau, vùng đó ít thay đổi và có thể dư thừa:

$$
S_i^{\mathrm{temporal}} = 1 - \cos(v_i^{(f)},v_i^{(f\pm1)}).
$$

Nhờ đó:

* background tĩnh nhận điểm thấp;
* chuyển động, cử chỉ hoặc thay đổi cảnh nhận điểm cao.

### Intra-frame spatial complexity

Trong từng spatial crop, phương pháp tính similarity của token với các token xung quanh. Variance cao cho thấy token nằm ở vùng có cấu trúc phức tạp hoặc biên semantic quan trọng:

$$
S_i^{\mathrm{spatial}} = \operatorname{Var} \left( \{\cos(v_i,v_j): v_j \in \text{crop}\} \right).
$$

### Fusion

Ba score được chuẩn hóa trong từng frame rồi kết hợp:

$$
S_i = w_g\tilde S_i^{\mathrm{global}} + w_t\tilde S_i^{\mathrm{temporal}} + w_s\tilde S_i^{\mathrm{spatial}}.
$$

Sau đó giữ top token theo score tổng hợp. ([arXiv][8])

Điểm khác biệt quan trọng là:

* SpecVLM ưu tiên **target attention + spatial coverage**.
* HIPPO ưu tiên **semantic relevance + temporal change + spatial complexity**.

HIPPO phù hợp hơn với trường hợp hành động quan trọng nhưng không nhận attention cao nhất, hoặc video có position bias mạnh. Đây hiện là preprint; paper báo cáo tối đa khoảng ($3.51\times$), nhưng con số này không nên so sánh trực tiếp với các paper khác vì backbone, GPU và pipeline song song khác nhau. ([arXiv][9])

---

# 6. Hướng 4 — Dùng sự thay đổi alignment qua các layer

## UV-Prune trong ParallelVLM

ParallelVLM cho rằng raw attention trả lời câu hỏi:

> Model đang nhìn token nào?

nhưng chưa chắc trả lời tốt:

> Token nào đang trở nên liên quan hơn với câu hỏi khi đi qua model?

Phương pháp vì vậy tính cosine similarity giữa video token $V_i$ và text token $X_j$ tại các early layers:

$$
S_{ij}^{l} = \frac{V_i^l \cdot X_j^l}{|V_i^l||X_j^l|}.
$$

Sau đó đo tổng mức tăng similarity qua các layer:

$$
\overline{\Delta S_i} = \sum_j\sum_l \left( S_{ij}^{l} - S_{ij}^{l-1} \right).
$$

Token có (\overline{\Delta S_i}) lớn là token đang được target biến đổi thành representation ngày càng phù hợp với câu hỏi. Drafter giữ top-(K) theo score này. ([arXiv][4])

### Tại sao điều này giảm positional bias?

Một token nhận attention cao chỉ vì vị trí có thể vẫn không tăng semantic alignment qua layer. Ngược lại, một frame giữa video chứa hành động được hỏi có thể dần alignment mạnh với prompt, dù raw attention ban đầu không cao.

Do đó UV-Prune cố giữ:

* semantic relevance;
* temporal coherence;
* mid-video evidence;
* alignment với distribution của target.

Đây là cách nhìn pruning như **knowledge transfer từ target sang drafter**, thay vì chỉ compression.

---

# 7. Hướng 5 — Không cho drafter nhìn raw video tokens

Pruning vẫn có một giới hạn: dù chỉ giữ 10% của 25K tokens, drafter vẫn phải xử lý 2.5K visual tokens. Với video cực dài, lượng này vẫn lớn.

Sparrow đưa ra một hướng khác:

$$
\boxed{\text{Target xử lý video}; \quad \text{drafter chỉ nhận visual semantics đã được target tích hợp}.}
$$

## Visual semantic internalization

Sparrow phân tích layer-wise và nhận thấy:

* early/middle layers cần raw visual tokens để thực hiện cross-modal interaction;
* ở các layer sâu, visual semantics quan trọng đã được tích hợp vào hidden states tại text positions;
* loại visual flow sau một độ sâu nhất định ít ảnh hưởng đến prediction.

Từ đó, raw video tokens được xem là cần thiết cho target nhưng không nhất thiết cần thiết cho một drafter hoạt động trên representation sâu. ([arXiv][3])

## Hidden-state reuse

Tại decoding step (t), drafter nhận:

* embedding của token hiện tại (e_t);
* hidden state text của target ở timestep trước (h_{t-1}^{T}).

Hai representation được ghép và chiếu:

$$
z_t = FC(e_t \oplus h_{t-1}^{T}).
$$

Hidden state của target đã chứa visual semantics, nên nó đóng vai trò như một “glimpse” nén của toàn bộ video.

## Text-anchored attention

Drafter bỏ toàn bộ visual KV cache và chỉ attention trên text positions:

$$
\operatorname{Attention}_{\mathrm{VATA}} = \operatorname{Softmax} \left( \frac{Q_tK_{\mathcal T}^{\top}}{\sqrt d} \right) V_{\mathcal T}.
$$

Chi phí attention của drafter thay đổi từ:

$$ O((L_{\mathrm{vis}}+L_{\mathrm{text}})^2) $$

xuống gần:

$$ O(L_{\mathrm{text}}^2). $$

([arXiv][3])

## Intermediate-layer visual-state bridging

Sparrow vẫn dùng visual information khi huấn luyện drafter. Thay vì raw vision embeddings, nó lấy intermediate target states tại các layer có cross-modal interaction mạnh để supervision cho drafter.

Điều này tạo ra một sự tách biệt:

* **Training:** drafter học alignment từ semantic-rich visual states.
* **Inference:** drafter không trực tiếp đọc raw visual tokens.

Paper cũng dùng multi-token prediction để giảm chênh lệch giữa huấn luyện one-step và inference nhiều token liên tiếp.

Sparrow được công bố tại ACL 2026 và báo cáo decoding speedup trung bình ($2.82\times$) ở thiết lập 25K visual tokens; end-to-end speedup được báo cáo riêng và thấp hơn vì target prefill vẫn phải xử lý full video. ([ACL Anthology][10])

### Trade-off

**Ưu điểm**

* Draft latency gần như không tăng theo số visual tokens.
* Tránh attention dilution.
* Không cần visual KV cache cho drafter.

**Hạn chế**

* Phải huấn luyện một drafter chuyên biệt.
* Drafter gắn chặt với target architecture và hidden dimension.
* Có dependency giữa target timestep và draft input.
* Khó dùng một drafter chung cho nhiều target models.

---

# 8. Hướng 6 — Chạy drafter và verifier song song

Pruning chỉ giảm $T_d$, nhưng pipeline cổ điển vẫn cộng:

$$ \gamma T_d + T_v^\gamma. $$

Các phương pháp parallel cố chuyển thành gần:

$$ \max(\gamma T_d,T_v^\gamma). $$

## Parallel prefill

Target prefill full video thường lâu hơn draft prefill. Vì vậy có thể chạy:

$$
\begin{cases}
\text{Target: prefill full video} \\
\text{Draft: pruning + compact prefill + startup drafting}
\end{cases}
$$

cùng lúc.

Khi target hoàn thành prefill:

* draft KV cache đã sẵn sàng;
* một window candidate đầu tiên đã được sinh;
* target có thể verify ngay.

ParallelVLM thực hiện chính xác pipeline này, trong đó target truyền early-layer representations cho UV-Prune trong khi target prefill vẫn tiếp tục. ([arXiv][4])

## Parallel decoding

Ở round $i$:

* target verify candidate của round $i-1$;
* drafter đồng thời sinh candidate cho round $i$.

$$
\begin{cases}
\text{Target}: \operatorname{Verify}(C_{i-1}) \\
\text{Draft}: \operatorname{Generate}(C_i)
\end{cases}
$$

Nếu candidate trước bị reject, candidate tiếp theo có thể phải rollback hoặc bỏ đi. Vì vậy parallel speculative decoding đổi thời gian chờ thành một trade-off giữa:

* overlap computation;
* wasted drafting sau rejection.

## Chọn draft window theo speed ratio

ParallelVLM đặt window gần tỷ lệ tốc độ:

$$
\gamma \approx c^* = \frac{T_p}{T_q(\alpha)},
$$

trong đó (T_q(\alpha)) là draft latency sau pruning ratio (\alpha).

Ví dụ paper đưa ra:

* chưa pruning: (T_q=78) ms, (T_p=420) ms, nên (\gamma\approx5);
* pruning 90%: (T_q=47) ms, nên (\gamma\approx9).

Như vậy pruning không chỉ giảm draft latency mà còn cho phép speculation xa hơn. ([arXiv][4])

## Adaptive optimistic/conservative scheduling trong HIPPO

HIPPO không luôn chạy cùng một kiểu song song.

### Optimistic mode

Khi round trước được chấp nhận hoàn toàn:

* target verify current batch;
* drafter đồng thời sinh batch tiếp theo.

Nếu current batch tiếp tục được chấp nhận, batch tiếp theo đã sẵn sàng.

### Conservative mode

Nếu round trước có rejection:

* target kiểm tra token draft đầu tiên;
* drafter sinh phần còn lại song song;
* nếu token đầu bị reject, drafting bị dừng sớm để tránh lãng phí.

Cơ chế này là một dạng adaptive draft scheduling dựa trên acceptance history. ([arXiv][8])

### So sánh hai hướng parallel

| Thành phần      | HIPPO                                | ParallelVLM                             |
| --------------- | ------------------------------------ | --------------------------------------- |
| Pruning signal  | Global + temporal + spatial          | Layer-wise vision–text alignment change |
| Prefill overlap | Có                                   | Có                                      |
| Decode overlap  | Adaptive optimistic/conservative     | Pipelined windows                       |
| Draft window    | Thích nghi gián tiếp theo acceptance | Chọn theo measured speed ratio          |
| Trạng thái      | Preprint                             | CVPR 2026                               |

ParallelVLM là phương pháp peer-reviewed tại CVPR 2026 và báo cáo tốc độ tối đa ($3.36\times$) trên LLaVA-OneVision-72B và ($2.42\times$) trên Qwen2.5-VL-32B trong thiết lập của paper. ([Open Access CVF][11])

---

# 9. Hướng 7 — Nới lỏng verifier thay vì chỉ cải thiện drafter

Các phương pháp trên cố tăng acceptance bằng cách làm drafter giống target hơn. Nhưng strict speculative decoding dừng tại mismatch đầu tiên, kể cả khi hai chuỗi chỉ khác về cách diễn đạt.

Ví dụ:

* Draft: “A man is **riding** a bicycle.”
* Target: “A man **rides** a bicycle.”

Strict matching reject tại “riding/rides”, dù semantic gần như giống nhau.

## LVSpec: strict trên visual anchors, loose trên linguistic fillers

LVSpec đưa ra quan sát rằng output Video-LLM gồm:

$$
\text{sparse visual anchors} + \text{many linguistic fillers}.
$$

Các visual anchors như tên vật thể, hành động, số lượng hoặc màu sắc ít nhưng quyết định grounding. Các token chức năng như mạo từ, giới từ hoặc các cách diễn đạt tương đương có thể được kiểm tra lỏng hơn. ([arXiv][12])

## Xác định visual relevance của text token

Target giữ visual hidden states $H_V$. Với candidate text hidden state $H_D^s$, LVSpec tính cosine similarity với visual tokens:

$$
C = \cos(H_D^s,H_V).
$$

Để tránh background video làm pha loãng score, nó không trung bình toàn bộ visual tokens mà lấy trung bình Top-$N$ similarities:

$$
A_s = \frac{1}{N} \sum_{n=1}^{N} \operatorname{TopN}(C,N).
$$

Token có score cao được xem là visual-relevant.

### Verification

* Visual-relevant token: strict verification.
* Visual-irrelevant token: loose verification.
* Candidate có positional shift nhỏ có thể được “cứu” nếu xuất hiện trong một vùng lân cận của target sequence.

([arXiv][12])

### Điểm cần đặc biệt cẩn thận

LVSpec không còn lossless theo nghĩa chuẩn của speculative decoding. Nó báo cáo giữ hơn 99.8% downstream performance, nhưng loose acceptance có thể làm output khác target autoregressive distribution. ([arXiv][13])

Do đó cần phân biệt:

$$
\begin{array}{ll}
\textbf{Exact/lossless SD:} & P_{\mathrm{output}} = P_{\mathrm{target}}, \\[2mm]
\textbf{High-fidelity loose SD:} & \operatorname{Quality}_{\mathrm{output}} \approx \operatorname{Quality}_{\mathrm{target}}.
\end{array}
$$

LVSpec đổi guarantee chặt lấy accepted length cao hơn.

---

# 10. Các phương pháp hiện tại khác nhau ở đâu?

| Hướng             | Drafter nhận gì?                   | Cách giảm chi phí                          |       Cần train? | Guarantee                            |
| ----------------- | ---------------------------------- | ------------------------------------------ | ---------------: | ------------------------------------ |
| Sparse-to-Dense   | Full prefix nhưng sparse visual KV | Top-(K) attention                          |            Không | Exact                                |
| SpecVLM           | Khoảng 10% visual tokens           | Target-attention pruning                   |            Không | Exact qua full verifier              |
| HIPPO             | Visual tokens được chọn holistic   | Semantic + temporal + spatial pruning      |            Không | Exact qua full verifier              |
| Sparrow           | Target text hidden states          | Bỏ raw visual KV khỏi drafter              |               Có | Có thể giữ exact với strict verifier |
| ParallelVLM       | UV-Pruned visual tokens            | Semantic pruning + pipeline overlap        |            Không | Exact                                |
| LVSpec            | Tùy drafter nền                    | Loose visual-aware verification            |            Không | High-fidelity, không exact           |
| VideoSpeculateRAG | Các answer candidates              | Answer-level parallel generation/reranking | Không nhất thiết | Không phải token-level exact SD      |

VideoSpeculateRAG nên được xem là một nhánh lân cận: drafter tạo nhiều **câu trả lời hoàn chỉnh** theo các retrieved documents, sau đó verifier chấm và chọn bằng reliability cùng CLIP alignment. Nó không phải speculative decoding token-level với rejection sampling theo nghĩa chặt. ([arXiv][14])

---

# 11. Tiến trình phát triển của nhánh Video-LLM speculative decoding

Có thể tóm tắt sự phát triển thành năm bước.

### Giai đoạn 1: Giảm KV được đọc

$$
\text{full attention} \rightarrow \text{sparse top-}K
$$

Sparse-to-Dense chứng minh same-model sparse drafting khả thi.

### Giai đoạn 2: Giảm số visual tokens của drafter

$$
V_{\mathrm{full}} \rightarrow V_{\mathrm{important}} + V_{\mathrm{uniform}}
$$

SpecVLM khai thác verifier-guided pruning.

### Giai đoạn 3: Chọn token theo semantics tốt hơn

$$
\text{raw attention} \rightarrow \text{semantic + temporal + spatial alignment}
$$

HIPPO và ParallelVLM xử lý position bias và semantic incompleteness.

### Giai đoạn 4: Loại raw video khỏi drafter

$$
V_{\mathrm{raw}} \rightarrow h_{\mathrm{text}}^{\mathrm{target}}(V,X)
$$

Sparrow chuyển từ visual-token compression sang semantic-state reuse.

### Giai đoạn 5: Tối ưu execution và verification

$$
\text{sequential} \rightarrow \text{parallel draft/verify}
$$

và:

$$
\text{strict exact match} \rightarrow \text{visual-aware loose matching}.
$$

ParallelVLM/HIPPO xử lý scheduling; LVSpec xử lý acceptance ceiling.

---

# 12. Insight kỹ thuật quan trọng nhất

## 12.1. Video speculative decoding thực chất là bài toán KV management

Trong text-only SD, draft model nhỏ thường đồng nghĩa với draft nhanh. Trong video:

$$
\text{small drafter} \not\Rightarrow \text{fast drafter},
$$

vì nó vẫn phải đọc visual KV cache rất dài.

Do đó, kích thước context mà drafter nhìn thấy có thể quan trọng hơn số parameter.

## 12.2. Visual information nên được xử lý một lần bởi target

Sparrow gợi ý một thiết kế mạnh:

$$
\text{target performs perception} \rightarrow \text{drafter consumes semantic state}.
$$

Điều này tránh việc một draft model nhỏ phải lặp lại cross-modal reasoning mà target đã thực hiện.

## 12.3. Pruning objective không phải task accuracy

Visual-token pruning thông thường tối ưu:

$$ \operatorname{Accuracy}(M(V')). $$

Pruning cho speculative decoding nên tối ưu:

$$
\operatorname{Alignment} \left( q(\cdot\mid V'), p(\cdot\mid V) \right),
$$

hoặc trực tiếp:

$$ \mathbb{E}[\text{accepted prefix length}]. $$

Một compressed drafter có thể trả lời video kém khi chạy độc lập nhưng vẫn là drafter tốt nếu prediction distribution đủ giống target tại các token quan trọng.

## 12.4. Pruning và parallelization có hiệu ứng cộng hưởng

Pruning làm (T_q) nhỏ hơn. Khi (T_q) nhỏ hơn:

* window (\gamma) có thể tăng;
* draft work dễ ẩn sau target verification hơn;
* pipeline utilization tốt hơn.

Do đó, parallel execution không chỉ là phần tối ưu hệ thống đặt sau pruning; hai thành phần phải được đồng thiết kế.

## 12.5. Long video cần dynamic representation

Các phương pháp hiện tại vẫn chủ yếu chọn visual token subset một lần tại prefill. Nhưng visual evidence cần thiết thay đổi theo token đang sinh:

* sinh tên người cần frame chứa người;
* sinh hành động cần temporal segment;
* sinh mô tả nền cần spatial context;
* sinh token chức năng gần như không cần video.

Một hướng tự nhiên tiếp theo là:

$$
V_t^* = \operatorname{Route}(V, h_t, Q, \text{acceptance history}),
$$

nghĩa là chọn visual context động theo decoding state.

---

# 13. Khoảng trống nghiên cứu còn rõ

### Adaptive visual budget theo decoding step

Hầu hết phương pháp dùng pruning ratio cố định như 90%. Một controller có thể quyết định:

$$
r_t = f(\text{draft entropy}, \text{visual relevance}, \text{recent acceptance}).
$$

Khi sinh visual anchor, giữ nhiều video tokens; khi sinh linguistic filler, giảm budget.

### Joint optimization của pruning và acceptance

Các phương pháp hiện tại thường thiết kế pruning heuristic rồi đánh giá accepted length. Có thể tối ưu trực tiếp:

$$
\max_\theta \frac{\mathbb{E}[\tau_\theta]}{T_{\mathrm{draft},\theta} + T_{\mathrm{verify},\theta}}.
$$

### Temporal-unit selection thay vì token-level selection

Pruning từng patch có thể phá cấu trúc hành động. Một lựa chọn tốt hơn có thể là hierarchy:

$$
\text{segment} \rightarrow \text{frame} \rightarrow \text{region} \rightarrow \text{patch}.
$$

### Streaming video

Các công trình hiện tại chủ yếu giả định video đã có đầy đủ trước prefill. Streaming cần xử lý:

* visual KV cập nhật liên tục;
* invalidation của draft cache;
* temporal retrieval;
* speculative generation khi video chưa kết thúc.

### Đánh giá end-to-end

Một số paper báo cáo decode speedup, một số báo cáo wall-clock/end-to-end speedup. Với video dài:

$$
T_{\mathrm{prefill}} \gg T_{\mathrm{decode}}
$$

có thể xảy ra, nên ($3\times$) decode speedup không đồng nghĩa ($3\times$) end-to-end speedup. Sparrow chẳng hạn báo cáo riêng decoding speedup và end-to-end speedup, cho thấy khoảng cách đáng kể khi target vẫn phải prefill 25K visual tokens. ([arXiv][3])

## Kết luận

Các phương pháp Video-LLM speculative decoding hiện nay đang hội tụ vào một kiến trúc:

$$
\boxed{
\begin{aligned}
&\text{Target xử lý full video}, \\
&\text{Target cung cấp semantic guidance}, \\
&\text{Drafter dùng compact/implicit visual context}, \\
&\text{Draft và verify chạy chồng lấp}, \\
&\text{Verification nghiêm ngặt tại visual anchors}.
\end{aligned}
}
$$

Ba hướng mạnh nhất hiện tại là:

1. **Verifier-guided semantic pruning** như ParallelVLM/HIPPO.
2. **Target-state reuse, bỏ visual KV khỏi drafter** như Sparrow.
3. **Visual-aware adaptive verification và scheduling** như LVSpec kết hợp parallel SD.

Một phương pháp mới có khả năng tạo đóng góp tốt nên không chỉ đề xuất thêm một heuristic pruning, mà cần đồng tối ưu **visual representation, acceptance length và execution schedule**.

[1]: https://aclanthology.org/2025.acl-short.59/?utm_source=chatgpt.com "Sparse-to-Dense: A Free Lunch for Lossless Acceleration ..."
[2]: https://arxiv.org/html/2508.16201v1 "\scalerel* SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning"
[3]: https://arxiv.org/html/2602.15318 "Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs"
[4]: https://arxiv.org/html/2603.19610 "ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding"
[5]: https://arxiv.org/html/2505.19155 "Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs"
[6]: https://arxiv.org/abs/2505.19155?utm_source=chatgpt.com "Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs"
[7]: https://aclanthology.org/2025.emnlp-main.366/?utm_source=chatgpt.com "Enhancing Speculative Decoding of Video LLMs via ..."
[8]: https://arxiv.org/html/2601.08273 "HIPPO: Accelerating Video Large Language Models Inference via Holistic-aware Parallel Speculative Decoding"
[9]: https://arxiv.org/abs/2601.08273?utm_source=chatgpt.com "HIPPO: Accelerating Video Large Language Models Inference via Holistic-aware Parallel Speculative Decoding"
[10]: https://aclanthology.org/2026.acl-long.450/?utm_source=chatgpt.com "Sparrow: Text-Anchored Window Attention with Visual- ..."
[11]: https://openaccess.thecvf.com/content/CVPR2026/html/Kong_ParallelVLM_Lossless_Video-LLM_Acceleration_with_Visual_Alignment_Aware_Parallel_Speculative_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[12]: https://arxiv.org/html/2604.05650v2 "See the Forest for the Trees See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs"
[13]: https://arxiv.org/abs/2604.05650?utm_source=chatgpt.com "See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs"
[14]: https://arxiv.org/abs/2601.01513?utm_source=chatgpt.com "FastV-RAG: Towards Fast and Fine-Grained Video QA with Retrieval-Augmented Generation"
