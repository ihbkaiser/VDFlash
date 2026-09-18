Ý tưởng cốt lõi của bạn rất đáng kiểm chứng: visual tokens có thể không còn cần thiết về mặt nội dung, nhưng một số visual slots vẫn có thể hữu ích vì chúng tiếp tục tham gia vào mẫu số của softmax. Tuy nhiên, cần tách rõ ba cơ chế: thông tin thị giác, số lượng attention slots và thay đổi vị trí positional encoding.

## 1. Cơ chế cần kiểm chứng

Với một query $q_i$, attention được tính bởi:

$$
a_{ij}
=
\frac{\exp(q_i k_j/\sqrt d)}
{\sum_{l}\exp(q_i k_l/\sqrt d)}
$$

và đầu ra là:

$$
o_i=\sum_j a_{ij}v_j.
$$

Giả sử thông tin thị giác đã được mã hóa vào các text hidden states. Khi đó, visual values có thể trở nên dư thừa:

$$
v_j^{visual}\approx 0.
$$

Tuy nhiên, nếu visual keys vẫn được giữ lại, attention mass trên các visual slots vẫn tham gia vào mẫu số:

$$
m_{visual}=\sum_{j\in V}a_{ij}.
$$

Khi đó:

$$
o_i
=
\sum_{j\in T}a_{ij}v_j
+
\underbrace{\sum_{j\in V}a_{ij}0}_{=0}.
$$

Visual slots không đóng góp nội dung, nhưng làm giảm attention mass dành cho text tokens. Đây có thể được xem là một dạng regularization thông qua mẫu số của softmax.

Từ đó xuất hiện trade-off:

- Xóa toàn bộ visual slots: text tokens nhận gần như toàn bộ attention mass, dễ dẫn đến text over-attention hoặc attention concentration.
- Giữ quá nhiều visual slots rỗng: attention bị phân tán vào nhiều slots không mang thông tin, gây attention dilution.
- Giữ một lượng trung gian: vừa tránh tập trung quá mức vào text, vừa không làm loãng attention quá nhiều.

Dự đoán tổng quát là quan hệ giữa số lượng null visual slots và hiệu năng có dạng inverted-U.

### Một lưu ý kỹ thuật quan trọng

Để kiểm chứng đúng cơ chế này, không được đơn giản đặt toàn bộ visual embedding đầu vào bằng zero. Cách đó thường làm cả key và value thay đổi.

Intervention đúng cần là:

- giữ visual key;
- đặt visual value bằng zero;
- giữ nguyên causal mask;
- giữ nguyên vị trí M-RoPE hoặc positional IDs của các token còn lại.

Nếu loại bỏ visual tokens khỏi sequence, text tokens phía sau có thể bị thay đổi vị trí. Khi đó, hiệu ứng quan sát được có thể đến từ positional shift chứ không phải từ softmax denominator.

Nên phân biệt ba điều kiện:

| Điều kiện | Visual key | Visual value | Visual slot |
|---|---:|---:|---:|
| Full visual | Có | Có | Có |
| Remove visual | Không | Không | Không |
| Null visual slots | Có | Bằng 0 | Có |

Nếu chỉ zero input embedding, điều kiện đó không còn là phép kiểm tra thuần túy về vai trò của denominator.

## 2. Làm rõ giả định “text đã chứa câu trả lời”

Câu này nên được viết lại chính xác hơn:

> Sau một số layer của target model, text-conditioned hidden states có thể đã chứa đủ thông tin ngữ nghĩa cần thiết để dự đoán token tiếp theo, bao gồm một phần visual semantics.

Điều này khác với việc “text prompt ban đầu đã chứa câu trả lời”.

Đối với MVBench:

- câu hỏi và các lựa chọn thường không chứa trực tiếp đáp án;
- nhưng hidden states của text tokens sau khi target xử lý toàn bộ visual context có thể đã internalize visual information.

Do đó, giả định cần kiểm định là:

$$
I(\text{answer}; H_{\text{text}}) > 0
$$

và có thể đủ lớn để drafter dự đoán token tiếp theo mà không cần raw visual values.

Có thể kiểm chứng bằng:

- linear probe trên text hidden states để dự đoán đáp án;
- so sánh target logits khi giữ/bỏ visual values;
- cosine similarity giữa text-hidden representation và visual-conditioned representation;
- KL divergence giữa phân phối token của các điều kiện.

## 3. Các metric nên dùng

Không nên chỉ đo “độ lớn của attention map”. Mỗi hàng softmax đã được chuẩn hóa nên tổng attention thường bằng 1. Cần đo các đại lượng sau.

### Attention mass

Với query $i$:

$$
M_V(i)=\sum_{j\in V}a_{ij},
\qquad
M_T(i)=\sum_{j\in T}a_{ij}.
$$

Nên tách thêm:

- instruction mass;
- visual mass;
- ordinary text mass.

### Attention concentration

Để đo over-attention vào một vài text tokens:

$$
C_T(i)=\sum_{j\in T}p_j^2
$$

với $p_j$ là attention đã chuẩn hóa trong nhóm text.

Có thể dùng thêm:

- maximum text attention;
- normalized entropy;
- effective number of attended tokens;
- attention mass trên top-1/top-5 text tokens.

Attention mass cao chưa chắc là over-attention. Over-attention nên được định nghĩa là:

> text attention mass hoặc text concentration tăng bất thường, đồng thời draft logits/hidden states lệch xa target hoặc training distribution.

### Representation shift

“Output embedding rơi khỏi phân phối gốc” cần được operationalize bằng:

- hidden norm;
- mean và covariance theo layer;
- cosine với baseline full-visual condition;
- Mahalanobis distance tới training distribution;
- MMD hoặc covariance shift;
- KL divergence giữa draft logits và target logits;
- top-1 agreement giữa drafter và target.

Chỉ thu thập mean hidden là chưa đủ. Mean có thể giữ nguyên trong khi variance hoặc covariance thay đổi mạnh. Nên ước lượng mean/covariance theo:

- layer;
- vị trí token;
- loại token: instruction, text, answer;
- training set và evaluation set.

## 4. Hypothesis 1: Over-attention và attention dilution trên MVBench

### Research question

Liệu số lượng visual slots rỗng trong draft context có tạo ra trade-off giữa text over-attention và attention dilution hay không?

### Hypothesis

Khi số lượng null visual slots tăng từ 0 lên quá lớn:

- text attention concentration giảm;
- visual/null-slot mass tăng;
- attention dilution tăng;
- acceptance length giảm.

Khi số lượng null visual slots bằng 0:

- text attention concentration tăng;
- hidden/logit distribution lệch khỏi baseline;
- acceptance length giảm do over-attention.

Vì vậy, một lượng null visual slots trung gian sẽ cho acceptance length cao nhất.

### Thiết kế

Giữ nguyên:

- target model;
- draft checkpoint;
- prompt;
- frame sampling;
- visual token budget;
- decoding temperature;
- target full visual context;
- random seed.

Chỉ thay đổi draft-side visual interface:

$$
z \in \{0,1\%,5\%,10\%,25\%,50\%,100\%\}
$$

trong đó $z$ là tỷ lệ visual slots được giữ lại dưới dạng null slots.

Để tránh nhầm lẫn giữa số lượng slots và nội dung visual, nên có ít nhất bốn condition:

1. Full visual values.
2. Remove toàn bộ visual slots.
3. Giữ null visual slots với nhiều mức $z$.
4. Giữ một số visual values thật, nếu muốn đo riêng giá trị nội dung của visual tokens.

### Metric chính

Đối với speculative decoding:

$$
\tau_{\mathrm{prop}}=\mathbb{E}[k_t]
$$

và

$$
\tau_{\mathrm{eff}}=\mathbb{E}[k_t+1],
$$

trong đó $k_t$ là số proposal được chấp nhận ở round $t$.

Metric chính nên là $\tau_{\mathrm{eff}}$ hoặc decode speedup. MVBench accuracy chỉ nên là guardrail nếu decoding lossless, vì khi target không bị thay đổi thì kết quả cuối cùng phải giống target.

Nếu muốn đo answer quality của drafter, cần chạy thêm một thí nghiệm draft-only, tách biệt với speculative decoding lossless.

### Kết quả kỳ vọng

| Quan sát | Diễn giải |
|---|---|
| Acceptance có dạng inverted-U; $z=0$ và $z=100\%$ đều kém hơn vùng giữa | Ủng hộ Hypothesis 1 và cơ chế denominator regularization |
| $z$ giảm làm text concentration tăng, hidden shift tăng, acceptance giảm | Ủng hộ nhánh over-attention |
| $z$ tăng làm null visual mass tăng, text mass giảm, acceptance giảm | Ủng hộ nhánh attention dilution |
| Acceptance tăng đơn điệu khi xóa visual slots | Hypothesis 1 bị bác bỏ; visual slots có thể chỉ là noise hoặc positional effect |
| Attention thay đổi rõ nhưng acceptance không thay đổi | Có hiệu ứng attention nhưng chưa chứng minh hiệu ứng functional |
| Acceptance thay đổi nhưng attention không giải thích được | Có thể do positional shift, KV-cache mismatch hoặc thay đổi representation khác |
| Hidden shift tăng nhưng acceptance không giảm | “Out-of-distribution” chưa phải mediator của acceptance trong thí nghiệm này |

Trong repo, runner `draft_attention` hiện đã ghi `visual_mass`, `text_mass`, `instruction_mass` và `visual_entropy` [tại đây](/home/hust/Phuc/VDFlash/src/analyze/Validate_Sparrow_hypothesises/run_draft_attention.py:262). Tuy nhiên, để kiểm tra Hypothesis 1 đầy đủ cần bổ sung hidden/logit statistics và kiểm soát key-value intervention.

## 5. Hypothesis 2: Vì sao VDC không cải thiện khi giảm visual tokens?

### Hypothesis 2.1

VDC khác MVBench ở chỗ text prompt tự nhiên có thể không chứa đủ semantic information để trả lời. Vì vậy:

> Giảm visual tokens giúp giảm attention dilution, nhưng đồng thời loại bỏ nguồn thông tin cần thiết để dự đoán câu trả lời. Hiệu ứng có lợi và có hại triệt tiêu nhau, nên acceptance không tăng so với baseline hoặc thậm chí giảm.

Nói chính xác hơn, vấn đề không phải là prompt “không chứa literal answer”, mà là:

$$
H_{\text{text}}
$$

có thể chưa phải là sufficient statistic cho đáp án trên VDC.

### Thiết kế

Dùng thiết kế factorial:

- Prompt type:
  - natural VDC prompt;
  - answer-augmented prompt.
- Draft visual condition:
  - full;
  - partial;
  - zero/remove.

Target luôn phải nhận cùng một prompt đầy đủ và cùng full visual context. Chỉ draft-side context được thay đổi.

Answer-augmented prompt có thể chứa một phần reference answer theo các mức:

$$
0\%, 25\%, 50\%, 100\%.
$$

Tuy nhiên, condition này phải được ghi rõ là diagnostic leakage condition, không được báo cáo như kết quả benchmark tự nhiên. Nếu chèn reference answer trực tiếp, cần kiểm soát prompt length và vị trí token để tránh confound do sequence length.

### Metric chính

$$
\Delta\tau_{\mathrm{eff}}
=
\tau_{\mathrm{eff}}(\text{reduced})
-
\tau_{\mathrm{eff}}(\text{full}).
$$

Nên dùng paired bootstrap confidence interval hoặc equivalence test. Không nên kết luận “không tốt hơn” chỉ vì p-value không có ý nghĩa. Cần định nghĩa trước một margin $\epsilon$, chẳng hạn 5% relative improvement hoặc một số lượng accepted tokens cụ thể.

Metric phụ:

- text/visual attention mass;
- text attention concentration;
- draft-target KL;
- VDC answer quality;
- exact target-prefix agreement;
- draft latency.

### Kết quả kỳ vọng

| Kết quả | Kết luận |
|---|---|
| Natural VDC: giảm visual không tốt hơn; answer-augmented VDC: giảm visual cải thiện rõ | Ủng hộ Hypothesis 2.1 và cho thấy text semantic có thể thay thế visual khi đã đủ thông tin |
| Natural VDC cũng cải thiện khi giảm visual | Hypothesis 2.1 không được ủng hộ; VDC prompt có thể đã đủ thông tin hoặc visual input chủ yếu gây nhiễu |
| Cả natural và answer-augmented đều không cải thiện | Trade-off denominator không phải cơ chế chính, hoặc intervention đang làm thay đổi position/KV/representation |
| Answer-augmented condition cải thiện nhưng hidden vẫn lệch mạnh | Có lợi ích về acceptance nhưng chưa chứng minh semantic compression giữ được phân phối representation |
| Final target answer thay đổi giữa các condition | Target isolation hoặc lossless decoding bị vi phạm |

## 6. Hypothesis 3.1: Acceptance chỉ cao ở phần sau câu trả lời

### Hypothesis

Visual information quan trọng hơn ở các token đầu của câu trả lời. Sau khi target đã sinh một phần answer prefix, các token tiếp theo trở nên dễ dự đoán hơn từ language context.

Do đó, việc cắt visual tokens có thể:

- làm acceptance thấp hơn ở đầu câu trả lời;
- nhưng làm acceptance cao hơn ở phần sau;
- khiến acceptance trung bình cuối cùng tăng.

Nên viết lại thành:

> Visual dependency của target giảm theo độ dài answer prefix; vì vậy, lợi ích của việc loại bỏ visual context có thể chỉ xuất hiện ở các vị trí sinh về sau.

Không nên dùng cụm “target đã trả lời một nửa” như định nghĩa chính. Cần đo theo vị trí token.

### Logging cần bổ sung

Với mỗi acceptance round $t$, ghi:

- `round_id`;
- `output_start_position`;
- `proposal_count`;
- `matched_proposals`;
- `effective_emitted_tokens`;
- `first_reject_position`;
- candidate token IDs;
- target token IDs;
- normalized output position.

Nếu $k_t$ là số proposal được chấp nhận:

$$
s_t=\sum_{u<t}(k_u+1)
$$

là vị trí bắt đầu của round $t$. Acceptance của proposal thứ $r$ được gán vào:

$$
p=s_t+r.
$$

Sau đó tính acceptance rate theo:

- absolute position;
- relative position trong câu trả lời;
- các bin 0–10%, 10–25%, 25–50%, 50–75%, 75–100%;
- nhóm task hoặc loại câu trả lời.

`acceptance_trace` hiện đã được ghi trong runtime [tại đây](/home/hust/Phuc/VDFlash/src/analyze/Validate_Sparrow_hypothesises/runtime.py:1074), nhưng trace hiện tại chủ yếu là số accepted proposals theo round. Để kiểm tra Hypothesis 3.1 cần thêm vị trí bắt đầu của từng round.

### Kết quả kỳ vọng

| Quan sát | Kết luận |
|---|---|
| Condition cắt visual có acceptance thấp hơn ở đầu nhưng cao hơn ở cuối | Ủng hộ Hypothesis 3.1 |
| Acceptance cao hơn đồng đều ở mọi vị trí | Hypothesis 3.1 bị bác bỏ; lợi ích là global noise reduction |
| Chỉ token đầu tiên cải thiện, phần còn lại không đổi | Hiệu ứng visual có thể nằm ở initial conditioning thay vì “nửa sau câu trả lời” |
| Acceptance tăng nhưng target visual sensitivity không giảm theo vị trí | Cơ chế visual dependency chưa được chứng minh |

Nên đo thêm target visual sensitivity bằng:

$$
D_{\mathrm{visual}}(p)
=
\mathrm{KL}
\left(
p_{\mathrm{target}}(\cdot|x_{\mathrm{full}},p)
\|
p_{\mathrm{target}}(\cdot|x_{\mathrm{text-only}},p)
\right).
$$

Nếu $D_{\mathrm{visual}}(p)$ cao ở đầu và giảm về cuối, điều đó hỗ trợ trực tiếp cho giả thuyết rằng visual information quan trọng hơn ở các token đầu.

## 7. Hypothesis 3.2: DFlash nhiều layer hơn nên hiểu visual tốt hơn MSD

### Hypothesis

DFlash có nhiều draft layers hơn MSD. Vì vậy, DFlash có thể thực hiện thêm các bước biến đổi và tích hợp visual semantics, trong khi MSD với một draft layer có thể không đủ năng lực xử lý visual context.

Tuy nhiên, giả thuyết này chưa chắc đúng. Một draft layer vẫn có thể nhận target hidden states đã được target model tích hợp visual information từ trước.

Đặc biệt, trong DFlash hiện tại, số lượng draft layers và số lượng target features là hai khái niệm khác nhau. Một model một draft layer vẫn có thể nhận nhiều target hidden features từ các layer khác nhau. Do đó, “một layer không đọc được visual” không thể được suy ra chỉ từ kiến trúc.

### Thiết kế

Huấn luyện ba model:

$$
L\in\{1,3,5\}
$$

với:

- cùng target model;
- cùng số target features;
- cùng selected target layer IDs;
- cùng block size và anchor configuration;
- cùng dữ liệu;
- cùng số optimizer steps;
- cùng learning-rate schedule;
- cùng hai phase training;
- cùng seed policy.

Mỗi phiên bản phải được train riêng từ đầu cho cả hai phase:

1. Stage 1: text-only hoặc language initialization.
2. Stage 2: multimodal training.

Không nên lấy checkpoint 5-layer rồi cắt xuống 1-layer, vì điều đó tạo ra khác biệt về initialization và training history.

### Metric chính

Đo acceptance theo visual condition:

$$
\tau_{\mathrm{eff}}(L,\text{full}),
\quad
\tau_{\mathrm{eff}}(L,\text{reduced}),
\quad
\tau_{\mathrm{eff}}(L,\text{zero}).
$$

Đại lượng quan trọng nhất là interaction:

$$
\Delta_{\mathrm{visual}}(L)
=
\tau_{\mathrm{eff}}(L,\text{full})
-
\tau_{\mathrm{eff}}(L,\text{zero}).
$$

Nếu nhiều layer thực sự giúp xử lý visual semantics, hiệu ứng của visual context nên tốt hơn khi $L$ tăng.

Cần báo cáo thêm:

- draft latency;
- số parameters;
- memory;
- training loss;
- target-draft KL;
- visual attention mass;
- acceptance theo vị trí;
- visual sensitivity của draft logits.

### Kết quả kỳ vọng

| Quan sát | Kết luận |
|---|---|
| $L$ tăng làm acceptance tăng chủ yếu ở full-visual condition | Ủng hộ Hypothesis 3.2 |
| $L$ tăng làm giảm chênh lệch giữa full và zero visual | Dấu hiệu draft nhiều layer khai thác visual context tốt hơn |
| $L$ tăng cải thiện đồng đều cả full và zero visual | Depth giúp language modeling nói chung, chưa chứng minh visual understanding |
| 1-layer gần bằng 3/5-layer khi target features giữ nguyên | Hypothesis 3.2 bị bác bỏ; target hidden states đã chứa đủ visual semantics |
| 5-layer acceptance cao hơn nhưng latency tăng mạnh | Có lợi ích representation nhưng cần đánh giá acceptance trên mỗi đơn vị draft cost |
| 5-layer không tốt hơn sau khi chuẩn hóa theo compute | Nhiều layer không mang lại lợi ích thực tế cho speculative decoding |

## 8. Thứ tự thực nghiệm nên thực hiện

Không nên bắt đầu bằng việc thu thập toàn bộ hidden states trên toàn bộ test set bằng B200. Nên đi theo các mức:

1. Một sample: kiểm tra key/value intervention, position IDs và lossless parity.
2. 5–10 samples: kiểm tra attention statistics, hidden shift và acceptance trace.
3. Pilot 30–50 samples: kiểm tra xu hướng và ước lượng variance.
4. Full cohort: chỉ chạy khi pilot cho thấy metric có tín hiệu rõ.
5. Training depth ablation $L=1,3,5$: thực hiện sau khi acceptance-position logging và visual-retention protocol đã ổn định.

Trong báo cáo cuối cùng, nên phân biệt ba mức kết luận:

- Evidence về attention mechanism.
- Evidence về drafter behavior, chẳng hạn acceptance length.
- Evidence về task quality.

Một kết quả chỉ nên được gọi là “xác nhận hypothesis” khi các metric thuộc cả ba mức không mâu thuẫn với nhau.