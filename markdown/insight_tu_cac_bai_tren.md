# Các nhận định tổng hợp về speculative decoding cho VLM

Qua các nhánh paper từ 2024 đến **31/07/2026**, có thể thấy speculative decoding cho VLM không còn đơn thuần là “lấy một model nhỏ dự đoán trước cho model lớn”. Bài toán thực chất đang chuyển thành:

$$
\boxed{
\text{Phân phối đúng lượng thông tin thị giác}
+
\text{tạo proposal phù hợp}
+
\text{tối ưu hành vi chấp nhận của verifier}
}
$$

Nói cách khác, trọng tâm của lĩnh vực đã dịch chuyển từ **thiết kế một drafter nhỏ** sang **thiết kế giao diện thông tin giữa video/image, drafter và target model**.

---

## 1. Drafter cần visual semantics, không nhất thiết cần visual tokens

Nhận định quan trọng nhất từ các kết quả hiện tại là:

> Drafter cần biết nội dung thị giác nào ảnh hưởng đến token tiếp theo, nhưng không nhất thiết phải trực tiếp xử lý toàn bộ visual-token sequence.

Các nghiên cứu ban đầu cho thấy language-only drafter vẫn có thể đạt mức tăng tốc đáng kể, đặc biệt khi phần lớn nội dung sinh ra được quyết định bởi language prior. Tuy nhiên, benchmark MMSpec cho thấy những phương pháp không nhận biết thị giác thường kém ổn định hơn trong các tác vụ phụ thuộc mạnh vào ảnh và khi batch size tăng. ([arXiv][1])

Điều này giải thích một mâu thuẫn bề ngoài trong literature:

* Một số paper cho thấy drafter không cần ảnh.
* Một số paper lại cho thấy vision-aware drafter tốt hơn.
* Một số paper thậm chí cho thấy đưa raw visual tokens vào drafter có thể làm kết quả tệ hơn.

Ba kết quả này không thực sự mâu thuẫn. Chúng cho thấy visual information có ba thành phần:

$$
V =
V_{\text{relevant}}
+
V_{\text{redundant}}
+
V_{\text{misaligned}}.
$$

Chỉ $V_{\text{relevant}}$ giúp dự đoán token. Phần dư thừa làm tăng attention cost, còn phần không tương thích với representation space của drafter có thể làm giảm acceptance rate. HiViS thậm chí quan sát thấy EAGLE-style drafter dùng text-only input có thể tốt hơn khi trực tiếp đưa visual tokens của target vào, do semantic-space mismatch và KV-cache contamination. ([arXiv][2])

Vì vậy, câu hỏi nghiên cứu tốt hơn không còn là:

> “Có nên đưa ảnh vào drafter không?”

mà là:

> “Đâu là visual sufficient statistic nhỏ nhất để drafter dự đoán giống target?”

---

## 2. “Nhiều thông tin thị giác hơn” không đồng nghĩa với “drafter tốt hơn”

Trong VLM chính, model lớn có đủ độ sâu để lọc và tích hợp visual information. Nhưng một drafter rất nông hoặc nhỏ có thể không đủ khả năng thực hiện quá trình lọc đó. ViSpec đưa ra giả thuyết rằng target VLM có thể dần loại bỏ thông tin ảnh dư thừa qua nhiều layer, trong khi drafter nhỏ gặp khó khăn nếu phải xử lý trực tiếp toàn bộ image sequence. ([arXiv][3])

Điều này dẫn đến hiện tượng có thể gọi là **negative visual gain**:

$$
\text{thêm visual tokens}
\Rightarrow
\begin{cases}
\text{nhiều grounding information hơn}, \\
\text{nhưng attention cost cao hơn}, \\
\text{nhiều noise hơn}, \\
\text{representation mismatch lớn hơn}.
\end{cases}
$$

Sparrow quan sát hiện tượng này đặc biệt rõ trong Video-LLM: KV cache quá lớn và attention dilution có thể khiến drafter hoạt động kém hơn khi nhận thêm raw video tokens. Paper gọi đây là “negative visual gain” và thay thế raw visual context bằng text hidden states đã internalize visual semantics. ([arXiv][4])

Như vậy, visual compression trong speculative decoding không chỉ là tối ưu FLOPs. Nó còn có thể là một dạng **representation regularization** cho drafter: loại bỏ thông tin mà drafter không đủ khả năng khai thác.

---

## 3. Trade-off thật sự là acceptance gain trên mỗi đơn vị draft cost

Một drafter tốt không nhất thiết là drafter có acceptance rate cao nhất. Ví dụ, đưa toàn bộ ảnh hoặc video vào một draft VLM lớn có thể làm acceptance tăng, nhưng chi phí drafting cũng tăng mạnh.

Một đại lượng khái niệm hữu ích hơn là:

$$
\eta_{\mathrm{draft}}
=
\frac{
\mathbb{E}[\text{accepted tokens per round}]
}{
T_{\mathrm{draft}}
+
T_{\mathrm{verification}}
+
T_{\mathrm{overhead}}
}.
$$

Một phương pháp có thể chấp nhận ít token hơn nhưng vẫn nhanh hơn nếu drafting đủ rẻ. Ngược lại, một drafter rất chính xác nhưng gần bằng target model về chi phí thường không đem lại lợi ích.

Đây là lý do nhiều phương pháp gần đây chọn:

* Nén visual tokens.
* Chỉ dùng global visual features.
* Tái sử dụng target hidden states.
* Sparse attention.
* Bỏ hoàn toàn raw visual tokens khỏi drafter.

SpecVLM dành cho video cho thấy drafter có thể chịu pruning mạnh hơn target; framework loại tới 90% video tokens khỏi drafter trong khi verifier vẫn dùng context đầy đủ. ([arXiv][5]) HiViS đẩy ý tưởng này xa hơn khi giảm explicit prefill sequence của drafter xuống khoảng 0,7–1,3% chiều dài target input bằng cách chỉ giữ text tokens và dùng hidden states của target làm visual guidance. ([arXiv][2])

Insight ở đây là:

> Mục tiêu không phải tối đa hóa acceptance rate riêng lẻ, mà tối đa hóa acceptance rate sau khi chuẩn hóa theo chi phí proposal.

---

## 4. Với Video-LLM, context length quan trọng hơn số tham số của drafter

Trong LLM thuần văn bản, dùng một model ít tham số hơn thường làm drafting nhanh hơn đáng kể. Nhưng với Video-LLM, ngay cả drafter nhỏ vẫn có thể rất chậm nếu phải đọc hàng chục nghìn video tokens.

Chi phí decoding attention của drafter phụ thuộc gần đúng vào:

$$
T_{\mathrm{draft/token}}
\propto
L_{\mathrm{draft}}
\times
d_{\mathrm{draft}}
\times
N_{\mathrm{context}},
$$

trong đó $N_{\mathrm{context}}$ có thể bị chi phối hoàn toàn bởi video tokens.

Do đó:

$$
\text{small parameters}
\not\Rightarrow
\text{cheap drafter},
$$

nếu context vẫn quá dài.

Các phương pháp Video-LLM đang hội tụ vào ba chiến lược:

1. **Giảm context:** verifier-guided token pruning.
2. **Giảm attention:** sparse top-(K) attention.
3. **Thay đổi representation:** dùng text hidden states hoặc intermediate semantic states thay raw visual tokens.

Sparse-to-Dense dùng cùng một Video-LLM dưới hai chế độ: sparse attention để draft và dense attention để verify, đạt tăng tốc mà không cần một model nhỏ độc lập. ([ACL Anthology][6]) SpecVLM dùng pruning được hướng dẫn bởi verifier, còn Sparrow chuyển visual semantics sang text-anchored hidden states và duy trì mức tăng tốc trung bình được báo cáo ngay cả với khoảng 25 nghìn visual tokens. ([arXiv][5])

Điều này cho thấy với video dài, phương pháp phân loại theo “draft model lớn hay nhỏ” không còn đủ hữu ích. Cần phân loại thêm theo:

$$
\boxed{
\text{Context seen by drafter}
}
$$

và

$$
\boxed{
\text{Representation used by drafter}
}
$$

---

## 5. Visual relevance thay đổi theo từng token sinh ra

Hầu hết các visual compressor ban đầu chọn một tập visual tokens cố định trước khi bắt đầu decoding. Nhưng mức phụ thuộc vào hình ảnh không cố định trong suốt câu trả lời.

Ví dụ trong câu:

> “The animal standing next to the red car is a dog, and it appears to be waiting for its owner.”

Các token “animal”, “red car” và “dog” có thể phụ thuộc mạnh vào ảnh. Trong khi “and it appears to be waiting for its owner” phần lớn được language prior hỗ trợ.

Có thể mô hình hóa:

$$
r_t
=
I(y_t; V\mid y_{<t},x),
$$

trong đó $r_t$ là mức liên quan thị giác tại decoding step $t$. Giá trị này thay đổi trong quá trình sinh.

MMSpec dựa trên nhận xét đó để đề xuất ViSkip: khi token state phụ thuộc mạnh vào visual context, hệ thống tạm ngừng speculation và cho target sinh trực tiếp; khi mức phụ thuộc thấp, nó sử dụng speculative decoding. ([arXiv][7])

TIGER tiến thêm một bước: thay vì chỉ bật hoặc tắt drafting, nó chọn động một tập visual tokens dựa trên textual state hiện tại của drafter. Do đó visual interface thay đổi trong từng giai đoạn sinh, thay vì dùng một compressed representation cố định. ([arXiv][8])

Từ đây có thể rút ra một insight mạnh:

> Multimodal speculative decoding nên là một chính sách tuần tự, không phải một cấu hình preprocessing cố định.

Policy tương lai có thể quyết định tại mỗi step:

$$
a_t =
(\text{visual budget},
\text{draft length},
\text{tree width},
\text{draft/target mode}).
$$

---

## 6. Training bằng cross-entropy chưa trực tiếp tối ưu speedup

Phần lớn drafter được huấn luyện bằng cách bắt chước logits hoặc next-token prediction của target:

$$
\mathcal{L}_{\mathrm{CE}}
=
-\sum_t\log q(y_t\mid y_{<t},x,V).
$$

Nhưng speculative decoding không thực sự tối ưu token accuracy độc lập. Một proposal chỉ đem lại lợi ích nếu verifier chấp nhận được **một prefix dài liên tiếp**.

Ví dụ:

* Drafter A dự đoán đúng 8/10 token rời rạc nhưng thường sai ở token thứ hai.
* Drafter B dự đoán đúng 7/10 token nhưng thường đúng 6 token đầu liên tiếp.

Drafter B có thể đem lại speedup cao hơn nhiều.

Do đó, utility thực tế gần với:

$$
U(z)
=
\operatorname{AcceptedPrefixLength}(z,p),
$$

hơn là tổng token likelihood.

VSD chỉ ra hai misalignment:

1. **Deterministic training vs. stochastic/tree decoding.**
2. **Token-level likelihood vs. path-level utility.**

Paper báo cáo rằng trong thiết lập phân tích của họ, nhiều greedy paths được tối ưu trong training thậm chí bị loại khi dựng draft tree; accepted path cũng thường không phải greedy path. ([arXiv][9])

TIGER áp dụng cùng tinh thần vào VLM bằng verifier-derived reward dựa trên accepted prefix length, thay vì chỉ distill target probabilities. ([arXiv][8])

Đây có thể là chuyển dịch quan trọng tiếp theo của lĩnh vực:

$$
\text{Teacher imitation}
\rightarrow
\text{Acceptance optimization}
\rightarrow
\text{Direct latency optimization}.
$$

Tuy nhiên, TIGER hiện vẫn được ghi là work in progress, vì vậy các kết luận về độ tổng quát cần được kiểm chứng thêm. ([arXiv][8])

---

## 7. Target-aware drafter có lợi thế cấu trúc so với independent drafter

Independent draft VLM có ưu điểm modular: có thể thay draft hoặc target tương đối dễ dàng. Nhưng nó gặp hai vấn đề:

* Phải tính lại hoặc tự xử lý visual information.
* Representation space của drafter có thể không khớp target.

Các phương pháp target-aware hoặc self-speculative tái sử dụng:

* Hidden states.
* Intermediate features.
* KV cache.
* Vision encoder outputs.
* Một phần layer của target.

Cách tiếp cận này có hai lợi ích đồng thời:

$$
\text{giảm computation duplication}
+
\text{giảm semantic mismatch}.
$$

HiViS dùng last-layer textual hidden states của target làm implicit visual input. Sparrow tái sử dụng hidden states và intermediate visual states. FastVLM dùng imitation network để truyền thông tin từ các layer sâu sang nhánh draft. ([arXiv][2])

Từ các kết quả này, có thể suy luận rằng **self-speculative và target-assisted drafting có khả năng trở thành kiến trúc mặc định** cho VLM lớn, đặc biệt khi vision encoder và visual KV cache đắt.

Đổi lại, chúng làm giảm tính modular:

* Drafter gắn chặt với target architecture.
* Khó thay target model.
* Có thể cần train lại khi target được cập nhật.
* Khó phục vụ nhiều target models bằng một drafter chung.

Vì vậy, independent drafter vẫn có thể phù hợp cho hệ thống multi-model hoặc device–server, trong khi target-aware drafter phù hợp cho một deployment stack cố định.

---

## 8. “Lossless” cần được tách thành nhiều cấp độ

Trong các paper, từ “lossless” đôi khi được sử dụng cho các khái niệm khác nhau.

### Cấp 1: Distribution-preserving speculative verification

Verifier dùng target model và quy tắc acceptance/correction chính xác, do đó output distribution giống autoregressive target decoding.

### Cấp 2: Greedy-equivalent

Với greedy decoding, output cuối giống target sinh tuần tự.

### Cấp 3: Benchmark-equivalent

Phương pháp thay đổi model hoặc visual tokens nhưng downstream accuracy gần như không giảm.

Chỉ hai cấp đầu là lossless theo nghĩa chặt của speculative decoding.

Một điểm tinh tế là có thể prune rất mạnh context **của drafter** nhưng vẫn giữ output lossless nếu target verifier tiếp tục dùng full context và correction rule chính xác. SpecVLM–Video thuộc kiểu này: pruning làm proposal rẻ hơn, nhưng target vẫn xác nhận bằng video context đầy đủ. ([arXiv][5])

Ngược lại, nếu visual tokens của chính target bị xóa hoặc target attention bị xấp xỉ mà không có cơ chế correction tương đương, phương pháp chỉ có thể được gọi là quality-preserving hoặc approximately lossless.

Do đó, khi đọc paper cần hỏi chính xác:

> Thành phần nào được nén: drafter, verifier hay cả hai?

---

## 9. Speedup không phải một con số có thể so trực tiếp giữa các paper

Một con số như $2.5\times$ có thể là:

* Decode-only speedup.
* End-to-end speedup.
* Tokens-per-second throughput.
* Speedup ở batch size 1.
* Speedup ở batch lớn.
* Memory-bound theoretical speedup.
* Wall-clock speedup trên phần cứng cụ thể.

MMSpec cho thấy phương pháp có throughput speedup cao hơn không nhất thiết có latency distribution tốt hơn. Một số phương pháp nhanh trung bình nhưng có tail latency lớn trên những sample khó. ([arXiv][7])

Benchmark này cũng cho thấy ngay cả các phương pháp vision-aware như MSD và ViSpec vẫn biến động mạnh giữa loại tác vụ và kiến trúc model; ví dụ speedup có thể thay đổi đáng kể giữa captioning, VQA, OCR và complex reasoning. ([arXiv][7])

Một protocol đánh giá đầy đủ nên báo cáo ít nhất:

$$
\begin{aligned}
&\text{TTFT}, \\
&\text{TPOT}, \\
&\text{end-to-end latency}, \\
&\text{p50/p95 latency}, \\
&\text{throughput}, \\
&\text{accepted prefix length}, \\
&\text{acceptance rate}, \\
&\text{draft/verify latency breakdown}, \\
&\text{peak memory}, \\
&\text{quality equivalence}.
\end{aligned}
$$

Đặc biệt, cần chia kết quả theo output length. Với VQA chỉ sinh vài token, thời gian prefill có thể chiếm phần lớn tổng latency, nên decode acceleration cao chưa chắc tạo ra end-to-end speedup đáng kể. ViSpec cũng chỉ ra sự thiếu hụt dữ liệu multimodal có response dài và phải xây dựng dữ liệu bổ sung phục vụ huấn luyện drafter. ([arXiv][3])

---

## 10. Batch size làm thay đổi vai trò của vision awareness

Một kết quả đáng chú ý từ MMSpec là vision-aware drafting trở nên quan trọng hơn khi batch size tăng. Những phương pháp không dùng visual information có xu hướng suy giảm speedup rõ hơn khi xử lý batch multimodal lớn và không đồng nhất. ([arXiv][7])

Một cách giải thích hợp lý là:

* Batch lớn chứa nhiều loại ảnh và câu hỏi hơn.
* Language prior dùng chung không còn đủ để dự đoán chính xác.
* Sai khác giữa draft và target tăng.
* Acceptance rate giảm.
* Verification rounds tăng.

Điều này cho thấy thiết kế tối ưu cho interactive inference, thường batch 1, có thể không tối ưu cho production serving.

Có thể hình dung hai chế độ:

| Deployment                  | Thiết kế phù hợp hơn                              |
| --------------------------- | ------------------------------------------------- |
| Batch 1, response ngắn      | Text-heavy hoặc self-speculative drafter đơn giản |
| Batch lớn, workload đa dạng | Vision-aware hoặc dynamic visual routing          |
| Video dài                   | Compressed/implicit visual context                |
| Reasoning dài               | Acceptance-aligned tree speculation               |

Do đó, không tồn tại một drafter tối ưu phổ quát; thiết kế cần gắn với serving regime.

---

# Cách phân loại phù hợp hơn

Thay vì chia paper thành các nhánh độc lập theo tên phương pháp, có thể phân loại mọi hệ thống trên ba trục.

## Trục 1: Visual interface của drafter

$$
\text{None}
\rightarrow
\text{Raw}
\rightarrow
\text{Compressed}
\rightarrow
\text{Implicit}
\rightarrow
\text{Dynamically routed}.
$$

| Interface              | Đặc điểm                                          |
| ---------------------- | ------------------------------------------------- |
| Không dùng vision      | Rẻ nhưng dễ sai ở token cần grounding             |
| Raw visual tokens      | Đầy đủ nhưng đắt và có semantic mismatch          |
| Compressed vision      | Cân bằng cost–grounding                           |
| Implicit target states | Rẻ, aligned nhưng phụ thuộc target                |
| Dynamic routing        | Thích nghi theo decoding state nhưng phức tạp hơn |

## Trục 2: Nguồn proposal

$$
\text{Independent model}
\rightarrow
\text{target-assisted}
\rightarrow
\text{self-speculative}
\rightarrow
\text{sparse/dense same model}.
$$

## Trục 3: Cơ chế điều khiển

$$
\text{Static}
\rightarrow
\text{confidence-adaptive}
\rightarrow
\text{vision-adaptive}
\rightarrow
\text{acceptance-aligned}
\rightarrow
\text{hardware-aware}.
$$

Cách phân loại này giải thích paper tốt hơn vì hai phương pháp cùng thuộc “vision-aware drafting” vẫn có thể rất khác nhau: một phương pháp dùng compressed tokens cố định, trong khi phương pháp khác dùng target hidden states hoặc route visual tokens theo từng step.

---

# Hướng nghiên cứu nổi bật nhất

Từ các kết quả hiện tại, hướng có tiềm năng nhất không phải là một drafter cố định, mà là một **adaptive speculative controller** đồng thời quyết định:

$$
\pi_\theta(s_t)
\rightarrow
(B_t,\gamma_t,W_t,M_t),
$$

trong đó:

* $B_t$: visual-token budget.
* $\gamma_t$: draft depth.
* $W_t$: tree width hoặc số candidate.
* $M_t$: lựa chọn speculative hay target-only decoding.

State $s_t$ có thể chứa:

* Text hidden state hiện tại.
* Cross-modal attention.
* Draft entropy.
* Acceptance history.
* Video complexity.
* KV-cache size.
* GPU utilization và batch size.

Objective cuối cùng nên gần với:

$$
\min_\pi
\mathbb{E}
\left[
T_{\mathrm{end-to-end}}
+
\lambda_1 M_{\mathrm{peak}}
+
\lambda_2 C_{\mathrm{energy}}
\right]
$$

với ràng buộc:

$$
P_{\pi}(y\mid x,V)
=
P_{\mathrm{target}}(y\mid x,V)
$$

nếu yêu cầu exact lossless decoding.

## Nhận định tổng kết

Literature hiện tại đang hội tụ về bốn kết luận:

1. **Raw visual tokens không phải interface tối ưu cho drafter.**
2. **Video speculative decoding chủ yếu là bài toán quản lý context và KV cache.**
3. **Visual dependency thay đổi theo từng decoding step, nên compression cần động.**
4. **Drafter nên được tối ưu theo accepted sequence và latency, không chỉ token likelihood.**

Do đó, thế hệ phương pháp tiếp theo nhiều khả năng sẽ kết hợp:

$$
\boxed{
\text{target-state reuse}
+
\text{dynamic visual routing}
+
\text{acceptance-aware training}
+
\text{system-aware control}
}
$$

thay vì chỉ thu nhỏ target VLM hoặc gắn thêm một draft model độc lập.

[1]: https://arxiv.org/abs/2404.08856?utm_source=chatgpt.com "On Speculative Decoding for Multimodal Large Language Models"
[2]: https://arxiv.org/html/2509.23928v1 "HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models"
[3]: https://arxiv.org/abs/2509.15235?utm_source=chatgpt.com "ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding"
[4]: https://arxiv.org/abs/2602.15318 "[2602.15318] Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs"
[5]: https://arxiv.org/abs/2508.16201?utm_source=chatgpt.com "SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning"
[6]: https://aclanthology.org/people/cunxiao-du/?utm_source=chatgpt.com "Cunxiao Du"
[7]: https://arxiv.org/html/2603.14989v1 "MMSpec: Benchmarking Speculative Decoding for Vision-Language Models"
[8]: https://arxiv.org/abs/2607.11131 "[2607.11131] TIGER: Text-Conditioned Visual Gated Routing with Acceptance Alignment for Multimodal Speculative Decoding"
[9]: https://arxiv.org/html/2602.05774v1 "Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance"
