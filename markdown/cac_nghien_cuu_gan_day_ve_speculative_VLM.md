Dưới đây là bản phân loại các paper về **speculative decoding cho Vision–Language Models**, cập nhật đến **31/07/2026**. Tôi tập trung vào VLM/MLLM sinh văn bản từ ảnh hoặc video; không đưa các paper speculative decoding cho autoregressive image generation vào nhóm chính.

## 1. Bức tranh tổng thể

Có thể phân loại literature hiện tại theo nút thắt mà mỗi phương pháp xử lý:

1. **Drafter độc lập:** dùng language model hoặc VLM nhỏ để dự đoán token.
2. **Vision-aware drafter:** thiết kế drafter nhận visual information hiệu quả hơn.
3. **Target-aware/self-speculative:** tái sử dụng hidden states, KV cache hoặc các layer của target.
4. **Visual-context reduction:** giảm visual tokens hoặc sparse attention, đặc biệt cho Video-LLM.
5. **Adaptive speculation:** tối ưu cây dự đoán, draft length hoặc kiến trúc theo độ khó.
6. **System và benchmark:** tối ưu deployment, device–edge hoặc đánh giá thống nhất.

Xu hướng phát triển khá rõ:

$$
\text{text-only drafter} \rightarrow \text{multimodal drafter} \rightarrow \text{compressed/implicit vision} \rightarrow \text{adaptive, hardware-aware speculation}.
$$

---

# 2. Nhóm I — Các paper nền tảng và drafter độc lập

## 2.1. On Speculative Decoding for Multimodal Large Language Models

**Gagrani et al., CVPR 2024 Workshop**

Đây là một trong những nghiên cứu đầu tiên áp dụng speculative decoding trực tiếp cho MLLM. Paper thử nghiệm trên LLaVA-7B và cho thấy một language-only draft model 115M vẫn có thể làm drafter mà không cần xử lý image tokens. Nhóm tác giả cũng thử compact LLaVA drafter có image adapter, nhưng lợi ích của visual input đối với drafter chỉ rõ hơn trong captioning và khá nhỏ ở các tác vụ khác. Paper báo cáo speedup memory-bound tối đa ($2.37\times$). ([arXiv][1])

**Ý nghĩa:** baseline quan trọng để kiểm tra câu hỏi:

> Drafter có thực sự cần nhìn ảnh không?

**Hạn chế:** thí nghiệm chủ yếu trên LLaVA cũ, model nhỏ và chưa xử lý bài toán visual-token explosion ở các VLM hiện đại.

---

## 2.2. In-batch Ensemble Drafting — IbED

**Lee et al., ICLR 2025 Workshop**

IbED tạo nhiều draft candidate trong cùng batch bằng cách chia sẻ tham số, thay vì chạy nhiều drafter độc lập. Mục tiêu chính là tăng robustness khi một drafter nhỏ không ổn định trên các loại multimodal prompts khác nhau. Đây là tiền thân của hướng ensemble và test-time adaptation cho speculative decoding trong LVLM. ([OpenReview][2])

**Phân loại:** independent drafter + ensemble drafting.

**Điểm đáng chú ý:** phù hợp khi ưu tiên throughput theo batch, nhưng chưa giải quyết triệt để chi phí visual context.

---

## 2.3. TABED: Test-Time Adaptive Ensemble Drafting

**Lee et al., Findings of EACL 2026**

TABED mở rộng hướng ensemble bằng cách thích nghi drafter ở test time. Paper benchmark các draft model nhỏ trên 11 dataset và xử lý vấn đề một drafter cố định thường không hoạt động tốt đồng đều giữa captioning, VQA, OCR và reasoning. ([ACL Anthology][3])

**Phân loại:** adaptive ensemble drafting.

**Quan hệ:**

$$
\text{IbED} \rightarrow \text{TABED: ensemble + test-time adaptation}.
$$

---

# 3. Nhóm II — Vision-aware multimodal drafter

Đây là nhóm quan trọng nhất đối với VLM ảnh. Các paper trong nhóm này cho rằng language-only drafter nhanh nhưng thiếu grounding; ngược lại, đưa toàn bộ visual tokens vào drafter lại quá đắt.

## 3.1. Multimodal Speculative Decoding — MSD

**Lin et al., 2025, arXiv**

MSD đưa ra hai nguyên tắc:

* Text tokens và visual tokens nên được xử lý khác nhau trong drafter.
* Drafter cần cả khả năng language modeling và visual perception.

Phương pháp tách riêng hai loại token trong drafting và huấn luyện hai giai đoạn: text instruction tuning trước, sau đó dần đưa multimodal data vào. Paper báo cáo speedup tối đa ($2.29\times$) cho LLaVA-1.5-7B và ($2.46\times$) cho LLaVA-1.5-13B. ([arXiv][4])

**Đóng góp chính:** thay vì hỏi “có đưa ảnh vào drafter hay không”, MSD hỏi “nên đưa ảnh vào theo giao diện nào”.

---

## 3.2. MASSV

**Ganesan et al., Findings of EMNLP 2025**

MASSV biến một small language model có sẵn thành multimodal drafter bằng:

1. Kết nối vision encoder của target VLM với drafter qua lightweight projector.
2. Self-data distillation bằng câu trả lời do chính target VLM tạo ra.

MASSV báo cáo accepted length tăng tối đa 30% và end-to-end speedup tối đa ($1.46\times$) so với text-only drafting trên Qwen2.5-VL và Gemma 3. ([ACL Anthology][5])

**Ưu điểm:** không cần huấn luyện một draft VLM hoàn chỉnh từ đầu.

**Hạn chế:** drafter vẫn phải nhận một dạng visual sequence nên chi phí tăng khi độ phân giải hoặc số ảnh lớn.

---

## 3.3. ViSpec

**Kang et al., NeurIPS 2025**

ViSpec dùng hai cơ chế visual grounding:

* Một vision adaptor nén image tokens thành biểu diễn nhỏ hơn.
* Một global visual vector được cộng hoặc chèn vào các text-token states tiếp theo.

Paper cũng tạo synthetic long-response multimodal data vì các benchmark VQA thông thường có output quá ngắn để huấn luyện speculative drafter tốt. ([arXiv][6])

**Phân loại:** compressed vision-aware feature drafter.

**Điểm mạnh:** thiết kế cân bằng giữa ba mục tiêu:

$$
\text{draft latency} \quad\leftrightarrow\quad \text{visual grounding} \quad\leftrightarrow\quad \text{acceptance length}.
$$

Đây là một paper nên đọc sớm vì nó phân tích khá trực tiếp tại sao EAGLE/Medusa chuyển nguyên trạng từ LLM sang VLM không hiệu quả.

---

## 3.4. SpecVLM: Fast Speculative Decoding in Vision-Language Models

**Huang et al., 2025, arXiv**

Không nên nhầm paper này với SpecVLM dành cho Video-LLM.

Paper xây dựng:

* **EagleVLM:** baseline EAGLE-2 cho VLM.
* **Elastic visual compressor:** tự chọn pruning, pooling, convolution hoặc resampler.
* **Online-logit distillation:** lấy teacher logits và penultimate features trực tiếp khi train, không phải lưu offline distillation corpus.

Paper báo cáo EagleVLM đạt khoảng ($1.5$)–($2.3\times$) end-to-end speedup, còn framework đầy đủ đạt ($2.5$)–($2.9\times$) trong các thiết lập của họ. ([arXiv][7])

**Phân loại:** EAGLE-style drafter + learnable visual compression.

**Điểm đáng nghiên cứu:** compressor không cố định một toán tử duy nhất cho mọi input.

---

## 3.5. Spec-LLaVA

**Huo et al., 2025, preprint/ICML workshop version**

Spec-LLaVA huấn luyện các draft VLM nhỏ, khoảng 68M hoặc 160M tham số, rồi sinh một cây candidate thay vì một chuỗi duy nhất. Cây được mở rộng hoặc cắt dựa trên confidence của drafter. Paper báo cáo decoding speedup tối đa ($3.28\times$) trên LLaVA-1.5 với output không thay đổi. ([arXiv][8])

**Phân loại:** compact draft VLM + dynamic tree verification.

**Hạn chế cần chú ý:** phần đánh giá tập trung nhiều vào LLaVA-1.5 và MS COCO, vì vậy khả năng tổng quát sang OCR, chart hoặc multi-image reasoning cần được kiểm chứng thêm.

---

# 4. Nhóm III — Target-aware và self-speculative decoding

Nhóm này cố tránh phải duy trì hai VLM hoàn chỉnh. Drafter tái sử dụng thông tin đã được target tính toán.

## 4.1. DREAM

**Hu et al., NeurIPS 2025**

DREAM có ba thành phần chính:

1. Cross-attention để đưa intermediate target features vào drafter.
2. Chọn layer feature theo attention entropy.
3. Visual-token compression để giảm draft latency.

Paper thử trên LLaVA, Pixtral, SmolVLM và Gemma 3, báo cáo speedup tối đa ($3.6\times$). ([OpenReview][9])

**Phân loại:** target-feature-assisted drafting.

**Ý tưởng cốt lõi:** drafter không cần tự tái khám phá toàn bộ visual semantics nếu target đã tính được representation tốt.

---

## 4.2. AASD

**Yang et al., DAC 2025**

AASD lấy KV cache lớp cuối của target làm nguồn thông tin cho speculation. Vì multimodal KV cache rất dài, phương pháp dùng KV Projector để nén trước khi thực hiện target–draft attention. Paper báo cáo speedup tối đa khoảng ($2\times$) mà không làm giảm accuracy trong thiết lập của họ. ([OpenReview][10])

**Phân loại:** target-KV-assisted drafter.

**Khác DREAM:** DREAM tập trung vào intermediate feature selection; AASD tập trung vào compressed KV cache và target–draft alignment.

---

## 4.3. FastVLM: Self-Speculative Decoding

**Bajpai and Hanawal, IJCNLP–AACL 2025**

FastVLM dùng một imitation network nhẹ để bắt chước thông tin từ các layer sâu hơn. Các layer giữa draft và target được chia sẻ, cho phép tái sử dụng KV cache trong verification. Paper báo cáo speedup khoảng ($1.55$)–($1.85\times$) với suy giảm hiệu năng nhỏ. ([ACL Anthology][11])

**Phân loại:** early-exit/self-speculative VLM.

**Điểm mạnh:** giảm memory duplication so với hai-model speculative decoding.

---

## 4.4. TwigVLM

**Shao et al., ICCV 2025**

TwigVLM thêm một nhánh nhỏ, gọi là “twig”, vào VLM gốc. Nhánh này đồng thời:

* Hướng dẫn visual-token pruning.
* Đóng vai trò self-speculative drafter.

Trên LLaVA-1.5-7B, paper báo cáo loại 88.9% visual tokens, giữ khoảng 96% hiệu năng gốc và tăng tốc đáng kể khi sinh câu trả lời dài. ([Open Access CVF][12])

**Phân loại:** self-speculation + token pruning đồng thiết kế.

**Lưu ý:** đây không phải hoàn toàn lossless về task quality vì visual-token pruning có thể làm thay đổi prediction.

---

## 4.5. HiViS

**Xie et al., CVPR 2026 Findings**

HiViS đưa ra một kết quả khá khác thường: **xóa toàn bộ visual tokens khỏi explicit input của drafter**. Thay vì bỏ hẳn visual information, drafter nhận visual semantics một cách ngầm qua last-layer hidden states của target.

Độ dài prefill của drafter chỉ còn khoảng 0.7–1.3% input length của target và paper báo cáo speedup tối đa ($2.65\times$), trong khi verifier vẫn bảo đảm output lossless. ([arXiv][13])

**Phân loại:** hidden-vision, target-state-assisted drafting.

**Thông điệp quan trọng:**

> Raw visual tokens có thể là một interface không phù hợp cho shallow drafter; compressed target semantics có thể hữu dụng hơn.

---

# 5. Nhóm IV — Video-LLM và long visual context

Đây là nhánh liên quan trực tiếp nhất đến xử lý video. Khi video tạo hàng chục nghìn token, draft model nhỏ không còn “nhanh” nếu nó vẫn phải attention trên toàn bộ visual context.

## 5.1. Sparse-to-Dense

**Zhang et al., ACL 2025 Short Paper**

Không dùng hai model khác kích thước. Thay vào đó:

* Sparse top-(K) attention version của Video-LLM làm drafter.
* Dense full-attention version của cùng model làm verifier.

Phương pháp training-free, plug-and-play và báo cáo wall-clock speedup tối đa ($1.94\times$). ([ACL Anthology][14])

**Phân loại:** self-speculative decoding bằng attention sparsification.

---

## 5.2. SpecVLM: Enhancing Speculative Decoding of Video LLMs

**Ji et al., EMNLP 2025 Main**

Đây là paper Video-LLM có tên SpecVLM.

Phương pháp dựa trên quan sát rằng drafter chịu ảnh hưởng ít hơn target khi visual tokens bị prune. SpecVLM dùng verifier attention để hướng dẫn staged token pruning và có thể loại đến khoảng 90% video tokens khỏi drafter. ([arXiv][15])

**Phân loại:** training-free, verifier-guided video-token pruning.

**Điểm mạnh:** verifier vẫn dùng context đầy đủ, do đó quá trình speculative verification có thể giữ output distribution của target.

---

## 5.3. Sparrow

**Zhang et al., ACL 2026 Long Paper**

Sparrow xử lý hai vấn đề của Video-LLM:

* Attention dilution.
* KV-cache explosion và mismatch giữa context của draft/target.

Paper khai thác hiện tượng **visual semantic internalization**: sau các layer sâu, visual semantics quan trọng đã được mã hóa một phần trong text hidden states. Drafter dùng text-anchored window attention, hidden-state reuse và intermediate visual-state bridging thay vì tiếp tục đọc toàn bộ raw visual tokens. Paper báo cáo average speedup ($2.82\times$), kể cả trường hợp khoảng 25K visual tokens. ([ACL Anthology][16])

**Phân loại:** implicit visual semantics + windowed speculation cho long video.

---

## 5.4. ParallelVLM

**Kong et al., 2026, arXiv**

ParallelVLM nhận xét draft và verification thường phải chờ nhau tuần tự. Framework song song hóa hai giai đoạn và sử dụng **Unbiased Verifier-Guided Pruning** nhằm tránh positional bias trong attention-guided token selection. ([arXiv][17])

**Phân loại:** parallel speculative pipeline + visual alignment-aware pruning.

**Khác SpecVLM:** SpecVLM chủ yếu làm drafter rẻ hơn; ParallelVLM còn nhắm đến mutual waiting và hardware utilization.

---

## 5.5. Loosely Speculative Decoding via Visual-Semantic Guidance

**Ji et al., 2026, arXiv**

Paper này nới lỏng sự phụ thuộc vào một draft Video-LLM đầy đủ bằng cách dùng visual-semantic guidance. Nó đánh giá cả standard speculative decoding với target/draft cùng family và các thiết lập giảm visual burden cho drafter. ([arXiv][18])

**Phân loại:** visual-semantic guidance, loosely coupled drafting.

---

## 5.6. HIPPO

**2026, arXiv**

HIPPO tiếp tục hướng parallel speculative decoding cho Video-LLM, tập trung vào holistic video context và song song hóa draft–verify. Hiện đây nên được xem là recent preprint hơn là baseline đã được cộng đồng kiểm chứng rộng. ([arXiv][19])

---

# 6. Nhóm V — Adaptive tree, architecture search và acceptance optimization

## 6.1. SAGE

**Tong et al., 2026, arXiv**

SAGE không giữ cây speculative cố định:

* Entropy thấp: cây sâu và hẹp.
* Entropy cao: cây nông và rộng.

Paper báo cáo decoding speedup tối đa ($3.36\times$) trên LLaVA-OneVision-72B và ($3.18\times$) trên Qwen2.5-VL-72B. ([arXiv][20])

**Phân loại:** entropy-guided dynamic tree speculation.

---

## 6.2. DREAM-S

**Liu et al., ACL 2026 Long Paper**

DREAM-S dùng neural architecture search để đồng thời chọn:

* Kiến trúc drafter.
* Cách tương tác giữa target và drafter.
* Input-pruning ratio.
* Cấu hình phù hợp với phần cứng.

Framework còn dùng attention-entropy-guided intermediate feature distillation và báo cáo speedup tối đa ($3.85\times$). ([ACL Anthology][21])

**Phân loại:** hardware-aware searchable drafting.

**Quan hệ với DREAM:**

$$
\text{DREAM: thiết kế drafter thủ công} \rightarrow \text{DREAM-S: tìm kiếm drafter và interface tự động}.
$$

---

## 6.3. MMSpec và ViSkip

**Shen et al., 2026, arXiv**

MMSpec là benchmark chuyên biệt cho VLM speculative decoding:

* 600 multimodal samples.
* 6 nhóm tác vụ.
* 10 speculative decoding algorithms trong cùng framework.

Các kết luận chính gồm:

* Phương pháp thiết kế cho text-only LLM thường suy giảm trong multimodal setting.
* Vision awareness quan trọng hơn khi batch size tăng.
* Throughput speedup không phản ánh đầy đủ latency.

Từ đó, paper đề xuất ViSkip, một phương pháp plug-and-play thích nghi speculation theo vision tokens. ([arXiv][22])

**Phân loại:** benchmark + adaptive speculation.

Đây là paper quan trọng nhất nếu mục tiêu của bạn là **thiết kế thực nghiệm công bằng**, không chỉ đề xuất một method mới.

---

## 6.4. Variational Speculative Decoding — VSD

**Zou et al., 2026, arXiv**

VSD thay objective huấn luyện drafter từ token likelihood sang xác suất một **draft path được verifier chấp nhận**. Phương pháp xây dựng objective dạng variational inference và tối ưu path-level utility. Paper đánh giá trên cả LLM và MLLM, báo cáo cải thiện speedup so với EAGLE-3 và ViSpec trong thiết lập của họ. ([arXiv][23])

**Phân loại:** acceptance-aware draft training.

Đây là thay đổi quan trọng về objective:

$$
\underbrace{\max \sum_t \log q(y_t)}_{\text{token imitation}} \quad\longrightarrow\quad \underbrace{\max \mathbb{E}[\text{accepted prefix length}]}_{\text{actual inference objective}}.
$$

---

## 6.5. TIGER

**Vo et al., tháng 7/2026, arXiv**

TIGER chọn visual tokens động theo textual state hiện tại của drafter, thay vì dùng một visual-token set cố định cho toàn bộ generation. Drafter còn được tối ưu bằng verifier-derived reward dựa trên accepted prefix length, với distillation warm-start và KL regularization. ([arXiv][24])

**Phân loại:** text-conditioned visual routing + acceptance-aligned policy training.

Đây là một trong những preprint mới nhất và có hướng nghiên cứu đáng chú ý vì nó kết hợp:

$$
\text{dynamic visual routing} + \text{verifier feedback} + \text{acceptance-aware optimization}.
$$

---

# 7. Nhóm VI — System và deployment

## CoVSpec

**Jia et al., 2026, arXiv**

CoVSpec thiết kế speculative decoding cho device–edge co-inference:

* Mobile device chạy draft VLM với visual tokens đã prune.
* Edge server chạy full target VLM.
* Draft length và verification frequency được điều chỉnh động.
* Verification và correction được tách để giảm thời gian chờ và truyền dữ liệu.

Paper báo cáo throughput tăng tối đa ($2.21\times$) và communication overhead giảm hơn 96% so với các baseline được đánh giá. ([arXiv][25])

**Phân loại:** distributed speculative decoding.

---

# 8. Bảng phân loại rút gọn

| Paper                             | Năm/venue           | Ảnh/video | Drafter                 | Xử lý visual context          | Training                       |
| --------------------------------- | ------------------- | --------- | ----------------------- | ----------------------------- | ------------------------------ |
| On Speculative Decoding for MLLMs | CVPRW 2024          | Ảnh       | LM nhỏ hoặc compact VLM | Có thể bỏ image tokens        | Có                             |
| IbED                              | ICLR Workshop 2025  | Ảnh       | Ensemble                | Không phải trọng tâm          | Có/không tùy drafter           |
| MSD                               | arXiv 2025          | Ảnh       | Multimodal drafter      | Tách text và vision           | Hai giai đoạn                  |
| MASSV                             | Findings EMNLP 2025 | Ảnh       | Adapted SLM             | Lightweight projector         | Có                             |
| DREAM                             | NeurIPS 2025        | Ảnh       | Target-aware            | Compression + target features | Có                             |
| ViSpec                            | NeurIPS 2025        | Ảnh       | Vision-aware            | Adaptor + global feature      | Có                             |
| FastVLM                           | IJCNLP–AACL 2025    | Ảnh       | Self-speculative        | Shared model states           | Có                             |
| TwigVLM                           | ICCV 2025           | Ảnh       | Self-speculative twig   | Token pruning                 | Có                             |
| SpecVLM–general                   | arXiv 2025          | Ảnh/VLM   | EAGLE-style             | Elastic compressor            | Có                             |
| HiViS                             | CVPR 2026           | Ảnh       | Target-state assisted   | Bỏ raw visual tokens          | Có                             |
| Sparse-to-Dense                   | ACL 2025            | Video     | Sparse same-model       | Top-(K) attention             | Không                          |
| SpecVLM–video                     | EMNLP 2025          | Video     | Smaller Video-LLM       | Verifier-guided pruning       | Không                          |
| Sparrow                           | ACL 2026            | Video     | Hidden-state-assisted   | Text window + visual glimpses | Có                             |
| ParallelVLM                       | arXiv 2026          | Video     | Parallel drafter        | Unbiased token pruning        | Không                          |
| SAGE                              | arXiv 2026          | Ảnh/VLM   | Tree drafter            | Không phải trọng tâm          | Tùy base                       |
| DREAM-S                           | ACL 2026            | Ảnh/VLM   | NAS-selected            | Search pruning ratio          | Có                             |
| MMSpec/ViSkip                     | arXiv 2026          | Ảnh/VLM   | Plug-and-play           | Vision-adaptive skipping      | Không hoặc nhẹ                 |
| TIGER                             | arXiv 2026          | Ảnh/VLM   | Learned drafter         | Text-conditioned routing      | Distillation + policy training |
| CoVSpec                           | arXiv 2026          | Ảnh/VLM   | Device drafter          | On-device pruning             | Không/tuỳ module               |

---

# 9. Phân loại theo câu hỏi nghiên cứu

## Nếu nghiên cứu “drafter nên nhìn ảnh thế nào?”

Ưu tiên:

1. On Speculative Decoding for MLLMs.
2. MSD.
3. MASSV.
4. ViSpec.
5. HiViS.
6. TIGER.

Các paper này tạo thành một chuỗi lập luận rất rõ:

$$
\text{no vision} \rightarrow \text{full vision} \rightarrow \text{compressed vision} \rightarrow \text{implicit vision} \rightarrow \text{dynamic vision routing}.
$$

## Nếu nghiên cứu Video-LLM

Ưu tiên:

1. Sparse-to-Dense.
2. SpecVLM–Video.
3. Sparrow.
4. ParallelVLM.
5. Loosely Speculative Decoding.

Ở đây nút thắt chính không chỉ là độ chính xác của drafter mà còn là:

$$
T_{\text{draft}} \propto |\mathrm{KV}_{\text{video}}|,
$$

nên một drafter nhỏ nhưng đọc toàn bộ video tokens vẫn có thể rất chậm.

## Nếu nghiên cứu draft training objective

Ưu tiên:

1. DREAM.
2. ViSpec.
3. SpecVLM–general.
4. VSD.
5. TIGER.

Xu hướng mới nhất là chuyển từ **logit imitation** sang tối ưu trực tiếp accepted prefix length hoặc latency.

## Nếu nghiên cứu system/hardware

Ưu tiên:

1. MMSpec.
2. DREAM-S.
3. ParallelVLM.
4. CoVSpec.

---

# 10. Những khoảng trống nghiên cứu còn rõ

### End-to-end speedup chưa được chuẩn hóa

Nhiều paper báo cáo “speedup” nhưng có thể đo các đại lượng khác nhau:

* Decode-only latency.
* End-to-end latency.
* Throughput.
* Tokens/s.
* Speedup với batch size 1 hoặc batch lớn.

MMSpec chỉ ra rằng throughput speedup không nhất thiết phản ánh latency thực tế. Vì vậy không nên xếp hạng trực tiếp các con số ($2.5\times$), ($3.6\times$) hay ($3.85\times$) giữa các paper. ([arXiv][22])

### Output của benchmark VLM thường quá ngắn

VQA thường chỉ sinh vài token. Trong trường hợp đó:

$$
T_{\text{prefill}} \gg T_{\text{decode}},
$$

nên speculative decoding có thể tăng decoding throughput nhưng gần như không làm end-to-end latency tốt hơn. ViSpec phải tạo long-response data một phần vì vấn đề này. ([arXiv][6])

### Visual-token compression và acceptance chưa được đồng tối ưu đầy đủ

Phần lớn phương pháp chọn compression ratio trước, rồi mới chạy speculation. TIGER và một số hướng 2026 mới bắt đầu dùng accepted prefix length làm feedback để chọn visual evidence. ([arXiv][24])

### Thiếu đánh giá video dài thực sự

Nhiều hệ thống được đánh giá trên clip ngắn hoặc số frame cố định. Sparrow tiến gần hơn tới long-video setting với khoảng 25K visual tokens, nhưng streaming video, multi-turn questions và cache reuse vẫn là các bài toán mở. ([ACL Anthology][16])

### Lossless cần được định nghĩa cẩn thận

Có ít nhất ba cách dùng từ “lossless”:

1. Giữ đúng phân phối sampling của target.
2. Greedy output giống target.
3. Benchmark accuracy không giảm đáng kể.

Chỉ hai cách đầu mới là lossless theo nghĩa speculative decoding chặt chẽ; token-pruning methods có thể giữ accuracy nhưng vẫn thay đổi output.

---

# 11. Danh sách đọc khuyến nghị

Để nắm lĩnh vực nhanh và đúng flow, thứ tự tốt nhất là:

1. **On Speculative Decoding for Multimodal Large Language Models** — baseline đầu tiên.
2. **ViSpec** — phân tích rõ visual redundancy và shallow drafter.
3. **DREAM** — target-aware feature drafting.
4. **HiViS** — loại raw visual tokens khỏi drafter.
5. **SpecVLM–Video** — token pruning cho Video-LLM.
6. **Sparrow** — long-video và hidden-state reuse.
7. **MMSpec** — cách benchmark và so sánh công bằng.
8. **DREAM-S hoặc TIGER** — hướng nghiên cứu mới: hardware-aware và acceptance-aware.

Từ literature hiện tại, hướng có tiềm năng mạnh nhất là:

$$
\boxed{\text{adaptive visual routing} + \text{target-state reuse} + \text{acceptance-aware draft training} + \text{hardware-aware scheduling}}
$$

Thay vì chỉ xây dựng một draft model nhỏ hơn, hệ thống sẽ quyết định động **drafter cần nhìn phần nào của video, nhìn ở representation nào, dự đoán bao xa và chạy trên tài nguyên nào**.

[1]: https://arxiv.org/abs/2404.08856?utm_source=chatgpt.com "On Speculative Decoding for Multimodal Large Language Models"
[2]: https://openreview.net/pdf?id=ffDhpmwqdu&utm_source=chatgpt.com "IN-BATCH ENSEMBLE DRAFTING: ROBUST SPECULA"
[3]: https://aclanthology.org/2026.findings-eacl.205.pdf?utm_source=chatgpt.com "TABED: Test-Time Adaptive Ensemble Drafting for Robust ..."
[4]: https://arxiv.org/abs/2505.14260?utm_source=chatgpt.com "Speculative Decoding Reimagined for Multimodal Large Language Models"
[5]: https://aclanthology.org/2025.findings-emnlp.656/?utm_source=chatgpt.com "MASSV: Multimodal Adaptation and Self-Data Distillation ..."
[6]: https://arxiv.org/html/2509.15235v1 "ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding"
[7]: https://arxiv.org/abs/2509.11815?utm_source=chatgpt.com "SpecVLM: Fast Speculative Decoding in Vision-Language Models"
[8]: https://arxiv.org/abs/2509.11961?utm_source=chatgpt.com "Spec-LLaVA: Accelerating Vision-Language Models with Dynamic Tree-Based Speculative Decoding"
[9]: https://openreview.net/forum?id=M5jz47umjR&utm_source=chatgpt.com "DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding | OpenReview"
[10]: https://openreview.net/forum?id=M4EySXQdo4&utm_source=chatgpt.com "Accelerate Inference by Aligning Speculative Decoding in ..."
[11]: https://aclanthology.org/2025.ijcnlp-long.64/?utm_source=chatgpt.com "Self-Speculative Decoding for Fast Vision-Language ..."
[12]: https://openaccess.thecvf.com/content/ICCV2025/papers/Shao_Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models_ICCV_2025_paper.pdf?utm_source=chatgpt.com "Growing a Twig to Accelerate Large Vision-Language Models"
[13]: https://arxiv.org/html/2509.23928v1 "HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models"
[14]: https://aclanthology.org/2025.acl-short.59/?utm_source=chatgpt.com "Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs - ACL Anthology"
[15]: https://arxiv.org/abs/2508.16201?utm_source=chatgpt.com "Enhancing Speculative Decoding of Video LLMs via ..."
[16]: https://aclanthology.org/2026.acl-long.450/?utm_source=chatgpt.com "Sparrow: Text-Anchored Window Attention with Visual- ..."
[17]: https://arxiv.org/abs/2603.19610?utm_source=chatgpt.com "ParallelVLM: Lossless Video-LLM Acceleration with Visual ..."
[18]: https://arxiv.org/html/2604.05650v1?utm_source=chatgpt.com "Loosely Speculative Decoding via Visual-Semantic ..."
[19]: https://arxiv.org/html/2601.08273v1?utm_source=chatgpt.com "HIPPO: Accelerating Video Large Language Models ..."
[20]: https://arxiv.org/abs/2602.00523?utm_source=chatgpt.com "SAGE: Accelerating Vision-Language Models via Entropy-Guided Adaptive Speculative Decoding"
[21]: https://aclanthology.org/2026.acl-long.2177.pdf?utm_source=chatgpt.com "DREAM-S: Speculative Decoding with Searchable Drafting ..."
[22]: https://arxiv.org/abs/2603.14989?utm_source=chatgpt.com "MMSpec: Benchmarking Speculative Decoding for Vision-Language Models"
[23]: https://arxiv.org/abs/2602.05774?utm_source=chatgpt.com "Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance"
[24]: https://arxiv.org/abs/2607.11131?utm_source=chatgpt.com "TIGER: Text-Conditioned Visual Gated Routing with Acceptance Alignment for Multimodal Speculative Decoding"
[25]: https://arxiv.org/abs/2605.02218?utm_source=chatgpt.com "CoVSpec: Efficient Device-Edge Co-Inference for Vision-Language Models via Speculative Decoding"
