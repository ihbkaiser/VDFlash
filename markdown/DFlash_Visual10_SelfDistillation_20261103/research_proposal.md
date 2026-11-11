# DFlash 10% visual với hai nguồn supervision: target model và full-visual DFlash

**Ngày cập nhật:** 2026-11-03.  
**Trạng thái:** tài liệu ý tưởng và thiết kế thực nghiệm; chưa huấn luyện student hoặc đo kết quả mới.  
**Phạm vi chính:** một target Qwen2.5-VL đóng băng, một full-visual DFlash teacher đóng băng và một DFlash student được huấn luyện với 10% visual positions.

## 1. Tóm tắt ý tưởng

Huấn luyện DFlash student chỉ truy cập 10% visual positions bằng hai nguồn supervision bổ sung cho nhau. Target model cung cấp token chuẩn để student học cách đề xuất token được verifier chấp nhận. Full-visual DFlash cung cấp phân phối proposal để truyền hành vi drafting khi có đầy đủ visual context. Student khởi tạo từ cùng checkpoint với full-visual DFlash teacher.

Mục tiêu là kết hợp acceptance tốt của full-visual DFlash với chi phí conditioning thấp của student 10%. Trong inference, full-visual DFlash teacher được bỏ khỏi hệ thống; target full visual xác minh các block do student 10% đề xuất.

Phương pháp cơ bản dùng target-token cross-entropy (CE) cùng KL từ full-visual DFlash. Một biến thể thêm gate theo target để hạn chế student bắt chước những dự đoán sai của full teacher. Teacher DFlash low-retention là một mở rộng riêng, không thuộc phương pháp hai nguồn trình bày ở đây.

## 2. Động lực từ thực nghiệm

Nguồn nội bộ: [tổng hợp năm cấu hình](../Baocao_DFlash_4Config_20261029/tong_hop_so_sanh_4_config_dflash.md), [distribution replay](../DFlash_VisualCut_Distribution_20261102/final_report.md) và [workflow](../../WORKFLOW_NOTES.md).

### 2.1. Acceptance tương đối ổn định khi giảm visual context

Qua năm cấu hình high_only, middle_high, middle_high_2, shallow_high và balanced, contrast τ_effective(cut − full) có pattern theo visual budget tương đối nhất quán:

| Visual budget | Mean Δτ qua 5 config | Quan sát |
|---:|---:|---|
| 400 | −0,053 | Full nhỉnh hơn; một config có CI hoàn toàn âm |
| 3.000 | −0,078 | Cut giảm có ý nghĩa ở 4/5 config |
| 13.000 | −0,006 | Chênh lệch rất nhỏ, cả năm CI chứa 0 |
| 25.000 | +0,033 | Cut gần full hoặc cao hơn; middle_high_2 tăng +0,115 |

Các mean trên được tính từ contrast đã làm tròn trong báo cáo, dùng để mô tả xu hướng. Năm config dùng chung cohort VDC50 và target, nên đây là tính ổn định qua layer selection trong cùng study. CI chứa 0 cho biết chưa phát hiện contrast khác 0; kiểm định tương đương cần biên được định trước.

Retention sweep cho thấy giữ 25–75% visual có thể cải thiện τ ở context dài. Mức giữ 10% chưa được kiểm tra trong campaign đó.

### 2.2. Phân phối khác nhau dù acceptance trung bình gần nhau

Distribution replay balanced ghi nhận JSD full/cut tăng từ 0,0526 lên 0,1522 bit theo budget; top-1 khác nhau ở 24,0–49,6% draft positions. Tuy nhiên, chênh lệch target-hit@1 chỉ tối đa 0,74 điểm phần trăm.

Full và cut có những vị trí đúng khác nhau. Do đó, bắt student khớp toàn bộ phân phối full teacher có thể làm mất một số dự đoán đúng vốn xuất hiện khi giảm context. Target-token supervision cung cấp tín hiệu để student tiếp tục ưu tiên khả năng khớp verifier.

Hai campaign cần đối chiếu checkpoint fingerprint, layer indexing, preprocessing và output contract trước khi xem như một phép đo thống nhất trên cùng weights.

### 2.3. Giả thuyết nghiên cứu

**Giả thuyết chính:** ở mức giữ 10% visual positions, student được huấn luyện bằng target-token CE và full-DFlash KL sẽ giữ acceptance gần full, đồng thời cải thiện so với student 10% chưa huấn luyện.

**Giả thuyết về hai nguồn:** full-DFlash KL có giá trị bổ sung so với chỉ fine-tune student bằng target-token CE.

**Giả thuyết về gate:** giảm KD tại vị trí full teacher sai target giúp giữ các dự đoán hữu ích của low-retention student và cải thiện accepted-prefix so với KD không gate.

Các giả thuyết trên chưa được xác nhận. Chênh lệch phân phối và τ hiện có tạo động lực thực nghiệm, chưa bảo đảm student sẽ học được lợi ích mong muốn.

## 3. Ba mô hình và vai trò từng nguồn thông tin

| Thành phần | Visual context | Thông tin cung cấp | Cập nhật weights? | Có trong inference? |
|---|---|---|---|---|
| Target model M | Full | Token chuẩn, embeddings và target hidden states; verifier khi decode | Đóng băng | Có |
| DFlash teacher F | Full visual positions và toàn bộ text context hợp lệ | Phân phối proposal tại các draft offsets | Đóng băng | Không |
| DFlash student S | 10% visual positions và toàn bộ text context hợp lệ | Proposal blocks cần được target accept | Có | Có |

Target có hai vai trò cần phân biệt. Token chuẩn của target là supervision cho CE. Hidden states và embeddings của target là conditioning/input cho DFlash. Phương pháp cơ bản chưa cần thêm KL trực tiếp từ full-vocabulary logits của target; nhánh KL sử dụng phân phối full-visual DFlash.

Từ “self-distillation” chỉ việc teacher và student DFlash cùng kiến trúc và cùng checkpoint khởi tạo, nhưng khác lượng conditioning context. Sau fine-tuning, student có weights riêng.

~~~mermaid
flowchart TD
    X["Video + câu hỏi"] --> M["Target full visual — đóng băng"]
    M --> Y["Target-generated token labels"]
    M --> H["Target hidden states"]
    H --> F["DFlash teacher: full context — đóng băng"]
    H --> K["Giữ text + chọn 10% visual positions"]
    K --> S["DFlash student — cập nhật weights"]
    A["Cùng anchors và mask layout"] --> F
    A --> S
    Y --> CE["Target-token CE"]
    S --> CE
    F --> KD["KL từ full-DFlash teacher"]
    S --> KD
    CE --> LOSS["Loss huấn luyện student"]
    KD --> LOSS
~~~

**Sơ đồ inference:** video/câu hỏi → target full visual → chọn conditioning 10% → student draft block → target verify → cập nhật tiền tố và lặp.

## 4. Input, ký hiệu và ràng buộc nhân quả

Với input x, target sinh response y bằng greedy decoding. Chọn anchor a và block size B; anchor chứa token sạch của target, còn B−1 vị trí tiếp theo được dự đoán song song. Ký hiệu k = 1,…,B−1 là draft offset sau anchor.

Teacher và student nhận cùng sample, prefix hợp lệ tại anchor, anchor token, mask token IDs, block size và supervised offsets. Đặt:

- p_F,a,k: phân phối full-visual DFlash tại offset k.
- p_S,a,k: phân phối student 10% tại cùng offset.
- y_a+k: token trên greedy response của target, dùng làm nhãn.
- Ω: tập các cặp anchor–offset hợp lệ, loại anchor, padding và vị trí ngoài response.

Token y_a+k được target sinh theo tiền tố sạch đến vị trí trước nó. DFlash dự đoán cả block từ conditioning hợp lệ tại anchor. Vì thế, target-generated labels và full-DFlash proposals có vai trò khác nhau ngay cả khi dùng chung target features.

Trong training trên toàn clean sequence, hidden states của tương lai chứa thông tin về nhãn. Mask phải chặn các context keys không hợp lệ với anchor theo contract DFlash hiện có. Teacher/student dùng chung anchors; không đưa các future hidden states vào student thông qua bước compact hoặc selector.

Nếu dùng selector dựa trên features, chỉ được sử dụng prompt features hoặc các features thuộc verified prefix hiện có. Lấy mẫu visual prompt thuần temporal/spatial đáp ứng điều kiện này.

## 5. Student giữ 10% visual như thế nào?

Gọi V là tập visual placeholder positions sau preprocessing, N_v = |V| và:

$$
K=\lceil 0.1N_v\rceil.
$$

Student giữ đúng K positions từ V, cùng toàn bộ text positions hợp lệ. Báo cáo tỷ lệ thực tế K/N_v và quy ước visual delimiters; teacher và target vẫn dùng full visual.

Sampler khởi đầu phân quota cho các temporal groups từ video_grid_thw, rồi lấy spatial coverage trong từng group. Tổng quota nguyên phải đúng K. Nếu K nhỏ hơn số temporal groups, chọn các nhóm rải đều thời gian. Temporal grid có thể đã merge nhiều raw frames, nên cần dùng đúng ánh xạ của processor.

Trong training có thể thay mẫu visual với seed đã ghi; khi đánh giá ghép cặp, cố định indices giữa các student recipes. Baseline prefix retention vẫn được giữ để đo riêng tác dụng của sampler.

Text hidden states do target tạo ra đã được tính với full visual. Đây là một cơ chế khả dĩ giúp student giữ thông tin thị giác gián tiếp dù chỉ truy cập một phần visual positions trực tiếp.

Giữ tọa độ M-RoPE gốc của các positions được chọn và tọa độ sinh của target. Compact cache indices cần ánh xạ riêng tới original indices. Việc bỏ visual positions chỉ tác động conditioning của draft.

## 6. Objective sử dụng hai nguồn supervision

### 6.1. Target-token loss

$$
\mathcal L_{\mathrm{target}}
=
\frac{\sum_{(a,k)\in\Omega}w_k[-\log p_{S,a,k}(y_{a+k})]}
{\sum_{(a,k)\in\Omega}w_k}.
$$

Loss này dạy student dự đoán token của target tại tất cả supervised offsets. Nó vẫn hoạt động khi full-DFlash teacher dự đoán sai.

w_k giữ loss-decay theo draft offset của cấu hình DFlash hiện tại. Vị trí đầu block được ưu tiên vì accepted-prefix kết thúc tại lỗi đầu tiên. Việc tăng per-position hit vẫn cần được kiểm chứng bằng decode thực tế.

### 6.2. Full-DFlash distillation loss

Với logits z và temperature T, đặt p^(T) = softmax(z/T). Dùng hướng KL teacher → student:

$$
\mathcal L_{\mathrm{KD}}
=
\frac{\sum_{(a,k)\in\Omega}w_kT^2
D_{\mathrm{KL}}(p_{F,a,k}^{(T)}\parallel p_{S,a,k}^{(T)})}
{\sum_{(a,k)\in\Omega}w_k}.
$$

Teacher logits được detach. Loss dùng full vocabulary; top-100 artifacts hiện tại không đủ để tính exact full-vocabulary KL. Có thể tính theo block chunks để giới hạn peak memory.

KL truyền cả cấu trúc xác suất của proposal distribution. Full teacher là một draft model nên cũng có lỗi; độ gần với teacher cần được đánh giá cùng độ khớp target.

### 6.3. Phương pháp cơ bản

$$
\boxed{\mathcal L_{\mathrm{base}}
=\mathcal L_{\mathrm{target}}+\lambda\mathcal L_{\mathrm{KD}}.}
$$

CE giữ mục tiêu khớp verifier; KL truyền hành vi drafting của full-visual DFlash. Student vẫn phải thích nghi với thông tin đầu vào bị giảm.

Khởi đầu đề xuất T=1, λ=0,5; kiểm tra thêm λ=0,1 với cùng data/steps nếu cần. Đây là starting points để chọn trên validation, chưa phải hyperparameters đã tối ưu. Giữ nguyên loss-decay và các thiết lập khác trong phép so sánh đầu tiên.

### 6.4. Biến thể KD có gate theo target

Đặt gate được tính từ teacher ở T=1:

$$
g_{a,k}=\mathbf 1[\arg\max p_{F,a,k}=y_{a+k}].
$$

$$
\mathcal L_{\mathrm{gated}}
=
\mathcal L_{\mathrm{target}}
+
\lambda
\frac{\sum_{(a,k)\in\Omega}w_kg_{a,k}T^2
D_{\mathrm{KL}}(p_{F,a,k}^{(T)}\parallel p_{S,a,k}^{(T)})}
{\sum_{(a,k)\in\Omega}w_k}.
$$

Gate dùng nhãn target lúc training và không nhận gradient. Mẫu số vẫn là tổng weight của mọi vị trí hợp lệ, nên KD có mức đóng góp thấp hơn khi full teacher ít hit. Cần log weighted gate coverage và CE/KD riêng để diễn giải ảnh hưởng này; một control nhân KD thường với cùng mean gate coverage giúp tách tác dụng chọn vị trí khỏi tác dụng giảm cường độ KD.

| Tình huống tại cùng anchor/offset | CE target | KD trong biến thể gated |
|---|---|---|
| Full teacher đúng, student sai | Dạy token target | Truyền phân phối full teacher |
| Full teacher đúng, student đúng | Tiếp tục củng cố token target | Học cấu trúc proposal distribution |
| Full teacher sai, student đúng | Củng cố token target | Tắt |
| Full teacher sai, student sai | Dạy token target | Tắt |

Gate chỉ cần biết teacher đúng hay sai; trạng thái student trong bảng dùng để giải thích tác động. Student chưa cần một low-retention teacher riêng.

Gate giúp giảm xung đột supervision, nhưng có thể bỏ qua soft information hữu ích khi teacher top-1 sai. Chia sẻ weights cũng có thể làm thay đổi các dự đoán student vốn đúng. Vì vậy base và gated phải được so sánh, không coi gate là cải thiện đã được chứng minh.

## 7. Quy trình training và inference

### 7.1. Training

1. Chọn một checkpoint balanced có fingerprint rõ ràng; tạo full-visual teacher F và student S cùng weights khởi tạo. Đóng băng F và target M.
2. Chuẩn bị video/question và target-generated response. Split train/validation/test theo video/source, tránh clip gần trùng.
3. Chạy target full visual hoặc đọc feature cache hợp lệ để lấy labels, embeddings và selected-layer hidden states. Reuse target features cho F và S.
4. Lấy cùng anchors, noise/mask layout và supervised offsets cho hai nhánh.
5. Chọn K visual prompt positions; xây student context/masks từ original indices. Giữ text context hợp lệ.
6. Chạy F full context với no-grad và S 10% context với gradient.
7. Tính CE, KL và gate nếu recipe sử dụng. Tính logits/loss theo chunks khi cần, normalize trên toàn valid offsets.
8. Backpropagate và cập nhật student. Giữ nguyên quy ước embedding/lm_head của pipeline và đóng băng các weights thuộc target.
9. Đánh giá student theo checkpoint trên full validation ở các mốc bước cố định; chọn recipe/λ dựa trên validation acceptance và latency.
10. Lưu student weights cùng checkpoint fingerprint, selector contract, retained indices/seed policy, layer selection và hyperparameters cần tái lập.

Target-generated trajectories có thể chuẩn bị offline, trong khi logits của full-DFlash teacher được tính on-the-fly. Không cần lưu full-vocabulary logits trên đĩa.

### 7.2. Inference

1. Target chạy prefill full visual, tạo features và anchor token.
2. Chọn 10% visual positions cho draft; tạo context/cache compact của student.
3. Student đề xuất B−1 tokens song song quanh anchor sạch.
4. Target verify block, accept prefix và cung cấp token tiếp theo theo decode contract.
5. Cập nhật target cache, student cache và verified text features; lặp tới EOS hoặc giới hạn sinh.

Inference dùng target M và student S. λ, teacher logits và gate chỉ phục vụ training.

Nếu offline training cho tín hiệu tốt, có thể refresh anchors và first-rejection cases từ student chạy với verifier. Với greedy lossless, verified trajectory vẫn theo target; student làm thay đổi block boundaries và tần suất gặp các offsets. Các draft tokens đã bị reject không được đưa vào clean prefix.

## 8. Thực nghiệm cần có để chứng minh phương pháp

### 8.1. Các đối chứng

Giữ cùng target, student architecture, checkpoint khởi tạo, selector, preprocessing, block size và generation limit, trừ biến đang khảo sát:

| ID | Cấu hình | Câu hỏi được kiểm tra |
|---|---|---|
| F0 | Full-visual DFlash gốc | Acceptance và runtime tham chiếu |
| C0 | Cut 0% gốc | Điểm tham chiếu từ campaign trước |
| R25 | Retained 25% gốc | Điểm tham chiếu retention trung gian |
| S0 | Student 10% chưa fine-tune | Chỉ giảm context có đủ chưa? |
| S-CE | Student 10%, target CE | Tác dụng fine-tuning theo target |
| S-KD | Student 10%, chỉ full-DFlash KL | Mức đóng góp của supervision teacher |
| S-Dual | Student 10%, CE + KL | Hai nguồn có bổ sung cho nhau? |
| S-Gated | Student 10%, CE + gated KL | Chọn vị trí teacher đúng có giúp không? |

Chạy baseline trước, rồi S-CE/S-KD/S-Dual, tiếp theo S-Gated và control KD cùng cường độ nếu cần. So sánh matched data, optimizer steps, valid supervised positions; báo cáo GPU-hours vì các recipe có chi phí khác nhau. Sampler prefix và temporal/spatial là ablation riêng.

### 8.2. Data và khả năng khái quát

Train gồm video ở bốn dải visual budget, cân bằng sample-level contribution để video dài không chi phối loss. Bổ sung nhiệm vụ temporal/action và chi tiết nhỏ giúp đánh giá những trường hợp 10% dễ mất thông tin.

VDC50 đã được dùng để xây giả thuyết nên được dùng làm regression set; cần thêm held-out video mới để xác nhận. Chạy nhiều seeds cho recipe tốt nhất, sau đó thêm layer selection khác. Báo cáo theo từng budget để phát hiện giảm ở vùng 3K.

### 8.3. Metrics

**Acceptance:** τ_effective, τ_proposal, target-hit@1 theo offset, first-rejection histogram và accepted-prefix survival. Với L là số proposal tokens được accept trong một valid round:

$$
\tau_{\mathrm{proposal}}=\mathbb E[L]
=\sum_{k=1}^{B-1}P(L\ge k),\qquad
\tau_{\mathrm{effective}}=1+\mathbb E[L].
$$

Công thức cộng một bonus token áp dụng theo valid-round contract; cần xử lý nhất quán round cuối, EOS và truncation.

**Phân phối:** CE với target, target-token probability, KL/JSD teacher–student và bốn nhóm teacher/student đúng–sai. JSD giảm chỉ cho biết phân phối gần teacher hơn.

**Runtime:** draft prefill, draft forward/round, verifier/round, selection time, TTFT, decode TPS, end-to-end latency, peak memory và KV bytes riêng target/draft. Giữ hardware/dtype, warmup và generation settings cố định; lặp timing và bootstrap ghép cặp cấp video. Tắt attention capture khi đo tốc độ.

**Output contract:** kiểm tra greedy output parity với target theo cùng EOS/length contract. Các báo cáo cũ ghi lossless rate thấp dù BLEU/ROUGE gần nhau, nên cần audit artifact trước campaign xác nhận; độ gần các metric ngữ nghĩa không thay thế output parity.

### 8.4. Tiêu chí thành công đề xuất

- Student dùng K = ceil(0,1 × actual_visual_count); target luôn full visual.
- Lower bound CI của Δτ(student − full) > −0,05 token/vòng theo từng budget. Biên 0,05 là đề xuất ban đầu cần cố định trước khi đánh giá xác nhận.
- S-Dual hoặc S-Gated cải thiện so với S0 và S-CE để xác nhận giá trị bổ sung của KD, thay vì chỉ lợi ích thêm training.
- Runtime tốt hơn full với CI/độ lặp đủ phân biệt khỏi noise; selection overhead được tính vào phép đo.
- Kết luận về output preservation dựa trên verifier contract và parity đã kiểm tra.

Mục tiêu cao hơn là giữ hoặc vượt baseline tốt hơn giữa F0 và S0 theo từng budget. Chọn baseline/recipe dựa trên validation rồi xác nhận trên test, tránh chọn max theo nhiễu test.

## 9. Khả thi trong repository và giới hạn hiệu quả

| Thành phần | Điểm tích hợp hiện có | Công việc khi triển khai |
|---|---|---|
| Training objective/masks | [dflash_family_model.py](../../src/train_Dflash_SpecForge/specforge/algorithms/common/dflash_family_model.py) | Reuse anchors; compact mapping; thêm KL/gate và metrics |
| Training config | [schema.py](../../src/train_Dflash_SpecForge/specforge/config/schema.py) | Retention, selector, frozen teacher, λ/T và recipe |
| DFlash wiring | [providers.py](../../src/train_Dflash_SpecForge/specforge/algorithms/dflash/providers.py) | Shared target features và teacher no-grad |
| Sampler baseline | [run_dflash_h21_h31.py](../../src/analyze/Validate_Sparrow_hypothesises/run_dflash_h21_h31.py) | Giữ prefix sampler làm control; thêm temporal/spatial selector |
| Compact inference | [qwen25vl_dflash_compare.py](../../src/infer/qwen25vl_dflash_compare.py) | Context/cache compact, M-RoPE và append verified text |
| Attention/KV | [dflash.py](../../src/train_Dflash_SpecForge/specforge/modeling/draft/dflash.py) | Phân biệt dense KV rồi gather với compact cache thật |
| Replay evaluation | [run_dflash_distribution_replay.py](../../src/analyze/Validate_Sparrow_hypothesises/run_dflash_distribution_replay.py) | Thêm student recipes, gate coverage và offset metrics |
| Free-running evaluation | [run_dflash_visualcut_attention.py](../../src/analyze/Validate_Sparrow_hypothesises/run_dflash_visualcut_attention.py) | Thêm retention 10%, timing độc lập capture và parity |

Helper build_visual_retention_mask trong [dflash_runtime.py](../../src/analyze/Validate_Sparrow_hypothesises/dflash_runtime.py) zero các rows; thao tác này chưa tự compact context. Một nhánh attention context_keep_indices lưu dense KV rồi gather. Claim giảm draft KV cần dùng đúng đường compact vật lý và đo bytes thực tế.

Trước training, kiểm tra parity giữa dense masked attention và compact attention trên cùng selected positions; kiểm tra causal mask, M-RoPE mapping, gradient chỉ vào student và KL finite. GPU runtime pilot dùng context thật để xác nhận throughput/VRAM. Chưa có số VRAM/GPU-hours đáng tin cậy từ nghiên cứu đọc code.

Nếu block size cố định, tốc độ decode có thể gần đúng bằng:

$$
\frac{\mathrm{TPS}_S}{\mathrm{TPS}_F}
\approx
\frac{\tau_S}{\tau_F}
\frac{t_{\mathrm{draft},F}+t_{\mathrm{verify},F}}
{t_{\mathrm{draft},S}+t_{\mathrm{verify},S}}.
$$

End-to-end còn gồm target prefill và preprocessing. Giảm 90% visual rows của draft chỉ giảm phần visual conditioning của draft; target KV và toàn hệ thống không giảm theo cùng tỷ lệ.

## 10. Cơ sở tài liệu và phạm vi đóng góp

[DFlash](https://arxiv.org/html/2602.06036v1) cung cấp kiến trúc block-parallel, target-feature conditioning, random anchors và loss-decay. Phương pháp đề xuất giữ các thành phần này và nghiên cứu distillation khi student giảm visual context.

[DistillSpec](https://arxiv.org/abs/2310.08461) hỗ trợ việc thiết kế distillation theo decoding strategy và dữ liệu draft gặp khi inference. [GKD](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html) nghiên cứu mismatch giữa dữ liệu cố định và student-generated sequences; áp dụng vào DFlash cần giữ verified-prefix contract.

[FastVID](https://arxiv.org/abs/2503.11187) và [DivPrune](https://openaccess.thecvf.com/content/CVPR2025/papers/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.pdf) tạo cơ sở cho temporal coverage và feature diversity trong chọn visual tokens. Hiệu quả pruning VLM từ các paper này chưa xác nhận trực tiếp hiệu quả conditioning 10% của DFlash.

Phạm vi nghiên cứu ở đây là khả năng phối hợp target supervision và full-DFlash supervision khi student chỉ truy cập 10% visual positions. CE + KL là kỹ thuật distillation đã có; đóng góp thực nghiệm cần chứng minh giá trị của hai nguồn, ảnh hưởng của gate tới accepted-prefix và lợi ích runtime trong pipeline video cụ thể.

## 11. Các mở rộng sau phương pháp chính

| Mở rộng | Điều kiện nên thử | Thay đổi so với phương pháp chính |
|---|---|---|
| Teacher DFlash low-retention riêng | Có nhiều vị trí low-only đúng tại early offsets | Thêm nguồn proposal supervision thứ ba |
| Selector dựa trên diversity/relevance | 10% phủ đều thiếu thông tin | Thay selector, tính thêm overhead và ablation |
| Curriculum 50% → 25% → 10% | Train trực tiếp 10% khó hội tụ | Thay lịch retention, so matched training compute |
| Refresh anchors theo student | Offline KD có gain nhưng còn lệch first-rejection | Thu data từ student + target verifier |
| Adaptive retention/fallback full | Cho phép vượt 10% ở một số vòng | Thay ràng buộc deployment, báo cáo mean/p95 retention |

Những mở rộng này được đánh giá sau khi hai nguồn supervision cơ bản đã có đối chứng rõ ràng.

## 12. Nội dung trình bày ngắn

> Chúng tôi đề xuất huấn luyện DFlash student với 10% visual positions bằng hai nguồn supervision: token chuẩn từ target model full visual và phân phối proposal từ full-visual DFlash đóng băng. Target-token loss hướng student tới các dự đoán được verifier chấp nhận, còn distillation loss truyền hành vi drafting khi có đầy đủ visual context. Một biến thể dùng target labels để chỉ distill tại các vị trí full teacher dự đoán đúng. Khi inference, chỉ student 10% và target full visual được sử dụng. Thực nghiệm sẽ kiểm tra liệu hai nguồn supervision có giúp student giữ accepted-prefix gần full, cải thiện so với fine-tuning CE-only và giảm thời gian drafting/end-to-end hay không.

Các lựa chọn vận hành cần chốt khi chuyển sang implementation gồm checkpoint/fingerprint, nguồn train video, GPU/training budget, cadence validation và biên non-inferiority cuối cùng. Tài liệu này hoàn thiện phương pháp đề xuất; các lợi ích của student 10% vẫn là giả thuyết cần đo.
