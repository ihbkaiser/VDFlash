# Kế hoạch thực nghiệm kiểm chứng các hypothesis về visual context

Ngày chốt thiết kế: 2026-09-07

Tài liệu này chuyển các suy luận hiện tại thành protocol có thể chạy và quy định
trước cách diễn giải kết quả. Đây là kế hoạch thực nghiệm, chưa phải báo cáo kết
quả. Các kết quả cũ trong `results/` chỉ được dùng làm exploratory evidence cho
đến khi khớp đúng condition, cohort và audit của protocol này.

## 1. Câu hỏi trung tâm

Visual information có thể đã được target model nén vào text-conditioned hidden
states. Khi đó visual slots có thể còn hai vai trò khác nhau:

1. mang visual content thông qua value vectors;
2. chiếm attention mass trong mẫu số softmax dù value của chúng không còn hữu ích.

Vai trò thứ hai tạo ra trade-off:

- xoá toàn bộ visual slots có thể làm text branch được khuếch đại quá mức
  (text over-weighting);
- giữ quá nhiều visual slots không hữu ích có thể làm loãng contribution của text
  semantics (attention dilution);
- một lượng visual slots trung gian có thể tối ưu acceptance.

## 2. Quy ước và bất biến chung

### 2.1. Các condition

| Ký hiệu | Condition | Ý nghĩa |
|---|---|---|
| F | Full visual | Giữ đầy đủ visual keys và values |
| R | Reduced visual | Giữ một tỷ lệ visual context, ví dụ 25%, 10%, 5% |
| D | Deleted visual | Xoá visual slots khỏi draft context |
| Z | Null visual slots | Giữ visual keys và positions, đặt visual values bằng 0 |

F/R/D là các condition benchmark tự nhiên. Z là control cơ chế và cần được
implement riêng nếu muốn kết luận về softmax denominator.

Khi chỉ nghiên cứu drafter:

- target luôn nhận full visual context;
- chỉ draft-side context bị can thiệp;
- target input fingerprint và target greedy output phải giống nhau giữa các
  condition;
- temperature bằng 0;
- prompt, frame sampling, visual-token budget, max output length và checkpoint
  được giữ cố định.

Nếu final speculative output khác target output thì đó là lỗi losslessness hoặc
target isolation, không phải bằng chứng task quality giảm.

### 2.2. Attention và representation metrics

Với một query position (i):

\[
M_V(i)=\sum_{j\in V}a_{ij},\qquad
M_T(i)=\sum_{j\in T}a_{ij}.
\]

Ngoài global mass, phải ghi:

- conditional text/visual entropy;
- attention concentration và top-k mass;
- effective number of attended tokens;
- text contribution norm \(\|o_T\|\);
- visual contribution norm \(\|o_V\|\);
- draft-target KL và top-1 agreement.

Global text mass tăng không tự động chứng minh text over-attention. Nếu chỉ thay
đổi mẫu số, các text weights có thể cùng được scale lên mà phân phối tương đối
giữa các text tokens không đổi.

### 2.3. Acceptance metric

Với acceptance round (t), gọi (k_t) là số draft proposals được target chấp
nhận. Báo cáo cả:

\[
\tau_{prop}=E[k_t],\qquad
\tau_{eff}=E[k_t+1].
\]

Acceptance là metric chính của speculative decoding. Task accuracy là guardrail
khi target decoding lossless; nếu cần đánh giá chất lượng riêng của drafter,
phải chạy thêm draft-only evaluation.

### 2.4. Thống kê

- Mọi condition được chạy paired theo sample.
- Bootstrap theo sample, không bootstrap độc lập từng token.
- Báo cáo mean, median, 95% CI và phân phối per-sample.
- Với các claim “không tốt hơn”, dùng equivalence margin thay vì suy luận từ
  p-value không có ý nghĩa.
- Margin mặc định cho acceptance là `max(0.1 token, 5% baseline)` và phải được
  khóa trước full run.

## 3. Experiment E0 — Instrumentation và parity gate

### Mục tiêu

Kiểm tra rằng intervention tác động đúng key/value/position trước khi chạy
benchmark.

### Thiết kế

Chạy một số sample với F, R, D và Z. Kiểm tra:

- condition Z giữ nguyên key nhưng value bằng zero;
- position IDs/M-RoPE của text tokens không bị shift ngoài chủ ý;
- attention rows có tổng bằng 1;
- target fingerprint và greedy output không đổi;
- acceptance trace lặp lại được;
- không có stale KV cache giữa các condition.

### Tiêu chí pass

E0 pass khi toàn bộ parity checks đạt trên ít nhất 3 sample. Nếu fail, dừng các
experiment sau và sửa instrumentation trước.

### Dữ liệu

`RUNNABLE_NOW` về mặt dataset và checkpoint, nhưng còn phụ thuộc environment/model
cache của runtime.

## 4. Experiment E1 — H1a: text over-attention và representation shift

### Câu hỏi

Xoá visual slots có làm draft text branch được khuếch đại, khiến hidden states và
logits lệch khỏi phân phối huấn luyện không?

### Condition

F, R, D; thêm Z nếu null-slot intervention đã pass E0.

### Quan sát chính

\[
D \rightarrow M_T\uparrow,\|o_T\|\uparrow
\rightarrow D_{hidden}\uparrow
\rightarrow KL_{draft\text{-}target}\uparrow
\rightarrow \tau_{eff}\downarrow.
\]

### Hidden protocol

Thu draft hidden states ở:

- last instruction position;
- token đầu tiên của answer;
- 4–16 answer positions đầu;
- một số answer positions giữa câu.

Trên training data, tính mean, variance và covariance theo layer và token role.
Trên MVBench, tính Mahalanobis distance tới training distribution và paired
distance tới F condition.

### Metric

Primary: paired hidden-distribution shift giữa D và F.

Guardrails: (M_T), \(\|o_T\|\), draft-target KL, acceptance và target exact
parity.

### Decision rule

Ủng hộ H1a nếu D có hidden shift lớn hơn F với 95% CI không chứa zero và đồng thời
có text contribution/logit mismatch hoặc acceptance giảm.

Nếu chỉ (M_T) tăng nhưng hidden và logits không đổi, chỉ kết luận rằng text
nhận nhiều global attention mass hơn; chưa kết luận được over-attention gây hại.

### Data dependency

**Bị block một phần bởi training data.** Checkout hiện có:

- một file ShareGPT khoảng 10.5 KB, 55 dòng vật lý;
- không có full 68K ShareGPT manifest/records theo preset training;
- không có multimodal LLaVA 68K source và images;
- không có teacher-cache đầy đủ dưới `artifacts/`.

Vì vậy:

- có thể chạy paired MVBench diagnostic D-vs-F nếu checkpoint và runtime sẵn sàng;
- không được gọi là so sánh đầy đủ với training distribution nếu chỉ dùng sample
  ShareGPT hiện có;
- training-distribution claim cần full stage-1/stage-2 data hoặc một cached
  hidden-distribution artifact có provenance rõ ràng.

## 5. Experiment E2 — H1b: visual attention dilution

### Câu hỏi

Visual slots có chiếm attention mass nhưng không đóng góp visual information hữu
ích, từ đó làm giảm text contribution và acceptance không?

### Phần A: null-slot sweep

Giữ visual slots, đặt value bằng zero, thay đổi tỷ lệ null slots:

\[
z\in\{0,1\%,5\%,10\%,25\%,50\%,100\%\}.
\]

Chỉ thay đổi (z); key, position và text context giữ nguyên.

### Phần B: content-control

So sánh F, Z và D để phân biệt:

- visual content thật;
- visual slots không có content;
- không có visual slots.

### Metric

Primary: \(\tau_{eff}(z)\), với kiểm định inverted-U. Fit một quadratic curve
và yêu cầu đỉnh nằm trong khoảng interior, không ở endpoint.

Guardrails:

- (M_V), (M_T);
- \(\|o_T\|\), \(\|o_V\|\);
- visual-zero logit effect;
- draft-target KL;
- latency và target exact parity.

### Kiểm tra visual attention hữu ích hay lãng phí

Tính:

\[
\Delta_{visual}=KL(p_{full}\|p_{visual-zero}).
\]

Visual mass cao nhưng \(\Delta_{visual}\) thấp là mẫu hình phù hợp với dilution.
Visual mass cao và \(\Delta_{visual}\) cao cho thấy visual information vẫn hữu
ích, không nên gọi trực tiếp là dilution.

### Decision rule

Ủng hộ H1b nếu:

- acceptance cao nhất ở vùng (z) trung gian;
- full/null condition làm giảm text contribution;
- visual value ablation không làm logits thay đổi đáng kể hoặc làm mất rất ít
  thông tin dự đoán;
- effect lặp lại trên nhiều sample/task type.

### Data dependency

`RUNNABLE_NOW` về dataset: phần MSD đã chạy được trên VDC-50; MVBench vẫn có
1,000 records trong `dataset/MVBench/classified/selected.jsonl` nhưng nhánh
DFlash Qwen2.5-VL chưa chạy được vì thiếu base-model snapshot. Không cần
training data cho acceptance/attention sweep. Z và value-path hook đã được
implement và pass unit test; vẫn cần full-cohort execution.

## 6. Experiment E3 — H2.1: VDC cần visual information

### Câu hỏi

VDC không cải thiện khi giảm visual tokens vì text prompt tự nhiên chưa đủ chứa
visual semantics cần thiết hay không?

### Thiết kế factorial

| Prompt | F | R | D |
|---|---:|---:|---:|
| Natural VDC | ✓ | ✓ | ✓ |
| Answer-augmented VDC | ✓ | ✓ | ✓ |

Answer-augmented prompt sử dụng reference answer ở các mức 25%, 50% và 100%.
Đây là leakage diagnostic, không phải benchmark result. Prompt length và vị trí
answer hint phải được kiểm soát để tránh nhầm hiệu ứng sequence length với hiệu
ứng semantic sufficiency.

### Metric

Primary interaction:

\[
I_{answer}=
[\tau_{eff}(D)-\tau_{eff}(F)]_{augmented}
-
[\tau_{eff}(D)-\tau_{eff}(F)]_{natural}.
\]

Guardrails:

- natural VDC phải đạt equivalence với claim “không tốt hơn”;
- target output exact parity trong từng prompt type;
- VDC answer quality;
- attention mass và draft-target KL.

### Decision rule

Ủng hộ H2.1 nếu:

- trong natural VDC, D/R không cải thiện F vượt qua equivalence margin;
- trong answer-augmented VDC, D/R cải thiện acceptance;
- interaction (I_{answer}) dương và vượt margin.

Nếu natural VDC cũng cải thiện, H2.1 không được ủng hộ. Nếu cả hai prompt type
đều không cải thiện, cần xem xét lại denominator mechanism hoặc positional/KV
confound.

### Data dependency

`RUNNABLE_NOW` về evaluation data: VDC-50 hiện có 50 video, `test.jsonl`,
`subset_manifest.jsonl` và calibration files. Reference answers nằm trong test
data nên có thể dựng answer-augmented diagnostic. Không cần full training data cho
E3.

## 7. Experiment E4 — H3.1: acceptance theo vị trí answer

### Câu hỏi

Việc cắt visual tokens có làm acceptance thấp ở đầu câu trả lời nhưng cao hơn ở
phần sau không?

### Logging protocol

Mỗi acceptance round phải ghi:

- `round_id`;
- `output_start_position`;
- `proposal_count`;
- `matched_proposals`;
- `effective_emitted_tokens`;
- `first_reject_position`;
- proposal và target token IDs.

Nếu round (t) chấp nhận (k_t) proposals:

\[
s_t=\sum_{u<t}(k_u+1)
\]

là vị trí bắt đầu round. Proposal thứ (r) được gán vào answer position
(s_t+r).

### Metric

Primary position interaction:

\[
I_{position}=\Delta_{late}-\Delta_{early},
\]

trong đó \(\Delta\) là reduced-minus-full acceptance rate.

Phân tích ở cả absolute positions và normalized bins: 0–10%, 10–25%, 25–50%,
50–75%, 75–100%.

### Decision rule

Ủng hộ H3.1 nếu:

\[
\Delta_{early}<0,
\qquad
\Delta_{late}>0,
\qquad
I_{position}>\epsilon.
\]

Nên đo thêm target visual sensitivity theo position bằng KL giữa full-visual và
text-only target logits. Nếu sensitivity cao ở đầu và giảm về cuối, cơ chế của
H3.1 được hỗ trợ mạnh hơn.

### Data dependency

`RUNNABLE_DIAGNOSTIC`: VDC-50 và MSD runtime hiện đủ để chạy acceptance study.
Runtime đã bổ sung `acceptance_by_position`, nhưng log hiện ghi proposal
positions/fallback positions chứ chưa lưu đầy đủ proposal/target token IDs như
protocol tối đa ở trên; cần coi token-level correctness là phần mở rộng nếu cần.

Không cần training data mới nếu chỉ đánh giá MSD hiện tại.

## 8. Experiment E5 — H3.2: draft depth và visual understanding

### Câu hỏi

DFlash nhiều draft layers hơn MSD có thực sự khai thác visual semantics tốt hơn,
hay chỉ cải thiện language modeling nói chung?

### Training conditions

Train riêng:

\[
L\in\{1,3,5\}.
\]

Mỗi model đi qua hai phase:

1. Stage 1 text-only;
2. Stage 2 multimodal.

Giữ cố định:

- target model;
- số target features;
- selected target layer IDs;
- block/anchor configuration;
- data order và số optimizer steps;
- learning-rate schedule;
- batch size và seed policy.

Không được thay đổi đồng thời `num_draft_layers` và `num_target_features`.

### Evaluation

Mỗi DFlash depth chạy F, R và D/Z. MSD được giữ làm external baseline để so sánh
mô tả; nguyên nhân nhân quả về depth chỉ được kết luận từ DFlash 1/3/5.

### Metric

Primary visual-depth interaction:

\[
G_{visual}(L)=
\tau_{eff}(L,full)-\tau_{eff}(L,zero),
\]

\[
I_{depth}=G_{visual}(5)-G_{visual}(1).
\]

Guardrails:

- acceptance per draft latency;
- parameters và memory;
- draft-target KL;
- position-wise acceptance;
- training loss và checkpoint reload parity.

### Decision rule

Ủng hộ H3.2 nếu depth tăng làm visual interaction tăng: model nhiều layers có lợi
thế rõ hơn trong full visual so với zero visual. Nếu depth cải thiện đồng đều cả
full và zero visual, đó là language-modeling gain, chưa phải visual-understanding
gain.

### Data dependency

**BLOCKED cho full retraining.** Hiện có các checkpoint DFlash 5-layer:

- `dataset/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt`;
- `dataset/qwen25vl-3b-dflash-20e-sharegpt68k-latest/training_state.pt`;
- các checkpoint 6e tương ứng.

Nhưng checkout không có:

- full ShareGPT 68K source tương ứng;
- full LLaVA multimodal 68K source và images;
- full prepared manifests;
- teacher-cache của hai phase;
- checkpoint 1-layer và 3-layer đã train.

Do đó hiện chỉ có thể:

- đánh giá exploratory checkpoint 5-layer;
- chạy architecture/config smoke với sample nhỏ;
- so sánh mô tả với MSD nếu các model/runtime sẵn sàng.

Chưa thể trả lời đầy đủ H3.2 cho đến khi khôi phục training data hoặc cung cấp
teacher-cache/manifest có provenance.

## 9. Data inventory và trạng thái hiện tại

| Tài nguyên | Trạng thái | Sử dụng |
|---|---|---|
| MVBench selected manifest | Có, 1,000 records | E0, E1, E2 |
| MVBench videos | Có, khoảng 1,272 files | E0, E1, E2 |
| VDC-50 videos | Có, 50 files | E3, E4 |
| VDC test/reference answers | Có | E3 answer diagnostic |
| VDC calibration | Có | E3, E4 |
| DFlash 5-layer checkpoints | Có | E1–E3 exploratory/evaluation |
| Full ShareGPT 68K training records | Không có | E1 reference, E5 retraining |
| Full LLaVA 68K multimodal data/images | Không có | E1 reference, E5 retraining |
| Prepared training manifests | Không có trong checkout | E1, E5 |
| Teacher hidden-state cache | Không có trong checkout | E1, E5 |
| DFlash 1/3-layer checkpoints | Không có | E5 |
| Position-wise MSD acceptance trace | Có, đã pilot | E4 |
| Key-preserved/value-zero hook | Có, unit-tested và đã pilot | E0/E2 |

## 10. Thứ tự thực hiện và promotion gates

### R0 — Protocol freeze

Khóa condition, primary metric, equivalence margin, seed, cohort và decision rule.

### R1 — Static/config sanity

Kiểm tra paths, model/checkpoint, manifest, output schema và intervention config.

### R2 — One-sample parity

Chạy E0. Không chuyển tiếp nếu target output hoặc position/KV parity fail.

### R3 — Small pilot

Chạy E1–E4 trên 5–20 samples. Kiểm tra signal, variance và độ đầy đủ của logs.

### R4 — Cohort pilot

Chạy khoảng 50 samples MVBench/VDC. Chỉ dùng để xác nhận xu hướng và ước lượng
margin; không claim hypothesis đã được xác nhận.

### R5 — Full inference study

Chạy full cohort cho E1–E4 khi audit pass và không còn silent failure.

### R6 — Training depth study

Chỉ bắt đầu E5 sau khi full stage-1/stage-2 data hoặc teacher-cache được cung cấp.
Train L=1,3,5 với cùng protocol, sau đó mới so sánh với MSD.

## 11. Kết luận về khả năng thực hiện hiện tại

- E0: one-sample smoke đã đạt native-prefill parity và losslessness; gate đầy đủ
  theo protocol (ít nhất 3 samples và mọi condition) chưa được gọi là pass.
- E1: chạy được paired MVBench diagnostic; training-distribution claim bị block
  bởi thiếu full training data hoặc cached hidden statistics.
- E2: chạy được trên MSD/VDC pilot với F/R/D/Z; full-cohort và logit-KL vẫn cần chạy.
- E3: chạy được trên VDC-50; answer augmentation chỉ là diagnostic leakage.
- E4: đã chạy pilot sau khi bổ sung position-wise acceptance logging; cần cohort đủ lớn.
- E5: full experiment bị block bởi thiếu full training data, teacher-cache,
  manifests và DFlash 1/3-layer checkpoints.

Không được dùng các checkpoint 5-layer hiện có để suy ra rằng DFlash nhiều layers
hơn MSD là nguyên nhân của acceptance gain. Đó chỉ là một hypothesis cần E5 kiểm
định.
