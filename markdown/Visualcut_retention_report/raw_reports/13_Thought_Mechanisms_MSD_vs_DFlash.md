# Thought - 2/11/2026

> Ghi chú gốc và hình minh họa được giữ để theo dõi lập luận. Xem phần điều chỉnh 6/11 bên dưới để biết phạm vi kết luận và giả thuyết hiện hành; các quan sát từ nghiên cứu liên quan chưa phải thực nghiệm xác nhận cơ chế trên DFlash của workspace.

Hiện tại chúng ta có thể confirm được một số insight

- **các text hidden state của model ở các high layer là đủ để bảo toàn performance của model**
  - Thực nghiệm chỉ giữ lại visual pos của hidden tại các layer khác nhau
  - nếu loại bỏ các visual tok ở các layer cao đi thì performance của model gần như không đổi
  - Thực nghiệm nữa để kiểm chứng vấn đề này là ở các layer cao → attn của target model vào các vị trí visual gần như rất ít (nhiều nhất ở layer 18)

Tuy nhiện hiện tại mới có thể kết luận được

- Trong inference
- Đối với LLM model lớn
- Thực tế là trong quá trình huấn luyện thì sparrow vẫn sử dụng hidden state của middle layer (giàu attn vào visual nhất)→ kết quả tốt hơn huấn luyện bằng hidden text thuần . Nhưng chưa cho thấy được cải thiện lớn

| Training/inference input | ChartQA τ | AI2D τ |
| ------------------------ | ---------- | ------- |
| Visual + Text            | 4.25       | 4.28    |
| 0% Visual + Text         | 4.23       | 4.25    |
| 50% Visual + Text        | 4.21       | 4.24    |

![image.png](figures/image.png)

Sự khác biệt của MSD và Dflash

![image.png](<figures/image 1.png>)

![image.png](<figures/image 2.png>)

- Đầu vào:
  - MSD nhận vào visual embed thô + text token + high hidden target for text tok  → Đã chữa cả thông tin về visual rồi
  - Dflash nhận vào mixed hidden of low to high layer for visual + text token
- Số layer : 1 và 5
- Decoding style: AR và Block Diffusion

## Main problem: tại sao thực nghiệm visual cut , visual retention trên MSD cho kết quả cải thiện đáng kể nhưng trên Dflash thì không thấy được xu hướng đó

## Điều chỉnh giả thuyết — 6/11/2026

**Giả thuyết hiện hành:** thông tin video có thể đi vào draft qua text hidden states do target tạo ra, bên cạnh đường visual positions trực tiếp. Khi đường text đã cung cấp đủ thông tin cho proposal, cắt bớt visual positions chỉ làm mất phần thông tin bổ sung nhỏ; tác động còn phụ thuộc layer selection, fusion và checkpoint. Điều này có thể giải thích việc logits thay đổi mạnh nhưng acceptance ít đổi. Chưa có can thiệp tách riêng hai đường để xác nhận cơ chế.

Ba điểm cần diễn đạt chính xác:

- DFlash trong code hiện tại **ghép hidden states từ các layer target**, chiếu qua `fc`/normalization, rồi các attention layer của draft tạo K/V riêng. Cách gọi “nhận KV ở nhiều layer target” là cách nói tắt, dễ nhầm với tái sử dụng nguyên target KV. Xem `extract_context_feature`, `DFlashDraftModel.forward` và các phép `k_proj`/`v_proj` trong [implementation](../../src/train_Dflash_SpecForge/specforge/modeling/draft/dflash.py).
- Text token đứng sau video có quyền nhìn video và prefix đứng trước nó theo causal mask. Layer sâu hơn có thể tích hợp thông tin tốt hơn nhưng không mở quyền nhìn tương lai. “Có thông tin cả câu” chỉ phù hợp khi nói về phần prompt/prefix đã có; chưa thể áp dụng cho toàn bộ đáp án chưa sinh. Trong một block, các slot draft dùng chung verified target context; bidirectional anchor/mask attention không cung cấp các token đúng ở slot sau.
- Quan sát text states ở high layer đủ giữ performance trong cấu hình nghiên cứu liên quan là đầu mối cho giả thuyết. Attention visual thấp tự nó chưa chứng minh visual không có đóng góp. Kết quả inference đó cũng chưa suy ra rằng huấn luyện bỏ visual hay thay layer input sẽ tương đương.

**Điều dữ liệu DFlash hiện hỗ trợ:** Cut/retention chưa cho xu hướng nhất quán, hiệu ứng phụ thuộc budget/config; matching token ID × slot trên subset 24 không xác nhận tương tác lexical-overlap dương. Các kết quả này phù hợp với giả thuyết dư thừa thông tin, nhưng cũng có thể xuất hiện do khác biệt phân bố attention, OOD hoặc trade-off giữa lợi ích thông tin và chi phí context. H3.1 early/late chưa ủng hộ quy tắc đơn giản “token càng muộn càng ít cần visual”.

**Câu hỏi cần thay thế cho so sánh MSD–DFlash:** khi giữ token IDs, target trajectory và visual rows cố định, việc giảm điều kiện video trong text states có làm lợi ích giữ visual positions tăng không? Khác biệt MSD–DFlash còn đồng thời gồm nguồn visual feature, kiến trúc, decode và training; chưa thể quy nguyên nhân cho multi-layer fusion.

**Thực nghiệm kế tiếp đã thiết kế:** pilot 12 video, 3K, cùng checkpoint; crossed visual Full/Cut × text states gốc/giảm điều kiện video, thêm control nhiễu text cùng độ lớn. Metric chính là mức tăng của `(Full−Cut)` trên `τ_proposal` khi đường text bị can thiệp. Cần control OOD, giữ nguyên causal prefix và dựng lại KV từng điều kiện. Xem [protocol chi tiết](../DFlash_TextState_Intervention_20261106/protocol.md) và [sổ nghiên cứu](../proposed_exp.md); chưa triển khai runner hoặc chạy pilot can thiệp.
