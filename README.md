# Báo Cáo Tóm Tắt Đồ Án FontDiffusion

## Giới Thiệu
FontDiffusion dùng để tạo 1 ảnh mới từ 2 ảnh đầu vào, gọi là ảnh 1 và ảnh 2, sao cho ảnh được tạo ra có content (chữ) của ảnh 1 nhưng với style viết/vẽ của ảnh 2. Ảnh 1 gọi là ảnh content, ảnh 2 gọi là ảnh reference. Cốt lõi cần trích xuất content features từ ảnh 1 và style features từ ảnh 2 và merge chúng lại để tạo ra ảnh mới.

Dưới đây là mô tả chi tiết về các thành phần cốt lõi của hệ thống: inference_pipeline, kiến trúc mã nguồn trong `src/`, cũng như các entrypoint chính cho training và testing.

## Chi tiết các module

### 1. `inference/`
Thư mục chứa toàn bộ quy trình sinh ảnh (inference) sử dụng pre-trained diffusion.
- Nơi đây có chứa nhiều mode sampling khác nhau: `sample_optimized.py` (tối ưu hóa tốc độ trên 1 GPU), `sample_batch.py` (xử lý lượng lớn ảnh/batch), và `sample_distributed.py` (hỗ trợ phân tán tải ra nhiều GPU dùng Accelerate, tối đa throughput khi tạo bộ dataset).
- Các script trong thư mục này chịu trách nhiệm khởi tạo class `DPM_Solver`, truyền inputs qua UNet để giảm nhiễu từng bước, decode bằng VAE và lưu thẳng xuống đĩa cứng hoặc JSON file meta (`results_checkpoint.json`).
> Chỉ cần dùng sample_distributed là đủ (các file là legacy code)

### 2. Các subfolders trong thư mục hệ thống `src/`
Đây là mã nguồn lưu trữ toàn bộ các class về kiến trúc mô hình, dữ liệu, loss cũng như các tool utils phục vụ training và validation:
- **`src/ablation/`**: Các thí nghiệm, phân tích ablation study để bóc tách tính năng đánh giá sự ảnh hưởng của từng thành phần lên cả bộ model.
- **`src/builders/`**: Factory functions đảm nhiệm việc khởi tạo linh hoạt các model block, bộ loss function, optimizers/schedulers và metric đánh giá dựa trên `configs/`. Các module được build bởi các function trong folder này trước kh truyền chúng vô các class để khởi tạo.
- **`src/configs/`**: Phụ trách định nghĩa Hyperparameters và các argument parser (như trong `fontdiffuser.py`). File cấu hình là nơi mọi scripts sẽ gọi để duy trì single truth of source cho config.
- **`src/dataset/`**: Các object liên quan đến xử lý dữ liệu đầu vào. Gồm Pytorch Dataset classes (FontDataset) cũng như các `CollateFN` có nhiệm vụ pack batch dữ liệu (images + captions) để chuẩn bị cho dataloaders. Đảm bảo hỗ trợ load multi-scale hay condition linh hoạt.
- **`src/dpm_solver/`**: Chứa thuật toán sample (Diffusion Probabilistic Models Solver). Thường không chạm vào code trong folder này.
- **`src/losses/`**: Tổng hợp các hàm Loss tuỳ chỉnh, đặc thù cho bài toán chuyển phong và diffusion: hàm perceptual loss (LPIPS), content preservation (Offset loss), SC (Style Consistency loss) và Identity loss.
- **`src/modules/`**: Lưu trữ network architectures cốt lõi: UNet 2D cải tiến, Content Encoder, Style Encoder (Attention-based), Font Style Transformation modules. Có thêm 1 số cải tiến từ UniCalli và Matryoshka nhưng chưa integrate vào.
- **`src/tools/`**: Một bộ các scripts hỗ trợ I/O, download/upload HuggingFace, metadata conversion, tạo bộ chia splits (`create_hf_dataset.py`, `export_dataset_ultra.py`) và caching/hashing tên file ảnh output (`filename_utils.py`) để tránh duplicate generations.
- **`src/trainers/`**: Các class quản lý vòng lặp huấn luyện (Training Loops). Lớp Trainer này encapsulate quá trình forward-pass, tính loss tổng hợp, log kết quả lên Wandb, và tự động lưu checkpoints sau các epoch đã định (e.g. `trainer.py`, `trainer_fst.py`).

### 3. `run_inference.py`
Là point-of-entry chính giúp người dùng chạy dễ dàng các script trong `inference/`. Script này parse các target arguments quan trọng như `--characters`, `--style_images`, `--ckpt_dir` rồi mapping linh hoạt vào các mode (`sample_batch` hoặc `sample_distributed`).

### 4. `train_fst.py`
Là file orchestrator để Train cho module FST (Font Style Transfer). Script này chịu trách nhiệm: 
- Bootstrapping các thiết lập thư viện Accelerate.
- Khởi tạo Logging (W&B).
- Load params từ config `configs/fontdiffuser.py`.
- Build Dataset và Models từ các build functions. Sau đó truyền tham số khởi tạo đối tượng `FontDiffuserFSTTrainer` lưu ở `src/trainers/`.

### 5. `src/trainers/trainer_fst.py`
Một subclass của hệ thống Trainer tuỳ chỉnh chuyên xử lý kịch bản cho module FST. Chịu trách nhiệm cho quá trình Forward logic bao gồm: 
- Lên lịch trình Phase 1 và Phase 2 của bài toán (VD dùng auxiliary losses, perceptual losses, tính gradient clip). 
- Mapping các tensors embedding giữa Content và Style, tính độ chênh lệch feature sau các block CNN.
- Cập nhật trọng số của `unet`, `style_encoder` thông qua Optimizer.

### 6. `font_diffusion.ipynb`
**Orchestrator** để gọi train và inference, đã install các library version phù hợp ở các cell tuỳ  môi trường Kaggle hay Google Colab.
- Auto-setup môi trường: Chứa cell cài cắm library (diffusers, accelerate, torch), login HuggingFace, Wandb tự động bằng secret key / tokens.
- Workflow chạy đồng bộ: Dùng shell magics (`!accelerate launch`) thực hiện pull dataset từ Hub (`download_from_hf`), load các module checkpoint, push dữ liệu lại lên cloud.
- Post-processing: Thu gom, nén ảnh kết quả (`zip_folder`).

File notebook đã chạy được trên cả Colab và Kaggle nên chỉ cần nhập đúng các path (checkpoint path, output path, etc.) vào các cell là được.

### 6.5 Data
Đang dùng dataset với config là **paper** và split là **train**. Ngoài ra còn vài split khác như:
- test_seen_style_unseen_char: Style có trong train set nhưng content nằm ngoài train set.
- test_unseen_style_seen_char: Style nằm ngoài train set nhưng content có trong train set.
- test_unseen_style_unseen_char: Cả style và content đều nằm ngoài train set.

### 7. Cải tiến so với FontDiffuser
- Thêm module Font Style Transformations và Consistency Loss từ paper FSTDiff. Paper này không có code nên code này được vibe bằng Claude.
- Thêm Frequency Decomposition bằng biến đổi Fourier để chia ra 2 miền tần số cao (style) và tần số thấp (content), dùng để disentangle giữa content features và style features giữa ảnh content và ảnh reference để chống lại problem khi style của ảnh content bị leak vào ảnh kết quả, còn content của ảnh reference không leak vào được do đã được diffusion noise loss khống chế. Có thêm Multi Scale Style Encoder để áp dụng multi scale cho cả style features. Content đã được multi-scale (Multi Scale Content Aggregation thừa kế từ FontDiffuser) Bổ sung thêm 1 số loss liên quan đến Fourier.
- Dự định apply Matryoshka Representation Learning để dùng ý tưởng nested embeddings của nó để tăng sức represent cho model nhưng xem kỹ hơn về kỹ thuật này thì Matryoshka thường dùng để tối ưu inference-time cho các model lớn chứ usecase không phải để style transfer nên chưa khảo sát tác dụng của phương pháp này (còn bị 1 vài bug dimension).
- Paper UniCalli đề xuất framework phối hợp giữa generation và recognition và train đồng thời 2 module này với nhau dành cho chữ Trung. Tiềm năng để thừa kế vào đồ án này.

### 8. Các checkpoint quan trọng
Pipeline training được kế thừa từ FontDiffuser: gồm 2 phase
- Phase 1: train các unet, content encoder, style encoder chủ yếu để cho model học được embeddings.
- Phase 2: có thêm Style Contrastive Refinement (SCR) để làm giám sát (module này bị tắt gradient vì ko train mà chỉ để giám sát) để tăng độ khác nhau giữa các style bằng contrastive learning.

Các checkpoint ở thư mục gốc: pretrained model của FontDiffuser: chú thích FST là Font Style Transformation, FFT là Fast Fourier Transform
- FST: FST-paper-experiment/checkpoint_step_9000. Này đáng lẽ train tới 15k steps nhưng các step cuối bị mất do kaggle quá session mà quên train tiếp.
- FST + FFT: FFT-paper-experiment/phase2/checkpoint_step_15000.
- Các folder có tên DRO*, FFT-aux-loss, FFT-identity*, FST* là các checkpoint train giữa chừng đã failed, có thể chạy inference thử xem kết quả như nào.
- finetuned-5P1-5P2: train tiếp pretrained của FontDiffuser thêm 5000 steps cho mỗi phase, tương tự cho finetuned-5P1 (này chỉ train 5000 steps phase 1).
- pretrained-15P1: train from scratch 15k steps phase 1
- phase-2: train tiếp 15k steps cho phase 2 từ cái checkpoint ở pretrained-15P1.

### 9. Unsuccessful attempts
- Skeleton distance transform: đã thử mà không thành công khi ảnh generate ra bị lỗi chưa denoise được hết điểm ảnh nhiễu. Cần setup cái này cả trong training và inference như 1 module tiền xử lý cho tấm ảnh.
- Đổi dấu SSIM và LPIPS làm loss function: generate ra ảnh có style rất giống với ảnh reference, gần như là hoàn hảo, nhưng nhiều chữ có nét phức tạp bị mất nhiều nét quan trọng và gần như khác khá nhiều (40-50%) so với ảnh content ban đầu. Nói cách khác, model đã sacrifice nét chữ để có style rất giống với ảnh reference.
- Thêm 1 vài loss mới để giữ lại content features không bị mất mát như dùng SSIM và LPIPS nhưng ra kết quả tệ, mô hình không tạo ảnh bình thường được mà toàn đen => mode collapse.

### 9. Một số paper liên quan:
- [ ]  https://www.alphaxiv.org/abs/2510.13745
- [ ]  https://www.alphaxiv.org/abs/2602.18874
- [ ]  https://www.alphaxiv.org/abs/2509.16632
- [ ]  https://www.alphaxiv.org/abs/2404.06779
- [ ]  https://www.alphaxiv.org/abs/2501.08062
- [ ]  FSTDiff: paper chứa Font Style Transformation nằm trong folder docs vì này hông có trên arxiv (này anh Phát cho em)