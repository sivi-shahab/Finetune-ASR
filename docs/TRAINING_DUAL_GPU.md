# Training Whisper di 2× NVIDIA L40S

Versi: 2026-10-28 · Terkait: [SPEC.md](SPEC.md), [CONTEXT_ENGINEERING.md](CONTEXT_ENGINEERING.md)

Training memakai **DDP (Distributed Data Parallel)** dengan 2 proses, satu per GPU. Masing-masing GPU memegang salinan penuh model, memproses separuh batch, lalu gradien disinkronkan antar GPU. Whisper large-v3 (±1,55 miliar parameter) cukup untuk full fine-tuning di 48 GB per GPU dengan bf16 dan gradient checkpointing, jadi FSDP atau DeepSpeed belum diperlukan.

---

## 1. Ringkasan konfigurasi

| Item | Nilai | Alasan |
| --- | --- | --- |
| Strategi | DDP, `torchrun --nproc_per_node=2` | Model muat di satu GPU; DDP paling sederhana dan stabil |
| Base model | `openai/whisper-large-v3` | `large-v3-turbo` hanya punya 4 layer decoder, jadi kemampuannya memakai konteks perlu diuji terpisah |
| Presisi | bf16 + TF32 | L40S (Ada) mendukung bf16 |
| Batch per GPU | 8 | Titik awal; naikkan jika memori cukup |
| Gradient accumulation | 2 | Batch efektif = 8 × 2 × 2 GPU = **32** |
| Gradient checkpointing | Ya, `use_reentrant=False` | Hemat memori aktivasi; wajib non-reentrant untuk DDP |
| Optimizer | AdamW (torch fused) | Fallback: AdamW 8-bit jika OOM |
| LR / scheduler | 1e-5, linear, warmup 500 step | Turunkan ke 5e-6 jika WER validation naik |
| Epoch | 3 (rentang 2–5) | Checkpoint dipilih berdasarkan metrik entitas |
| Dataloader | 6 worker per GPU | 24 core CPU, sisakan untuk proses utama |

## 2. Environment

Python bawaan server adalah 3.13, sedangkan sebagian library audio/ASR belum stabil di 3.13. Gunakan **dua environment terpisah**:

| Env | Python | Isi | Alasan |
| --- | --- | --- | --- |
| `whisper-train` | 3.11 | torch, transformers, datasets, accelerate, soundfile, jiwer, tensorboard | Training |
| `whisperx-infer` | 3.11 | whisperx (versi dipin), faster-whisper, ctranslate2 | WhisperX mem-pin versi torch dan CUDA sendiri |

Partisi `/home` hanya tersisa ±9 GB, jadi env dibuat dengan `conda create -p` di `.envs/` proyek ini, dan cache pip/HF disimpan di `.cache/`:

```bash
bash scripts/setup_envs.sh train   # .envs/whisper-train  ← requirements-train.txt (sudah terpasang)
bash scripts/setup_envs.sh infer   # .envs/whisperx-infer ← requirements-infer.txt (belum)
```

Jalankan keduanya satu per satu. Kalau dijalankan paralel, conda berebut cache paket yang sama dan instalasi bisa gagal.

Driver 580 mendukung build CUDA 12.x maupun 13.x. Sebelum memilih wheel, cek kompatibilitas ctranslate2 dengan versi CUDA/cuDNN yang terpasang. Catat versi final semua paket di `requirements-train.txt` dan `requirements-infer.txt`.

Cek cepat bahwa kedua GPU terlihat dan NCCL berjalan:

```bash
.envs/whisper-train/bin/torchrun --nproc_per_node=2 src/nccl_check.py
# hasil 2026-10-28: rank 0 dan 1 (NVIDIA L40S) → 3.0, lolos tanpa NCCL_P2P_DISABLE
```

## 3. Memori per GPU (estimasi)

Whisper selalu mem-pad audio ke 30 detik (3.000 frame mel, 128 bin untuk large-v3). Karena itu, memori encoder per sampel **konstan** berapa pun durasi clip, dan pemakaian memori mudah diprediksi.

| Komponen | Ukuran (perkiraan) |
| --- | --- |
| Bobot fp32 (master) | ±6,2 GB |
| Gradien fp32 | ±6,2 GB |
| State AdamW (2 × fp32) | ±12,4 GB |
| Subtotal statis | **±25 GB** |
| Aktivasi (batch 8, checkpointing, ≤ 448 token decoder) | Diukur saat smoke test |
| Kapasitas L40S | 48 GB |

Angka di atas adalah perkiraan kasar. Ukur pemakaian sebenarnya pada smoke test dengan batch terburuk (label 448 token).

Jika terjadi OOM, coba langkah berikut berurutan:

1. Turunkan batch per GPU menjadi 4 dan naikkan accumulation menjadi 4 (batch efektif tetap 32).
2. Gunakan `optim="adamw_bnb_8bit"`: state optimizer turun ke ±3 GB.
3. Gunakan DeepSpeed ZeRO-2 untuk membagi state optimizer dan gradien ke 2 GPU.

## 4. Interkoneksi PCIe (tanpa NVLink)

`nvidia-smi topo -m` menunjukkan `PHB`: kedua GPU berkomunikasi lewat PCIe host bridge. Setiap optimizer step, DDP melakukan all-reduce gradien fp32 (±6,2 GB). Konsekuensinya:

- Gradient accumulation mengurangi frekuensi sinkronisasi. Trainer otomatis memakai `no_sync()` pada langkah akumulasi, jadi all-reduce hanya terjadi sekali per optimizer step.
- Jika GPU util turun setiap step karena menunggu komunikasi, naikkan `gradient_accumulation_steps` menjadi 4. Batch efektif ikut naik menjadi 64, jadi jumlah step per epoch berkurang separuh; sesuaikan `warmup_steps` dan LR bila perlu.
- Jika NCCL hang saat inisialisasi, set `NCCL_P2P_DISABLE=1` (komunikasi lewat memori host) dan `NCCL_DEBUG=INFO` untuk diagnosis.

## 5. Data pipeline

- Manifest `dataset/train.jsonl` dibaca dengan `datasets.load_dataset("json", ...)`. Audio **tidak** dimuat ke RAM sekaligus (RAM 62 GB).
- Collator (`src/train.py`) per sampel:
    1. Baca FLAC dengan `soundfile` → `WhisperFeatureExtractor` (128 mel).
    2. Pilih variasi prompt (lihat CONTEXT_ENGINEERING.md bagian 5.2) → `context_builder.build_prompt(...)`.
    3. Susun `decoder_input_ids` + `labels` dengan masking `-100` untuk prompt dan prefix.
    4. Padding kanan: `decoder_input_ids` di-pad dengan `<|endoftext|>`, `labels` dengan `-100`.
- `remove_unused_columns=False` **wajib**. Tanpa ini, Trainer membuang kolom manifest (`prev_text`, `crm`, `text_raw`) sebelum sampai ke collator.
- Oversampling segmen PII dilakukan dengan menduplikasi baris di dataset train (bukan sampler khusus), agar tetap kompatibel dengan `DistributedSampler`.
- SpecAugment opsional: `model.config.apply_spec_augment=True`, `mask_time_prob=0.05`. Aktifkan jika validation loss mulai naik sementara train loss terus turun.

## 6. Kode inti

```python
# src/train.py (sketsa bagian konfigurasi)
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3",
                                             language="indonesian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model.config.forced_decoder_ids = None
model.config.use_cache = False                 # wajib saat gradient checkpointing
model.generation_config.language = "id"
model.generation_config.task = "transcribe"

args = Seq2SeqTrainingArguments(
    output_dir="outputs/run-001",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=500,
    num_train_epochs=3,
    bf16=True,
    tf32=True,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    dataloader_num_workers=6,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    label_names=["labels"],
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=25,
    report_to="tensorboard",                   # lokal saja, data berisi PII
    seed=42,
)
```

Evaluasi dibagi dua:

| Tahap | Apa | Di mana |
| --- | --- | --- |
| Selama training | `eval_loss` (teacher forcing, dengan prompt sesuai variasi) | Trainer, setiap 500 step |
| Per checkpoint | WER + entity exact match, dengan prompt domain | `src/evaluate.py` di GPU yang sedang kosong, atau setelah training |
| Checkpoint final | Konteks penuh (loop berurutan) + ablation | `src/infer_context.py` + `src/evaluate.py` |

Prompt per sampel berbeda-beda, sedangkan `generate()` di Hugging Face hanya menerima satu prompt per batch. Karena itu, WER tidak dihitung di dalam Trainer.

## 7. Menjalankan

```bash
cd /data/finetune_whisper

# 1) Smoke test: 200 sampel, 50 step, satu GPU
bash scripts/run_train.sh smoke

# 2) Smoke test DDP: sama, dua GPU
bash scripts/run_train.sh smoke-ddp

# 3) Training penuh (di dalam tmux/screen agar tidak putus); argumen tambahan diteruskan ke train.py
bash scripts/run_train.sh full run-001
bash scripts/run_train.sh full run-002 --lr 5e-6 --pii_oversample 3

# Lanjut dari checkpoint terakhir
bash scripts/run_train.sh resume run-001
```

Smoke test lolos jika:

- Loss turun mendekati nol pada 200 sampel (model mampu overfit).
- `generate()` dengan prompt menghasilkan teks yang sama dengan label untuk beberapa sampel train.
- Puncak memori per GPU tercatat (`torch.cuda.max_memory_allocated()`) dan masih di bawah ±42 GB.
- Pada run DDP, waktu per step tercatat sebagai dasar estimasi durasi training penuh.

Estimasi jumlah step: `step per epoch = jumlah segmen train (setelah oversampling) / 32`. Contoh: 36.000 segmen → 1.125 step per epoch.

Pemantauan: `nvidia-smi dmon -s um` (utilisasi dan memori), `tensorboard --logdir outputs/ --bind_all` (hanya di jaringan internal).

## 8. Disk dan checkpoint

| Item | Ukuran (perkiraan) |
| --- | --- |
| Satu checkpoint (bobot fp32 + optimizer + scheduler) | ±19 GB |
| `save_total_limit=2` | ±38 GB |
| Model final hasil konversi CT2 (float16) | ±3 GB |
| Clip FLAC dataset | Bergantung jumlah jam; ±50–60 GB per 1.000 jam |

Sisa disk `/data` saat ini ±113 GB. Hitung ulang sebelum training penuh, dan pindahkan checkpoint lama ke penyimpanan internal lain jika perlu.

## 9. Setelah training: konversi untuk WhisperX

```bash
# simpan model + processor final (tanpa state optimizer)
python - <<'EOF'
from transformers import WhisperForConditionalGeneration, WhisperProcessor
m = WhisperForConditionalGeneration.from_pretrained("outputs/run-001/checkpoint-BEST")
p = WhisperProcessor.from_pretrained("outputs/run-001")
m.save_pretrained("outputs/whisper-telemarketing"); p.save_pretrained("outputs/whisper-telemarketing")
EOF

conda activate whisperx-infer
ct2-transformers-converter --model outputs/whisper-telemarketing \
  --output_dir outputs/whisper-telemarketing-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
```

Inferensi memakai 2 GPU dengan satu worker per GPU yang mengambil panggilan dari antrean bersama (lihat CONTEXT_ENGINEERING.md bagian 6).

## 10. Opsi bila data sedikit

| Data train (jam) | Rekomendasi awal |
| --- | --- |
| < 20 | Bekukan encoder (`model.model.encoder.requires_grad_(False)`), latih decoder saja; LR 1e-5 |
| 20–100 | Full fine-tune, 3–5 epoch, SpecAugment aktif |
| > 100 | Full fine-tune, 2–3 epoch |

Ambang di tabel ini adalah titik awal, bukan aturan pasti. LoRA tidak direkomendasikan sebagai pilihan pertama: adaptasi akustik ke audio telepon lebih efektif dengan melatih bobot penuh, dan adapter harus di-merge dulu sebelum konversi CT2.

## 11. Troubleshooting

| Gejala | Kemungkinan penyebab | Tindakan |
| --- | --- | --- |
| NCCL hang saat start | P2P lewat PCIe bermasalah | `NCCL_P2P_DISABLE=1`, `NCCL_DEBUG=INFO` |
| `Expected to mark a variable ready only once` | Checkpointing reentrant + DDP | `use_reentrant=False`, `ddp_find_unused_parameters=False` |
| OOM | Batch terlalu besar | Lihat bagian 3 |
| GPU util < 70% | Dataloader lambat | Tambah worker, cek I/O disk, atau pre-compute fitur mel |
| Loss NaN | LR terlalu tinggi | LR 5e-6, cek label kosong |
| Output mengabaikan prompt | `decoder_input_ids` tidak dikirim | Cek collator: kirim `decoder_input_ids` **dan** `labels` |
| Output menyalin prompt | Porsi variasi "tanpa prompt" kurang | Naikkan porsi tanpa prompt; cek penjaga di inferensi |
| `KeyError: prev_text` di collator | Kolom dibuang Trainer | `remove_unused_columns=False` |
