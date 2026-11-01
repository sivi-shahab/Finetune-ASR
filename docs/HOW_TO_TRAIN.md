# How To: Training Whisper dengan Dataset QC Dashboard + Audio

Versi: 2026-11-01 · Terkait: [SPEC.md](SPEC.md), [CONTEXT_ENGINEERING.md](CONTEXT_ENGINEERING.md), [TRAINING_DUAL_GPU.md](TRAINING_DUAL_GPU.md)

Panduan langkah demi langkah dari audio mentah sampai model terpakai di WhisperX. Semua perintah dijalankan dari `/data/finetune_whisper`. Langkah yang membutuhkan audio, review manusia, atau unduhan diberi tanda ⚠️.

| # | Langkah | Perintah utama | Hasil |
| --- | --- | --- | --- |
| 0 | Prasyarat | `bash scripts/setup_envs.sh infer` | Env WhisperX + model Whisper ter-cache |
| 1 | Masukkan audio ⚠️ | `bash scripts/qc_dataset.sh import-audio <folder>` | `raw/audio/<ticket_id>.flac` |
| 2 | Bangun dataset | `bash scripts/qc_dataset.sh build` | `dataset/*.jsonl`, clip 16 kHz |
| 3 | Review manusia ⚠️ | `bash scripts/qc_dataset.sh review <nama> queue` / `spotcheck` | `dataset/reviews.jsonl` |
| 4 | Cek siap training | `bash scripts/qc_dataset.sh check` | `READY` |
| 5 | Smoke test | `bash scripts/run_train.sh smoke` lalu `smoke-ddp` | Bukti pipeline jalan + puncak memori |
| 6 | Training penuh | `bash scripts/run_train.sh full run-001` | `outputs/run-001/checkpoint-*` |
| 7 | Pilih checkpoint | `src/evaluate.py predict` + `score` | Checkpoint terbaik |
| 8 | Konversi + evaluasi konteks | `scripts/convert_ct2.sh`, `scripts/run_infer_2gpu.sh manifest` | Model CT2 + laporan test |
| 9 | Pakai di WhisperX | `scripts/run_infer_2gpu.sh audio` | Transkrip + speaker |

---

## 0. Prasyarat (sekali)

| Item | Status 2026-11-01 | Cara |
| --- | --- | --- |
| Env training `.envs/whisper-train` | Terpasang | — |
| Model `openai/whisper-large-v3` (±3 GB) | Belum diunduh | Terunduh otomatis saat smoke test pertama ke `.cache/huggingface` |
| Env inferensi `.envs/whisperx-infer` | Belum | `bash scripts/setup_envs.sh infer`, lalu pin versi di `requirements-infer.txt` |
| Token Hugging Face untuk diarization pyannote | Belum | Terima syarat model pyannote di HF, lalu `export HF_TOKEN=...` sebelum langkah 9 |
| Persetujuan kepatuhan/DPO (data PII) | — | Wajib sebelum langkah 1 (SPEC bagian 6) |
| Disk | Sisa ±113 GB | Raw FLAC ±12 GB + clip ±16 GB + 2 checkpoint ±38 GB + model CT2 ±3 GB |

Selalu set cache ke partisi `/data` (partisi `/home` hanya ±9 GB). Skrip di `scripts/` sudah melakukannya otomatis:

```bash
export HF_HOME=/data/finetune_whisper/.cache/huggingface
```

## 1. Masukkan audio ⚠️

1. Pilih panggilan dari `raw/audio_list.csv` (kolom `ticket_id, file_path, campaign, qc_status, qc_tl_status, duration_sec`). Baris tanpa `file_path` tidak bisa dipakai.
2. Ambil file GSM-nya ke satu folder, misalnya `/data/tmp/audio_qc/`. Nama file boleh tetap nama asli dari server (`20260901135012-201122-xxxxxxxxxx.gsm`) atau diganti menjadi `<ticket_id>.gsm`.
3. Impor (konversi lokal dengan ffmpeg, tanpa akses jaringan):

```bash
bash scripts/qc_dataset.sh import-audio /data/tmp/audio_qc
# INFO {"found": 2068, "imported_or_existing": 2068, "failed": 0, "unknown_ticket": 0, "still_missing": 227}
```

- `failed` > 0: lihat `raw/import_failed.jsonl` (biasanya file rusak atau kosong).
- `unknown_ticket`: file yang namanya tidak cocok dengan `audio_list.csv`.
- `still_missing`: tiket yang belum punya audio. Tiket ini dilewati di langkah 2 (`missing_pair`).
- Folder sumber boleh dihapus setelah impor. `raw/audio/` sudah menjadi salinan lossless.

Untuk mengecek jumlahnya: `ls raw/audio/*.flac | wc -l`.

## 2. Bangun dataset

```bash
bash scripts/qc_dataset.sh build        # --pii-policy confidence (default)
```

Yang terjadi (SPEC bagian 4 dan 7):
- Audio di-resample ke 16 kHz dan dipotong menjadi segmen 1–30 detik.
- Teks dinormalisasi ke format akhir (`Rp15.000.000`, `17-08-1985`).
- PII dideteksi, lalu QC dijalankan dan data dibagi per agent.
- Hasilnya ditulis ke `dataset/`.

Untuk ±2.000 panggilan, prosesnya hitungan menit sampai puluhan menit, tergantung disk.

Cek `dataset/stats.json`. Angka acuan dari dry-run 2026-11-01:

| Kunci | Acuan |
| --- | --- |
| `splits.train.hours` | ±265 jam |
| `splits.train.pii_segment_share` | ±0,23 |
| `review_queue_by_split` | test ±1.100, validation ±400, train ±12.400 (opsional) |
| `rejected` | `pii_unreviewed`, `cps_outlier`, `bad_timing`, `eval_unreviewed`, `duration`, `secret`, ... |

Jika angkanya jauh berbeda dari acuan, periksa dulu sebelum lanjut, misalnya `duration_mismatch` yang besar bisa berarti audio salah pasangan.

## 3. Review manusia ⚠️

Jalankan di server, lalu buka dari laptop lewat SSH tunnel:

```bash
bash scripts/qc_dataset.sh review andi queue          # di server
ssh -L 8765:127.0.0.1:8765 <user>@<server>            # di laptop → http://127.0.0.1:8765
```

| Tahap | Mode | Target | Wajib |
| --- | --- | --- | --- |
| a | `queue` | Semua segmen **test** lalu **validation** (muncul paling dulu) | Ya |
| b | `spotcheck` | Minimal 100 segmen train acak, termasuk `pii_auto` | Ya (untuk audit) |
| c | `queue` | Segmen PII train berkeyakinan rendah | Opsional: menambah data PII |

Aturan untuk reviewer:
- Dengarkan setiap digit.
- Tulis dalam format akhir (SPEC 5.2).
- Gunakan **Buang segmen** untuk audio rusak, dan **Rahasia** untuk PIN/OTP/CVV yang nilainya disebut.

Progres dan persentase koreksi bisa dilihat di `http://127.0.0.1:8765/stats`.

Setelah sesi review, terapkan hasilnya:

```bash
bash scripts/qc_dataset.sh build
```

Jika spotcheck menunjukkan koreksi > 10%, label otomatis kurang bagus. Ketatkan ambang (`build --pii-min-logprob -0.05`) atau review lebih banyak.

## 4. Cek siap training

```bash
bash scripts/qc_dataset.sh check
# READY
#   train         5xxxx segmen   26x.x jam  PII  1xxxx  direview   xxxx
#   validation      xxx segmen     x.x jam  ...
#   test           xxxx segmen     x.x jam  ...
#   WARN  ...
```

`NOT READY` disertai daftar `ERROR`, misalnya validation/test belum direview, audio hilang, panggilan bocor antar split, atau ada nilai rahasia. Perbaiki lalu ulangi langkah 2 sampai 4. Ringkasan lengkapnya ada di `dataset/READY.json`.

## 5. Smoke test

```bash
bash scripts/run_train.sh smoke         # 1 GPU, 200 sampel, 50 step (unduh model ±3 GB pertama kali)
bash scripts/run_train.sh smoke-ddp     # 2 GPU, sama
bash scripts/run_tests.sh               # 4 test yang tadinya dilewati sekarang ikut jalan
```

Lolos jika:
- `train_loss` turun di log (`logs/smoke.log`).
- Tidak ada OOM.
- Baris `puncak memori GPU` di akhir log < ±42 GiB per GPU.
- Kedua rank DDP selesai.

Catat detik per step dari `smoke-ddp` untuk memperkirakan durasi training penuh.

Jika OOM: tambahkan `--batch_size 4 --grad_accum 4` (batch efektif tetap 32) atau `--optim adamw_bnb_8bit`. Lihat TRAINING_DUAL_GPU.md bagian 3.

## 6. Training penuh

Jalankan di `tmux` agar tidak putus saat SSH terputus:

```bash
tmux new -s train
bash scripts/run_train.sh full run-001
# Ctrl-b d untuk keluar; tmux attach -t train untuk kembali
```

Perkiraan jumlah step:
- ±56 ribu segmen train, ditambah oversampling PII ×2 (±13 ribu salinan), menjadi ±69 ribu sampel per epoch.
- Dengan batch efektif 32, itu **±2.200 step per epoch**, atau ±6.500 step untuk 3 epoch.
- Kalikan dengan detik per step dari smoke test untuk mendapat estimasi waktu.

| Pantau | Cara |
| --- | --- |
| Loss & eval_loss | `.envs/whisper-train/bin/tensorboard --logdir outputs/run-001 --bind_all` (jaringan internal saja) |
| GPU | `nvidia-smi dmon -s um` (utilisasi < 70% → dataloader lambat) |
| Log | `tail -f logs/run-001.log` |

Melanjutkan setelah terhenti: `bash scripts/run_train.sh resume run-001`.

Opsi yang bisa dicoba di run berikutnya (argumen diteruskan ke `src/train.py`):

```bash
bash scripts/run_train.sh full run-002 --lr 5e-6                 # validation memburuk
bash scripts/run_train.sh full run-003 --pii_oversample 3        # akurasi entitas masih rendah
bash scripts/run_train.sh full run-004 --spec_augment            # overfitting
bash scripts/run_train.sh full run-005 --epochs 5
```

## 7. Pilih checkpoint

Trainer menyimpan checkpoint setiap 500 step, dengan maksimal 2 checkpoint (`save_total_limit`) agar disk aman. Hapus `--save_total_limit` hanya jika disk cukup. Untuk setiap checkpoint, jalankan prediksi pada validation dengan prompt domain:

```bash
PY=.envs/whisper-train/bin/python
for ck in outputs/run-001/checkpoint-*; do
  n=$(basename $ck)
  $PY src/evaluate.py predict --model $ck --manifest dataset/validation.jsonl \
      --prompt domain --out outputs/preds/val-$n.jsonl
  $PY src/evaluate.py score --manifest dataset/validation.jsonl \
      --pred outputs/preds/val-$n.jsonl --out outputs/reports/val-$n.json
done
```

Pilih checkpoint dengan **entity exact match `_all`** tertinggi, dengan syarat WER-nya tidak lebih buruk. Setelah training selesai, model akhir ada di `outputs/run-001/final/`. Model ini bisa saja bukan yang terbaik, jadi selalu bandingkan.

## 8. Konversi ke WhisperX dan evaluasi akhir

```bash
# Baseline (sistem A): Whisper large-v3 asli, tanpa prompt
.envs/whisper-train/bin/python src/evaluate.py predict --model openai/whisper-large-v3 \
    --manifest dataset/test.jsonl --prompt none --out outputs/preds/test-A.jsonl

# Konversi checkpoint terbaik ke CTranslate2 (env whisperx-infer)
bash scripts/convert_ct2.sh outputs/run-001/checkpoint-XXXX outputs/whisper-telemarketing-ct2

# Sistem C: inferensi berurutan dengan konteks (riwayat dari prediksi model sendiri), 2 GPU
bash scripts/run_infer_2gpu.sh manifest --model outputs/whisper-telemarketing-ct2 \
    --manifest dataset/test.jsonl --out outputs/preds/test-C4.jsonl

# Skor (juga untuk test_pii.jsonl)
for s in A C4; do
  .envs/whisper-train/bin/python src/evaluate.py score --manifest dataset/test.jsonl \
      --pred outputs/preds/test-$s.jsonl --out outputs/reports/test-$s.json
done
```

Ablation konteks (CONTEXT_ENGINEERING.md bagian 7): ulangi mode `manifest` dengan `--no-history`, `--no-crm`, `--no-domain`, atau `--no-guard`, lalu bandingkan.

**Kriteria lolos** (SPEC bagian 8):
- Entity exact match sistem C lebih tinggi dari sistem A dan dari ElevenLabs untuk kategori `nominal`, `tanggal`, `no_kartu`, `no_hp`, dan `nik`.
- WER tidak lebih buruk.
- Uji hafalan PII lolos.

## 9. Pakai di WhisperX

```bash
export HF_TOKEN=...            # untuk diarization pyannote
bash scripts/run_infer_2gpu.sh audio --model outputs/whisper-telemarketing-ct2 \
    --audio-dir /path/audio_baru --out-dir outputs/transcripts --crm raw/crm.jsonl
# → outputs/transcripts/<call_id>.json : segmen + kata + speaker + flag retried/low_confidence
```

`--level 1` memakai WhisperX batch dengan prompt domain saja (lebih cepat, tanpa riwayat). `--level 2` (default) adalah inferensi berurutan dengan konteks percakapan. Sebelum dipakai di produksi, uji `get_vad_chunks()` dan `load_diarizer()` pada versi WhisperX yang dipin, karena API internalnya berbeda antar versi.

## Masalah umum

| Gejala | Langkah | Tindakan |
| --- | --- | --- |
| `still_missing` besar | 1 | Cek pencocokan nama file dengan `audio_list.csv` |
| Banyak `duration_mismatch` | 2 | Audio tidak cocok dengan transkrip (salah file atau terpotong) |
| `NOT READY: validation ... belum direview` | 4 | Selesaikan mode `queue` untuk test/validation, lalu `build` ulang |
| `tokenizer hanya punya N token` | 5 | Model belum terunduh atau tidak ada akses internet; cek `HF_HOME` |
| NCCL hang | 5–6 | `NCCL_P2P_DISABLE=1 bash scripts/run_train.sh ...` |
| OOM | 5–6 | `--batch_size 4 --grad_accum 4` atau `--optim adamw_bnb_8bit` |
| Output menyalin prompt / berulang | 8 | Periksa flag `retried`; naikkan porsi tanpa prompt (CONTEXT_ENGINEERING 5.2) |

Detail lainnya: TRAINING_DUAL_GPU.md bagian 11.
