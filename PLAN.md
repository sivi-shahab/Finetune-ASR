# Plan: Fine-tune Whisper untuk WhisperX – Telemarketing

Versi: 2026-10-28 · Spec: [docs/SPEC.md](docs/SPEC.md) · Konteks: [docs/CONTEXT_ENGINEERING.md](docs/CONTEXT_ENGINEERING.md) · Training: [docs/TRAINING_DUAL_GPU.md](docs/TRAINING_DUAL_GPU.md)

**Tujuan:** model ASR untuk WhisperX yang menulis PII dan angka dalam format akhir dengan benar dan memakai konteks percakapan. Model ini harus lebih akurat dari Whisper large-v3 asli dan dari ElevenLabs untuk entity exact match di `test_pii`.

**Status 2026-10-28:** kode Fase 0, 2, 3, 5, dan 6 sudah ditulis dan sebagian besar diuji tanpa data asli; Fase 1 menunggu data asli. Env WhisperX belum dipasang.

Fase disusun berurutan. Fase 2 dan 4 bisa berjalan paralel dengan review manual di Fase 3.

---

## Fase 0 – Persiapan & persetujuan

- [ ] Persetujuan tim kepatuhan/DPO untuk training dengan PII (**blocker untuk Fase 3 dan seterusnya**)
- [x] Env `whisper-train` (Python 3.11, torch 2.11 cu128, transformers 5.21) di `.envs/`, versi dipin di `requirements-train.txt`
- [ ] Env `whisperx-infer`: skrip siap (`scripts/setup_envs.sh infer`), **belum dipasang**
- [x] Cek NCCL 2 GPU lolos (`src/nccl_check.py`), tanpa `NCCL_P2P_DISABLE`
- [x] Struktur direktori; `raw/`, `dataset/`, `outputs/` chmod 700
- [ ] Hitung kebutuhan disk dari total jam audio (sisa ±113 GB)

**Selesai jika:** kedua env berjalan, `torchrun` 2 GPU lolos, dan persetujuan tertulis sudah ada.

## Fase 1 – Audit data

**Sumber data: telemarketing-qc-dashboard** (SPEC bagian 2a)
- [x] Petakan sumber: DB DWH `dashboard` (results, qc_status_requests, result_speech_to_text, recording_tms_api)
- [x] Export read-only 2.295 panggilan + `raw/audio_list.csv` (2026-11-01)
- [x] Dry-run segmentasi/normalisasi pada transkrip asli; perbaikan: `bad_timing`, aturan `secret`, split berbasis jam
- [x] Keputusan 2026-11-01: `--pii-policy confidence` (default) — PII train diterima bila logprob kata angka ≥ −0,1, diaudit lewat spotcheck; validation/test tetap wajib review
- [ ] Ambil audio sesuai `raw/audio_list.csv` (oleh tim) → `scripts/qc_dataset.sh import-audio`
- [ ] Build + review validation/test + spotcheck → `scripts/qc_dataset.sh check` = READY

**Audit (lanjutan, setelah audio tersedia)**

- [ ] Salin data ke `raw/audio/` dan `raw/transcripts/`
- [ ] Cocokkan format JSON ElevenLabs dengan SPEC bagian 2 menggunakan 5 sampel; perbarui SPEC jika berbeda
- [ ] Hitung statistik: jumlah panggilan, total jam, sample rate, mono/stereo, jumlah speaker per panggilan
- [ ] Ambil 30 panggilan acak, dengarkan, lalu catat jenis PII, gaya penyebutan angka, dan kesalahan ElevenLabs yang sering muncul
- [ ] Jawab pertanyaan terbuka di SPEC bagian 10

**Deliverable:** `docs/DATA_AUDIT.md`. **Selesai jika:** semua pertanyaan terbuka terjawab atau punya pemilik.

## Fase 2 – Normalisasi & context builder

- [x] `src/normalize.py`: konversi teks ElevenLabs → format akhir (SPEC 5.2), deteksi `pii_types`, deteksi `secret`, penanda `needs_review`
- [x] `tests/test_normalize.py`: satu test per baris tabel SPEC 5.2 + format digit ElevenLabs + kasus tepi
- [ ] Tambah contoh nyata dari audit (menunggu Fase 1)
- [x] `src/context_builder.py` + `tests/test_context_builder.py` (CONTEXT_ENGINEERING bagian 4)
- [x] `configs/domain_prompt.txt` (berisi contoh format akhir)
- [ ] Verifikasi ≤ 40 token dengan tokenizer Whisper asli (test dilewati sampai tokenizer diunduh)

**Selesai jika:** semua test lolos, dan 100 kalimat ber-angka dari data asli dicek manual dengan akurasi konversi ≥ 98%.

## Fase 3 – Build dataset & review manual

- [x] `src/build_dataset.py`: validasi pasangan, resample ke FLAC 16 kHz, segmentasi, `prev_text`, QC, split per `call_id`, `stats.json` (diuji dengan data sintetis)
- [x] Alat review: `src/review_server.py`
- [ ] Review: seluruh validation + test, lalu semua segmen PII di train
- [ ] Bangun `test_pii` (≥ 500 entitas, semua kategori)
- [ ] Jika porsi PII < 20%: rekam skrip simulasi dengan data fiktif

**Selesai jika:** checklist acceptance di SPEC bagian 9 terpenuhi.

## Fase 4 – Baseline

- [x] `src/evaluate.py`: WER/CER ternormalisasi, entity exact match, value match, digit error rate, metrik per posisi chunk dan speaker, konsistensi nama
- [ ] Sistem A: Whisper large-v3 asli lewat WhisperX batch
- [ ] Sistem D: ElevenLabs (transkrip yang sudah ada, dibandingkan dengan referensi hasil review)
- [ ] Sistem A + prompt domain (Level 1), untuk melihat efek prompt tanpa training

**Deliverable:** tabel baseline di `docs/RESULTS.md`.

## Fase 5 – Training

- [x] `src/train.py`: collator dengan variasi prompt, DDP, konfigurasi sesuai TRAINING_DUAL_GPU bagian 6 (belum dijalankan)
- [ ] Smoke test 1 GPU, lalu smoke test DDP 2 GPU (kriteria di TRAINING_DUAL_GPU bagian 7)
- [ ] Run-001: full fine-tune large-v3, 3 epoch
- [ ] Evaluasi tiap checkpoint dengan prompt domain; pilih checkpoint terbaik berdasarkan entity exact match di validation
- [ ] Jika hasil belum cukup: atur LR, porsi PII, atau porsi variasi prompt untuk run-002

**Selesai jika:** checkpoint terbaik mengalahkan sistem A pada entity exact match di validation.

## Fase 6 – Inferensi dengan konteks

- [ ] Konversi checkpoint terbaik ke CT2 float16
- [x] `src/infer_context.py`: Level 1/2 + mode manifest untuk ablation (loop konteks diuji dengan ASR tiruan)
- [ ] Uji `get_vad_chunks()` dan `load_diarizer()` pada versi WhisperX yang dipin
- [x] Runner 2 GPU: `scripts/run_infer_2gpu.sh` (panggilan dibagi berdasarkan hash)
- [ ] Ukur real-time factor per panggilan

**Selesai jika:** pipeline menghasilkan transkrip lengkap dengan label speaker untuk seluruh test set tanpa error.

## Fase 7 – Evaluasi akhir & ablation

- [ ] Sistem B dan C di `test` dan `test_pii`
- [ ] Ablation konteks C0–C5 (CONTEXT_ENGINEERING bagian 7)
- [ ] Uji hafalan PII (audio hening dan audio non-PII)
- [ ] Uji regresi di 1 jam audio umum non-telemarketing
- [ ] Analisis kesalahan: 50 kesalahan entitas teratas, dikelompokkan per penyebab

**Selesai jika:** kriteria lolos di SPEC bagian 8 terpenuhi. **Deliverable:** `docs/RESULTS.md` lengkap.

## Fase 8 – Serah terima

- [ ] Model CT2 + `domain_prompt.txt` + versi paket dipin
- [ ] Panduan menjalankan inferensi dan batas kemampuan model (kategori entitas yang masih lemah)
- [ ] Rencana retraining: alur data baru → review → retrain

---

## Risiko utama

| Risiko | Dampak | Mitigasi |
| --- | --- | --- |
| Persetujuan PII tertunda | Fase 3+ terhambat | Ajukan di Fase 0; Fase 1–2 tetap berjalan dengan akses terbatas |
| Kapasitas reviewer kurang | Data PII sedikit | Review hanya rentang entitas, prioritaskan test, tambah data simulasi |
| Label ElevenLabs salah di digit | Model belajar salah | Review wajib untuk semua segmen PII |
| Konteks menyebarkan kesalahan | Akurasi turun di akhir panggilan | Penjaga + variasi `text_raw`; ukur per posisi chunk |
| Disk penuh | Training berhenti | FLAC, `save_total_limit=2`, pantau `df -h` |
| API WhisperX berubah | Inferensi rusak | Pin versi dan tes integrasi di Fase 6 |
| Model menghafal PII | Kebocoran data | Uji hafalan, akses terbatas, tidak dibagikan ke luar |

## Keputusan yang sudah diambil

| Keputusan | Pilihan |
| --- | --- |
| PII di data training | Dipertahankan verbatim; PIN/OTP/CVV/password dibuang |
| Format angka | Format akhir langsung dari model (`Rp15.000.000`, `17-08-1985`) |
| Konteks | Prompt berlapis (domain + CRM + riwayat), inferensi berurutan per chunk |
| Strategi multi-GPU | DDP 2× L40S, bf16, batch efektif 32 |
| Base model | `openai/whisper-large-v3` |
