# Spec Dataset Training Whisper – Percakapan Telemarketing

Versi: 2026-10-28 · Status: draft

Spec ini mengubah rekaman telemarketing (`.wav`) dan transkrip ElevenLabs (JSON dengan diarization) menjadi dataset fine-tuning Whisper. Model hasilnya menulis PII dan angka dalam **format akhir** dan memakai **konteks percakapan sebelumnya**. Detail konteks ada di [CONTEXT_ENGINEERING.md](CONTEXT_ENGINEERING.md), dan detail training ada di [TRAINING_DUAL_GPU.md](TRAINING_DUAL_GPU.md).

---

## 1. Ruang lingkup

- **Yang dilatih:** hanya model ASR Whisper (audio → teks). VAD, alignment wav2vec2, dan diarization pyannote tetap memakai bawaan WhisperX.
- **Fokus utama:** akurasi entitas PII dan angka (nominal, tanggal lahir, nomor kartu/rekening/HP, NIK, nama, alamat, email) dalam format akhir.
- **Konteks:** model dilatih memakai prompt berisi teks percakapan sebelumnya dan konteks domain.
- **Peran diarization ElevenLabs:** untuk memotong segmen, metadata speaker (agent/nasabah), dan evaluasi. Label speaker **tidak** dimasukkan ke teks target.
- **Bahasa:** Indonesia, termasuk campuran istilah Inggris dan bahasa percakapan.
- **Di luar cakupan:** fine-tuning model alignment dan diarization.

## 2. Data sumber

Setiap panggilan = satu pasang file dengan nama dasar yang sama: `raw/audio/<call_id>.wav` + `raw/transcripts/<call_id>.json`.

Format di bawah mengikuti output standar ElevenLabs Scribe dengan `diarize=true`. **Format ini perlu dicocokkan dengan 5 sampel asli sebelum pipeline dibangun.**

```json
{
  "language_code": "ind",
  "language_probability": 0.98,
  "text": "Selamat pagi, Bapak. Saya dari ...",
  "words": [
    {"text": "Selamat", "start": 0.32, "end": 0.71, "type": "word", "speaker_id": "speaker_0", "logprob": -0.05},
    {"text": " ", "start": 0.71, "end": 0.74, "type": "spacing", "speaker_id": "speaker_0"},
    {"text": "(tertawa)", "start": 5.10, "end": 5.80, "type": "audio_event", "speaker_id": "speaker_1"}
  ]
}
```

| Field | Dipakai untuk | Aturan |
| --- | --- | --- |
| `words[].text` | Teks target | Gabungkan token `word` + `spacing` sesuai urutan |
| `words[].start` / `end` (detik) | Batas potong segmen | Wajib ada; kata tanpa timestamp → panggilan dikarantina |
| `words[].type` | Filter | `audio_event` dibuang dari teks |
| `words[].speaker_id` | Metadata & giliran bicara | Dipetakan ke `agent` / `customer` |
| `words[].logprob` | Skor kepercayaan | Dipakai di QC |
| `language_code` | Filter | Selain Indonesia → tinjau manual |

**Kontrak audio:** WAV PCM, ≥ 8 kHz, mono atau stereo. Jika rekaman stereo dengan agent dan nasabah di kanal terpisah, catat di metadata.

**Penting:** transkrip ElevenLabs adalah *pseudo-label*. Melatih langsung dari label ini membuat model meniru kesalahannya, terutama pada digit. Segmen PII serta seluruh validation dan test wajib dikoreksi manusia.

## 2a. Sumber data produksi: telemarketing-qc-dashboard

Data diambil dari DB DWH (`10.155.32.28`, schema `dashboard`) lewat container `telemarketing-qc-api-api-1`, **read-only** (`default_transaction_read_only=on`). Skripnya `src/qc_export_db.py` dan `src/qc_export.py`, dijalankan dengan `scripts/qc_dataset.sh export`.

| Kebutuhan | Tabel / sumber | Catatan |
| --- | --- | --- |
| Daftar panggilan QC | `results.source_files` (status `done`) | `call_id` = stem PDF = `ticket_id`; satu result bisa berisi beberapa panggilan |
| Transkrip per kata + diarization | `result_speech_to_text.asr_raw_response` (`asr_backend='elevenlabs'`) | Respons mentah ElevenLabs Scribe; diambil baris terbaru per `audio_file` |
| Path audio | `recording_tms_api.file_path` | File GSM 8 kHz di server rekaman; nama file **berbeda** dari `ticket_id` |
| Nama nasabah (lapisan CRM) dan agent | `recording_tms_api.cust_name`, `agent_id` | `agent_id` dipakai untuk split per agent |
| Feedback QC | `qc_status_requests.requested_status`, `tl_qc_status` | Manual Status PASS/FAIL per tiket |

Snapshot 2026-11-01:
- 2.325 file panggilan di QC dashboard; 2.295 di antaranya punya JSON ElevenLabs, total ±412 jam.
- 2.068 punya `file_path`. Kampanye Collection hampir semuanya tanpa path.
- 622 panggilan sudah dinilai QC: 580 PASS dan 42 FAIL.

**Batasan feedback QC:** Manual Status dan banding error code menilai **kepatuhan agent**, bukan ketepatan transkrip. Per 2026-11-01, tabel `error_code_appeals` dan `qc_manual_checks` masih kosong. Karena itu, label transkrip tetap berasal dari ElevenLabs dan dicek manusia lewat `review_server.py`. Status QC hanya dibawa sebagai metadata (`meta.qc_status`) untuk filter dan analisis.

**Audio tidak diambil otomatis.** `raw/audio_list.csv` berisi `ticket_id,file_path,campaign,qc_status,qc_tl_status,duration_sec`. Tim mengambil sendiri audio yang dipilih, lalu `scripts/qc_dataset.sh import-audio <folder>` mengonversinya ke `raw/audio/<ticket_id>.flac`. Nama file boleh berupa `ticket_id` atau nama asli dari `file_path`.

**Temuan pada data asli** (dry-run 2.295 transkrip):
- Ada kata ElevenLabs dengan timestamp rusak (durasi 30 sampai 190 detik untuk satu kata). Kata seperti ini diisolasi dan segmennya ditolak (`bad_timing`).
- Kata "PIN" hampir selalu muncul di instruksi ("buat PIN", "ketik PIN spasi ..."). Karena itu, segmen dianggap `secret` hanya bila nilainya ikut disebut (lihat bagian 7).

## 3. Format output dataset

```
dataset/
├── clips/<call_id>/<call_id>_<idx:04d>.flac   # 16 kHz mono
├── train.jsonl
├── validation.jsonl
├── test.jsonl            # + test_pii.jsonl (subset)
├── rejected.jsonl        # segmen dibuang + alasan
└── stats.json            # jam audio, porsi PII, distribusi durasi, seed
```

Clip disimpan sebagai **FLAC** (lossless, sekitar separuh ukuran WAV). Sebagai gambaran, WAV 16 kHz mono 16-bit butuh ±115 GB per 1.000 jam, sedangkan sisa disk saat ini ±113 GB.

Satu baris manifest = satu segmen:

```json
{"id": "CALL123_0007", "audio_filepath": "dataset/clips/CALL123/CALL123_0007.flac", "call_id": "CALL123", "seg_index": 7, "start": 41.20, "end": 63.85, "duration": 22.65, "prev_text": "Selamat pagi, dengan Ibu Sari Wulandari? Saya Dina dari Bank Mega, Bu. Ibu ada penawaran kenaikan limit kartu kredit.", "prev_text_raw": "Selamat pagi, dengan Ibu Sari Wulandari? Saya Dina dari Bank Mega, Bu. Ibu ada penawaran kenaikan limit kartu kredit.", "text": "Baik, Bu. Limitnya jadi Rp15.000.000, untuk verifikasi tanggal lahir Ibu 17-08-1985 ya?", "text_raw": "Baik, Bu. Limitnya jadi lima belas juta, untuk verifikasi tanggal lahir Ibu 17 Agustus 1985 ya?", "crm": {"customer_name": "Sari Wulandari", "product": "kenaikan limit kartu kredit"}, "speakers": ["agent", "customer"], "n_turns": 3, "pii_types": ["nominal", "tanggal"], "language": "id", "sample_rate": 16000, "mean_logprob": -0.12, "reviewed": true, "split": "train"}
```

| Field | Tipe | Wajib | Keterangan |
| --- | --- | --- | --- |
| `id` | string | ya | `<call_id>_<idx>`, unik |
| `audio_filepath` | string | ya | Relatif terhadap root project |
| `call_id` | string | ya | Kunci split |
| `seg_index` | int | ya | Urutan segmen dalam panggilan, mulai dari 0 |
| `start`, `end`, `duration` | float (detik) | ya | Posisi di audio asli; `duration` ≤ 30 |
| `prev_text` | string | ya | Teks final segmen-segmen sebelumnya di panggilan yang sama, maksimal 1.500 karakter terakhir (pemotongan ke 223 token dilakukan `context_builder`). `""` untuk segmen pertama |
| `prev_text_raw` | string | ya | Sama seperti `prev_text`, tapi dari `text_raw`; dipakai variasi prompt "noisy" saat training |
| `meta` | object / null | tidak | Dari `raw/meta.jsonl`: `result_id`, `campaign`, `qc_status`, `qc_tl_status`, `agent_id`, `product`, `source` |
| `min_logprob` | float | ya | Logprob terendah ElevenLabs di segmen |
| `entity_min_logprob` | float | ya | Logprob terendah dari kata angka; bila tidak ada kata angka, sama dengan `min_logprob` |
| `pii_auto` | bool | ya | `true` bila segmen PII masuk train tanpa review (kebijakan `confidence`) |
| `text` | string | ya | Teks target dalam format akhir |
| `text_raw` | string | ya | Teks asli ElevenLabs, untuk audit dan augmentasi konteks |
| `crm` | object / null | tidak | Nama nasabah & produk jika tersedia |
| `speakers`, `n_turns` | list, int | ya | Role yang bicara di segmen ini |
| `pii_types` | list | ya | Kategori PII; `[]` jika tidak ada |
| `mean_logprob` | float | ya | Rata-rata `logprob` kata |
| `reviewed` | bool | ya | `true` jika sudah dikoreksi manusia |
| `split` | string | ya | `train` / `validation` / `test` |

Aturan `prev_text`:

- Diambil dari kolom `text` (sudah direview), bukan dari `text_raw`.
- Segmen yang dibuang QC tetap ikut dihitung, karena saat inferensi percakapan juga tidak terputus.
- **Pengecualian:** teks segmen berlabel `secret` (PIN, OTP, CVV, password) tidak pernah masuk ke `prev_text`.

## 4. Pipeline konversi

Segmen dipotong mendekati cara WhisperX membaca audio: VAD lalu chunk sampai 30 detik, dan satu chunk bisa berisi lebih dari satu pembicara.

1. **Validasi pasangan.** Cocokkan `.wav` dan `.json` berdasarkan `call_id`. File tanpa pasangan, JSON rusak, atau selisih durasi audio dengan kata terakhir > 2 detik → `rejected.jsonl`.
2. **Resample.** Ubah ke 16 kHz, mono, 16-bit (`ffmpeg -ar 16000 -ac 1`). Audio telepon 8 kHz cukup di-upsample.
3. **Ambil token kata.** Token `spacing` tidak dipakai untuk hitungan waktu, tapi spasinya dipertahankan saat teks disusun ulang.
4. **Petakan speaker.** `speaker_0`/`speaker_1` → `agent`/`customer`. Aturan: speaker dengan total waktu bicara terlama = `agent`. Speaker yang bicara pertama tidak dipakai karena di panggilan keluar nasabah biasanya menyapa duluan ("Halo?"). Verifikasi pada sampel.
5. **Cari titik potong.** Titik potong yang sah = jeda antar kata ≥ 0,3 detik atau pergantian speaker. Jangan memotong di tengah kata atau di tengah **deretan angka** (nomor, NIK, nominal, tanggal), walaupun ada jeda.
6. **Susun segmen (greedy).**
    - Target 10–28 detik; minimum 1 detik; maksimum keras 30 detik.
    - Padding 0,1 detik di awal dan akhir tanpa melewati kata tetangga.
    - Jeda > 3 detik selalu memisah segmen.
    - 10–20% segmen pendek (1–5 detik), misalnya satu giliran nasabah.
7. **Normalisasi** teks ke format akhir (bagian 5).
8. **Isi `prev_text`**, lalu jalankan QC (bagian 7) dan tulis clip + manifest.

## 5. Normalisasi & konvensi format akhir

Teks target ortografis: huruf besar-kecil dan tanda baca dipertahankan. Model menulis **format akhir** secara langsung, tanpa skrip ITN setelah WhisperX.

### 5.1 Aturan umum

| Kasus | Aturan | Contoh |
| --- | --- | --- |
| Event non-speech (`audio_event`) | Hapus | `(tertawa)` → dihapus |
| Filler | Pertahankan, ejaan baku tetap | `eh`, `em`, `hmm`, `ya` |
| Singkatan produk | Daftar kata baku | `KTA`, `CC`, `cicilan 0%` |
| Kata Inggris | Ejaan asli | `limit`, `cashback`, `annual fee` |
| Bahasa gaul | Sesuai ucapan, jangan dibakukan | `nggak`, `udah`, `gimana` |
| Ucapan tumpang-tindih | Gabung sesuai urutan waktu | — |
| Tidak terdengar | Segmen dibuang, jangan diberi tag | — |
| Spasi & tanda baca | Rapikan spasi ganda dan spasi sebelum tanda baca | `iya , Pak` → `iya, Pak` |

### 5.2 Angka & PII

| Entitas | Diucapkan | Ditulis di label | Aturan |
| --- | --- | --- | --- |
| Nominal uang | "lima ratus ribu rupiah" | `Rp500.000` | `Rp` + titik ribuan, tanpa spasi; kata "rupiah" tidak ditulis |
| Nominal tanpa "rupiah" | "limitnya lima belas juta" | `limitnya Rp15.000.000` | Tetap `Rp` jika konteksnya uang |
| Nominal campuran/pecahan | "dua setengah juta" | `Rp2.500.000` | Dihitung jadi angka penuh |
| Nominal desimal | "seribu lima ratus koma lima puluh" | `Rp1.500,50` | Koma desimal |
| Persen/bunga | "nol koma lima persen" | `0,5%` | `%` menempel pada angka |
| Tanggal lengkap | "tujuh belas Agustus sembilan belas delapan puluh lima" | `17-08-1985` | `DD-MM-YYYY` |
| Tanggal, tahun 2 digit | "tujuh belas, nol delapan, delapan lima" | `17-08-1985` | `YY` > dua digit tahun berjalan → `19YY`, selain itu `20YY` |
| Tanggal tanpa tahun | "tanggal dua lima" | `tanggal 25` | Tahun tidak dikarang |
| Masa berlaku kartu | "nol delapan dua delapan" | `08/28` | `MM/YY` |
| Jam | "jam dua siang" | `jam 14.00` | 24 jam, `HH.MM` |
| Nomor kartu/rekening/HP/NIK | "nol delapan satu dua, tiga empat ..." | `081234...` | Digit menyambung, tanpa spasi/tanda hubung |
| "kosong" / "nol" | "kosong delapan" | `08` | Keduanya `0` |
| "dobel" / "tripel" | "dobel tiga" | `33` | Diekspansi |
| Tenor/jumlah | "dua belas bulan" | `12 bulan` | Digit + satuan |
| Angka kecil non-entitas | "dua orang" | `2 orang` | Tetap digit |
| Email | "andi titik budi at gmail titik com" | `andi.budi@gmail.com` | Bentuk tertulis |
| Nama & alamat | "Jalan Mawar nomor dua belas, RT tiga" | `Jalan Mawar No. 12, RT 03` | Kapital nama diri; RT/RW dua digit |

- **Pengulangan atau koreksi:** tulis semua yang diucapkan, misalnya `812, eh, 813`. Model tidak memutuskan sendiri nilai mana yang benar.
- `src/normalize.py` mengubah output ElevenLabs (yang formatnya campuran) ke tabel di atas. Setiap baris tabel wajib punya unit test di `tests/`.
- Kasus yang tidak bisa diputuskan otomatis ditandai `needs_review`.
- **`pii_types`:** `nominal`, `tanggal`, `no_kartu`, `no_rekening`, `no_hp`, `nik`, `nama`, `alamat`, `email`. Deteksi awal memakai regex digit dan kata kunci (`tanggal lahir`, `nomor kartu`, `rekening`, `NIK`, `alamat`, `email`, `juta`, `ribu`), lalu dikonfirmasi reviewer.
- **Normalisasi evaluasi** (WER umum): `BasicTextNormalizer` Whisper + pemetaan varian ejaan. Metrik entitas dihitung pada output mentah.

## 6. Privasi & PII

PII **dipertahankan** di data training agar model belajar menangkap data nasabah dengan tepat. Dataset dan model berstatus data rahasia sesuai UU PDP dan kebijakan internal bank. **Persetujuan tim kepatuhan/DPO wajib didapat sebelum training.**

- Teks dan audio verbatim: tidak ada masking, beep, atau tag.
- **Dibuang:** PIN, OTP, CVV/CVC, dan password (alasan `secret`), termasuk dari `prev_text`.
- Training, anotasi, dan penyimpanan hanya di server internal.
- Akses `raw/`, `dataset/`, dan `outputs/` dibatasi ke tim training. Model diperlakukan serahasia datanya.
- Prediksi berisi PII tidak ditulis ke dashboard bersama.
- **Uji hafalan:** jalankan model pada audio hening dan audio non-PII, lalu pastikan tidak ada nomor atau nama dari data training yang keluar.
- Clip turunan dihapus mengikuti masa retensi rekaman sumber.

## 7. Quality control & split

| Cek | Ambang awal | Alasan di `rejected.jsonl` |
| --- | --- | --- |
| Durasi | 1–30 detik | `duration` |
| Panjang label | Prompt + prefix + teks ≤ 448 token | `too_many_tokens` |
| Kecepatan bicara | 4–25 karakter/detik | `cps_outlier` |
| Kepercayaan (non-PII) | `mean_logprob` ≥ −0,5 | `low_confidence` |
| PII belum direview | `pii_types` ≠ `[]` dan `reviewed: false` | `pii_unreviewed` |
| Validation/test belum direview | `split` ≠ `train` dan `reviewed: false` | `eval_unreviewed` |
| Teks kosong | Minimal 1 kata | `empty` |
| Audio buruk | RMS < −50 dBFS atau > 1% sampel clipping | `bad_audio` |
| PIN/OTP/CVV/password | Kata kunci **dan** deret ≥ 3 digit (di luar nominal, tanggal, jam) di segmen yang sama, atau di segmen tepat setelah pertanyaan OTP/CVV | `secret` |
| Timestamp kata rusak | Ada kata berdurasi > 3 detik | `bad_timing` |
| Dibuang reviewer | Aksi "Buang segmen" di `review_server.py` | `human_rejected` |
| Bahasa | `language_code` bukan Indonesia | `language` |
| Duplikat | File audio identik (hash SHA-256), seluruh panggilan ditolak | `duplicate` |

Ambang ini titik awal. Setel ulang setelah melihat `stats.json` dan mendengarkan ±50 sampel yang ditolak. Segmen yang sudah direview tidak dikenai cek `low_confidence` dan `cps_outlier`.

**Alur pengecekan manusia** (`src/review_server.py`, `scripts/qc_dataset.sh review <nama> [queue|spotcheck]`):

| Mode | Isi | Tujuan |
| --- | --- | --- |
| `queue` | `review_queue.jsonl`: segmen yang tertahan karena belum direview, urut test → validation → train | Membuka segmen PII dan validation/test |
| `spotcheck` | Sampel acak 3% (`--rate`) segmen train yang lolos otomatis, termasuk `pii_auto` | Mengukur persentase label otomatis yang perlu dikoreksi |

- Aksi: **Simpan**, **Rahasia**, **Buang segmen**, **Lewati**. Setiap aksi ditulis ke `reviews.jsonl` beserta `auto_text`, `edited`, `mode`, `reviewer`.
- Halaman `/stats` menampilkan progres per reviewer dan persentase koreksi per mode.
- Setelah review, jalankan ulang `build_dataset.py`. Review dicocokkan dengan id + start/end segmen, jadi parameter segmentasi tidak boleh diubah setelah review dimulai.

**Siap training:** `src/check_ready.py` (`scripts/qc_dataset.sh check`) menulis `READY.json` dan keluar dengan kode 0 hanya bila semua syarat berikut terpenuhi:
- Ketiga split terisi.
- Tidak ada panggilan yang bocor antar split.
- Validation/test 100% direview, dan PII train sudah direview atau `pii_auto`.
- Tidak ada nilai rahasia di `text`/`prev_text`.
- Semua audio 16 kHz mono, 1 sampai 30 detik, dengan durasi sesuai manifest.

Peringatan muncul bila spotcheck masih < 100 segmen atau koreksinya > 10%.

- **Review PII:** semua segmen ber-`pii_types` dikoreksi manusia sebelum masuk train. Reviewer cukup memeriksa rentang entitas sambil mendengar audio.
- **Porsi PII:** target ≥ 20% segmen train berisi PII, dengan oversampling 2–3× untuk segmen PII yang sudah direview, hanya di train. Jika masih kurang, rekam skrip simulasi dengan data **fiktif**.
- **Split per agent** (`--split-by agent`, default bila `meta.agent_id` ada; selain itu per panggilan). Kelompok diurutkan acak-deterministik lalu diisi ke test sampai `--test-hours` (default 4 jam audio panggilan), kemudian validation sampai `--val-hours` (default 2 jam), sisanya train. Agent yang akan membuat split melebihi 1,5x target tetap di train. Target berbasis jam, bukan persentase, supaya beban review validation/test tetap kecil.
- **Kebijakan PII train** (`--pii-policy`):
    - `confidence` (**default, diputuskan 2026-11-01**): segmen PII di train diterima tanpa review bila `entity_min_logprob` ≥ `--pii-min-logprob` (default −0,1) dan tidak ada `needs_review`. Ditandai `pii_auto: true`.
    - `review`: semua segmen PII wajib direview (±25 ribu segmen di train; tidak realistis untuk data saat ini).
    - Validation/test selalu wajib direview.
    - Kualitas `pii_auto` diaudit dengan mode spotcheck.
    - Pada data 2026-11-01, kebijakan `confidence` menghasilkan train ±265 jam dengan 22,7% segmen PII; antrean review wajib validation/test ±1.500 segmen (±6 jam).
- **`test_pii`:** subset test dengan ≥ 500 entitas yang tersebar di semua kategori.
- Validation dan test 100% `reviewed: true`.
- Seed dicatat di `stats.json`.

## 8. Evaluasi

| Sistem | Keterangan |
| --- | --- |
| A | Whisper large-v3 asli, WhisperX batch |
| B | Fine-tune, WhisperX batch + prompt domain statis |
| C | Fine-tune, inferensi berurutan dengan konteks penuh |
| D | ElevenLabs (referensi) |

- **Umum:** WER/CER ternormalisasi per segmen dan per panggilan, terpisah untuk agent dan nasabah.
- **PII (`test_pii`, metrik utama), dihitung pada output mentah:**
    - *Entity exact match* per kategori.
    - *Value match*: nilai benar meski formatnya salah. Selisihnya dengan exact match = kesalahan yang murni soal format.
    - *Digit error rate* di dalam entitas.
- **Efek konteks:** metrik per posisi chunk (awal, tengah, akhir panggilan), plus konsistensi ejaan nama dalam satu panggilan.
- **Kriteria lolos:** C > A dan C > D untuk entity exact match di `nominal`, `tanggal`, `no_kartu`, `no_hp`, `nik`; WER C ≤ WER B; tidak ada penurunan di 1 jam audio umum non-telemarketing.

## 9. Checklist acceptance dataset

- [ ] Format JSON dicocokkan dengan 5 sampel asli
- [ ] Persetujuan kepatuhan/DPO untuk training dengan PII
- [ ] `normalize.py` lolos unit test untuk setiap baris tabel 5.2
- [ ] Semua clip 16 kHz mono, 1–30 detik, total token ≤ 448
- [ ] Tidak ada `call_id` di lebih dari satu split
- [ ] Semua segmen PII serta seluruh validation/test `reviewed: true`
- [ ] Tidak ada PIN/OTP/CVV/password, termasuk di `prev_text`
- [ ] `test_pii` ≥ 500 entitas
- [ ] `stats.json` lengkap
- [ ] 50 segmen PII acak didengarkan, setiap digit cocok dengan audio
- [ ] Baseline sistem A tercatat

## 10. Pertanyaan terbuka

- Total jam audio dan jumlah panggilan, serta perkiraan porsi PII?
- Apakah rekaman stereo dengan kanal terpisah per speaker?
- Apakah format akhir di 5.2 sudah sesuai sistem hilir (CRM, QA call)?
- Apakah data CRM (nama nasabah, produk) bisa diakses saat inferensi?
- Berapa batas waktu proses per panggilan?
- Siapa reviewer PII, dan berapa kapasitasnya per minggu?
