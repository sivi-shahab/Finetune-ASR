# Context Engineering: Agar WhisperX Mengetahui Konteks Percakapan

Versi: 2026-10-28 · Terkait: [SPEC.md](SPEC.md), [TRAINING_DUAL_GPU.md](TRAINING_DUAL_GPU.md)

Whisper hanya punya satu jalur untuk menerima konteks: **prompt teks** sebelum token `<|startoftranscript|>`, maksimal **223 token**. Context engineering di proyek ini berarti mengatur apa saja yang masuk ke 223 token itu, dengan urutan dan format yang sama persis saat training dan inferensi.

---

## 1. Masalah: WhisperX default tidak membawa konteks

| Perilaku default WhisperX | Akibat |
| --- | --- |
| Audio dipotong VAD menjadi chunk ≤ 30 detik | Satu panggilan 5 menit = ±10–20 chunk |
| Chunk didekode secara **batch** (paralel) | Chunk ke-7 tidak tahu isi chunk 1–6 |
| `condition_on_previous_text=False` | Tidak ada teks sebelumnya yang dipakai sebagai prompt |
| `initial_prompt` (jika diisi) sama untuk semua chunk | Hanya bisa membawa konteks statis |

Contoh dampaknya: di chunk ke-7 nasabah berkata "yang lima belas itu, Mbak". Tanpa konteks, model tidak tahu bahwa ini merujuk ke `Rp15.000.000` (limit) yang disebut di chunk ke-3. Nama nasabah juga bisa dieja berbeda-beda di tiap chunk.

## 2. Cara Whisper memakai prompt

```
<|startofprev|> [prompt ≤ 223 token] <|startoftranscript|><|id|><|transcribe|><|notimestamps|> [output]
```

Tiga sifat penting:

1. **Model meniru gaya prompt.** Jika prompt berisi `Rp15.000.000` dan `17-08-1985`, output cenderung memakai format yang sama. Jika prompt huruf kecil tanpa tanda baca, output ikut begitu. Karena itu, prompt **selalu ditulis dalam format akhir**.
2. **Kosakata di prompt lebih mudah dikenali.** Nama produk, nama nasabah, dan istilah bank yang muncul di prompt lebih jarang salah eja.
3. **Prompt bisa disalin.** Pada audio hening atau tidak jelas, model kadang mengulang isi prompt. Ini harus dijaga (bagian 6) dan dilatih (bagian 5).

Whisper asli hanya dilatih memakai prompt berupa teks sebelumnya. Prompt dengan struktur khusus (glosarium, data CRM) baru efektif setelah model **dilatih dengan format prompt yang sama**.

## 3. Lapisan konteks

Prompt disusun dari empat lapisan dengan prioritas tetap. Jika token tidak cukup, lapisan berprioritas paling rendah dipotong lebih dulu.

| Prioritas | Lapisan | Isi | Budget | Sumber |
| --- | --- | --- | --- | --- |
| 1 | **Domain** | Jenis percakapan + contoh format akhir + istilah produk | ≤ 40 token | Statis, `configs/domain_prompt.txt` |
| 2 | **Sesi (CRM)** | Nama nasabah, produk yang ditawarkan, nama agent | ≤ 30 token | CRM sebelum panggilan (opsional) |
| 3 | **Riwayat percakapan** | Ekor transkrip panggilan sejauh ini | Sisa budget (±150 token) | Output chunk sebelumnya |
| — | Total | | ≤ 223 token | |

Riwayat dipotong **dari depan**, pada batas kata. Bagian terbaru percakapan selalu dipertahankan.

### Template

```
[domain]  Telemarketing Bank Mega: kartu kredit, KTA, limit Rp15.000.000, cicilan 0%, tanggal lahir 17-08-1985.
[sesi]    Nasabah: Sari Wulandari. Produk: kenaikan limit kartu kredit.
[riwayat] Selamat pagi, dengan Ibu Sari Wulandari? ... Baik, Bu, limitnya jadi Rp15.000.000.
```

Ketiga baris digabung menjadi satu string dengan pemisah spasi. Label `[domain]`, `[sesi]`, dan `[riwayat]` hanya penjelasan di dokumen ini dan tidak ikut masuk ke prompt.

Aturan penulisan:

- Contoh angka di lapisan domain adalah **contoh format, bukan data nasabah asli**.
- Jangan memakai label seperti `Agent:` atau `Nasabah:` di riwayat. Output Whisper tidak berisi label speaker, jadi label di prompt akan bertentangan dengan gaya output.
- Teks segmen `secret` (PIN, OTP, CVV, password) tidak pernah masuk ke riwayat.
- Tidak ada lapisan "memori entitas" (daftar nomor yang sudah disebut) pada versi awal. Lapisan seperti ini berisiko membuat model mengulang nomor lama ketika nasabah menyebut nomor baru. Uji sebagai eksperimen terpisah.

## 4. Satu modul untuk training dan inferensi

Ketidakcocokan format prompt antara training dan inferensi adalah sumber bug paling umum. Karena itu, **hanya `src/context_builder.py` yang boleh menyusun prompt**, baik di data collator training maupun di loop inferensi.

```python
# src/context_builder.py (sketsa)
MAX_PROMPT_TOKENS = 223  # batas prompt Whisper: n_text_ctx // 2 - 1

def build_prompt(tokenizer, domain: str, crm: dict | None, history: str,
                 max_tokens: int = MAX_PROMPT_TOKENS) -> list[int]:
    """Kembalikan token id prompt (tanpa <|startofprev|>), maksimal max_tokens."""
    parts = [domain.strip()]
    if crm:
        crm_txt = []
        if crm.get("customer_name"):
            crm_txt.append(f"Nasabah: {crm['customer_name']}.")
        if crm.get("product"):
            crm_txt.append(f"Produk: {crm['product']}.")
        parts.append(" ".join(crm_txt))
    head = tokenizer.encode(" " + " ".join(p for p in parts if p),
                            add_special_tokens=False)
    if len(head) >= max_tokens:
        return head[:max_tokens]
    hist = tokenizer.encode(" " + history.strip(), add_special_tokens=False) if history.strip() else []
    room = max_tokens - len(head)
    return head + hist[-room:]  # riwayat dipotong dari depan
```

Catatan implementasi:

- Pemotongan `hist[-room:]` bisa memotong di tengah kata. Versi final sebaiknya memotong pada batas spasi: decode, cari spasi pertama, lalu encode ulang.
- Unit test di `tests/test_context_builder.py` minimal mencakup: hasil ≤ 223 token, riwayat terbaru selalu ada, domain tidak pernah terpotong jika budget cukup, dan tidak ada teks `secret`.

## 5. Training: model belajar memakai konteks

### 5.1 Susunan sampel

```
decoder_input_ids = [<|startofprev|>] + prompt + [SOT, <|id|>, <|transcribe|>, <|notimestamps|>] + teks
labels            = digeser 1 token; semua posisi yang memprediksi prompt dan prefix = -100
```

```python
SOP  = tok.convert_tokens_to_ids("<|startofprev|>")
SOT  = tok.convert_tokens_to_ids("<|startoftranscript|>")
LANG = tok.convert_tokens_to_ids("<|id|>")
TASK = tok.convert_tokens_to_ids("<|transcribe|>")
NOTS = tok.convert_tokens_to_ids("<|notimestamps|>")
EOT  = tok.convert_tokens_to_ids("<|endoftext|>")

def make_example(prompt_ids: list[int], text: str):
    pre  = ([SOP] + prompt_ids) if prompt_ids else []
    pfx  = [SOT, LANG, TASK, NOTS]
    body = tok.encode(" " + text.strip(), add_special_tokens=False) + [EOT]
    seq  = pre + pfx + body
    assert len(seq) <= 448
    dec_in = seq[:-1]
    labels = seq[1:]
    n_mask = len(pre) + len(pfx) - 1          # posisi yang memprediksi prompt/prefix
    labels = [-100] * n_mask + labels[n_mask:]  # loss hanya pada teks + EOT
    return dec_in, labels
```

`decoder_input_ids` dan `labels` **keduanya** dikirim ke model. Jika hanya `labels` yang dikirim, Hugging Face akan membuat `decoder_input_ids` dengan menggeser `labels`, sehingga prompt hilang.

### 5.2 Variasi prompt (dipilih acak per sampel, per epoch)

| Variasi | Porsi | Tujuan |
| --- | --- | --- |
| Domain + (CRM) + riwayat penuh dari `prev_text` | 50% | Kondisi normal inferensi |
| Domain + (CRM) + riwayat dipotong acak (0–100% panjang) | 15% | Awal panggilan, riwayat pendek |
| Domain + riwayat dari `text_raw` (belum direview) | 10% | Riwayat yang mengandung kesalahan ASR |
| Domain saja | 10% | Chunk pertama |
| Tanpa prompt | 15% | Tetap jalan jika konteks dimatikan |

- **(CRM)** disertakan dengan probabilitas 0,6 jika sampel punya field `crm`.
- Variasi "riwayat dari `text_raw`" melatih model agar tidak menyalin kesalahan dari riwayat. Saat inferensi, riwayat berasal dari output model sendiri yang bisa salah.
- Variasi dipilih ulang setiap epoch (di collator, bukan saat membuat manifest), sehingga satu segmen muncul dengan konteks berbeda-beda.

### 5.3 Evaluasi selama training

`model.generate(prompt_ids=...)` di Hugging Face hanya menerima satu prompt untuk seluruh batch. Karena itu:

- **Eval di Trainer** memakai prompt domain saja. Prompt ini sama untuk semua sampel, jadi bisa di-batch.
- **Eval konteks penuh** dijalankan offline dengan loop inferensi (bagian 6) pada checkpoint terpilih.

## 6. Inferensi: tiga level

| Level | Cara | Konteks | Kecepatan | Kapan dipakai |
| --- | --- | --- | --- | --- |
| 1 | WhisperX batch + `asr_options={"initial_prompt": DOMAIN}` | Domain saja | Cepat (batch) | Baseline, volume besar |
| 2 | Loop berurutan per chunk dengan `context_builder` | Domain + CRM + riwayat | Lebih lambat | **Target produksi** |
| 3 | Level 2 + diarization dulu, lalu chunk per giliran bicara | + batas speaker | Paling lambat | Eksperimen lanjutan |

### Level 1 – prompt statis (tanpa kode tambahan)

```python
import whisperx
model = whisperx.load_model("outputs/whisper-telemarketing-ct2", "cuda",
                            compute_type="float16", language="id",
                            asr_options={"initial_prompt": DOMAIN_PROMPT})
result = model.transcribe(audio, batch_size=16)
```

### Level 2 – riwayat bergulir (`src/infer_context.py`)

```python
import whisperx
from faster_whisper import WhisperModel
from context_builder import build_prompt

audio = whisperx.load_audio(path)                 # float32, 16 kHz
chunks = get_vad_chunks(audio)                    # [(start, end), ...] ≤ 30 dtk, urut waktu
asr = WhisperModel("outputs/whisper-telemarketing-ct2", device="cuda", compute_type="float16")

def decode(chunk, prompt_ids):
    segs, _ = asr.transcribe(chunk, language="id", initial_prompt=prompt_ids,
                             condition_on_previous_text=False, without_timestamps=True,
                             vad_filter=False, beam_size=5)
    return list(segs)

history, out = "", []
for s, e in chunks:
    chunk = audio[int(s * 16000):int(e * 16000)]
    segs = decode(chunk, build_prompt(tok, DOMAIN_PROMPT, crm, history))
    text = " ".join(x.text.strip() for x in segs)
    if is_bad(segs, text, history):               # penjaga, lihat di bawah
        segs = decode(chunk, build_prompt(tok, DOMAIN_PROMPT, crm, ""))
        text = " ".join(x.text.strip() for x in segs)
        # hasil dekode ulang tetap disimpan, tapi TIDAK masuk riwayat
    elif not is_secret(text):
        history = f"{history} {text}".strip()
    out.append({"start": s, "end": e, "text": text})

align_model, meta = whisperx.load_align_model(language_code="id", device="cuda")
aligned = whisperx.align(out, align_model, meta, audio, "cuda")
diarize = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device="cuda")
result = whisperx.assign_word_speakers(diarize(audio), aligned)
```

Catatan versi (wajib dicek pada versi WhisperX yang dipin):

- `get_vad_chunks`: salin logika VAD + `merge_chunks(chunk_size=30)` dari `FasterWhisperPipeline.transcribe` di versi WhisperX yang dipakai. API internal ini berubah antar versi.
- `initial_prompt` di faster-whisper menerima string atau list token id. Kirim token id dari `context_builder` agar pemotongannya sama dengan saat training.
- Lokasi `DiarizationPipeline` (`whisperx.DiarizationPipeline` atau `whisperx.diarize.DiarizationPipeline`) berbeda antar versi.
- Model alignment untuk `id`: cek apakah WhisperX punya default. Jika tidak, berikan model wav2vec2 Indonesia lewat `model_name`.

### Penjaga (`is_bad`)

Chunk dianggap buruk jika salah satu kondisi berikut terpenuhi:

| Sinyal | Ambang awal | Menangkap |
| --- | --- | --- |
| `compression_ratio` | > 2,4 | Pengulangan atau halusinasi berulang |
| `avg_logprob` | < −1,0 | Model tidak yakin |
| `no_speech_prob` tinggi + teks tidak kosong | > 0,6 | Halusinasi di audio hening |
| Overlap 8-gram dengan riwayat di prompt | ≥ 1 n-gram panjang tersalin | Model menyalin prompt |

Hasil yang buruk tetap dicoba ulang tanpa riwayat. Hasil yang masih buruk ditandai `low_confidence` di output dan tidak dipakai sebagai konteks. Tujuannya agar satu kesalahan tidak menyebar ke seluruh panggilan.

### Throughput dengan 2 GPU

Loop berurutan tidak bisa di-batch per chunk. Paralelisme dilakukan **per panggilan**: jalankan dua worker (satu per GPU, `CUDA_VISIBLE_DEVICES=0` dan `=1`) yang mengambil panggilan dari antrean yang sama. Kecepatan sebenarnya (real-time factor) perlu diukur.

## 7. Mengukur apakah konteks membantu

Jalankan ablation di `test` dan `test_pii` dengan checkpoint yang sama:

| Run | Domain | CRM | Riwayat | Penjaga |
| --- | --- | --- | --- | --- |
| C0 | – | – | – | – |
| C1 | ✓ | – | – | – |
| C2 | ✓ | ✓ | – | – |
| C3 | ✓ | – | ✓ | ✓ |
| C4 | ✓ | ✓ | ✓ | ✓ |
| C5 | ✓ | ✓ | ✓ | – |

Metrik untuk tiap run: WER, entity exact match per kategori, konsistensi ejaan nama dalam satu panggilan, dan jumlah chunk yang ditangkap penjaga. Bandingkan juga per posisi chunk; manfaat konteks seharusnya paling terlihat di bagian tengah dan akhir panggilan. Perbandingan C4 dengan C5 menunjukkan seberapa besar kontribusi penjaga.

## 8. Risiko

| Risiko | Mitigasi |
| --- | --- |
| Kesalahan menyebar lewat riwayat | Penjaga + variasi `text_raw` saat training |
| Model menyalin prompt | Variasi tanpa prompt, deteksi n-gram, `no_speech_prob` |
| Format prompt beda antara training dan inferensi | Satu modul `context_builder` + unit test |
| Nomor lama terulang | Tidak ada memori entitas di versi awal; diukur di ablation |
| PII rahasia masuk konteks | Filter `secret` di riwayat |
| Inferensi lambat | Paralel per panggilan di 2 GPU; Level 1 sebagai fallback |
