"""Test end-to-end build_dataset dengan data sintetis (JSON format ElevenLabs + audio noise)."""
import json

import numpy as np
import pytest
import soundfile as sf

import build_dataset as bd

SCRIPT = [
    # (speaker, kalimat, jeda setelahnya dalam detik)
    ("speaker_1", "Halo?", 0.6),
    ("speaker_0", "Selamat pagi, dengan Ibu Sari Wulandari? Saya Dina dari Bank Mega, Bu.", 0.5),
    ("speaker_1", "Iya, saya sendiri.", 0.5),
    ("speaker_0", "Ibu ada penawaran kenaikan limit kartu kredit, limitnya jadi lima belas juta.", 0.4),
    ("speaker_0", "Untuk verifikasi, tanggal lahir Ibu?", 0.5),
    ("speaker_1", "Tujuh belas Agustus sembilan belas delapan puluh lima.", 0.5),
    ("speaker_0", "Nomor HP yang terdaftar?", 0.4),
    ("speaker_1", "Nol delapan satu dua, tiga empat lima enam, tujuh delapan sembilan nol.", 0.6),
    ("speaker_0", "Baik, terima kasih Ibu, nanti kartunya dikirim ke alamat terdaftar ya.", 0.5),
    ("speaker_1", "Oke, makasih Mbak.", 0.3),
]
WORD_DUR, WORD_GAP = 0.35, 0.05


def make_call(raw, call_id, rng):
    words, t = [], 0.5
    for spk, sentence, pause in SCRIPT:
        for k, w in enumerate(sentence.split()):
            words.append({"text": w, "start": round(t, 3), "end": round(t + WORD_DUR, 3), "type": "word",
                          "speaker_id": spk, "logprob": -0.05})
            t += WORD_DUR + WORD_GAP
            words.append({"text": " ", "start": round(t - WORD_GAP, 3), "end": round(t, 3), "type": "spacing",
                          "speaker_id": spk})
        t += pause
    (raw / "transcripts" / f"{call_id}.json").write_text(
        json.dumps({"language_code": "ind", "language_probability": 0.99, "text": "", "words": words}))
    audio = (rng.standard_normal(int((t + 0.5) * 8000)) * 0.05).astype(np.float32)
    sf.write(raw / "audio" / f"{call_id}.wav", np.stack([audio, audio], axis=1), 8000)  # 8 kHz stereo


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw"
    (raw / "audio").mkdir(parents=True)
    (raw / "transcripts").mkdir()
    rng = np.random.default_rng(0)
    for i in range(12):
        make_call(raw, f"CALL{i:03d}", rng)
    (raw / "audio" / "ORPHAN.wav").write_bytes(b"")
    (raw / "crm.jsonl").write_text(json.dumps({"call_id": "CALL000", "customer_name": "Sari Wulandari"}) + "\n")
    out = tmp_path / "dataset"
    stats = bd.main(["--raw-dir", str(raw), "--out-dir", str(out), "--test-hours", "0.02", "--val-hours", "0.01",
                     "--pii-policy", "review", "--workers", "2"])
    return out, stats


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def test_output_lengkap(dataset):
    out, stats = dataset
    for name in ("train", "validation", "test", "test_pii", "rejected", "review_queue"):
        assert (out / f"{name}.jsonl").exists()
    assert stats["rejected"]["missing_pair"] == 1


def test_segmen_dalam_batas_dan_clip_16k(dataset):
    out, _ = dataset
    for r in rows(out / "train.jsonl"):
        assert 1.0 <= r["duration"] <= 30.0
        data, sr = sf.read(r["audio_filepath"])
        assert sr == 16000 and data.ndim == 1


def test_segmen_pii_belum_review_tidak_masuk_train(dataset):
    out, _ = dataset
    assert all(not r["pii_types"] for r in rows(out / "train.jsonl"))
    queue = rows(out / "review_queue.jsonl")
    assert any(r["pii_types"] for r in queue)
    # validation/test kosong sebelum review
    assert rows(out / "validation.jsonl") == [] and rows(out / "test.jsonl") == []


def test_split_per_call_tidak_bocor(dataset):
    out, _ = dataset
    calls = {}
    for r in rows(out / "rejected.jsonl"):
        if "split" in r:
            calls.setdefault(r["call_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in calls.values())


def test_deretan_angka_tidak_terpotong(dataset):
    out, _ = dataset
    for r in rows(out / "rejected.jsonl") + rows(out / "train.jsonl"):
        if "081234567890" in r.get("prev_text", "") or "Nol" in r.get("text_raw", ""):
            if "Nol" in r.get("text_raw", ""):
                assert "081234567890" in r["text"], r["text"]


def test_prev_text_berurutan(dataset):
    out, _ = dataset
    segs = sorted((r for r in rows(out / "rejected.jsonl") + rows(out / "train.jsonl")
                   if r.get("call_id") == "CALL001" and "seg_index" in r), key=lambda r: r["seg_index"])
    assert segs[0]["prev_text"] == ""
    for a, b in zip(segs, segs[1:]):
        assert b["prev_text"].endswith(a["text"])


def test_review_membuka_segmen_pii(dataset, tmp_path):
    out, _ = dataset
    queue = rows(out / "review_queue.jsonl")
    reviews = [{"id": r["id"], "start": r["start"], "end": r["end"], "text": r["text"], "reviewed": True}
               for r in queue]
    (out / "reviews.jsonl").write_text("\n".join(json.dumps(r) for r in reviews) + "\n")
    stats = bd.main(["--raw-dir", str(tmp_path / "raw"), "--out-dir", str(out), "--test-hours", "0.02", "--val-hours", "0.01",
                     "--pii-policy", "review", "--workers", "2"])
    assert stats["review_queue"] == 0
    train = rows(out / "train.jsonl")
    assert any("Rp15.000.000" in r["text"] for r in train)
    assert stats["splits"]["validation"]["segments"] + stats["splits"]["test"]["segments"] > 0


# --- Aturan dari data asli -------------------------------------------------

def test_kata_bertimestamp_rusak_diisolasi():
    W = bd.Word
    words = [W("halo", 0.0, 0.3, "s0", -0.1), W("Bu", 0.35, 0.6, "s0", -0.1),
             W("rusak", 0.65, 120.0, "s0", -0.1),                       # 119 detik: timestamp rusak
             W("baik", 120.05, 120.4, "s1", -0.1), W("ya", 120.45, 120.7, "s1", -0.1)]
    spans = bd.segment_words(words, __import__("random").Random(0))
    assert (2, 3) in spans                                              # kata rusak sendirian


def test_assign_splits_target_jam_dan_agent_utuh():
    keys = {f"c{i}": f"agent:A{i % 10}" for i in range(200)}            # 10 agent, @20 panggilan
    hours = {c: 0.1 for c in keys}                                      # 2 jam per agent
    sp = bd.assign_splits(keys, hours, seed=1, test_hours=2.0, val_hours=2.0)
    by_split = {}
    for c, s in sp.items():
        by_split.setdefault(s, set()).add(keys[c])
    assert len(by_split["test"]) == 1 and len(by_split["validation"]) == 1
    assert not (by_split["test"] & by_split["train"])                   # agent tidak bocor


def test_rahasia_dijawab_di_segmen_berikutnya(tmp_path):
    raw = tmp_path / "raw"
    (raw / "audio").mkdir(parents=True)
    (raw / "transcripts").mkdir()
    lines = [("speaker_0", "Untuk verifikasi, boleh sebutkan kode OTP nya Bu?", 3.5),
             ("speaker_1", "Satu dua tiga empat lima enam.", 3.5),
             ("speaker_0", "Baik terima kasih, silakan ketik PIN spasi enam angka.", 0.5)]
    words, t = [], 0.5
    for spk, sent, pause in lines:
        for w in sent.split():
            words.append({"text": w, "start": round(t, 3), "end": round(t + 0.35, 3), "type": "word",
                          "speaker_id": spk, "logprob": -0.05})
            t += 0.4
        t += pause
    (raw / "transcripts" / "S1.json").write_text(json.dumps({"language_code": "ind", "words": words}))
    sf.write(raw / "audio" / "S1.wav", (np.random.default_rng(0).standard_normal(int((t + 1) * 8000)) * 0.05)
             .astype(np.float32), 8000)
    out = tmp_path / "ds"
    bd.main(["--raw-dir", str(raw), "--out-dir", str(out), "--test-hours", "0", "--val-hours", "0", "--workers", "1"])
    segs = sorted(rows(out / "rejected.jsonl") + rows(out / "train.jsonl"), key=lambda r: r.get("seg_index", 0))
    secret = {r["seg_index"]: "secret" in r.get("reject", []) for r in segs}
    assert secret == {0: False, 1: True, 2: False}                      # jawaban OTP dibuang, instruksi PIN tidak
    assert "123456" not in segs[2]["prev_text"]


def test_pii_policy_confidence(tmp_path):
    raw = tmp_path / "raw"
    (raw / "audio").mkdir(parents=True)
    (raw / "transcripts").mkdir()
    make_call(raw, "CALL000", np.random.default_rng(0))         # logprob semua kata -0.05
    out = tmp_path / "ds"
    args = ["--raw-dir", str(raw), "--out-dir", str(out), "--test-hours", "0", "--val-hours", "0", "--workers", "1"]
    bd.main(args + ["--pii-policy", "review"])
    assert not any(r["pii_types"] for r in rows(out / "train.jsonl"))
    bd.main(args)                                               # default = confidence, ambang -0.1
    pii = [r for r in rows(out / "train.jsonl") if r["pii_types"]]
    assert pii and all(r["pii_auto"] and not r["reviewed"] for r in pii)
    bd.main(args + ["--pii-policy", "confidence", "--pii-min-logprob", "-0.01"])   # ambang lebih ketat
    assert not any(r["pii_types"] for r in rows(out / "train.jsonl"))
