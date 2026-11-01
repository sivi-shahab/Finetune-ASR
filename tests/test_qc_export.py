"""Test qc_export: convert (transkrip, meta, crm, audio_list) dan import-audio (dua pola nama file)."""
import csv
import json
import shutil

import numpy as np
import pytest
import soundfile as sf

import build_dataset as bd
from qc_export import convert, import_audio
from test_build_dataset import SCRIPT, WORD_DUR, WORD_GAP

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg tidak ada")


def elevenlabs_words():
    words, t = [], 0.5
    for spk, sentence, pause in SCRIPT:
        for w in sentence.split():
            words.append({"text": w, "start": round(t, 3), "end": round(t + WORD_DUR, 3), "type": "word",
                          "speaker_id": spk, "logprob": -0.05})
            t += WORD_DUR + WORD_GAP
        t += pause
    return words, t


@pytest.fixture
def exported(tmp_path):
    words, dur = elevenlabs_words()
    rows = []
    for i in range(3):
        rows.append({"call_id": f"0101AAA{i}_2026090113501{i}", "result_id": f"r{i}", "campaign": "Cashline",
                     "qc_status": "PASS" if i else None, "qc_tl_status": None, "uploaded_at": "2026-09-01",
                     "file_path": f"/home/ecentrix/recording/2026/09/01/2026090113501{i}-201122-abc{i}.gsm",
                     "customer_name": "SARI WULANDARI", "agent_id": f"AG{i % 2}", "product": "CashLine NTB",
                     "elevenlabs": {"language_code": "ind", "audio_duration_secs": dur, "words": words}})
    src = tmp_path / "qc_export.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))
    raw = tmp_path / "raw"
    return raw, convert(src, raw), rows, dur


def test_convert(exported):
    raw, res, rows, _ = exported
    assert res["calls"] == 3
    assert len(list((raw / "transcripts").glob("*.json"))) == 3
    lst = list(csv.DictReader((raw / "audio_list.csv").open()))
    assert [r["ticket_id"] for r in lst] == [r["call_id"] for r in rows]
    assert lst[0]["file_path"].endswith(".gsm")
    crm = [json.loads(x) for x in (raw / "crm.jsonl").read_text().splitlines()]
    assert crm[0]["customer_name"] == "Sari Wulandari" and crm[0]["product"] == "Cashline"


def test_import_audio_dua_pola_nama_lalu_build(exported, tmp_path):
    raw, _, rows, dur = exported
    src = tmp_path / "ambil_manual"
    src.mkdir()
    audio = (np.random.default_rng(0).standard_normal(int((dur + 0.5) * 8000)) * 0.05).astype(np.float32)
    sf.write(src / f"{rows[0]['call_id']}.wav", audio, 8000)                    # nama = ticket_id
    sf.write(src / "2026090113501" "1-201122-abc1.wav", audio, 8000)            # nama asli server
    sf.write(src / "bukan_tiket.wav", audio, 8000)
    res = import_audio(src, raw, workers=2)
    assert res == {"found": 3, "imported_or_existing": 2, "failed": 0, "unknown_ticket": 1, "still_missing": 1}
    data, sr = sf.read(raw / "audio" / f"{rows[1]['call_id']}.flac")
    assert sr == 8000 and data.ndim == 1

    out = tmp_path / "dataset"
    stats = bd.main(["--raw-dir", str(raw), "--out-dir", str(out), "--test-hours", "0.02", "--val-hours", "0.01", "--workers", "1"])
    assert stats["rejected"].get("missing_pair") == 1        # tiket ketiga belum ada audionya
    segs = [json.loads(x) for f in ("train", "rejected", "review_queue") for x in (out / f"{f}.jsonl").read_text().splitlines()]
    seg = next(s for s in segs if s.get("meta"))
    assert seg["meta"]["campaign"] == "Cashline" and "agent_id" in seg["meta"]
    assert seg["crm"]["customer_name"] == "Sari Wulandari"
