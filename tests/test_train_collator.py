"""Test ContextCollator dengan processor Whisper asli (dilewati jika belum ada di cache HF)."""
import numpy as np
import pytest
import soundfile as sf

from context_builder import MAX_DECODER_TOKENS


def _processor():
    try:
        from transformers import WhisperProcessor
        proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3", local_files_only=True,
                                                language="indonesian", task="transcribe")
    except Exception:
        return None
    # transformers 5 membuat tokenizer kosong (tanpa error) bila file tidak ada di cache.
    return proc if len(proc.tokenizer) > 50_000 else None


PROC = _processor()
pytestmark = pytest.mark.skipif(PROC is None, reason="processor Whisper belum ada di cache")


@pytest.fixture
def batch(tmp_path):
    from train import ContextCollator
    rows = []
    for i in range(4):
        path = tmp_path / f"c{i}.flac"
        sf.write(path, (np.random.default_rng(i).standard_normal(16000 * (3 + i)) * 0.05).astype(np.float32),
                 16000, format="FLAC")
        rows.append({"id": f"C_{i:04d}", "audio_filepath": str(path), "text": "Limitnya Rp15.000.000, Bu.",
                     "prev_text": "Selamat pagi, Bu. " * (i * 30), "prev_text_raw": "selamat pagi bu",
                     "crm": {"customer_name": "Sari Wulandari"}, "pii_types": ["nominal"], "_eval": i % 2 == 0})
    return ContextCollator(PROC, "Telemarketing Bank Mega."), rows


def test_bentuk_batch(batch):
    collate, rows = batch
    out = collate(rows)
    assert out["input_features"].shape[:2] == (4, 128)        # large-v3: 128 mel
    assert out["input_features"].shape[2] == 3000             # selalu dipad ke 30 detik
    assert out["decoder_input_ids"].shape == out["labels"].shape == out["decoder_attention_mask"].shape
    assert out["decoder_input_ids"].shape[1] <= MAX_DECODER_TOKENS


def test_label_sama_dengan_teks_target(batch):
    collate, rows = batch
    out = collate(rows)
    tok = PROC.tokenizer
    for i in range(4):
        lab = [x for x in out["labels"][i].tolist() if x != -100]
        assert tok.decode(lab, skip_special_tokens=True).strip() == "Limitnya Rp15.000.000, Bu."
        # token sebelum label pertama adalah <|notimestamps|>
        first = next(k for k, x in enumerate(out["labels"][i].tolist()) if x != -100)
        assert out["decoder_input_ids"][i, first].item() == tok.convert_tokens_to_ids("<|notimestamps|>")


def test_eval_deterministik(batch):
    collate, rows = batch
    a = collate([rows[0], rows[2]])["decoder_input_ids"]
    b = collate([rows[0], rows[2]])["decoder_input_ids"]
    assert a.equal(b)
