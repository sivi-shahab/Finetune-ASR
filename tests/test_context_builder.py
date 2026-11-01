"""Test context_builder. Memakai tokenizer tiruan level karakter agar berjalan offline;
test dengan tokenizer Whisper asli dijalankan bila modelnya sudah ada di cache HF."""
import random

import pytest

from context_builder import (MAX_DECODER_TOKENS, MAX_PROMPT_TOKENS, DOMAIN_MAX_TOKENS, SPECIAL_TOKENS,
                             append_history, build_decoder_sequence, build_prompt_ids,
                             load_domain_prompt, sample_prompt)


class CharTokenizer:
    """1 karakter = 1 token. Meniru antarmuka tokenizer transformers."""
    SPECIAL = {tok: 100_000 + i for i, tok in enumerate(SPECIAL_TOKENS.values())}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i < 100_000)

    def convert_tokens_to_ids(self, tok):
        return self.SPECIAL[tok]


TOK = CharTokenizer()
DOMAIN = "Telemarketing Bank Mega."
SP = CharTokenizer.SPECIAL


def text_of(ids):
    return TOK.decode(ids)


# --- build_prompt_ids -----------------------------------------------------

def test_prompt_tidak_melebihi_budget():
    history = "kata " * 500
    ids = build_prompt_ids(TOK, DOMAIN, None, history)
    assert len(ids) <= MAX_PROMPT_TOKENS


def test_riwayat_terbaru_selalu_dipertahankan():
    history = " ".join(f"kalimat{i}" for i in range(200))
    ids = build_prompt_ids(TOK, DOMAIN, None, history)
    assert text_of(ids).endswith("kalimat199")
    assert text_of(ids).startswith(" " + DOMAIN)


def test_riwayat_dipotong_di_batas_kata():
    history = " ".join(f"kata{i:03d}" for i in range(100))
    out = text_of(build_prompt_ids(TOK, DOMAIN, None, history, max_tokens=60))
    hist = out[len(" " + DOMAIN):].split()
    assert all(w.startswith("kata") and len(w) == 7 for w in hist)


def test_crm_masuk_setelah_domain():
    crm = {"customer_name": "Sari Wulandari", "product": "kenaikan limit"}
    out = text_of(build_prompt_ids(TOK, DOMAIN, crm, "Halo Bu."))
    assert out == f" {DOMAIN} Nasabah: Sari Wulandari. Produk: kenaikan limit. Halo Bu."


def test_tanpa_lapisan_apa_pun():
    assert build_prompt_ids(TOK, "", None, "") == []


# --- append_history -------------------------------------------------------

def test_secret_dan_low_confidence_tidak_masuk_riwayat():
    h = append_history("", "Halo Bu.")
    assert append_history(h, "kode OTP nya 1234", secret=True) == "Halo Bu."
    assert append_history(h, "halusinasi", low_confidence=True) == "Halo Bu."
    assert append_history(h, "Baik.") == "Halo Bu. Baik."


# --- build_decoder_sequence -----------------------------------------------

def test_masking_label_hanya_pada_teks():
    prompt = TOK.encode(" konteks")
    dec_in, labels = build_decoder_sequence(TOK, prompt, "halo")
    assert len(dec_in) == len(labels)
    assert dec_in[0] == SP["<|startofprev|>"]
    body = TOK.encode(" halo") + [SP["<|endoftext|>"]]
    assert [x for x in labels if x != -100] == body
    # token terakhir input = token sebelum EOT; label terakhir = EOT
    assert labels[-1] == SP["<|endoftext|>"]
    # posisi label pertama yang dihitung tepat setelah <|notimestamps|>
    first = next(i for i, x in enumerate(labels) if x != -100)
    assert dec_in[first] == SP["<|notimestamps|>"]


def test_tanpa_prompt_tidak_ada_startofprev():
    dec_in, labels = build_decoder_sequence(TOK, [], "halo")
    assert dec_in[0] == SP["<|startoftranscript|>"]
    assert labels[:3] == [-100, -100, -100]


def test_prompt_dipangkas_agar_muat():
    prompt = TOK.encode(" x" * 300)
    dec_in, _ = build_decoder_sequence(TOK, prompt, "teks " * 40)
    assert len(dec_in) <= MAX_DECODER_TOKENS


def test_teks_terlalu_panjang_ditolak():
    with pytest.raises(ValueError):
        build_decoder_sequence(TOK, [], "a" * 500)


# --- sample_prompt --------------------------------------------------------

def test_distribusi_variasi_prompt():
    rng = random.Random(0)
    ex = {"prev_text": "a b c d", "prev_text_raw": "a b c x", "crm": {"customer_name": "X"}}
    counts = {}
    for _ in range(20_000):
        v = sample_prompt(ex, DOMAIN, rng).variant
        counts[v] = counts.get(v, 0) + 1
    for name, p in [("full", .50), ("truncated", .15), ("noisy", .10), ("domain_only", .10), ("none", .15)]:
        assert abs(counts[name] / 20_000 - p) < 0.02, name


def test_variasi_none_benar_benar_kosong():
    rng = random.Random(0)
    ex = {"prev_text": "a b", "crm": {"customer_name": "X"}}
    for _ in range(200):
        c = sample_prompt(ex, DOMAIN, rng)
        if c.variant == "none":
            assert build_prompt_ids(TOK, c.domain, c.crm, c.history) == []
            return
    pytest.fail("variasi none tidak pernah muncul")


def test_variasi_noisy_memakai_teks_mentah():
    rng = random.Random(1)
    ex = {"prev_text": "Rp15.000.000", "prev_text_raw": "lima belas juta"}
    seen = {sample_prompt(ex, DOMAIN, rng).history for _ in range(300)}
    assert "lima belas juta" in seen


# --- Tokenizer Whisper asli (opsional) ------------------------------------

def _whisper_tokenizer():
    try:
        from transformers import WhisperTokenizerFast
        tok = WhisperTokenizerFast.from_pretrained("openai/whisper-large-v3", local_files_only=True)
    except Exception:
        return None
    # transformers 5 membuat tokenizer kosong (tanpa error) bila file tidak ada di cache.
    return tok if len(tok) > 50_000 else None


@pytest.mark.skipif(_whisper_tokenizer() is None, reason="tokenizer Whisper belum ada di cache")
def test_domain_prompt_dalam_budget():
    tok = _whisper_tokenizer()
    ids = build_prompt_ids(tok, load_domain_prompt(), None, "")
    assert len(ids) <= DOMAIN_MAX_TOKENS
