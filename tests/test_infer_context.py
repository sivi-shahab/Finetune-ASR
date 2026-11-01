"""Test logika loop konteks (ContextDecoder) dengan ASR tiruan; tidak butuh WhisperX."""
from dataclasses import dataclass

from infer_context import ContextDecoder, Layers, copied_from_prompt


@dataclass
class Seg:
    text: str
    compression_ratio: float = 1.5
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.01


class CharHF:
    """Meniru tokenizers.Tokenizer: 1 karakter = 1 token."""
    SPECIAL = {"<|startofprev|>": 50361}

    class Enc:
        def __init__(self, ids):
            self.ids = ids

    def get_vocab_size(self):
        return 51866

    def encode(self, text, add_special_tokens=False):
        return self.Enc([ord(c) for c in text])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def token_to_id(self, tok):
        return self.SPECIAL.get(tok)


class FakeASR:
    """Mengembalikan output terjadwal per panggilan dan mencatat prompt yang diterima."""
    hf_tokenizer = CharHF()

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def transcribe(self, audio, initial_prompt=None, **kw):
        self.prompts.append("".join(chr(i) for i in (initial_prompt or [])))
        return [self.outputs.pop(0)], None


def test_riwayat_bertambah_dan_masuk_prompt():
    asr = FakeASR([Seg("Halo Bu."), Seg("Limitnya Rp15.000.000.")])
    res = ContextDecoder(asr, "Domain.", Layers()).run([("a", None), ("b", None)], {"customer_name": "Sari"})
    assert [r["text"] for r in res] == ["Halo Bu.", "Limitnya Rp15.000.000."]
    assert asr.prompts[0] == " Domain. Nasabah: Sari."
    assert asr.prompts[1] == " Domain. Nasabah: Sari. Halo Bu."


def test_hasil_buruk_didekode_ulang_tanpa_riwayat_dan_tidak_jadi_konteks():
    asr = FakeASR([Seg("Halo Bu."), Seg("ulang ulang ulang", compression_ratio=3.0), Seg("Iya."), Seg("Baik.")])
    res = ContextDecoder(asr, "D.", Layers(crm=False)).run([("a", 0), ("b", 0), ("c", 0)], None)
    assert res[1]["retried"] and res[1]["text"] == "Iya."
    assert asr.prompts[2] == " D."                 # dekode ulang tanpa riwayat
    assert asr.prompts[3] == " D. Halo Bu."        # "Iya." (hasil ulang) tidak masuk riwayat


def test_teks_rahasia_tidak_masuk_riwayat():
    asr = FakeASR([Seg("Kode OTP nya 123456."), Seg("Baik.")])
    res = ContextDecoder(asr, "D.", Layers()).run([("a", 0), ("b", 0)], None)
    assert res[0]["secret"]
    assert asr.prompts[1] == " D."


def test_ablation_tanpa_lapisan():
    asr = FakeASR([Seg("Halo."), Seg("Baik.")])
    ContextDecoder(asr, "D.", Layers(domain=False, crm=False, history=False)).run([("a", 0), ("b", 0)],
                                                                                   {"customer_name": "X"})
    assert asr.prompts == ["", ""]


def test_deteksi_salin_prompt():
    hist = "satu dua tiga empat lima enam tujuh delapan sembilan"
    assert copied_from_prompt("lalu satu dua tiga empat lima enam tujuh delapan", hist)
    assert not copied_from_prompt("satu dua tiga", hist)
