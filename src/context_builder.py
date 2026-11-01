"""Satu-satunya tempat prompt konteks Whisper disusun (docs/CONTEXT_ENGINEERING.md).

Dipakai oleh collator training (src/train.py) dan loop inferensi (src/infer_context.py),
sehingga format prompt di kedua sisi selalu identik.

Lapisan prompt, dari prioritas tertinggi:
    1. domain  : teks tetap dari configs/domain_prompt.txt
    2. sesi    : data CRM (nama nasabah, produk)
    3. riwayat : ekor transkrip panggilan sejauh ini, dipotong dari depan
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

MAX_PROMPT_TOKENS = 223    # batas prompt Whisper: n_text_ctx // 2 - 1
MAX_DECODER_TOKENS = 448   # n_text_ctx Whisper
DOMAIN_MAX_TOKENS = 40

SPECIAL_TOKENS = {
    "sop": "<|startofprev|>",
    "sot": "<|startoftranscript|>",
    "lang": "<|id|>",
    "task": "<|transcribe|>",
    "notimestamps": "<|notimestamps|>",
    "eot": "<|endoftext|>",
}

# Variasi prompt saat training (CONTEXT_ENGINEERING.md bagian 5.2).
PROMPT_VARIANTS = (
    ("full", 0.50),
    ("truncated", 0.15),
    ("noisy", 0.10),
    ("domain_only", 0.10),
    ("none", 0.15),
)
CRM_PROB = 0.6


# ---------------------------------------------------------------------------
# Adapter tokenizer: transformers (training) dan tokenizers.Tokenizer (faster-whisper)
# ---------------------------------------------------------------------------


class Tok:
    """Antarmuka minimal: encode tanpa token spesial, decode, dan id token spesial."""

    def __init__(self, tokenizer):
        self._t = tokenizer
        self._hf = hasattr(tokenizer, "convert_tokens_to_ids")
        size = len(tokenizer) if hasattr(tokenizer, "__len__") else getattr(tokenizer, "get_vocab_size", lambda: None)()
        if size is not None and size < 50_000:
            # transformers 5 membuat tokenizer kosong tanpa error bila file tidak ditemukan.
            raise ValueError(f"tokenizer hanya punya {size} token; file tokenizer Whisper tidak termuat")

    def encode(self, text: str) -> list[int]:
        if self._hf:
            return list(self._t.encode(text, add_special_tokens=False))
        return list(self._t.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self._t.decode(ids)

    def special(self, name: str) -> int:
        tok = SPECIAL_TOKENS[name]
        i = self._t.convert_tokens_to_ids(tok) if self._hf else self._t.token_to_id(tok)
        if i is None:
            raise KeyError(f"token {tok} tidak ada di tokenizer")
        return i


def as_tok(tokenizer) -> Tok:
    return tokenizer if isinstance(tokenizer, Tok) else Tok(tokenizer)


def load_domain_prompt(path: str | Path = Path(__file__).parent.parent / "configs" / "domain_prompt.txt") -> str:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    return " ".join(ln for ln in lines if ln and not ln.startswith("#"))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def crm_text(crm: dict | None) -> str:
    if not crm:
        return ""
    parts = []
    if crm.get("customer_name"):
        parts.append(f"Nasabah: {crm['customer_name']}.")
    if crm.get("product"):
        parts.append(f"Produk: {crm['product']}.")
    return " ".join(parts)


def tail_ids(tok: Tok, text: str, room: int) -> list[int]:
    """Token ekor `text` sebanyak <= room, dipotong pada batas kata."""
    text = " ".join(text.split())
    if not text or room <= 0:
        return []
    ids = tok.encode(" " + text)
    if len(ids) <= room:
        return ids
    words = tok.decode(ids[-room:]).split(" ")
    # Kata pertama bisa terpotong di tengah BPE: buang, lalu pastikan muat setelah encode ulang.
    words = [w for w in words[1:] if w]
    while words:
        ids = tok.encode(" " + " ".join(words))
        if len(ids) <= room:
            return ids
        words = words[1:]
    return []


def build_prompt_ids(tokenizer, domain: str, crm: dict | None, history: str,
                     max_tokens: int = MAX_PROMPT_TOKENS) -> list[int]:
    """Token id prompt (tanpa <|startofprev|>), maksimal max_tokens.

    Domain dan CRM tidak pernah dipotong kecuali melebihi budget sendiri;
    riwayat mengisi sisa budget dan dipotong dari depan.
    """
    tok = as_tok(tokenizer)
    head_text = " ".join(p for p in (domain.strip(), crm_text(crm)) if p)
    head = tok.encode(" " + head_text) if head_text else []
    if len(head) >= max_tokens:
        return head[:max_tokens]
    return head + tail_ids(tok, history, max_tokens - len(head))


def append_history(history: str, text: str, *, secret: bool = False, low_confidence: bool = False) -> str:
    """Tambahkan teks chunk ke riwayat, kecuali teks rahasia atau hasil yang tidak lolos penjaga."""
    if secret or low_confidence or not text.strip():
        return history
    return f"{history} {text.strip()}".strip()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class PromptChoice:
    variant: str
    domain: str
    crm: dict | None
    history: str


def sample_prompt(example: dict, domain: str, rng: random.Random) -> PromptChoice:
    """Pilih variasi prompt untuk satu sampel training (dipanggil di collator, tiap epoch)."""
    r, acc, variant = rng.random(), 0.0, PROMPT_VARIANTS[-1][0]
    for name, p in PROMPT_VARIANTS:
        acc += p
        if r < acc:
            variant = name
            break
    crm = example.get("crm") if rng.random() < CRM_PROB else None
    prev = example.get("prev_text") or ""
    if variant == "full":
        return PromptChoice(variant, domain, crm, prev)
    if variant == "truncated":
        words = prev.split()
        keep = int(len(words) * rng.random())
        return PromptChoice(variant, domain, crm, " ".join(words[len(words) - keep:]) if keep else "")
    if variant == "noisy":
        return PromptChoice(variant, domain, None, example.get("prev_text_raw") or prev)
    if variant == "domain_only":
        return PromptChoice(variant, domain, None, "")
    return PromptChoice(variant, "", None, "")


def build_decoder_sequence(tokenizer, prompt_ids: list[int], text: str,
                           max_len: int = MAX_DECODER_TOKENS) -> tuple[list[int], list[int]]:
    """Susun (decoder_input_ids, labels) untuk satu sampel.

    Urutan: [<|startofprev|> prompt] <|startoftranscript|><|id|><|transcribe|><|notimestamps|> teks <|endoftext|>
    Loss hanya pada token teks dan <|endoftext|>; posisi lain diberi -100.
    Jika terlalu panjang, prompt dipotong dari depan.
    """
    tok = as_tok(tokenizer)
    prefix = [tok.special(k) for k in ("sot", "lang", "task", "notimestamps")]
    body = tok.encode(" " + text.strip()) + [tok.special("eot")]
    room = max_len - len(prefix) - len(body) - 1   # -1 untuk <|startofprev|>
    if len(prefix) + len(body) > max_len:
        raise ValueError(f"teks terlalu panjang: {len(body)} token")
    prompt_ids = prompt_ids[-room:] if room > 0 else []
    pre = [tok.special("sop")] + prompt_ids if prompt_ids else []
    seq = pre + prefix + body
    n_mask = len(pre) + len(prefix) - 1
    labels = seq[1:]
    return seq[:-1], [-100] * n_mask + labels[n_mask:]
