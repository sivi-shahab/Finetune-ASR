"""Bangun dataset training dari audio + transkrip ElevenLabs (docs/SPEC.md bagian 2-7).

Contoh:
    python src/build_dataset.py --raw-dir raw --out-dir dataset --tokenizer openai/whisper-large-v3

Input:
    raw/audio/<call_id>.wav, raw/transcripts/<call_id>.json
    raw/audio/<call_id>.flac     (alternatif .wav; mis. hasil src/qc_export.py)
    raw/crm.jsonl                (opsional) {"call_id": ..., "customer_name": ..., "product": ...}
    raw/meta.jsonl               (opsional) {"call_id", "agent_id", "campaign", "qc_status", ...}
    dataset/reviews.jsonl        (opsional) hasil review manusia, ditulis src/review_server.py
Output (di --out-dir):
    clips/<call_id>/<id>.flac, train|validation|test|test_pii.jsonl,
    rejected.jsonl, review_queue.jsonl, stats.json

Segmentasi deterministik (seed + call_id), sehingga id segmen stabil antar-run
selama data dan parameter tidak berubah. Review dicocokkan dengan id + start/end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from normalize import (DIGIT_WORDS, REPEAT_WORDS, SCALES, SE_SCALES, UNITS, detect_pii, has_secret_value,
                       normalize)

log = logging.getLogger("build_dataset")
SR = 16000

# Parameter segmentasi (SPEC bagian 4)
MIN_PAUSE = 0.3          # detik; jeda minimum untuk titik potong
HARD_SPLIT_PAUSE = 3.0   # jeda yang selalu memisah segmen
LONG_WORD = 3.0          # kata lebih lama dari ini = timestamp ElevenLabs rusak (menelan musik/hening)
PAD = 0.1
MIN_DUR, MAX_DUR = 1.0, 30.0
TARGET_MIN, TARGET_MAX = 10.0, 28.0
SHORT_PROB, SHORT_MIN, SHORT_MAX = 0.15, 1.0, 5.0
PREV_TEXT_MAX_CHARS = 1500   # cukup untuk > 223 token; pemotongan final di context_builder

# QC (SPEC bagian 7)
CPS_MIN, CPS_MAX = 4.0, 25.0
MIN_LOGPROB = -0.5
RMS_MIN_DBFS = -50.0
CLIP_FRAC_MAX = 0.01
MAX_TEXT_TOKENS = 448 - 5   # prefix 4 token + <|endoftext|>
SPLITS = (("train", None), ("validation", None), ("test", None))   # target jam: --test-hours/--val-hours
# Penolakan yang bisa diselesaikan dengan review manusia
REVIEWABLE = {"pii_unreviewed", "eval_unreviewed", "low_confidence", "cps_outlier"}

NUMBER_WORDS = (set(UNITS) | set(DIGIT_WORDS) | set(REPEAT_WORDS) | set(SCALES) | set(SE_SCALES)
                | {"belas", "puluh", "ratus", "sepuluh", "sebelas", "seratus", "koma", "setengah"})


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str
    logprob: float | None


@dataclass
class Segment:
    id: str
    audio_filepath: str
    call_id: str
    seg_index: int
    start: float
    end: float
    duration: float
    prev_text: str
    prev_text_raw: str
    text: str
    text_raw: str
    crm: dict | None
    speakers: list[str]
    n_turns: int
    pii_types: list[str]
    language: str
    sample_rate: int
    mean_logprob: float | None
    min_logprob: float | None
    entity_min_logprob: float | None   # min logprob kata angka; tanpa kata angka = min_logprob
    reviewed: bool
    split: str
    secret: bool = False
    needs_review: list[str] = field(default_factory=list)
    meta: dict | None = None
    human_rejected: bool = False
    bad_timing: bool = False
    pii_auto: bool = False        # PII diterima tanpa review (kebijakan confidence); diaudit lewat spotcheck
    reject: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def load_words(path: Path) -> tuple[list[Word], str, float | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = []
    for w in data.get("words", []):
        if w.get("type", "word") != "word":
            continue   # spacing & audio_event tidak masuk teks
        if w.get("start") is None or w.get("end") is None:
            raise ValueError(f"kata tanpa timestamp: {w.get('text')!r}")
        words.append(Word(w["text"].strip(), float(w["start"]), float(w["end"]),
                          w.get("speaker_id") or "speaker_0", w.get("logprob")))
    words = [w for w in words if w.text]
    words.sort(key=lambda w: w.start)
    return words, data.get("language_code", ""), data.get("audio_duration_secs")


def speaker_roles(words: list[Word]) -> dict[str, str]:
    """Speaker dengan total waktu bicara terlama = agent (telemarketing: agent dominan).
    Bicara pertama tidak dipakai karena di panggilan keluar nasabah biasanya menyapa duluan."""
    talk = Counter()
    for w in words:
        talk[w.speaker] += w.end - w.start
    roles = {}
    for i, (spk, _) in enumerate(talk.most_common()):
        roles[spk] = "agent" if i == 0 else "customer"
    return roles


AUDIO_EXTS = (".wav", ".flac")


def audio_path(raw: Path, call_id: str) -> Path:
    for ext in AUDIO_EXTS:
        p = raw / "audio" / f"{call_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"audio {call_id} tidak ada")


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    channels = audio.shape[1]
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR)
    return audio.astype(np.float32), channels


# ---------------------------------------------------------------------------
# Segmentasi
# ---------------------------------------------------------------------------


def is_numberish(text: str) -> bool:
    t = re.sub(r"[^\w]", "", text.lower())
    return bool(t) and (t in NUMBER_WORDS or t.isdigit() or t == "rp")


def cut_allowed(words: list[Word], k: int) -> bool:
    """Boleh memotong di antara kata k dan k+1?"""
    a, b = words[k], words[k + 1]
    gap = b.start - a.end
    if gap >= HARD_SPLIT_PAUSE or a.end - a.start > LONG_WORD or b.end - b.start > LONG_WORD:
        return True   # kata bertimestamp rusak diisolasi agar hanya satu segmen yang terbuang
    if is_numberish(a.text) and is_numberish(b.text):
        return False   # jangan memotong deretan angka walaupun ada jeda
    return gap >= MIN_PAUSE or a.speaker != b.speaker


def segment_words(words: list[Word], rng: random.Random) -> list[tuple[int, int]]:
    """Kembalikan daftar (i, j) inklusif-eksklusif indeks kata per segmen."""
    spans, i, n = [], 0, len(words)
    while i < n:
        short = rng.random() < SHORT_PROB
        target = rng.uniform(SHORT_MIN, SHORT_MAX) if short else rng.uniform(TARGET_MIN, TARGET_MAX)
        last_cut, j = None, i
        while j + 1 < n:
            if (words[j + 1].start - words[j].end >= HARD_SPLIT_PAUSE
                    or words[j].end - words[j].start > LONG_WORD
                    or words[j + 1].end - words[j + 1].start > LONG_WORD):
                last_cut = j
                break
            dur_next = words[j + 1].end - words[i].start + 2 * PAD
            if cut_allowed(words, j):
                last_cut = j
                if words[j].end - words[i].start >= target:
                    break
            if dur_next > MAX_DUR:
                break
            j += 1
        if j + 1 >= n:
            last_cut = n - 1
        if last_cut is None:
            # Tidak ada titik potong sah dalam 30 detik: potong di jeda terbesar.
            gaps = [(words[k + 1].start - words[k].end, k) for k in range(i, max(i + 1, j))]
            last_cut = max(gaps)[1] if gaps else i
        spans.append((i, last_cut + 1))
        i = last_cut + 1
    return spans


# ---------------------------------------------------------------------------
# Per panggilan
# ---------------------------------------------------------------------------


def _u(key: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(call_keys: dict[str, str], call_hours: dict[str, float], seed: int,
                  test_hours: float, val_hours: float) -> dict[str, str]:
    """Split per kelompok (agent atau panggilan) dengan target JAM untuk test & validation.

    Kelompok diurutkan acak-deterministik lalu diisi ke test sampai test_hours, kemudian
    validation sampai val_hours, sisanya train. Kelompok yang akan membuat split melebihi
    1,5x target dilewati (tetap train), agar satu agent besar tidak membengkakkan beban review.
    """
    groups: dict[str, float] = defaultdict(float)
    for cid, key in call_keys.items():
        groups[key] += call_hours.get(cid, 0.0)
    split_of_key, filled = {}, {"test": 0.0, "validation": 0.0}
    for key in sorted(groups, key=lambda k: _u(k, seed)):
        h, dest = groups[key], "train"
        for name, target in (("test", test_hours), ("validation", val_hours)):
            if filled[name] < target and filled[name] + h <= 1.5 * target:
                dest = name
                filled[name] += h
                break
        split_of_key[key] = dest
    return {cid: split_of_key[key] for cid, key in call_keys.items()}


def audio_quality(clip: np.ndarray) -> str | None:
    if clip.size == 0:
        return "bad_audio"
    rms = float(np.sqrt(np.mean(clip ** 2)) + 1e-12)
    if 20 * np.log10(rms) < RMS_MIN_DBFS:
        return "bad_audio"
    if float(np.mean(np.abs(clip) >= 0.999)) > CLIP_FRAC_MAX:
        return "bad_audio"
    return None


def process_call(call_id: str, args: dict) -> list[dict]:
    raw = Path(args["raw_dir"])
    out = Path(args["out_dir"])
    words, lang, json_dur = load_words(raw / "transcripts" / f"{call_id}.json")
    if not words:
        return [{"id": f"{call_id}_call", "call_id": call_id, "reject": ["empty_call"]}]
    dry = args.get("dry_run", False)
    if dry:   # tanpa audio: hanya segmentasi, normalisasi, PII, QC teks
        audio, channels = None, 0
        audio_dur = float(json_dur or words[-1].end + 1.0)
    else:
        audio, channels = load_audio(audio_path(raw, call_id))
        audio_dur = len(audio) / SR
    if words[-1].end - audio_dur > 2.0:
        return [{"id": f"{call_id}_call", "call_id": call_id, "reject": ["duration_mismatch"]}]

    roles = speaker_roles(words)
    crm = args["crm"].get(call_id)
    reviews = args["reviews"]
    meta = args["meta"].get(call_id)
    split = args["splits"][call_id]
    rng = random.Random(f"{args['seed']}:{call_id}")
    tokenizer = _tokenizer(args.get("tokenizer"))

    rows, prev, prev_raw, secret_asked = [], "", "", False
    spans = segment_words(words, rng)
    for idx, (i, j) in enumerate(spans):
        ws = words[i:j]
        start = max(ws[0].start - PAD, words[i - 1].end if i > 0 else 0.0, 0.0)
        end = min(ws[-1].end + PAD, words[j].start if j < len(words) else audio_dur, audio_dur)
        seg_id = f"{call_id}_{idx:04d}"
        text_raw = " ".join(w.text for w in ws)
        norm = normalize(text_raw, crm_name=(crm or {}).get("customer_name"))
        # OTP/CVV sering ditanya di satu giliran dan dijawab di segmen berikutnya.
        if secret_asked and has_secret_value(norm.text):
            norm.secret = True
        secret_asked = norm.secret_mention and not norm.secret
        text, pii, reviewed, human_rejected = norm.text, norm.pii_types, False, False
        rev = reviews.get(seg_id)
        if rev and abs(rev.get("start", start) - start) < 0.05 and abs(rev.get("end", end) - end) < 0.05:
            text, reviewed = rev.get("text", text), bool(rev.get("reviewed", True))
            human_rejected = bool(rev.get("rejected"))
            pii = rev.get("pii_types") or detect_pii(text, (crm or {}).get("customer_name"))[0]
            norm.secret = bool(rev.get("secret", norm.secret))
        elif rev:
            log.warning("review %s dilewati: start/end berubah", seg_id)
        lps = [w.logprob for w in ws if w.logprob is not None]
        spk = [roles[w.speaker] for w in ws]
        seg = Segment(
            id=seg_id, audio_filepath=str(out / "clips" / call_id / f"{seg_id}.flac"),
            call_id=call_id, seg_index=idx, start=round(start, 3), end=round(end, 3),
            duration=round(end - start, 3), prev_text=prev[-PREV_TEXT_MAX_CHARS:],
            prev_text_raw=prev_raw[-PREV_TEXT_MAX_CHARS:], text=text, text_raw=text_raw, crm=crm,
            speakers=sorted(set(spk)), n_turns=1 + sum(a != b for a, b in zip(spk, spk[1:])),
            pii_types=pii, language="id", sample_rate=SR,
            mean_logprob=round(float(np.mean(lps)), 4) if lps else None,
            min_logprob=round(float(min(lps)), 4) if lps else None,
            entity_min_logprob=_entity_min_logprob(ws),
            reviewed=reviewed, split=split, secret=norm.secret, needs_review=norm.needs_review,
            meta=meta, human_rejected=human_rejected,
            bad_timing=any(w.end - w.start > LONG_WORD for w in ws),
        )
        clip = None if dry else audio[int(start * SR):int(end * SR)]
        seg.reject = qc(seg, clip, tokenizer, lang, args.get("pii_policy", "review"),
                        args.get("pii_min_logprob", -0.1))
        # Riwayat: semua segmen ikut (termasuk yang ditolak QC), kecuali rahasia.
        if not seg.secret:
            prev = f"{prev} {text}".strip()
            prev_raw = f"{prev_raw} {text_raw}".strip()
        if not dry and (not seg.reject or reviewable(seg.reject)) and not seg.secret:
            path = Path(seg.audio_filepath)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), clip, SR, format="FLAC")
        rows.append({**asdict(seg), "channels": channels})
    return rows


def _entity_min_logprob(ws: list[Word]) -> float | None:
    num = [w.logprob for w in ws if w.logprob is not None and is_numberish(w.text)]
    lps = num or [w.logprob for w in ws if w.logprob is not None]
    return round(float(min(lps)), 4) if lps else None


def reviewable(reject: list[str]) -> bool:
    r = set(reject)
    return r <= REVIEWABLE and bool(r & {"pii_unreviewed", "eval_unreviewed"})


def qc(seg: Segment, clip: np.ndarray | None, tokenizer, lang: str, pii_policy: str = "review",
       pii_min_logprob: float = -0.1) -> list[str]:
    r = []
    if not (MIN_DUR <= seg.duration <= MAX_DUR):
        r.append("duration")
    if not seg.text.strip():
        r.append("empty")
    elif not seg.reviewed and not (CPS_MIN <= len(seg.text) / max(seg.duration, 1e-3) <= CPS_MAX):
        r.append("cps_outlier")
    if tokenizer is not None and len(tokenizer.encode(" " + seg.text, add_special_tokens=False)) > MAX_TEXT_TOKENS:
        r.append("too_many_tokens")
    if not seg.pii_types and seg.mean_logprob is not None and seg.mean_logprob < MIN_LOGPROB and not seg.reviewed:
        r.append("low_confidence")
    if seg.pii_types and not seg.reviewed:
        # Kebijakan confidence: hanya train, dan hanya bila kata paling tidak yakin pun di atas ambang.
        if (pii_policy == "confidence" and seg.split == "train" and seg.entity_min_logprob is not None
                and seg.entity_min_logprob >= pii_min_logprob and not seg.needs_review):
            seg.pii_auto = True
        else:
            r.append("pii_unreviewed")
    if seg.split != "train" and not seg.reviewed:
        r.append("eval_unreviewed")
    if seg.secret:
        r.append("secret")
    if seg.human_rejected:
        r.append("human_rejected")
    if seg.bad_timing:
        r.append("bad_timing")
    if lang and not lang.startswith(("id", "ind")):
        r.append("language")
    q = audio_quality(clip) if clip is not None else None
    if q:
        r.append(q)
    return r


_TOK_CACHE: dict = {}


def _tokenizer(name: str | None):
    if not name:
        return None
    if name not in _TOK_CACHE:
        from transformers import WhisperTokenizerFast
        tok = WhisperTokenizerFast.from_pretrained(name)
        if len(tok) < 50_000:
            raise ValueError(f"tokenizer {name} kosong ({len(tok)} token); file tidak termuat")
        _TOK_CACHE[name] = tok
    return _TOK_CACHE[name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _call_hours(raw: Path, call_id: str) -> float:
    """Durasi panggilan (jam) dari transkrip, tanpa membaca audio."""
    data = json.loads((raw / "transcripts" / f"{call_id}.json").read_text(encoding="utf-8"))
    secs = data.get("audio_duration_secs")
    if secs is None:
        ws = [w for w in data.get("words", []) if w.get("end") is not None]
        secs = ws[-1]["end"] if ws else 0.0
    return float(secs) / 3600


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--crm", default=None, help="default: <raw-dir>/crm.jsonl jika ada")
    ap.add_argument("--reviews", default=None, help="default: <out-dir>/reviews.jsonl jika ada")
    ap.add_argument("--tokenizer", default=None, help="mis. openai/whisper-large-v3; untuk cek panjang token")
    ap.add_argument("--meta", default=None, help="default: <raw-dir>/meta.jsonl jika ada")
    ap.add_argument("--split-by", choices=["agent", "call"], default="agent",
                    help="agent: semua panggilan satu agent di split yang sama (butuh agent_id di meta)")
    ap.add_argument("--dry-run", action="store_true",
                    help="tanpa audio: perkirakan segmen, PII dan beban review dari transkrip saja")
    ap.add_argument("--pii-policy", choices=["review", "confidence"], default="confidence",
                    help="confidence (default, diputuskan 2026-11-01): PII train diterima bila "
                         "logprob kata angka >= --pii-min-logprob, diaudit lewat spotcheck; review: semua PII "
                         "wajib direview. "
                         "Validation/test selalu wajib direview.")
    ap.add_argument("--pii-min-logprob", type=float, default=-0.1,
                    help="ambang logprob kata angka (atau semua kata bila PII tanpa angka)")
    ap.add_argument("--test-hours", type=float, default=4.0, help="target jam audio panggilan untuk test")
    ap.add_argument("--val-hours", type=float, default=2.0, help="target jam audio panggilan untuk validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_ids = {p.stem for p in (raw / "transcripts").glob("*.json")}
    audio_ids = (set(json_ids) if args.dry_run
                 else {p.stem for ext in AUDIO_EXTS for p in (raw / "audio").glob(f"*{ext}")})
    rejected = [{"id": f"{c}_call", "call_id": c, "reject": ["missing_pair"]}
                for c in sorted(audio_ids ^ json_ids)]
    calls = []
    seen_hash: dict[str, str] = {}
    for c in sorted(audio_ids & json_ids):
        # Duplikat = file audio identik (rekaman yang sama masuk dua kali).
        src = raw / "transcripts" / f"{c}.json" if args.dry_run else audio_path(raw, c)
        h = hashlib.sha256(src.read_bytes()).hexdigest()
        if h in seen_hash:
            rejected.append({"id": f"{c}_call", "call_id": c, "reject": ["duplicate"], "duplicate_of": seen_hash[h]})
            continue
        seen_hash[h] = c
        calls.append(c)
    crm = {r["call_id"]: {k: r[k] for k in ("customer_name", "product") if r.get(k)}
           for r in read_jsonl(Path(args.crm) if args.crm else raw / "crm.jsonl")}
    reviews = {r["id"]: r for r in read_jsonl(Path(args.reviews) if args.reviews else out / "reviews.jsonl")}
    meta = {r["call_id"]: {k: v for k, v in r.items() if k not in ("call_id", "file_path")}
            for r in read_jsonl(Path(args.meta) if args.meta else raw / "meta.jsonl")}
    # Split dihitung sekali untuk seluruh data (per agent bila diketahui, agar suara agent
    # di test tidak pernah terdengar saat training), dengan target jam test/validation.
    hours = {c: _call_hours(raw, c) for c in calls}
    keys = {c: (f"agent:{meta[c]['agent_id']}" if args.split_by == "agent" and (meta.get(c) or {}).get("agent_id")
                else f"call:{c}") for c in calls}
    splits = assign_splits(keys, hours, args.seed, args.test_hours, args.val_hours)
    shared = {"raw_dir": str(raw), "out_dir": str(out), "crm": crm, "reviews": reviews, "meta": meta,
              "splits": splits,
              "split_by": args.split_by, "seed": args.seed, "tokenizer": args.tokenizer,
              "dry_run": args.dry_run, "pii_policy": args.pii_policy, "pii_min_logprob": args.pii_min_logprob}
    log.info("%d panggilan, %d review", len(calls), len(reviews))

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {c: ex.submit(process_call, c, shared) for c in calls}
        for c, fut in futs.items():
            try:
                rows.extend(fut.result())
            except Exception as e:   # noqa: BLE001 - satu panggilan rusak tidak menghentikan build
                log.error("%s: %s", c, e)
                rejected.append({"id": f"{c}_call", "call_id": c, "reject": ["invalid_input"], "error": str(e)})

    by_split: dict[str, list] = defaultdict(list)
    review_queue = []
    for r in rows:
        if r.get("reject"):
            rejected.append(r)
            if reviewable(r["reject"]) and not r.get("secret"):
                review_queue.append(r)
        else:
            by_split[r["split"]].append(r)
        if not r.get("reject") and r.get("needs_review") and not r.get("reviewed"):
            review_queue.append(r)

    drop = ("reject", "secret", "needs_review", "channels", "human_rejected", "bad_timing")
    # pii_auto tetap di manifest: collator bisa membedakan label PII yang belum dicek manusia
    for name, _ in SPLITS:
        write_jsonl(out / f"{name}.jsonl", [{k: v for k, v in r.items() if k not in drop} for r in by_split[name]])
    test_pii = [r for r in by_split["test"] if r["pii_types"]]
    write_jsonl(out / "test_pii.jsonl", [{k: v for k, v in r.items() if k not in drop} for r in test_pii])
    write_jsonl(out / "rejected.jsonl", rejected)
    write_jsonl(out / "review_queue.jsonl", review_queue)

    stats = build_stats(by_split, rejected, review_queue, test_pii, args.seed)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("selesai: %s", json.dumps({k: v["segments"] for k, v in stats["splits"].items()}))
    return stats


def build_stats(by_split, rejected, review_queue, test_pii, seed) -> dict:
    def summary(rs):
        durs = [r["duration"] for r in rs]
        pii = Counter(t for r in rs for t in r["pii_types"])
        return {
            "segments": len(rs),
            "hours": round(sum(durs) / 3600, 3),
            "calls": len({r["call_id"] for r in rs}),
            "pii_segment_share": round(sum(bool(r["pii_types"]) for r in rs) / len(rs), 4) if rs else 0.0,
            "pii_types": dict(sorted(pii.items())),
            "duration_hist": dict(Counter(f"{int(d // 5) * 5}-{int(d // 5) * 5 + 5}s" for d in durs)),
            "reviewed_share": round(sum(r["reviewed"] for r in rs) / len(rs), 4) if rs else 0.0,
        }
    return {
        "seed": seed,
        "splits": {name: summary(by_split[name]) for name, _ in SPLITS},
        "test_pii_entities_approx": sum(len(r["pii_types"]) for r in test_pii),
        "rejected": dict(Counter(reason for r in rejected for reason in r["reject"])),
        "review_queue": len(review_queue),
        "review_queue_by_split": dict(Counter(r.get("split") for r in review_queue)),
        "review_queue_hours": round(sum(r.get("duration", 0) for r in review_queue) / 3600, 2),
        "review_queue_pii_types": dict(Counter(t for r in review_queue for t in r.get("pii_types", []))),
        "params": {"min_pause": MIN_PAUSE, "pad": PAD, "target": [TARGET_MIN, TARGET_MAX],
                   "short_prob": SHORT_PROB, "min_logprob": MIN_LOGPROB, "cps": [CPS_MIN, CPS_MAX]},
    }


if __name__ == "__main__":
    main()
