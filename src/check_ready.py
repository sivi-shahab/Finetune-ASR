"""Periksa apakah dataset siap dipakai training (docs/SPEC.md bagian 9).

    python src/check_ready.py --dataset dataset [--tokenizer openai/whisper-large-v3] [--strict]

Keluar dengan kode 0 bila READY, 1 bila NOT READY. Ringkasan ditulis ke dataset/READY.json.
ERROR menghalangi training; WARN perlu diketahui tapi tidak menghalangi (kecuali --strict).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from normalize import SECRET_RE, has_secret_value  # noqa: E402

SPLITS = ("train", "validation", "test")
MAX_TEXT_TOKENS = 448 - 5
SPOTCHECK_MIN = 100           # sampel spotcheck minimum sebelum tingkat koreksinya dipercaya
SPOTCHECK_MAX_EDIT = 0.10     # > 10% segmen train dikoreksi saat spotcheck → label otomatis kurang bagus


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def check(dataset: Path, tokenizer=None, check_audio: bool = True) -> dict:
    errors, warns = [], []
    data = {s: read_jsonl(dataset / f"{s}.jsonl") for s in SPLITS}
    for s in SPLITS:
        if not data[s]:
            errors.append(f"{s}.jsonl kosong atau tidak ada")

    # 1) Kebocoran antar split: panggilan dan agent
    owner_call, owner_agent = defaultdict(set), defaultdict(set)
    for s, rows in data.items():
        for r in rows:
            owner_call[r["call_id"]].add(s)
            agent = (r.get("meta") or {}).get("agent_id")
            if agent:
                owner_agent[agent].add(s)
    leak_c = [c for c, v in owner_call.items() if len(v) > 1]
    leak_a = [a for a, v in owner_agent.items() if len(v) > 1]
    if leak_c:
        errors.append(f"{len(leak_c)} panggilan muncul di lebih dari satu split")
    if leak_a:
        warns.append(f"{len(leak_a)} agent muncul di lebih dari satu split (build dengan --split-by call?)")

    # 2) Review: validation/test 100%; PII train harus direview atau pii_auto
    for s in ("validation", "test"):
        un = sum(not r.get("reviewed") for r in data[s])
        if un:
            errors.append(f"{s}: {un} segmen belum direview manusia")
    pii_unchecked = [r for r in data["train"] if r.get("pii_types") and not r.get("reviewed") and not r.get("pii_auto")]
    if pii_unchecked:
        errors.append(f"train: {len(pii_unchecked)} segmen PII belum direview")
    pii_auto = sum(bool(r.get("pii_auto")) for r in data["train"])

    # 3) Rahasia, teks, panjang token
    for s, rows in data.items():
        for r in rows:
            for field in ("text", "prev_text"):
                t = r.get(field) or ""
                if SECRET_RE.search(t) and has_secret_value(t):
                    errors.append(f"{s} {r['id']}: kemungkinan nilai rahasia di {field}")
                    break
            if not r.get("text", "").strip():
                errors.append(f"{s} {r['id']}: teks kosong")
            if tokenizer is not None and len(tokenizer.encode(" " + r["text"], add_special_tokens=False)) > MAX_TEXT_TOKENS:
                errors.append(f"{s} {r['id']}: label > {MAX_TEXT_TOKENS} token")

    # 4) Audio: ada, 16 kHz mono, durasi cocok, 1-30 detik
    bad_audio = 0
    if check_audio:
        for s, rows in data.items():
            for r in rows:
                p = Path(r["audio_filepath"])
                try:
                    info = sf.info(str(p))
                    ok = (info.samplerate == 16000 and info.channels == 1
                          and abs(info.duration - r["duration"]) < 0.05 and 1.0 <= info.duration <= 30.0)
                except Exception:   # noqa: BLE001 - file hilang/rusak dihitung sebagai audio buruk
                    ok = False
                if not ok:
                    bad_audio += 1
                    if bad_audio <= 5:
                        errors.append(f"{s} {r['id']}: audio hilang/rusak/tidak sesuai ({p})")
        if bad_audio > 5:
            errors.append(f"... total {bad_audio} audio bermasalah")

    # 5) Kualitas label otomatis (dari spotcheck)
    spot = [r for r in read_jsonl(dataset / "reviews.jsonl") if r.get("mode") == "spotcheck"]
    spot_rate = (sum(bool(r.get("edited")) for r in spot) / len(spot)) if spot else None
    if len(spot) < SPOTCHECK_MIN:
        warns.append(f"spotcheck baru {len(spot)} segmen (< {SPOTCHECK_MIN}); kualitas label train otomatis belum terukur")
    elif spot_rate > SPOTCHECK_MAX_EDIT:
        warns.append(f"spotcheck: {spot_rate:.1%} segmen train dikoreksi (> {SPOTCHECK_MAX_EDIT:.0%})")
    if pii_auto and len(spot) < SPOTCHECK_MIN:
        warns.append(f"{pii_auto} segmen PII train diterima otomatis (pii_auto) tanpa audit spotcheck yang cukup")

    queue = read_jsonl(dataset / "review_queue.jsonl")
    summary = {
        s: {"segments": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
            "calls": len({r["call_id"] for r in rows}),
            "pii_segments": sum(bool(r.get("pii_types")) for r in rows),
            "reviewed": sum(bool(r.get("reviewed")) for r in rows)}
        for s, rows in data.items()
    }
    return {"ready": not errors, "errors": errors[:50], "n_errors": len(errors), "warnings": warns,
            "splits": summary, "pii_auto_train": pii_auto, "spotcheck_n": len(spot),
            "spotcheck_edit_rate": round(spot_rate, 4) if spot_rate is not None else None,
            "review_queue_remaining": len(queue)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--strict", action="store_true", help="WARN juga dianggap gagal")
    a = ap.parse_args(argv)
    tok = None
    if a.tokenizer:
        from transformers import WhisperTokenizerFast
        tok = WhisperTokenizerFast.from_pretrained(a.tokenizer)
        if len(tok) < 50_000:
            raise SystemExit(f"tokenizer {a.tokenizer} kosong; file tidak termuat")
    rep = check(Path(a.dataset), tok, check_audio=not a.skip_audio)
    if a.strict and rep["warnings"]:
        rep["ready"] = False
    (Path(a.dataset) / "READY.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("READY" if rep["ready"] else "NOT READY")
    for s, v in rep["splits"].items():
        print(f"  {s:<11} {v['segments']:>7} segmen {v['hours']:>7.1f} jam  PII {v['pii_segments']:>6}  "
              f"direview {v['reviewed']:>6}")
    for e in rep["errors"][:15]:
        print("  ERROR", e)
    for w in rep["warnings"]:
        print("  WARN ", w)
    return 0 if rep["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
