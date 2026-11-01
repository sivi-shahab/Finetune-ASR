"""Evaluasi: prediksi dari checkpoint HF dan skor WER/CER + metrik entitas (docs/SPEC.md bagian 8).

Prediksi (checkpoint Hugging Face, env whisper-train):
    python src/evaluate.py predict --model outputs/run-001/checkpoint-1000 \
        --manifest dataset/validation.jsonl --prompt domain --out outputs/preds/val-ckpt1000.jsonl

    --prompt none          : tanpa prompt
    --prompt domain        : prompt domain saja (bisa di-batch)
    --prompt gold_context  : domain + CRM + prev_text dari manifest (batas atas; batch 1)
    Konteks hasil prediksi sendiri (kondisi produksi) dijalankan lewat src/infer_context.py.

Skor (prediksi dari sumber mana pun: jsonl {"id", "text"}):
    python src/evaluate.py score --manifest dataset/test.jsonl --pred outputs/preds/test.jsonl \
        --out outputs/reports/test.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from normalize import extract_entities, normalize

PRIMARY_TYPES = ("nominal", "tanggal", "no_kartu", "no_hp", "nik")
SPELLING = {"gak": "nggak", "ga": "nggak", "enggak": "nggak", "engga": "nggak", "ngga": "nggak",
            "tdk": "tidak", "udh": "udah", "sdh": "sudah", "makasih": "makasih", "trims": "terima kasih"}


def read_jsonl(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Skor
# ---------------------------------------------------------------------------


def wer_normalize(text: str) -> str:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    t = BasicTextNormalizer()(text)
    return " ".join(SPELLING.get(w, w) for w in t.split())


def canonical(kind: str, value: str) -> str:
    """Nilai entitas tanpa format: nominal → angka bulat, lainnya → digit/huruf saja."""
    if kind == "nominal":
        whole = value.removeprefix("Rp").split(",")[0].replace(".", "")
        return whole.lstrip("0") or "0"
    if kind == "email":
        return value.lower()
    return re.sub(r"\D", "", value)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def entity_scores(pairs: list[tuple[str, str]]) -> dict:
    """pairs: [(ref, hyp)]. Exact match = entitas referensi muncul persis di hipotesis;
    value match = nilai sama setelah hipotesis dinormalisasi; digit error = edit distance digit."""
    per = defaultdict(lambda: {"n": 0, "exact": 0, "value": 0, "digit_err": 0, "digits": 0})
    for ref, hyp in pairs:
        ref_ents = extract_entities(ref)
        if not ref_ents:
            continue
        hyp_exact = [v for _, v in extract_entities(hyp)]
        hyp_norm = [(k, canonical(k, v)) for k, v in extract_entities(normalize(hyp).text)]
        for kind, val in ref_ents:
            s = per[kind]
            s["n"] += 1
            if val in hyp_exact:
                hyp_exact.remove(val)
                s["exact"] += 1
            cv = canonical(kind, val)
            hv = [h for _, h in hyp_norm]
            if cv in hv:
                s["value"] += 1
            if kind != "email":
                s["digits"] += len(cv)
                s["digit_err"] += min((edit_distance(cv, h) for h in hv if h.isdigit()), default=len(cv))
    out = {}
    for kind, s in sorted(per.items()):
        out[kind] = {
            "n": s["n"],
            "exact_match": round(s["exact"] / s["n"], 4),
            "value_match": round(s["value"] / s["n"], 4),
            "digit_error_rate": round(s["digit_err"] / s["digits"], 4) if s["digits"] else None,
        }
    tot = sum(s["n"] for s in per.values())
    if tot:
        out["_all"] = {
            "n": tot,
            "exact_match": round(sum(s["exact"] for s in per.values()) / tot, 4),
            "value_match": round(sum(s["value"] for s in per.values()) / tot, 4),
        }
    return out


def text_scores(pairs: list[tuple[str, str]]) -> dict:
    import jiwer
    refs, hyps = [], []
    for r, h in pairs:
        r, h = wer_normalize(r), wer_normalize(h)
        if r:
            refs.append(r)
            hyps.append(h)
    if not refs:
        return {"n": 0}
    return {"n": len(refs), "wer": round(jiwer.wer(refs, hyps), 4), "cer": round(jiwer.cer(refs, hyps), 4)}


def position_bucket(row: dict, last_index: dict) -> str:
    n = last_index.get(row["call_id"], 0)
    if n == 0:
        return "awal"
    x = row["seg_index"] / n
    return "awal" if x < 1 / 3 else ("tengah" if x < 2 / 3 else "akhir")


def speaker_bucket(row: dict) -> str:
    s = row.get("speakers") or []
    return s[0] if len(s) == 1 else "campuran"


def name_consistency(rows: list[dict], preds: dict) -> dict | None:
    hit = tot = 0
    for r in rows:
        name = (r.get("crm") or {}).get("customer_name")
        if name and name in r["text"]:
            tot += 1
            hit += name in preds.get(r["id"], "")
    return {"n": tot, "exact_spelling": round(hit / tot, 4)} if tot else None


def score(manifest: str, pred: str) -> dict:
    rows = read_jsonl(manifest)
    preds = {p["id"]: p["text"] for p in read_jsonl(pred)}
    missing = [r["id"] for r in rows if r["id"] not in preds]
    rows = [r for r in rows if r["id"] in preds]
    last = defaultdict(int)
    for r in rows:
        last[r["call_id"]] = max(last[r["call_id"]], r["seg_index"])
    pairs = [(r["text"], preds[r["id"]]) for r in rows]

    groups = defaultdict(list)
    for r in rows:
        groups[f"posisi:{position_bucket(r, last)}"].append((r["text"], preds[r["id"]]))
        groups[f"speaker:{speaker_bucket(r)}"].append((r["text"], preds[r["id"]]))
    return {
        "manifest": manifest,
        "pred": pred,
        "missing_predictions": len(missing),
        "text": text_scores(pairs),
        "entities": entity_scores(pairs),
        "groups": {g: {"text": text_scores(p), "entities": entity_scores(p).get("_all")}
                   for g, p in sorted(groups.items())},
        "name_consistency": name_consistency(rows, preds),
    }


def print_report(rep: dict) -> None:
    t = rep["text"]
    print(f"Segmen: {t.get('n')}  WER: {t.get('wer')}  CER: {t.get('cer')}  "
          f"(prediksi hilang: {rep['missing_predictions']})")
    print(f"{'entitas':<14}{'n':>6}{'exact':>9}{'value':>9}{'digit_err':>11}")
    for k, s in rep["entities"].items():
        print(f"{k:<14}{s['n']:>6}{s['exact_match']:>9.3f}{s['value_match']:>9.3f}"
              f"{(s.get('digit_error_rate') if s.get('digit_error_rate') is not None else float('nan')):>11.3f}")
    for g, s in rep["groups"].items():
        e = s["entities"] or {}
        print(f"  {g:<20} WER {s['text'].get('wer')}  entity exact {e.get('exact_match')}")


# ---------------------------------------------------------------------------
# Prediksi HF
# ---------------------------------------------------------------------------


def predict(model_path: str, manifest: str, prompt_mode: str, out: str, batch_size: int,
            domain_path: str, num_beams: int) -> None:
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from context_builder import Tok, build_prompt_ids, load_domain_prompt

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda").eval()
    tok = Tok(processor.tokenizer)
    sop, sot = tok.special("sop"), tok.special("sot")
    domain = load_domain_prompt(domain_path)
    rows = read_jsonl(manifest)

    def prompt_for(r):
        if prompt_mode == "none":
            return None
        if prompt_mode == "domain":
            ids = build_prompt_ids(tok, domain, None, "")
        else:
            ids = build_prompt_ids(tok, domain, r.get("crm"), r.get("prev_text", ""))
        return torch.tensor([sop] + ids, device="cuda")

    bs = 1 if prompt_mode == "gold_context" else batch_size
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in range(0, len(rows), bs):
            chunk = rows[b:b + bs]
            audio = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in chunk]
            feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
            prompt = prompt_for(chunk[0])
            n_prompt = 0 if prompt is None else len(prompt)
            with torch.no_grad():
                seqs = model.generate(feats.to("cuda", torch.bfloat16), prompt_ids=prompt, language="id",
                                      task="transcribe", num_beams=num_beams,
                                      max_new_tokens=440 - n_prompt)
            for r, seq in zip(chunk, seqs.tolist()):
                # Ambil hanya token setelah <|startoftranscript|> (buang prompt bila ikut dikembalikan).
                if sot in seq:
                    seq = seq[len(seq) - seq[::-1].index(sot):]
                text = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
                f.write(json.dumps({"id": r["id"], "text": text}, ensure_ascii=False) + "\n")
            print(f"{min(b + bs, len(rows))}/{len(rows)}", end="\r", flush=True)
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--prompt", choices=["none", "domain", "gold_context"], default="domain")
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--domain_prompt", default="configs/domain_prompt.txt")
    s = sub.add_parser("score")
    s.add_argument("--manifest", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        predict(a.model, a.manifest, a.prompt, a.out, a.batch_size, a.domain_prompt, a.num_beams)
    else:
        rep = score(a.manifest, a.pred)
        print_report(rep)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
