"""Ubah hasil ekspor QC dashboard menjadi input build_dataset di raw/.

Skrip ini TIDAK mengambil audio. Audio diambil sendiri oleh tim dari server rekaman
berdasarkan raw/audio_list.csv, lalu diimpor dengan perintah `import-audio`.

1) Tulis transkrip + metadata + daftar audio:
    python src/qc_export.py convert --input raw/qc_export.jsonl --raw-dir raw

   raw/transcripts/<call_id>.json   respons mentah ElevenLabs (format SPEC bagian 2)
   raw/crm.jsonl                    {"call_id", "customer_name", "product"} untuk lapisan CRM prompt
   raw/meta.jsonl                   {"call_id", "result_id", "campaign", "qc_status", "agent_id", ...}
   raw/audio_list.csv               ticket_id,file_path,campaign,qc_status,qc_tl_status,duration_sec
                                    (file_path = dashboard.recording_tms_api.file_path)

2) Setelah file audio (.gsm/.wav/.mp3/...) diambil ke satu folder, nama file = ticket_id + ekstensi:
    python src/qc_export.py import-audio --src /path/folder_audio --raw-dir raw

   Tiap file dikonversi lokal dengan ffmpeg ke raw/audio/<ticket_id>.flac (mono, sample rate asli;
   build_dataset me-resample ke 16 kHz). File yang sudah ada dilewati. Ticket yang tidak ada di
   meta.jsonl dilewati dan dilaporkan.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("qc_export")

META_FIELDS = ("call_id", "result_id", "campaign", "qc_status", "qc_tl_status", "uploaded_at",
               "agent_id", "product", "file_path")
AUDIO_LIST_FIELDS = ("ticket_id", "file_path", "campaign", "qc_status", "qc_tl_status", "duration_sec")
AUDIO_EXTS = {".gsm", ".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def convert(input_path: Path, raw: Path) -> dict:
    """Tulis transkrip, crm.jsonl, meta.jsonl dan audio_list.csv."""
    (raw / "transcripts").mkdir(parents=True, exist_ok=True)
    (raw / "audio").mkdir(parents=True, exist_ok=True)
    metas, crm, listing, no_path = [], [], [], 0
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["call_id"]
            if "/" in cid or cid.startswith("."):
                log.warning("call_id tidak valid, dilewati: %r", cid)
                continue
            el = r["elevenlabs"]
            (raw / "transcripts" / f"{cid}.json").write_text(json.dumps(el, ensure_ascii=False), encoding="utf-8")
            metas.append({k: r.get(k) for k in META_FIELDS} | {"source": "qc_dashboard"})
            if r.get("customer_name") or r.get("campaign"):
                crm.append({"call_id": cid,
                            "customer_name": (r.get("customer_name") or "").strip().title() or None,
                            "product": r.get("campaign") or r.get("product")})
            no_path += not r.get("file_path")
            listing.append({"ticket_id": cid, "file_path": r.get("file_path") or "",
                            "campaign": r.get("campaign") or "", "qc_status": r.get("qc_status") or "",
                            "qc_tl_status": r.get("qc_tl_status") or "",
                            "duration_sec": el.get("audio_duration_secs") or ""})
    write_jsonl(raw / "meta.jsonl", metas)
    write_jsonl(raw / "crm.jsonl", crm)
    with (raw / "audio_list.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUDIO_LIST_FIELDS)
        w.writeheader()
        w.writerows(listing)
    hours = sum(float(x["duration_sec"] or 0) for x in listing) / 3600
    return {"calls": len(metas), "without_file_path": no_path, "audio_hours": round(hours, 1)}


def _to_flac(src: Path, dst: Path) -> str | None:
    if dst.exists() and dst.stat().st_size > 0:
        return None
    part = dst.with_name(dst.stem + ".part.flac")
    p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                        "-ac", "1", "-c:a", "flac", str(part)], capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not part.exists():
        part.unlink(missing_ok=True)
        return (p.stderr or f"ffmpeg exit {p.returncode}").strip()[-300:]
    part.rename(dst)
    return None


def import_audio(src_dir: Path, raw: Path, workers: int) -> dict:
    """File boleh dinamai <ticket_id>.<ext> ATAU nama asli dari server rekaman
    (basename file_path, mis. 20260901135012-201122-xxxxxxxxxx.gsm)."""
    known = {m["call_id"] for m in read_jsonl(raw / "meta.jsonl")}
    by_remote = {}
    with (raw / "audio_list.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["file_path"]:
                by_remote[Path(r["file_path"]).stem] = r["ticket_id"]
    (raw / "audio").mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    todo, unknown = [], []
    for p in files:
        tid = p.stem if p.stem in known else by_remote.get(p.stem)
        (todo.append((p, tid)) if tid in known else unknown.append(p.name))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        errors = list(ex.map(lambda pt: (pt, _to_flac(pt[0], raw / "audio" / f"{pt[1]}.flac")), todo))
    failed = [{"ticket_id": t, "src": str(p), "error": e} for (p, t), e in errors if e]
    write_jsonl(raw / "import_failed.jsonl", failed)
    have = {p.stem for p in (raw / "audio").glob("*.flac") if not p.stem.endswith(".part")}
    return {"found": len(files), "imported_or_existing": len(todo) - len(failed), "failed": len(failed),
            "unknown_ticket": len(unknown), "still_missing": len(known - have)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--input", default="raw/qc_export.jsonl")
    c.add_argument("--raw-dir", default="raw")
    i = sub.add_parser("import-audio")
    i.add_argument("--src", required=True, help="folder berisi audio hasil ambil manual")
    i.add_argument("--raw-dir", default="raw")
    i.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "convert":
        res = convert(Path(a.input), Path(a.raw_dir))
    else:
        res = import_audio(Path(a.src), Path(a.raw_dir), a.workers)
    log.info("%s", json.dumps(res))
    return res


if __name__ == "__main__":
    main()
