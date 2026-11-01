"""Pengecekan dataset oleh manusia (docs/SPEC.md bagian 7).

    python src/review_server.py --dataset dataset --reviewer nama                    # antrean wajib
    python src/review_server.py --dataset dataset --reviewer nama --mode spotcheck   # cek acak train

Mode:
    queue      segmen yang tertahan sampai direview: PII, validation/test, dan needs_review.
               Urutan: test → validation → train.
    spotcheck  sampel acak (--rate, default 3%) segmen train yang lolos otomatis. Persentase
               yang dikoreksi = perkiraan tingkat kesalahan label ElevenLabs di train.

Aksi: Simpan (teks benar/dikoreksi) · Rahasia (PIN/OTP/CVV) · Buang (audio rusak / label tak
bisa diperbaiki) · Lewati. Setiap aksi menambah satu baris ke dataset/reviews.jsonl; jalankan
ulang src/build_dataset.py agar hasilnya masuk manifest. Ringkasan: http://host:port/stats

Buka http://127.0.0.1:8765 (lewat SSH tunnel: ssh -L 8765:127.0.0.1:8765 server).
Hanya library standar. Default bind 127.0.0.1: data berisi PII, jangan buka ke jaringan luas.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import threading
import time
import zlib
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))
from normalize import extract_entities  # noqa: E402

STYLE = """body{font:15px system-ui;max-width:920px;margin:24px auto;padding:0 16px}
textarea{width:100%;height:120px;font:16px monospace} .meta{color:#555;font-size:13px}
.ent{background:#fff3c4;padding:1px 4px;border-radius:3px;margin-right:6px}
button{font-size:15px;margin-right:8px} .raw{background:#f3f3f3;padding:8px;border-radius:6px}
.tag{display:inline-block;background:#e8eefc;border-radius:4px;padding:0 6px;margin-right:4px}
table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:4px 8px;text-align:right}"""

PAGE = """<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Review {done}/{total}</title>
<style>{style}</style></head><body>
<p class="meta">mode <b>{mode}</b> &middot; {done}/{total} selesai &middot; <a href="/stats">statistik</a></p>
<p><span class="tag">{id}</span><span class="tag">{split}</span><span class="tag">{campaign}</span>
<span class="tag">QC: {qc_status}</span><span class="tag">{speakers}</span>
<span class="meta">{start:.1f}&ndash;{end:.1f} dtk &middot; {reasons}</span></p>
<audio controls autoplay src="/audio?id={id_q}" style="width:100%"></audio>
<p><b>Konteks sebelumnya:</b> <span class="meta">{prev}</span></p>
<p><b>ElevenLabs (mentah):</b></p><div class="raw">{raw}</div>
<form method="post" action="/save">
<input type="hidden" name="id" value="{id_q}">
<p><b>Teks final</b> (format akhir SPEC 5.2: Rp15.000.000, 17-08-1985, 081234567890; dengarkan setiap digit):</p>
<textarea name="text" autofocus>{text}</textarea>
<p>Entitas terdeteksi: {ents}</p>
<button name="action" value="save">Simpan (Ctrl+Enter)</button>
<button name="action" value="secret">Rahasia (PIN/OTP/CVV)</button>
<button name="action" value="reject">Buang segmen</button>
<button name="action" value="skip">Lewati</button>
</form>
<script>document.addEventListener('keydown',e=>{{if(e.ctrlKey&&e.key==='Enter')document.querySelector('button[value=save]').click()}})</script>
</body></html>"""

DONE_PAGE = "<!doctype html><meta charset=utf-8><p>Semua segmen di mode ini sudah dicek. <a href=/stats>statistik</a></p>"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def in_sample(seg_id: str, rate: float) -> bool:
    """Sampel deterministik: segmen yang sama selalu terpilih untuk rate yang sama."""
    return zlib.crc32(seg_id.encode()) / 0xFFFFFFFF < rate


class Store:
    def __init__(self, dataset: Path, reviewer: str, mode: str = "queue", rate: float = 0.03):
        self.reviews_path = dataset / "reviews.jsonl"
        self.reviewer, self.mode = reviewer, mode
        self.lock = threading.Lock()
        if mode == "queue":
            items = read_jsonl(dataset / "review_queue.jsonl")
        else:
            items = [r for r in read_jsonl(dataset / "train.jsonl") if not r.get("reviewed") and in_sample(r["id"], rate)]
        order = {"test": 0, "validation": 1, "train": 2}
        items.sort(key=lambda r: (order.get(r.get("split"), 3), r["call_id"], r["seg_index"]))
        self.queue = items
        self.by_id = {r["id"]: r for r in items}
        self.reviews = read_jsonl(self.reviews_path)
        self.done = {r["id"] for r in self.reviews}
        self.skipped: set[str] = set()

    def next(self) -> dict | None:
        for r in self.queue:
            if r["id"] not in self.done and r["id"] not in self.skipped:
                return r
        return None

    def save(self, seg_id: str, text: str, action: str) -> None:
        r = self.by_id[seg_id]
        text = " ".join(text.split())
        row = {"id": seg_id, "start": r["start"], "end": r["end"], "text": text, "reviewed": True,
               "secret": action == "secret", "rejected": action == "reject", "action": action,
               "mode": self.mode, "auto_text": r["text"], "edited": text != r["text"],
               "reviewer": self.reviewer, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self.lock, self.reviews_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.done.add(seg_id)
            self.reviews.append(row)


def stats_page(store: Store) -> str:
    by_rev = defaultdict(Counter)
    by_mode = defaultdict(Counter)
    for r in store.reviews:
        a = r.get("action", "save")
        by_rev[r.get("reviewer", "?")][a] += 1
        m = by_mode[r.get("mode", "queue")]
        m["n"] += 1
        m[a] += 1
        m["edited"] += bool(r.get("edited"))
    rows = "".join(f"<tr><td style=text-align:left>{html.escape(k)}</td><td>{sum(c.values())}</td>"
                   f"<td>{c['save']}</td><td>{c['secret']}</td><td>{c['reject']}</td></tr>"
                   for k, c in sorted(by_rev.items()))
    modes = "".join(f"<tr><td style=text-align:left>{k}</td><td>{c['n']}</td><td>{c['edited']}</td>"
                    f"<td>{c['edited'] / c['n']:.1%}</td><td>{c['reject']}</td></tr>"
                    for k, c in sorted(by_mode.items()) if c["n"])
    remaining = sum(1 for r in store.queue if r["id"] not in store.done)
    return f"""<!doctype html><meta charset=utf-8><title>Statistik review</title><style>{STYLE}</style>
<p><a href=/>&larr; kembali</a></p><h3>Mode {store.mode}: sisa {remaining} dari {len(store.queue)}</h3>
<h4>Per reviewer</h4><table><tr><th>reviewer</th><th>total</th><th>simpan</th><th>rahasia</th><th>buang</th></tr>{rows}</table>
<h4>Per mode</h4><table><tr><th>mode</th><th>dicek</th><th>dikoreksi</th><th>% dikoreksi</th><th>dibuang</th></tr>{modes}</table>
<p class=meta>% dikoreksi di mode spotcheck ≈ tingkat kesalahan label ElevenLabs pada train yang lolos otomatis.</p>"""


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):   # jangan cetak URL/isi ke log (PII)
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/audio":
                r = store.by_id.get(parse_qs(u.query).get("id", [""])[0])
                if not r or not Path(r["audio_filepath"]).exists():
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, Path(r["audio_filepath"]).read_bytes(), "audio/flac")
            if u.path == "/stats":
                return self._send(200, stats_page(store).encode())
            r = store.next()
            if r is None:
                return self._send(200, DONE_PAGE.encode())
            meta = r.get("meta") or {}
            ents = extract_entities(r["text"])
            page = PAGE.format(
                style=STYLE, mode=store.mode, done=len(store.done & store.by_id.keys()), total=len(store.queue),
                id=html.escape(r["id"]), id_q=html.escape(r["id"], quote=True), split=r.get("split"),
                campaign=html.escape(str(meta.get("campaign") or "-")),
                qc_status=html.escape(str(meta.get("qc_status") or "belum dinilai")),
                start=r["start"], end=r["end"], speakers=", ".join(r.get("speakers", [])),
                reasons=html.escape(", ".join(r.get("reject", []) + r.get("needs_review", []))),
                prev=html.escape(r.get("prev_text", "")[-300:]), raw=html.escape(r.get("text_raw", "")),
                text=html.escape(r["text"]),
                ents=" ".join(f'<span class="ent">{html.escape(k)}: {html.escape(v)}</span>' for k, v in ents)
                or "-")
            self._send(200, page.encode())

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode("utf-8"))
            seg_id = form.get("id", [""])[0]
            action = form.get("action", ["save"])[0]
            if seg_id in store.by_id and action in ("save", "secret", "reject", "skip"):
                if action == "skip":
                    store.skipped.add(seg_id)
                else:
                    store.save(seg_id, form.get("text", [""])[0], action)
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--mode", choices=["queue", "spotcheck"], default="queue")
    ap.add_argument("--rate", type=float, default=0.03, help="porsi sampel untuk mode spotcheck")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    store = Store(Path(a.dataset), a.reviewer, a.mode, a.rate)
    print(f"mode {a.mode}: {len(store.queue)} segmen, {len(store.done & store.by_id.keys())} sudah dicek. "
          f"http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
