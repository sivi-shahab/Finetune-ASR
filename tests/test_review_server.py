"""Test pengecekan manusia: antrean, spotcheck, buang segmen, statistik, dan efeknya ke build_dataset."""
import json
import threading
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import build_dataset as bd
from review_server import Store, make_handler
from test_build_dataset import make_call


def serve(store):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def post(url, **form):
    urllib.request.urlopen(url + "/save", data=urllib.parse.urlencode(form).encode())


@pytest.fixture
def built(tmp_path):
    raw = tmp_path / "raw"
    (raw / "audio").mkdir(parents=True)
    (raw / "transcripts").mkdir()
    rng = np.random.default_rng(0)
    for i in range(6):
        make_call(raw, f"CALL{i:03d}", rng)
    out = tmp_path / "dataset"
    bd.main(["--raw-dir", str(raw), "--out-dir", str(out), "--test-hours", "0.02", "--val-hours", "0.01", "--workers", "1"])
    return raw, out


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def test_alur_queue(built):
    raw, out = built
    store = Store(out, "tester")
    srv, url = serve(store)
    try:
        first = store.next()
        page = urllib.request.urlopen(url + "/").read().decode()
        assert first["id"] in page and "Rp15.000.000" in page and "QC:" in page
        audio = urllib.request.urlopen(url + "/audio?id=" + urllib.parse.quote(first["id"])).read()
        assert audio[:4] == b"fLaC"
        post(url, id=first["id"], text="Teks  dikoreksi.", action="save")
        saved = rows(out / "reviews.jsonl")[-1]
        assert saved["id"] == first["id"] and saved["text"] == "Teks dikoreksi."
        assert saved["reviewed"] and saved["edited"] and saved["mode"] == "queue"
        second = store.next()
        post(url, id=second["id"], text=second["text"], action="reject")
        assert rows(out / "reviews.jsonl")[-1]["rejected"]
        stats = urllib.request.urlopen(url + "/stats").read().decode()
        assert "tester" in stats and "% dikoreksi" in stats
    finally:
        srv.shutdown()

    # Efek ke build: yang dibuang → human_rejected; yang disimpan keluar dari antrean.
    bd.main(["--raw-dir", str(raw), "--out-dir", str(out), "--test-hours", "0.02", "--val-hours", "0.01", "--workers", "1"])
    rej = {r["id"]: r["reject"] for r in rows(out / "rejected.jsonl")}
    assert "human_rejected" in rej[second["id"]]
    assert first["id"] not in {r["id"] for r in rows(out / "review_queue.jsonl")}


def test_spotcheck_mengambil_sampel_train(built):
    _, out = built
    train = rows(out / "train.jsonl")
    assert train
    store = Store(out, "tester", mode="spotcheck", rate=1.0)
    assert {r["id"] for r in store.queue} == {r["id"] for r in train}
    assert Store(out, "tester", mode="spotcheck", rate=0.0).queue == []
