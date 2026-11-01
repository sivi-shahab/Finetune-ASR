"""Test check_ready: NOT READY sebelum review, READY setelah validation/test direview."""
import json

import numpy as np

import build_dataset as bd
from check_ready import check
from test_build_dataset import make_call


def test_ready_setelah_review(tmp_path):
    raw = tmp_path / "raw"
    (raw / "audio").mkdir(parents=True)
    (raw / "transcripts").mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        make_call(raw, f"CALL{i:03d}", rng)
    out = tmp_path / "ds"
    args = ["--raw-dir", str(raw), "--out-dir", str(out), "--test-hours", "0.02", "--val-hours", "0.01",
            "--workers", "1"]
    bd.main(args)
    rep = check(out)
    assert not rep["ready"] and any("validation" in e or "test" in e for e in rep["errors"])

    queue = [json.loads(x) for x in (out / "review_queue.jsonl").read_text().splitlines()]
    (out / "reviews.jsonl").write_text("\n".join(json.dumps(
        {"id": r["id"], "start": r["start"], "end": r["end"], "text": r["text"], "reviewed": True}) for r in queue))
    bd.main(args)
    rep = check(out)
    assert rep["ready"], rep["errors"]
    assert rep["splits"]["test"]["segments"] > 0 and rep["splits"]["validation"]["segments"] > 0
    assert any("spotcheck" in w for w in rep["warnings"])
