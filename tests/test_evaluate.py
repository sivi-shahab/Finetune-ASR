"""Test metrik evaluasi (bagian score; tidak butuh model)."""
import json

import pytest

from evaluate import canonical, edit_distance, entity_scores, score


def test_canonical():
    assert canonical("nominal", "Rp15.000.000") == "15000000"
    assert canonical("nominal", "Rp1.500,50") == "1500"
    assert canonical("tanggal", "17-08-1985") == "17081985"


def test_edit_distance():
    assert edit_distance("0812", "0812") == 0
    assert edit_distance("0812", "0813") == 1
    assert edit_distance("0812", "") == 4


def test_exact_vs_value_match():
    ref = "limitnya Rp15.000.000, tanggal lahir 17-08-1985"
    # format salah tapi nilai benar → value match, bukan exact match
    s = entity_scores([(ref, "limitnya 15 juta, tanggal lahir 17 Agustus 1985")])
    assert s["nominal"]["exact_match"] == 0 and s["nominal"]["value_match"] == 1
    assert s["tanggal"]["exact_match"] == 0 and s["tanggal"]["value_match"] == 1
    s = entity_scores([(ref, ref)])
    assert s["_all"]["exact_match"] == 1


def test_digit_error_rate():
    s = entity_scores([("nomor HP 081234567890", "nomor HP 081234567899")])
    assert s["no_hp"]["exact_match"] == 0
    assert s["no_hp"]["digit_error_rate"] == pytest.approx(1 / 12, abs=1e-3)
    s = entity_scores([("nomor HP 081234567890", "nomor HP nya lupa")])
    assert s["no_hp"]["digit_error_rate"] == 1.0


def test_score_end_to_end(tmp_path):
    manifest = [
        {"id": "A_0000", "call_id": "A", "seg_index": 0, "text": "Selamat pagi, Bu.", "speakers": ["agent"]},
        {"id": "A_0001", "call_id": "A", "seg_index": 1, "text": "Limitnya Rp15.000.000.", "speakers": ["agent"],
         "crm": {"customer_name": "Sari"}},
        {"id": "A_0002", "call_id": "A", "seg_index": 2, "text": "Iya, Sari di sini.", "speakers": ["customer"],
         "crm": {"customer_name": "Sari"}},
    ]
    preds = [{"id": "A_0000", "text": "selamat pagi bu"}, {"id": "A_0001", "text": "Limitnya Rp15.000.000."},
             {"id": "A_0002", "text": "Iya, Sary di sini."}]
    m, p = tmp_path / "m.jsonl", tmp_path / "p.jsonl"
    m.write_text("\n".join(json.dumps(x) for x in manifest))
    p.write_text("\n".join(json.dumps(x) for x in preds))
    rep = score(str(m), str(p))
    assert rep["missing_predictions"] == 0
    assert rep["text"]["n"] == 3
    assert rep["entities"]["nominal"]["exact_match"] == 1
    assert rep["name_consistency"] == {"n": 1, "exact_spelling": 0.0}
    assert set(rep["groups"]) == {"posisi:awal", "posisi:tengah", "posisi:akhir", "speaker:agent", "speaker:customer"}
