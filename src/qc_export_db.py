"""Ekspor metadata + transkrip ElevenLabs panggilan QC dari DB telemarketing-qc (READ ONLY).

Dijalankan DI DALAM container telemarketing-qc-api (sudah punya SQLAlchemy, psycopg2 dan
kredensial DB di env), sehingga tidak perlu memasang driver atau menyalin kredensial:

    docker exec -i telemarketing-qc-api-api-1 python - --scope qc_reviewed \
        < src/qc_export_db.py > raw/qc_export.jsonl

Keluaran: satu baris JSON per file panggilan (stem PDF = stem audio = call_id):
    {"call_id", "result_id", "campaign", "qc_status", "qc_tl_status", "uploaded_at",
     "file_path", "customer_name", "agent_id", "product", "elevenlabs": {...respons mentah...}}

Hanya SELECT; sesi diset default_transaction_read_only=on.
Sumber: dashboard.results, qc_status_requests, recording_tms_api, result_speech_to_text.
"""
import argparse
import json
import os
import sys

from sqlalchemy import create_engine, text

SQL_CALLS = """
WITH r AS (
    SELECT id, campaign, uploaded_at, jsonb_array_elements_text(source_files) AS pdf
    FROM results
    WHERE status = 'done' AND source_files IS NOT NULL
)
SELECT DISTINCT ON (call_id)
       regexp_replace(r.pdf, '(\\s*\\(\\d+\\))?\\.pdf$', '', 'i') AS call_id,
       r.id::text AS result_id, r.campaign, r.uploaded_at,
       q.requested_status AS qc_status, q.tl_qc_status AS qc_tl_status
FROM r
LEFT JOIN qc_status_requests q ON q.result_id = r.id
ORDER BY call_id, r.uploaded_at DESC
"""

SQL_RECORDING = """
SELECT DISTINCT ON (tiket_id) tiket_id, file_path, cust_name, agent_id, campaign
FROM recording_tms_api
WHERE tiket_id = ANY(:ids)
ORDER BY tiket_id, inserted_at DESC NULLS LAST
"""

SQL_STT = """
SELECT DISTINCT ON (stem) stem, asr_raw_response
FROM (
    SELECT regexp_replace(audio_file, '\\.[^.]+$', '') AS stem, asr_raw_response, created_at
    FROM result_speech_to_text
    WHERE asr_backend = 'elevenlabs' AND asr_raw_response IS NOT NULL
      AND regexp_replace(audio_file, '\\.[^.]+$', '') = ANY(:ids)
) s
ORDER BY stem, created_at DESC
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["qc_reviewed", "qc_all"], default="qc_reviewed",
                    help="qc_reviewed: hanya tiket dengan Manual Status PASS/FAIL")
    ap.add_argument("--campaign", default=None, help="mis. Cashline")
    ap.add_argument("--since", default=None, help="uploaded_at >= YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args()

    url = (f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
           f"@{os.environ['POSTGRES_HOST']}:{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ['POSTGRES_DB']}")
    schema = os.environ.get("POSTGRES_SCHEMA", "dashboard")
    eng = create_engine(url, connect_args={
        "options": f"-csearch_path={schema} -cdefault_transaction_read_only=on"})

    with eng.connect() as c:
        assert c.execute(text("show transaction_read_only")).scalar() == "on"
        calls = [dict(r._mapping) for r in c.execute(text(SQL_CALLS))]
        if a.scope == "qc_reviewed":
            calls = [r for r in calls if r["qc_status"] in ("PASS", "FAIL")]
        if a.campaign:
            calls = [r for r in calls if (r["campaign"] or "").lower() == a.campaign.lower()]
        if a.since:
            calls = [r for r in calls if r["uploaded_at"] and str(r["uploaded_at"]) >= a.since]
        calls.sort(key=lambda r: r["call_id"])
        if a.limit:
            calls = calls[:a.limit]
        print(f"[qc_export_db] {len(calls)} file panggilan (scope={a.scope})", file=sys.stderr)

        written = missing_stt = 0
        for i in range(0, len(calls), a.batch):
            chunk = calls[i:i + a.batch]
            ids = [r["call_id"] for r in chunk]
            rec = {r.tiket_id: r for r in c.execute(text(SQL_RECORDING), {"ids": ids})}
            stt = {r.stem: r.asr_raw_response for r in c.execute(text(SQL_STT), {"ids": ids})}
            for r in chunk:
                el = stt.get(r["call_id"])
                if el is None:
                    missing_stt += 1
                    continue
                rc = rec.get(r["call_id"])
                out = {
                    "call_id": r["call_id"],
                    "result_id": r["result_id"],
                    "campaign": r["campaign"],
                    "qc_status": r["qc_status"],
                    "qc_tl_status": r["qc_tl_status"],
                    "uploaded_at": str(r["uploaded_at"]) if r["uploaded_at"] else None,
                    "file_path": rc.file_path if rc else None,
                    "customer_name": rc.cust_name if rc else None,
                    "agent_id": rc.agent_id if rc else None,
                    "product": rc.campaign if rc else None,
                    "elevenlabs": json.loads(el) if isinstance(el, str) else el,
                }
                sys.stdout.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
                written += 1
            print(f"[qc_export_db] {min(i + a.batch, len(calls))}/{len(calls)}", file=sys.stderr)
        print(f"[qc_export_db] selesai: {written} ditulis, {missing_stt} tanpa JSON ElevenLabs", file=sys.stderr)


if __name__ == "__main__":
    main()
