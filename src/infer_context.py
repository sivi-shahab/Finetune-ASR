"""Inferensi WhisperX dengan konteks percakapan (docs/CONTEXT_ENGINEERING.md bagian 6).

Jalan di env whisperx-infer. Dua mode:

1) Produksi: audio panggilan penuh → VAD → chunk ≤ 30 dtk → dekode berurutan dengan prompt
   (domain + CRM + riwayat) → align → diarization.
       python src/infer_context.py audio --model outputs/whisper-telemarketing-ct2 \
           --audio-dir raw/audio --out-dir outputs/transcripts --level 2

2) Evaluasi: segmen manifest per panggilan, berurutan, riwayat dari prediksi model sendiri.
   Hasilnya {"id","text"} untuk `src/evaluate.py score`. Flag --no-* untuk ablation C0–C5.
       python src/infer_context.py manifest --model outputs/whisper-telemarketing-ct2 \
           --manifest dataset/test.jsonl --out outputs/preds/test-C4.jsonl

Dua GPU: jalankan dua proses, masing-masing CUDA_VISIBLE_DEVICES=0/1 dan --worker-index 0/1
--num-workers 2 (lihat scripts/run_infer_2gpu.sh).

PERHATIAN: get_vad_chunks() dan load_diarizer() memakai API internal WhisperX yang berubah
antar versi. Uji pada versi yang dipin sebelum dipakai.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from context_builder import Tok, append_history, build_prompt_ids, load_domain_prompt  # noqa: E402
from normalize import SECRET_RE  # noqa: E402

log = logging.getLogger("infer_context")
SR = 16000

# Penjaga (CONTEXT_ENGINEERING.md bagian 6)
MAX_COMPRESSION = 2.4
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH = 0.6
COPY_NGRAM = 8


@dataclass
class Layers:
    domain: bool = True
    crm: bool = True
    history: bool = True
    guard: bool = True


# ---------------------------------------------------------------------------
# Dekode satu chunk dengan konteks
# ---------------------------------------------------------------------------


def copied_from_prompt(text: str, history: str, n: int = COPY_NGRAM) -> bool:
    out, hist = text.split(), history.split()
    if len(out) < n or len(hist) < n:
        return False
    grams = {tuple(hist[i:i + n]) for i in range(len(hist) - n + 1)}
    return any(tuple(out[i:i + n]) in grams for i in range(len(out) - n + 1))


def is_bad(segs, text: str, history: str) -> bool:
    if not text.strip():
        return False
    for s in segs:
        if s.compression_ratio > MAX_COMPRESSION or s.avg_logprob < MIN_AVG_LOGPROB:
            return True
        if s.no_speech_prob > MAX_NO_SPEECH:
            return True
    return copied_from_prompt(text, history)


class ContextDecoder:
    """Loop berurutan per chunk. Riwayat hanya diisi hasil yang lolos penjaga dan bukan rahasia."""

    def __init__(self, asr, domain: str, layers: Layers, beam_size: int = 5):
        self.asr = asr
        self.tok = Tok(asr.hf_tokenizer)
        self.domain = domain if layers.domain else ""
        self.layers = layers
        self.beam_size = beam_size

    def _decode(self, chunk, prompt_ids):
        segs, _ = self.asr.transcribe(
            chunk, language="id", task="transcribe", initial_prompt=prompt_ids or None,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
            beam_size=self.beam_size)
        segs = list(segs)
        return segs, " ".join(s.text.strip() for s in segs).strip()

    def run(self, chunks, crm: dict | None):
        """chunks: iterable (key, audio_float32). Hasil: list dict per chunk."""
        crm = crm if self.layers.crm else None
        history, out = "", []
        for key, audio in chunks:
            hist = history if self.layers.history else ""
            segs, text = self._decode(audio, build_prompt_ids(self.tok, self.domain, crm, hist))
            low_conf = False
            if self.layers.guard and is_bad(segs, text, hist):
                segs, text = self._decode(audio, build_prompt_ids(self.tok, self.domain, crm, ""))
                low_conf = is_bad(segs, text, "")
                retried = True
            else:
                retried = False
            secret = bool(SECRET_RE.search(text))
            # Hasil dekode ulang tidak pernah dipakai sebagai konteks.
            history = append_history(history, text, secret=secret, low_confidence=retried or low_conf)
            out.append({"key": key, "text": text, "retried": retried, "low_confidence": low_conf,
                        "secret": secret})
        return out


# ---------------------------------------------------------------------------
# WhisperX: VAD, align, diarization (API bergantung versi)
# ---------------------------------------------------------------------------


def get_vad_chunks(pipeline, audio, chunk_size: int = 30) -> list[tuple[float, float]]:
    """Meniru langkah VAD di FasterWhisperPipeline.transcribe. Cek ulang pada versi yang dipin."""
    import torch
    vad_model, params = pipeline.vad_model, pipeline._vad_params
    try:
        from whisperx.vads import Pyannote, Vad   # whisperx >= 3.4
        if isinstance(vad_model, Vad):
            waveform, merge = vad_model.preprocess_audio(audio), vad_model.merge_chunks
        else:
            waveform, merge = Pyannote.preprocess_audio(audio), Pyannote.merge_chunks
    except ImportError:                            # whisperx < 3.4
        from whisperx.vad import merge_chunks as merge
        waveform = torch.from_numpy(audio).unsqueeze(0)
    segs = vad_model({"waveform": waveform, "sample_rate": SR})
    segs = merge(segs, chunk_size, onset=params["vad_onset"], offset=params["vad_offset"])
    return [(float(s["start"]), float(s["end"])) for s in segs]


def load_diarizer(device: str):
    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError:
        from whisperx import DiarizationPipeline
    token = os.environ.get("HF_TOKEN")
    try:
        return DiarizationPipeline(use_auth_token=token, device=device)
    except TypeError:
        return DiarizationPipeline(token=token, device=device)


def load_aligner(device: str, model_name: str | None):
    import whisperx
    try:
        return whisperx.load_align_model(language_code="id", device=device, model_name=model_name)
    except ValueError as e:
        raise SystemExit(f"Model alignment untuk 'id' tidak tersedia ({e}). "
                         "Berikan model wav2vec2 Indonesia lewat --align-model.") from e


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


def shard(items: list, index: int, total: int) -> list:
    return [x for x in items if zlib.crc32(str(x).encode()) % total == index]


def read_crm(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["call_id"]: {k: r[k] for k in ("customer_name", "product") if r.get(k)} for r in rows}


def run_audio(a, layers: Layers) -> None:
    import whisperx
    domain = load_domain_prompt(a.domain_prompt)
    files = sorted(Path(a.audio_dir).glob("*.wav")) if a.audio_dir else [Path(p) for p in a.audio]
    files = shard(files, a.worker_index, a.num_workers)
    crm_all = read_crm(a.crm)
    asr_options = {"initial_prompt": domain} if a.level == 1 and layers.domain else {}
    pipeline = whisperx.load_model(a.model, a.device, compute_type=a.compute_type, language="id",
                                   asr_options=asr_options)
    align_model, align_meta = load_aligner(a.device, a.align_model)
    diarizer = None if a.no_diarize else load_diarizer(a.device)
    decoder = ContextDecoder(pipeline.model, domain, layers, a.beam_size)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        call_id = path.stem
        dst = out_dir / f"{call_id}.json"
        if dst.exists() and not a.overwrite:
            continue
        audio = whisperx.load_audio(str(path))
        if a.level == 1:
            segments = pipeline.transcribe(audio, batch_size=a.batch_size, language="id")["segments"]
            flags = [{} for _ in segments]
        else:
            chunks = get_vad_chunks(pipeline, audio)
            res = decoder.run(((c, audio[int(c[0] * SR):int(c[1] * SR)]) for c in chunks), crm_all.get(call_id))
            segments = [{"start": r["key"][0], "end": r["key"][1], "text": r["text"]} for r in res if r["text"]]
            flags = [{k: r[k] for k in ("retried", "low_confidence", "secret")} for r in res if r["text"]]
        aligned = whisperx.align(segments, align_model, align_meta, audio, a.device, return_char_alignments=False)
        if diarizer is not None:
            aligned = whisperx.assign_word_speakers(diarizer(audio, min_speakers=2, max_speakers=2), aligned)
        segs = aligned["segments"]
        if len(segs) == len(flags):   # align bisa memecah segmen; flag hanya ditempel bila jumlahnya sama
            for s, f in zip(segs, flags):
                s.update(f)
        dst.write_text(json.dumps({"call_id": call_id, "level": a.level, "model": a.model,
                                   "segments": segs, "text": " ".join(s["text"].strip() for s in segs)},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("%s: %d segmen", call_id, len(segs))


def run_manifest(a, layers: Layers) -> None:
    import soundfile as sf
    from faster_whisper import WhisperModel
    domain = load_domain_prompt(a.domain_prompt)
    rows = [json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    calls = defaultdict(list)
    for r in rows:
        calls[r["call_id"]].append(r)
    call_ids = shard(sorted(calls), a.worker_index, a.num_workers)
    asr = WhisperModel(a.model, device=a.device, compute_type=a.compute_type)
    decoder = ContextDecoder(asr, domain, layers, a.beam_size)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for n, cid in enumerate(call_ids, 1):
            segs = sorted(calls[cid], key=lambda r: r["seg_index"])
            chunks = ((r["id"], sf.read(r["audio_filepath"], dtype="float32")[0]) for r in segs)
            for res in decoder.run(chunks, segs[0].get("crm")):
                f.write(json.dumps({"id": res["key"], "text": res["text"], "retried": res["retried"],
                                    "low_confidence": res["low_confidence"]}, ensure_ascii=False) + "\n")
            log.info("%d/%d panggilan", n, len(call_ids))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("audio", "manifest"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="direktori model CTranslate2")
        p.add_argument("--domain-prompt", default="configs/domain_prompt.txt")
        p.add_argument("--device", default="cuda")
        p.add_argument("--compute-type", default="float16")
        p.add_argument("--beam-size", type=int, default=5)
        p.add_argument("--worker-index", type=int, default=0)
        p.add_argument("--num-workers", type=int, default=1)
        for layer in ("domain", "crm", "history", "guard"):
            p.add_argument(f"--no-{layer}", action="store_true", help=f"matikan lapisan {layer} (ablation)")
    pa = sub.choices["audio"]
    pa.add_argument("--audio", nargs="*", default=[])
    pa.add_argument("--audio-dir", default=None)
    pa.add_argument("--out-dir", default="outputs/transcripts")
    pa.add_argument("--crm", default="raw/crm.jsonl")
    pa.add_argument("--level", type=int, choices=[1, 2], default=2)
    pa.add_argument("--batch-size", type=int, default=16, help="hanya level 1")
    pa.add_argument("--align-model", default=None)
    pa.add_argument("--no-diarize", action="store_true")
    pa.add_argument("--overwrite", action="store_true")
    pm = sub.choices["manifest"]
    pm.add_argument("--manifest", required=True)
    pm.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    layers = Layers(domain=not a.no_domain, crm=not a.no_crm, history=not a.no_history, guard=not a.no_guard)
    (run_audio if a.mode == "audio" else run_manifest)(a, layers)


if __name__ == "__main__":
    main()
