"""Voice leg: caller text -> ElevenLabs TTS -> simulated phone line -> ElevenLabs STT (Scribe).

Writes data/audio/<id>.wav (8 kHz telephone-quality) and data/asr_turns.jsonl, which has the
same fields as turns.jsonl but with 'caller' replaced by the ASR transcript (original kept in
'caller_text'), plus timings.

Phone-line simulation: 300-3400 Hz band-pass, resample to 8 kHz, 8-bit mu-law companding
(what G.711 telephony does), and additive noise at 20 dB SNR.
"""
import io, json, os, sys, time, uuid, wave, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import env  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "data" / "audio"; AUD.mkdir(parents=True, exist_ok=True)
H = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}
# ElevenLabs premade voices, rotated so the set isn't one speaker
VOICES = ["JBFqnCBsd6RMkjVDRZzb", "21m00Tcm4TlvDq8ikWAM", "pNInz6obpgDQGcFmaJgB",
          "Xb7hH8MSUJpSbSDYk0k2", "onwK4e9ZLuTAKqWW03F9", "XrExE9yKIg1WjnnlVkGX",
          "nPczCjzI2devNBz1zQrb", "pFZP5JQG7iQjIQuC4Bku", "cjVigY5qzO86Huf0OWal", "FGY2WhTYpPnrIDTdsKH5"]
TTS_MODEL = "eleven_flash_v2_5"     # ElevenLabs' low-latency model, supports Hindi via language_code
STT_MODEL = os.environ.get("STT_MODEL", "scribe_v1")
rng = np.random.default_rng(0)      # seeded: the same noise is added on every re-run

TTS_RATE = 16000          # Hz, the PCM rate requested from ElevenLabs
PHONE_RATE = 8000         # Hz, narrow-band telephone audio (G.711)
PHONE_BAND = (300, 3400)  # Hz, the frequency band a phone line carries
MU = 255                  # mu-law companding constant used by G.711
SNR_DB = 20               # background noise level: audible but not severe
WAV_HEADER = 44           # bytes in a standard PCM .wav header
RETRIES, TIMEOUT_S = 4, 60


def _req(url, data, headers, timeout=TIMEOUT_S):
    for attempt in range(RETRIES):
        try:
            t = time.perf_counter()
            out = urllib.request.urlopen(urllib.request.Request(url, data, {**H, **headers}), timeout=timeout).read()
            return out, time.perf_counter() - t
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code not in (429, 500, 502, 503):
                raise RuntimeError(f"HTTP {e.code} {msg}")
        except Exception as e:  # noqa
            msg = str(e)
        time.sleep(2 * 2 ** attempt)
    raise RuntimeError(msg)


def tts(text, voice, lang):
    body = {"text": text, "model_id": TTS_MODEL, "language_code": lang}
    pcm, dt = _req(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_{TTS_RATE}",
                   json.dumps(body).encode(), {"Content-Type": "application/json"})
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768, dt   # 16-bit PCM -> [-1, 1]


def phone_line(x, sr=TTS_RATE, snr_db=SNR_DB):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / sr)
    X[(f < PHONE_BAND[0]) | (f > PHONE_BAND[1])] = 0
    y = np.fft.irfft(X, len(x))[::sr // PHONE_RATE]           # downsample to 8 kHz
    mu = MU; y = np.clip(y / (np.abs(y).max() + 1e-9) * 0.9, -1, 1)   # normalise to 90% of full scale
    q = np.round((np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)) * 127) / 127   # 8-bit mu-law
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    noise = rng.standard_normal(len(y)); noise = np.cumsum(noise) * 0.02 + noise   # white + a little low-frequency rumble, like line hiss
    noise *= np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return np.clip(y + noise, -1, 1), PHONE_RATE


def to_wav(y, sr):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    return b.getvalue()


def stt(wav):
    bd = uuid.uuid4().hex
    body = (f"--{bd}\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\n{STT_MODEL}\r\n"
            f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{bd}--\r\n".encode()
    out, dt = _req("https://api.elevenlabs.io/v1/speech-to-text", body,
                   {"Content-Type": f"multipart/form-data; boundary={bd}"})
    j = json.loads(out)
    return j.get("text", "").strip(), j.get("language_code"), dt


def process(t, i):
    p = AUD / f"{t['id']}.wav"
    lang = "en" if t["lang"] == "en" else "hi"
    rec = dict(t, caller_text=t["caller"], voice=VOICES[i % len(VOICES)])
    if p.exists():
        wav = p.read_bytes(); rec["tts_s"] = None
    else:
        x, rec["tts_s"] = tts(t["caller"], rec["voice"], lang)
        y, sr = phone_line(x)
        wav = to_wav(y, sr); p.write_bytes(wav)
    rec["audio_s"] = (len(wav) - WAV_HEADER) / 2 / PHONE_RATE   # 2 bytes per sample
    rec["caller"], rec["asr_lang"], rec["stt_s"] = stt(wav)
    return rec


if __name__ == "__main__":
    # usage: python voice.py [turns.jsonl] [asr_out.jsonl]
    src = sys.argv[1] if len(sys.argv) > 1 else "turns.jsonl"
    turns = [json.loads(l) for l in open(ROOT / "data" / src, encoding="utf-8")]
    out = ROOT / "data" / (sys.argv[2] if len(sys.argv) > 2 else "asr_turns.jsonl")
    with ThreadPoolExecutor(3) as ex, open(out, "w", encoding="utf-8") as f:
        for r in ex.map(lambda a: process(*a), [(t, i) for i, t in enumerate(turns)]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(r["id"], f"{r['audio_s']:.1f}s audio, stt {r['stt_s']:.2f}s |", r["caller_text"], "=>", r["caller"], flush=True)
