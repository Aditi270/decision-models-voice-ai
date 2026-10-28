"""Measure the other components of a voice turn so decision latency can be put in context.

- LLM reply generation: time to first streamed token (TTFT) and to full reply, per LLM
- ElevenLabs TTS: time to first audio byte (streaming endpoint), English and Hindi
Writes results/latency_budget.json
"""
import json, statistics, time, urllib.request
from pathlib import Path

import clients, voice

ROOT = Path(__file__).resolve().parent.parent
N = 20                 # samples per measurement; enough for a stable median
REPLY_MAX_TOKENS = 120  # a one-to-two sentence spoken reply
REPLY_SYS = ("You are a friendly receptionist voice agent for City Care Clinic in India. Reply in one or two "
             "short spoken sentences, in the caller's language.")
PROMPTS = [("Agent: How can I help you today?\nCaller: I need to cancel my appointment for tomorrow.", "en"),
           ("Agent: मैं आपकी क्या मदद कर सकती हूँ?\nCaller: मुझे कल का अपॉइंटमेंट कैंसल करना है।", "hi")]
REPLIES = {"en": "Sure, I can help with that. Could you please tell me the name the appointment is booked under?",
           "hi": "ज़रूर, मैं आपकी मदद करती हूँ। कृपया बताइए अपॉइंटमेंट किस नाम से बुक है?"}


def llm_stream(model, user):
    body = {"model": model, "max_tokens": REPLY_MAX_TOKENS, "stream": True,
            "messages": [{"role": "system", "content": REPLY_SYS}, {"role": "user", "content": user}],
            **clients.LLM_EXTRA.get(model, {})}
    req = urllib.request.Request(f"{clients.GW}/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {clients.KEY}", "Content-Type": "application/json"})
    t = time.perf_counter(); first = None
    with urllib.request.urlopen(req, timeout=clients.TIMEOUT_S) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                d = json.loads(line[len(b"data: "):])   # server-sent event line
                if first is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                    first = time.perf_counter() - t
    return first, time.perf_counter() - t


def tts_ttfb(text, lang, v):
    body = {"text": text, "model_id": voice.TTS_MODEL, "language_code": lang}
    req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{v}/stream?output_format=pcm_{voice.TTS_RATE}",
                                 json.dumps(body).encode(), {**voice.H, "Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=clients.TIMEOUT_S) as r:
        r.read(1); first = time.perf_counter() - t
        r.read()
    return first, time.perf_counter() - t


def summ(xs):
    xs = sorted(x for x in xs if x is not None)
    return dict(n=len(xs), median=statistics.median(xs), p90=xs[int(0.9 * (len(xs) - 1))], all=xs)


if __name__ == "__main__":
    out = {"llm_reply": {}, "tts": {}}
    for name, (kind, model) in clients.SYSTEMS.items():
        if kind != "llm":
            continue
        ttft, total = [], []
        for i in range(N):
            a, b = llm_stream(model, PROMPTS[i % 2][0]); ttft.append(a); total.append(b)
        out["llm_reply"][name] = dict(ttft=summ(ttft), total=summ(total))
        print(name, out["llm_reply"][name]["ttft"]["median"], out["llm_reply"][name]["total"]["median"], flush=True)
    for lang in ("en", "hi"):
        f, tot = [], []
        for i in range(N // 2):   # TTS is far less variable than LLMs; 10 per language is enough
            a, b = tts_ttfb(REPLIES[lang], lang, voice.VOICES[i % len(voice.VOICES)]); f.append(a); tot.append(b)
        out["tts"][lang] = dict(ttfb=summ(f), total=summ(tot))
        print("tts", lang, out["tts"][lang]["ttfb"]["median"], flush=True)
    from analyze import round3
    (ROOT / "results" / "latency_budget.json").write_text(json.dumps(round3(out), indent=1))
