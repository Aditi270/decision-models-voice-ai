"""Uniform callers for decision models and LLM baselines (all via Vercel AI Gateway).

Every system receives the *same* state text and the *same* question wording.
Each call returns a normalised dict:
  intent, intent_p (prob of chosen intent), p_esc, p_done, reply, reply_p,
  latency_s (wall clock, successful attempt), cost_usd, ok, error, raw
"""
import json, os, time, urllib.request, urllib.error

import env  # noqa: F401  (loads .env)

GW = "https://ai-gateway.vercel.sh/v1"
KEY = os.environ["AI_GATEWAY_API_KEY"]

TIMEOUT_S = 60          # generous: a stuck call is retried rather than counted as slow
RETRIES = 4             # attempts per call; waits 1.5 s, 3 s, 6 s between them
RETRY_CODES = (408, 409, 429, 500, 502, 503, 504)   # transient errors (timeout, rate limit, server side)
LLM_MAX_TOKENS = 200    # the JSON answer is ~20-60 tokens; leaves headroom without letting a model ramble

INTENT_DEFS = {
    "book": "Caller wants a new appointment or is in the middle of booking one",
    "change": "Caller wants to reschedule or cancel an existing appointment",
    "billing": "Bills, payments, fees, refunds or insurance",
    "reports": "Lab or test results, medical reports, or prescriptions",
    "emergency": "An urgent medical situation that may need immediate care",
    "info": "General clinic information: hours, location, services, doctor availability",
    "other": "Greetings, goodbyes, requests for a human, complaints, wrong numbers, or anything else",
}
Q_INTENT = "What is the caller's main purpose in this turn?"
Q_ESC = ("Should this call be handed to a human staff member right now? Yes if there is a possible "
         "medical emergency or risk to life, the caller asks for a human, the caller is angry or very "
         "frustrated or threatens a complaint or legal action, or the caller needs medical advice or "
         "interpretation of results that an automated agent must not give.")
Q_DONE = ("Has the caller finished their turn, so the agent should reply now? No if the caller trailed "
          "off mid-sentence, is still giving information such as digits, or asked for a moment to think "
          "or look something up.")
LANG_DEFS = {"english": "Reply in English",
             "hindi": "Reply in Hindi (colloquial Hindi or Hinglish is fine)"}
Q_LANG = ("Which language should the voice agent use for its next reply? Follow the caller's explicit "
          "request if they make one; otherwise match the language the caller is speaking.")

def state_text(agent, caller):
    return f"Phone call to City Care Clinic (India). Transcript of the latest exchange:\nAgent: {agent}\nCaller: {caller}"


def _post(path, body, timeout=TIMEOUT_S, retries=RETRIES):
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f"{GW}/{path}", data, {
            "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        t = time.perf_counter()
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            return out, time.perf_counter() - t, attempt, None
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            if e.code not in RETRY_CODES:
                break
        except Exception as e:  # timeouts, resets
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * 2 ** attempt)
    return None, None, retries, last


# ---------------------------------------------------------------- decision models
def _decision_questions():
    return [
        {"type": "choice", "name": "intent", "instructions": Q_INTENT,
         "choices": [{"value": k, "description": v} for k, v in INTENT_DEFS.items()]},
        {"type": "predicate", "name": "esc", "instructions": Q_ESC},
        {"type": "predicate", "name": "done", "instructions": Q_DONE},
        {"type": "choice", "name": "reply", "instructions": Q_LANG,
         "choices": [{"value": k, "description": v} for k, v in LANG_DEFS.items()]},
    ]


def call_decision(model, agent, caller):
    body = {"model": model, "input": state_text(agent, caller), "questions": _decision_questions()}
    raw, lat, retries, err = _post("decisions", body)
    res = dict(latency_s=lat, retries=retries, ok=raw is not None, error=err, raw=raw)
    if raw is None:
        return res
    try:
        for a in raw["answers"]:
            n = a["name"]
            if a["type"] == "predicate":
                res[f"p_{n}"] = float(a["probability"])
            else:
                res[n] = a["choice"]
                probs = {p["value"]: p["probability"] for p in a.get("probabilities", [])}
                res[f"{n}_p"] = float(probs.get(a["choice"], a.get("confidence", 0)))
                res[f"{n}_conf"] = a.get("confidence")
                res[f"{n}_probs"] = probs
        gw = raw.get("provider_metadata", {}).get("gateway", {})
        res["cost_usd"] = float(gw.get("cost", 0) or 0)
        res["in_tokens"] = raw.get("usage", {}).get("input_tokens")
        try:
            pa = gw["routing"]["modelAttempts"][-1]["providerAttempts"][-1]
            res["provider_s"] = (pa["endTime"] - pa["startTime"]) / 1000
        except (KeyError, IndexError, TypeError):
            pass
    except Exception as e:
        res.update(ok=False, error=f"parse: {e}")
    return res


# ---------------------------------------------------------------- LLM baselines
def _llm_schema():
    props = {"intent": {"type": "string", "enum": list(INTENT_DEFS)},
             "intent_confidence": {"type": "number"},
             "escalate_probability": {"type": "number"},
             "turn_complete_probability": {"type": "number"},
             "reply_language": {"type": "string", "enum": list(LANG_DEFS)},
             "reply_language_confidence": {"type": "number"}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _llm_system():
    return "\n\n".join([
        "You are the decision component of a voice agent answering phone calls for City Care Clinic "
        "in India. Callers speak English, Hindi or Hinglish. Read the latest exchange and answer the "
        "questions below. Respond only with JSON matching the schema. All probabilities and "
        "confidences are numbers between 0 and 1.",
        "intent: " + Q_INTENT + " Options:\n" + "\n".join(f"- {k}: {v}" for k, v in INTENT_DEFS.items()) +
        "\nintent_confidence: probability that your chosen intent is correct.",
        "escalate_probability: probability that the answer to this is yes: " + Q_ESC,
        "turn_complete_probability: probability that the answer to this is yes: " + Q_DONE,
        "reply_language: " + Q_LANG + " Options:\n" + "\n".join(f"- {k}: {v}" for k, v in LANG_DEFS.items()) +
        "\nreply_language_confidence: probability that your chosen language is correct.",
    ])


# GPT-6 Luna reasons by default (~45 hidden tokens, +0.3 s); turned off so it is compared like the
# other LLMs, which answer directly. temperature 0 everywhere = most repeatable answers.
LLM_EXTRA = {
    "openai/gpt-6-luna": {"reasoning_effort": "none"},
}


def call_llm(model, agent, caller):
    body = {"model": model, "max_tokens": LLM_MAX_TOKENS, "temperature": 0,
            "messages": [{"role": "system", "content": _llm_system()},
                         {"role": "user", "content": state_text(agent, caller)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "decision", "strict": True, "schema": _llm_schema()}},
            **LLM_EXTRA.get(model, {})}
    raw, lat, retries, err = _post("chat/completions", body)
    res = dict(latency_s=lat, retries=retries, ok=raw is not None, error=err, raw=raw)
    if raw is None:
        return res
    u = raw.get("usage", {})
    res["cost_usd"] = float(u.get("cost", 0) or 0)
    res["in_tokens"], res["out_tokens"] = u.get("prompt_tokens"), u.get("completion_tokens")
    try:
        txt = raw["choices"][0]["message"]["content"].strip()
        if txt.startswith("```"):
            txt = txt.strip("`").removeprefix("json").strip()
        d = json.loads(txt)
        res["intent"], res["intent_p"] = d["intent"], float(d["intent_confidence"])
        res["p_esc"] = float(d["escalate_probability"])
        res["p_done"] = float(d["turn_complete_probability"])
        res["reply"], res["reply_p"] = d["reply_language"], float(d["reply_language_confidence"])
        assert res["intent"] in INTENT_DEFS and res["reply"] in LANG_DEFS
    except Exception as e:
        res.update(ok=False, error=f"parse: {type(e).__name__}: {e}")
    return res


SYSTEMS = {
    # name: (kind, model id)
    "Jev": ("decision", "typesafe-ai/jev"),
    "OpenAI Decisions": ("decision", "openai/gpt-6-luna-decisions"),
    "GPT-6 Luna (LLM)": ("llm", "openai/gpt-6-luna"),
    "Claude Haiku 4.5 (LLM)": ("llm", "anthropic/claude-haiku-4.5"),
    "Gemini 3.1 Flash Lite (LLM)": ("llm", "google/gemini-3.1-flash-lite"),
}


def call(system, agent, caller):
    kind, model = SYSTEMS[system]
    return (call_decision if kind == "decision" else call_llm)(model, agent, caller)
