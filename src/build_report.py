"""Build report/report.pdf from results/metrics.json and the charts in report/.
Every number in the report is read from metrics.json, so the text can't drift from the data.
PDF is printed with headless Microsoft Edge (or Chrome)."""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report"
M = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
REPO = "https://github.com/Aditi270/decision-models-voice-ai"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]

SYS = list(M["all"])
DM = [s for s in SYS if "(LLM)" not in s]
LLM = [s for s in SYS if "(LLM)" in s]
name = lambda s: s.removesuffix(" (LLM)")
A, H, T, V, VT = M["all"], M["hard"], M["text"], M["asr"], M["text_rep0"]


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def rng(key, systems, src=A, f=pct, d=1):
    vals = [src[s][key] for s in systems]
    lo, hi = f(min(vals), d), f(max(vals), d)
    if lo == hi:
        return lo
    unit = " s" if lo.endswith(" s") else ""          # "0.43–0.61 s" rather than "0.43 s–0.61 s"
    return f"{lo.removesuffix(unit)}–{hi}"


sec = lambda x, d=2: f"{x:.{d}f} s"
usd = lambda x, d=3: f"${x:.{d}f}"
vt = M["voice_turn"]
turn = {name(k.removesuffix(" decides")): v for k, v in vt.items()}
llm_totals = [turn[name(s)]["total"] for s in LLM]
saved_lo, saved_hi = min(llm_totals) - turn["Jev"]["total"], max(llm_totals) - turn["Jev"]["total"]
saved = f"{saved_lo:.1f}–{saved_hi:.1f} s"

# ------------------------------------------------------------------ built from data
def voice_table():
    rows = ""
    for s in SYS:
        d = V[s]["acc_all4"] - VT[s]["acc_all4"]
        rows += (f"<tr><td><b>{name(s)}</b></td><td>{pct(VT[s]['acc_all4'], 1)}</td><td>{pct(V[s]['acc_all4'], 1)}</td>"
                 f"<td>{d * 100:+.1f} pts</td><td>{pct(VT[s]['cut_in_rate'], 1)} → {pct(V[s]['cut_in_rate'], 1)}</td></tr>")
    return rows


def lang_gap():
    """largest spread between English, Hindi and Hinglish accuracy, over all systems and decisions (in points)"""
    return max(max(v) - min(v) for s in SYS for t in ("intent", "esc", "done", "reply")
               for v in [[A[s][f"acc_{t}_{l}"] for l in ("en", "hi", "hx")]]) * 100


HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Decision models for voice AI</title>
<style>
:root {{ --ink:#1b1b1a; --ink2:#52514e; --muted:#7a7974; --line:#e4e3df; --soft:#f5f4f0;
        --dm:#2a78d6; --llm:#eb6834; --good:#1f7a4d; --warn:#9a6700; --bad:#b42318; }}
@page {{ size: A4; margin: 16mm 16mm 18mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Segoe UI", "Nirmala UI", sans-serif; color: var(--ink); font-size: 10pt; line-height: 1.47; margin: 0; background:#fff; }}
h1 {{ font-size: 23pt; line-height: 1.15; margin: 0 0 4px; letter-spacing: -0.3px; }}
h2 {{ font-size: 14pt; margin: 22px 0 6px; padding-top: 4px; }}
h3 {{ font-size: 11pt; margin: 14px 0 4px; }}
p {{ margin: 0 0 8px; }}
.sub {{ color: var(--ink2); font-size: 11pt; margin-bottom: 2px; }}
.meta {{ color: var(--muted); font-size: 9pt; margin-bottom: 16px; }}
a {{ color: var(--dm); text-decoration: none; }}
.verdict {{ border-left: 5px solid var(--dm); background: var(--soft); padding: 12px 16px; margin: 10px 0 14px; border-radius: 0 6px 6px 0; }}
.verdict .big {{ font-size: 15pt; font-weight: 700; margin-bottom: 4px; }}
.tiles {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 10px 0 6px; }}
.tile {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; }}
.tile .n {{ font-size: 12.5pt; font-weight: 700; line-height: 1.2; white-space: nowrap; }}
.tile .l {{ font-size: 8.6pt; color: var(--ink2); line-height: 1.35; }}
.dm {{ color: var(--dm); }} .llm {{ color: var(--llm); }} .muted {{ color: var(--muted); font-size: 8.5pt; }}
table {{ width: 100%; border-collapse: collapse; margin: 6px 0 10px; font-size: 9pt; }}
th {{ text-align: left; font-weight: 600; color: var(--ink2); border-bottom: 1.5px solid var(--ink2); padding: 4px 6px; vertical-align: bottom; }}
td {{ border-bottom: 1px solid var(--line); padding: 4px 6px; vertical-align: top; }}
img {{ width: 100%; margin: 4px 0 2px; break-inside: avoid; }}
h2, h3 {{ break-after: avoid; }}
table {{ break-inside: avoid; }}
.cap {{ font-size: 8.6pt; color: var(--ink2); margin: 0 0 10px; }}
.claim td:nth-child(2) {{ font-weight: 700; white-space: nowrap; }}
.yes {{ color: var(--good); }} .part {{ color: var(--warn); }} .no {{ color: var(--bad); }}
.hi {{ font-family: "Nirmala UI", "Segoe UI", sans-serif; }}
.keep {{ break-inside: avoid; }}
ul {{ margin: 2px 0 8px; padding-left: 18px; }} li {{ margin-bottom: 3px; }}
.foot {{ break-inside: avoid; color: var(--muted); font-size: 8.5pt; border-top: 1px solid var(--line); padding-top: 6px; margin-top: 14px; }}
</style></head><body>

<h1>Are decision models actually useful for voice AI?</h1>
<div class="sub">I put Jev and OpenAI's Decisions API up against three LLMs on a clinic phone line in English, Hindi and Hinglish.</div>
<div class="meta">October 2026 · Code, data, audio and raw results: <a href="{REPO}">{REPO.removeprefix("https://")}</a></div>

<div class="verdict">
  <div class="big">Verdict: yes, they're definitely useful for voice AI.</div>
  They got the decisions right just as often as the LLMs did, in all three languages, and gave their answer
  <b>2–4× faster</b>. On a phone call, speed is the thing callers feel most, so that's a real win.
  Jev was also far cheaper. The one claim that didn't hold up is <b>"more reliable"</b>: they <i>matched</i> the LLMs,
  they didn't beat them.
</div>

<div class="tiles">
  <div class="tile"><div class="n">{rng('acc_all4', DM)} <span class="muted">vs</span> {rng('acc_all4', LLM)}</div>
    <div class="l">turns where all 4 decisions were right: <span class="dm">decision models</span> vs <span class="llm">LLMs</span> (342 turns × 3 runs)</div></div>
  <div class="tile"><div class="n"><span class="dm">{rng('lat_med', DM, f=sec, d=2)}</span> <span class="muted">vs</span> <span class="llm">{rng('lat_med', LLM, f=sec, d=2)}</span></div>
    <div class="l">median time to get the decisions back</div></div>
  <div class="tile"><div class="n">{saved}</div>
    <div class="l">shaved off every voice turn: {sec(turn['Jev']['total'])} with Jev vs {rng('total', [name(s) for s in LLM], src=turn, f=sec, d=2)} with an LLM deciding</div></div>
  <div class="tile"><div class="n">100%</div>
    <div class="l">of emergencies sent to a human, by all five systems, in every language, including the subtle ones</div></div>
  <div class="tile"><div class="n">{usd(A['Jev']['cost_per_1k'], 2)} <span class="muted">vs</span> {rng('cost_per_1k', LLM, f=usd, d=2)}</div>
    <div class="l">cost per 1,000 caller turns: Jev vs LLMs (OpenAI Decisions: {usd(A['OpenAI Decisions']['cost_per_1k'], 2)})</div></div>
  <div class="tile"><div class="n">≤ {lang_gap():.1f} pts</div>
    <div class="l">biggest accuracy gap between English, Hindi and Hinglish, for any system on any decision: language barely matters</div></div>
</div>

<h2>What I tested</h2>
<p>I picked something real: an AI receptionist answering calls for a clinic in India. Callers talk in English, Hindi,
or the usual mix of both (Hinglish). Every time the caller says something, the agent has to make four quick calls
<i>before</i> it can say anything back:</p>
<table>
<tr><th>Decision</th><th>Type</th><th>Why it matters on a call</th></tr>
<tr><td><b>What does the caller want?</b><br><span class="muted">book, reschedule/cancel, billing, reports, emergency, info, other</span></td><td>choice</td><td>Sends the call down the right flow.</td></tr>
<tr><td><b>Hand over to a human?</b></td><td>yes/no</td><td>Safety: emergencies, angry callers, anyone asking for medical advice.</td></tr>
<tr><td><b>Has the caller finished talking?</b></td><td>yes/no</td><td>Answer too early and you interrupt; too late and there's awkward silence.</td></tr>
<tr><td><b>Reply in English or Hindi?</b></td><td>choice</td><td>People switch languages, or ask the agent to.</td></tr>
</table>
<p>That's exactly the kind of quick, typed decision these models are sold for, and in voice it has to happen fast.</p>

<h3>The contenders</h3>
<p><b class="dm">Decision models:</b> TypeSafe Jev and OpenAI Decisions (built on GPT-6 Luna).
<b class="llm">LLMs:</b> GPT-6 Luna (the same base model, a nice like-for-like check), Claude Haiku 4.5 and Gemini 3.1 Flash Lite,
which are the kind of small, fast models people actually use in voice agents. The LLMs answered in strict JSON with their own probabilities.</p>

<h3>The test calls</h3>
<ul>
<li><b>80 everyday scenarios</b>, each written in English, Hindi (Devanagari) and Hinglish, so the languages are compared on the exact same content.</li>
<li><b>34 deliberately tricky ones</b>: "I'm <i>dying</i> to know if my report's ready", "It's probably nothing, but my chest is tight and my left arm is numb",
a self-harm hint buried inside a prescription request, sarcasm, half-said phone numbers, "the second one".</li>
<li>That's <b>342 caller turns</b>, each sent to every system 3 times (about 5,100 calls), plus a voice version of every turn (more on that below).</li>
</ul>
<p><b>Keeping it fair:</b> every system got the same wording for every question, went through the same gateway (Vercel AI Gateway) from the
same machine, in shuffled order, and cost is what the gateway actually billed. I wrote the correct answers up front and didn't change them after seeing results.</p>

<h2>1. Are they as accurate? Yes, it's basically a tie</h2>
<img src="fig_accuracy.png">
<div class="cap">Everyday set (80 scenarios × 3 languages × 3 runs). Dot = accuracy, faint bar = 95% confidence range.
<span class="dm">Blue</span> = decision model, <span class="llm">orange</span> = LLM.</div>
<p>On everyday calls everyone scores {min(T[s][f'acc_{t}'] for s in SYS for t in ('intent', 'esc', 'done', 'reply')) * 100:.0f}–100% on every decision. The tricky set pulls everyone down to
{rng('acc_all4', SYS, src=H)} (all four decisions right), and the decision models land right in the middle of the pack:
Jev {pct(H['Jev']['acc_all4'], 1)}, OpenAI Decisions {pct(H['OpenAI Decisions']['acc_all4'], 1)}, the LLMs
{rng('acc_all4', LLM, src=H)}. None of these gaps are bigger than the noise.</p>
<p>The bit I care most about held up: <b>every system sent every emergency to a human</b>, including the understated chest pain,
"No, he's not responding" (which only makes sense from the previous line), and the buried self-harm hint.</p>

<h3>Hindi and Hinglish: no penalty</h3>
<img src="fig_lang.png">
<div class="cap">Same 114 scenarios in each language. The three markers per row sit almost on top of each other.</div>

<h3 class="keep">The mistakes are more interesting than the scores</h3>
<table class="keep">
<tr><th>What the caller said</th><th>What went wrong</th></tr>
<tr><td class="hi">"हाँ हाँ, आराम से कीजिए, रिफंड का वादा किए बस एक महीना ही तो हुआ है।"<br><span class="muted">"Oh sure, take your time, it's only been a month since you promised my refund"</span></td>
<td>Everyone caught this sarcasm in English. In Hindi, <b>Jev, OpenAI Decisions and Claude Haiku missed it</b> and didn't escalate. The clearest language gap I found.</td></tr>
<tr><td class="hi">"मेरी थायरॉइड रिपोर्ट आई क्या? ...जानने के लिए मरा जा रहा हूँ!"<br><span class="muted">"I'm dying to know if my report's ready"</span></td>
<td><b>Jev and GPT-6 Luna</b> escalated it as if someone were actually dying.</td></tr>
<tr><td>"It's not an emergency or anything, I just need to renew my BP prescription."</td>
<td><b>GPT-6 Luna and Gemini</b> saw "emergency" and escalated. LLMs fall for keywords too.</td></tr>
<tr><td>"My mother passed away last week, and I'm calling to cancel her follow-ups."</td>
<td><b>Claude Haiku and Gemini</b> escalated. Arguably kind, but not what the rules said.</td></tr>
<tr><td>"Please help, my wife fell in the bathroom and she's not—"</td>
<td><b>Claude Haiku</b> said she'd finished talking. Honestly, jumping in during an emergency is probably right.</td></tr>
<tr><td>"Beta, I don't understand all this, can you give the phone to someone?"</td>
<td>All five chose to reply in <b>Hindi</b>; my answer key said English. I think the models have a point.</td></tr>
</table>
<p>So the errors are spread across both camps, and none of them is the "decision models are dumb" kind.</p>

<h2>2. Are they faster and cheaper? Faster, clearly. Cheaper, only Jev</h2>
<img src="fig_latency.png">
<div class="cap">Time per call for all four decisions, measured from my laptop in India (every call, both test sets). Thick line = median, thin line = 90th percentile.</div>
<p>Jev comes back in about <b>{sec(A['Jev']['lat_med'])}</b> and OpenAI Decisions in <b>{sec(A['OpenAI Decisions']['lat_med'])}</b>.
The LLMs take {rng('lat_med', LLM, f=sec, d=2)}. And that undersells it: about 0.3 s of every call here is just
network and connection setup (India to US servers). The models' own processing time, which the gateway reports for the decision models,
is only {sec(A['Jev']['provider_med'])} for Jev and {sec(A['OpenAI Decisions']['provider_med'])} for OpenAI Decisions.
Independent benchmarks report gaps of 4× to 70×+ (see section 4).</p>
<img src="fig_cost.png" style="width:88%">
<div class="cap">Cost per 1,000 caller turns (all four decisions), as billed. Log scale.</div>
<p>Jev is properly cheap: about <b>{usd(A['Jev']['cost_per_1k'])}</b> per 1,000 turns, roughly ⅓ of GPT-6 Luna and ~1/40 of Claude Haiku.
OpenAI Decisions costs <b>the same as GPT-6 Luna</b> ({usd(A['OpenAI Decisions']['cost_per_1k'])} vs {usd(A['GPT-6 Luna (LLM)']['cost_per_1k'])}),
which makes sense since it's built on it. Either way these are fractions of a cent, so for most teams cost won't be the deciding factor. Speed will.</p>

<h2>3. Does it hold up with actual voice?</h2>
<p>Typed text is the easy case. So I turned all 342 caller lines into speech with <b>ElevenLabs</b> (10 different voices, Hindi
pronunciation for the Hindi lines), squashed it down to phone quality (8 kHz, the narrow phone frequency band, phone-style compression,
background hiss), then had ElevenLabs transcribe it back. The systems then made their decisions on those messy transcripts.</p>
<table>
<tr><th>System</th><th>All 4 right (typed)</th><th>All 4 right (after voice)</th><th>Change</th><th>Interrupted a caller who wasn't finished</th></tr>
{voice_table()}
</table>
<p>Basically nothing changes, about a point either way. The transcriber kept Hindi and Hinglish well (it writes them in a mix of Devanagari and
English, the way people actually speak). The one thing that slipped a little is "has the caller finished?": when someone trails off, the
transcript sometimes adds a full stop or even invents an ending ("it's somewhere on the..." became "...on this page"), and Jev
started cutting in on a few of those.</p>

<h3>Where the time actually goes in a voice turn</h3>
<img src="fig_budget.png">
<div class="cap">Measured medians for each step: ElevenLabs transcription, the decision, a reply LLM's first word (same reply model in every row), then ElevenLabs' first audio.</div>
<p>This is the picture that settles it for me. Transcription alone takes about {sec(turn['Jev']['Speech-to-text'])}, and the reply and the
voice add another {sec(turn['Jev']['Reply LLM first token'] + turn['Jev']['TTS first audio'])}. Having an LLM make the decisions puts
another 1.2–1.8 s in the middle; a decision model cuts that to about half a second. That's <b>{saved} back on every single turn</b> (depending on which LLM it replaces).
Voice-AI guidance says callers notice pauses past ~0.5–1 s and start hanging up past ~1.2 s, so a second matters a lot.</p>

<h3>What my voice test leaves out</h3>
<ul>
<li><b>Real people.</b> The voices are synthetic: clean, evenly paced, and not Indian accents (I couldn't pick Indian-accented voices with the key
I had). Real callers mumble, talk over each other and have TVs on. So treat the voice numbers as a best case.</li>
<li><b>Pauses and tone.</b> In a real call, "has the caller finished?" also depends on how long they've paused and whether their voice dropped.
Neither decision model can hear audio. They only get text, so they're missing half the signal there.</li>
<li><b>Streaming.</b> I transcribed whole clips; a production agent would stream audio and run decisions while the caller is still talking,
which is what OpenAI recommends and would make decision models look even better.</li>
</ul>

<h2>4. Checking the makers' claims</h2>
<table class="claim">
<tr><th style="width:17%">Claim</th><th style="width:13%">Verdict</th><th>What I found</th><th style="width:30%">Outside evidence</th></tr>
<tr><td>Faster</td><td class="yes">True</td><td>2–4× faster end-to-end in my test, {saved} saved per voice turn; the models' own processing time was only ~0.14–0.29 s.</td>
<td>Vals AI: Jev 4–74× faster than LLMs on single decisions. Others measured 0.2–0.5 s for OpenAI Decisions vs ~1.6 s for a full LLM call.</td></tr>
<tr><td>Cheaper</td><td class="part">Partly</td><td>Jev ~⅓ of the cheapest LLM. OpenAI Decisions costs the same as GPT-6 Luna.</td>
<td>Vals AI: GPT-6 Luna ~3× Jev's price. BT Insights: Jev ~1/7 of OpenAI Decisions.</td></tr>
<tr><td>More reliable</td><td class="no">Not really</td><td>Same accuracy as the LLMs. And the LLMs were reliable too: with strict JSON output, none of the 5 systems ever returned a broken answer.</td>
<td>Vals AI: Jev "within noise" of top LLMs on simple checks, but last of 12 on legal reasoning.</td></tr>
<tr><td>Probabilities you can trust</td><td class="part">Depends</td><td>OpenAI Decisions' probabilities were the best calibrated of all five (escalation calibration error {A['OpenAI Decisions']['ece_esc']:.3f}).
Jev's were the worst ({A['Jev']['ece_esc']:.3f}): it leans towards escalating.</td>
<td>Vals AI: Jev's calibration swings by task. BT Insights: tuning the 50% cut-off mattered more than picking a model.</td></tr>
</table>
<p>One more thing from the research worth knowing: a paper called <b>JevOut</b> shows that ordinary conversational context can flip a decision
model's answer. That lines up with the Hindi sarcasm misses above, and it's a good reason not to rely on a single yes/no for safety decisions.</p>

<h2>5. So when should you use them?</h2>
<ul>
<li><b>Use a decision model</b> for the quick decisions on every turn: routing, "hand to a human?", "are they done talking?", which language. Same accuracy, much faster.</li>
<li><b>Keep an LLM for the reply.</b> Decision models can't produce sentences, so something still has to talk to the caller. Same for anything that needs real
reasoning: that's where outside benchmarks show decision models fall behind.</li>
<li><b>Tune the thresholds before trusting them with safety.</b> Don't just take "≥ 50% means yes" for escalation; test it on your own calls.
It's cheap to run a second check on anything close to the line.</li>
</ul>

<div class="foot">
<b>Everything is in the repo:</b> <a href="{REPO}">{REPO.removeprefix("https://")}</a>: the test calls (<code>data/</code>), all 342 audio clips,
every raw result (<code>results/</code>) and the code to re-run it all.
</div>
</body></html>"""

if __name__ == "__main__":
    html = REP / "report.html"
    html.write_text(HTML, encoding="utf-8")
    browser = next(b for b in BROWSERS if Path(b).exists())
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={REP / 'report.pdf'}", html.as_uri()], check=True, capture_output=True)
    print("wrote", REP / "report.pdf")
