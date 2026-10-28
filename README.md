# Are decision models useful for voice AI?

Decision models (TypeSafe Jev, OpenAI Decisions API) return a typed answer with a probability instead of text, and
are claimed to be faster, cheaper and more reliable than LLMs. This project tests that claim for a voice agent on a
clinic phone line in India, with callers speaking English, Hindi or Hinglish.

On every caller turn the agent makes four decisions: **intent**, **escalate to a human?**, **has the caller finished
speaking?** and **reply language**. Two decision models (Jev, OpenAI Decisions) are compared with three LLMs
(GPT-6 Luna, Claude Haiku 4.5, Gemini 3.1 Flash Lite) on 342 caller turns for accuracy, latency, cost and calibration,
on typed text and on audio generated with ElevenLabs, passed through a simulated phone line and transcribed back.

## Run

Create `.env` in the project root:
```
AI_GATEWAY_API_KEY=...
ELEVENLABS_API_KEY=...
```

Then, from `src/`:
```
pip install numpy matplotlib

python dataset.py                      # test data -> data/turns.jsonl, data/turns_hard.jsonl
python run_text.py                     # 5 systems x 3 repetitions -> results/text_runs.jsonl
python run_text.py --input ../data/turns_hard.jsonl --out ../results/hard_runs.jsonl
python voice.py                        # speech -> phone line -> transcript -> data/asr_turns.jsonl
python voice.py turns_hard.jsonl asr_turns_hard.jsonl
cat ../data/asr_turns.jsonl ../data/asr_turns_hard.jsonl > ../data/asr_all.jsonl
python run_text.py --reps 1 --input ../data/asr_all.jsonl --out ../results/asr_runs.jsonl
python latency_budget.py               # timing of the rest of a voice turn
python analyze.py                      # metrics -> results/metrics.json, charts -> report/
python build_report.py                 # -> report/report.pdf (needs Edge or Chrome)
```
