# Agent Orchestration — Phase 0 Scaffold

**One LLM agent per stock. Agents talk to each other. Out comes a long/short
portfolio.**

This folder is the Phase 0 skeleton from the research plan
(`../agent-orchestration-gap-analysis.md`). Phase 0's only job is to get the
**plumbing** right: create one agent per stock, let them exchange structured
messages over a configurable communication network, and turn their final scores
into a long/short book.

To keep it easy to read and reason about, Phase 0 is deliberately dumb:
- the "market data" is **randomly generated** (seeded, so it's reproducible);
- the agents reason with a **simple arithmetic rule**, not an LLM;
- there are **zero third-party dependencies** — pure Python standard library.

That means it runs instantly, offline, and for free. Later phases swap the fake
data and the rule for real data and a real (local, open-source) LLM *without
changing any of the wiring*.

## Run it

```bash
# On NJIT HPC first:  module load Miniforge3 && conda activate agents
cd agent_orchestration
python run_demo.py
```

You'll see the three conversation rounds, the final 30-stock ranking, and the
long/short book. Change the behaviour by editing `config.py` (try
`TOPOLOGY = "full"` or `"sector"`, or bump `N_ROUNDS`).

---

## The picture: modules and steps

Real daily prices in, a graded long/short book out. One rebalance date walks
top-to-bottom through this:

```
                            config.py
            (universe = DJIA-30 + sectors, TOPOLOGY, N_ROUNDS,
                       N_LONG / N_SHORT, SEED)
                                │ settings
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  data_loader.py — REAL point-in-time price data                      │
│                                                                      │
│    Yahoo Finance daily adjusted close — free, no API key,            │
│    downloaded once from a login node                                 │
│                        │                                             │
│                        ▼                                             │
│    price_cache/{TICKER}_2024-10-01_2026-07-01.csv   (30 files)       │
│                        │  read from disk on every later run          │
│          ┌─────────────┴──────────────┐                              │
│          ▼                            ▼                              │
│   compute_features(asof)       forward_returns(asof, H)              │
│    momentum = 63d return        the NEXT H days' return              │
│    value    = 5d reversal       → evaluation.py ONLY,                │
│    (both tanh-z-scored)           never shown to an agent            │
│    prices dated <= asof only                                         │
└──────────────────────────────────────────────────────────────────────┘
             │ {ticker: {momentum, value, date}}
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  entry point:  run_demo · run_llm_demo · run_analysis ·              │
│                run_llm_analysis · run_daily_backtest                 │
│                   (wires everything together)                        │
└──────────────────────────────────────────────────────────────────────┘
        │                    │                          │
        ▼                    ▼                          ▼
  StockAgent × 30       topology.py               MessageBus +
  one per ticker      who-talks-to-whom       Round-/BatchScheduler
  rule or local LLM  full | sparse | sector   routes permitted msgs
        │                    │                          │
        └──────────┬─────────┴──────────────────────────┘
                   ▼
      ┌───────────────────────────────────────────────────────────────┐
      │                     THE CONVERSATION                          │
      │                                                               │
      │  Round 0   each agent reads ONLY its own stock's features     │
      │            and posts a StockMessage                           │
      │            {score, direction, confidence, thesis}             │
      │                     │                                         │
      │                     ▼                                         │
      │  Round 1   each agent reads its NEIGHBOURS' messages (the     │
      │   ..N      topology decides which) and REVISES its score so   │
      │            it is relative to peers, then re-posts.            │
      └───────────────────────────────────────────────────────────────┘
                   │  final message per stock (one score each)
                   ▼
       manager_agent.py           OR          aggregator.py
   reads all 30 finals, emits           sort by score, take top 5
   signed conviction weights            long / bottom 5 short,
   (longs +1, shorts -1)                equal weight
                   │
                   ▼
             Portfolio  ───────►  evaluation.py
        { longs, shorts }         rank_IC vs the forward return,
                                  long/short spread, turnover + cost,
                                  Monte-Carlo null → percentile, p
```

One caveat on that top box: `run_demo.py` is the lone exception — it still
fabricates seeded random features so the wiring can be exercised with no price
cache present at all. Every other entry point reads the real cached prices.

### Data that flows between the pieces: `StockMessage`

Every agent, every round, emits the *same* structured record (see
`messages.py`):

```
StockMessage
├─ ticker      "AAPL"
├─ score       0..100   (higher = more attractive to hold long)
├─ direction   LONG | SHORT | NEUTRAL
├─ confidence  0.0..1.0
├─ thesis      "<=50-word justification"
└─ round_num   0 = independent view, 1.. = revisions
```

Structured (not free-text) messages keep token counts tiny for small local
models **and** let us later measure how information moves through the network.

---

## What each file does

| File | Role in the picture above |
|------|---------------------------|
| `config.py` | All the knobs: the DJIA-30 universe + sectors, how many longs/shorts, which topology, how many rounds, the random seed. |
| `messages.py` | The `StockMessage` record + `Direction` enum. The common language agents speak. |
| `memory.py` | A tiny per-agent log of its own past calls (`AgentMemory`). Placeholder for FAgent's episodic memory. |
| `stock_agent.py` | `StockAgent`: owns one ticker; `initial_assessment()` then `revise()`. **This is the file Phase 2 will point at an LLM.** |
| `topology.py` | Builds the who-talks-to-whom graph: `full`, `sparse`, or `sector`. |
| `message_bus.py` | `MessageBus` routes each agent only the messages it's allowed to see; `RoundScheduler` runs round 0..N. |
| `aggregator.py` | `rank_and_split()` sorts the final scores into the long/short `Portfolio`. |
| `run_demo.py` | The entry point that connects all of the above and prints the result. |

**The important design property:** the *protocol* (rounds, topology,
aggregation) lives entirely outside `StockAgent`. So you can change how agents
communicate — the actual research question — without touching how an individual
agent thinks, and vice-versa.

---

## The libraries — what's used and how it works

### Phase 0 uses ONLY the Python standard library

No `pip install` needed. Here is every import and why it's there:

| Library | Where | What it does for us |
|---------|-------|---------------------|
| `dataclasses` | `messages.py`, `memory.py`, `aggregator.py` | The `@dataclass` decorator auto-writes the boring `__init__`/`__repr__` for a data-holding class, so `StockMessage` is a clean record with named fields instead of an untyped dict. `field(default_factory=list)` gives each object its own fresh list. |
| `enum` | `messages.py` | `Direction(str, Enum)` defines a fixed set of allowed values (LONG/SHORT/NEUTRAL). Inheriting from `str` means it prints and compares like a plain string but you can't typo an invalid stance. |
| `random` | `topology.py`, `run_demo.py` | `random.Random(seed)` is a *local* seeded generator (independent of global state) used to pick sparse neighbours and to fabricate placeholder market data — seeded so every run is identical and reproducible. |
| `typing` | most files | `Dict`, `List` are just hints that document what a function expects/returns. They don't change behaviour; they make the code (and editor autocomplete) clearer. |
| `logging` | `run_demo.py`, `message_bus.py` | Prints the round-by-round progress. Using `logging` instead of `print` lets you later raise/lower verbosity without editing code. |

### Libraries that arrive in later phases (all free & open-source)

These are **not installed yet** — they're documented in `requirements.txt`.

- **Phase 1 (real data):**
  - `pandas` / `numpy` — hold price & fundamental tables and compute the
    `momentum`/`value` features (and, for topology T4, a return-correlation
    matrix). NumPy is the fast numeric-array core; pandas is the labelled-table
    layer on top of it.
  - `yfinance` — free historical price downloads (or reuse the data already in
    `../FAgent/data/`).
- **Phase 2 (real reasoning, still zero-cost):** we reuse
  `../FAgent/utils/llm_utils.py`'s `LLMClient`, which never calls a paid API. It
  offers three backends chosen by the `LLM_BACKEND` environment variable:
  - `none` — no model at all; the rule-based fallback (what Phase 0 does).
  - `openai-compatible` — talks over HTTP (via the `requests` library) to a
    **local** server you run yourself: **Ollama** or **vLLM** serving an
    open-weight model like Qwen2.5. Free because it runs on your own GPU.
  - `transformers` — loads a **HuggingFace** model directly in-process with
    `transformers` + `torch` on the HPC GPU.
  - `sentence-transformers` (optional) — real text embeddings so agent memory
    can retrieve *similar* past situations; falls back to a hashing embedding if
    not installed.

When Phase 2 lands, only `stock_agent.py`'s two methods change: instead of the
arithmetic rule they will build a short prompt (the stock's data + the peer
messages) and call `LLMClient.generate_json(...)` to get back the same
`StockMessage` fields. Everything else in this folder stays exactly as it is.

---

## Phase 1 — real data + a preliminary analysis (added)

Phase 1 keeps every Phase-0 module untouched and adds three files:

| File | Role |
|------|------|
| `data_loader.py` | Downloads real daily adjusted-close for the DJIA-30 from Yahoo's free endpoint (via `requests`, **no `yfinance` needed**), caches to `price_cache/`, and computes **point-in-time** features (`momentum` = trailing 63-day return; `value` = a short-term reversal proxy — a true value factor needs fundamentals). |
| `evaluation.py` | The grading metrics: `rank_ic` (Spearman of scores vs next-week returns) and `long_short_spread`, plus a mean/std/Sharpe summary. Pure arithmetic, numpy-free. |
| `run_analysis.py` | Walk-forward loop: for each weekly rebalance date, build features, run every configuration, and grade it against the **next** week's return (never shown to agents). Prints a comparison table. |

Run it: `python run_analysis.py` (first run downloads + caches; later runs are offline).

**Point-in-time discipline built in:** features at date *t* use only prices dated
≤ *t*; the forward return used to grade them uses *t*→*t+1wk* and is fed only to
`evaluation.py`, never to an agent. The window starts after Oct-2024 (past
Qwen2.5's cutoff) so a future LLM agent can't have memorised the outcomes.

### Preliminary result (74 weekly rebalances, Dec-2024 → Jun-2026)

```
configuration             meanIC   IC>0   LS/wk%  Sharpe   cumLS%
random (floor)            -0.040   43%   -0.512   -1.45    -33.3
momentum-only              0.006   49%    0.021    0.04     -3.9
B2 no-comm (round 0)       0.012   47%   -0.111   -0.21    -12.6
comm: full                 0.012   47%   -0.114   -0.22    -12.8
comm: sparse               0.004   49%   -0.224   -0.41    -20.0
comm: sector               0.018   51%   -0.012   -0.02     -6.0
```

**How to read this (and what it does NOT say).** The agents are still the
Phase-0 **rule-based placeholders**, so this is a *pipeline + harness* check on
real data, not a test of the communication hypothesis. Three things it confirms:
1. The harness discriminates — the `random` floor is clearly the worst on every
   metric, so the evaluation isn't just noise.
2. Rule-based scores have near-zero rank-IC (~0.01). That is *expected and
   correct*: a fixed momentum/reversal blend has little weekly cross-sectional
   edge on liquid large-caps. There's nothing here for communication to
   amplify yet — which is exactly why Phase 2 needs real LLM reasoning.
3. The one suggestive dot: `comm: sector` edges out the others (meanIC 0.018,
   IC>0 51%), consistent with the plan's hypothesis that comparing a stock to
   its true economic peers helps. Far too small to lean on with rule-based
   agents — logged as a hypothesis for the LLM phase, not a finding.

## Phase 2 — real local-LLM agents (added)

The agents can now reason with a **local, open-source Qwen2.5** model instead
of the arithmetic rule. Two new files, and `stock_agent.py` gained an LLM path
alongside the rule-based one:

| File | Role |
|------|------|
| `llm.py` | Bridges to FAgent's zero-cost `LLMClient` (loaded by file path so it can't shadow our `config`). `make_llm_client()` returns a client, or `None` when `LLM_BACKEND` is unset — in which case agents silently fall back to the rule. |
| `run_llm_demo.py` | Same pipeline as `run_demo.py`, but agents use the LLM. Runs on real data at the latest leakage-safe date. |
| `stock_agent.py` | Now has `_llm_initial` / `_llm_revise` (build a compact prompt → `generate_json` → `StockMessage`). If the model returns junk, it falls back to the rule so a run never crashes. |

Enable it (zero API cost — the model runs on our own hardware):

```bash
export LLM_BACKEND=transformers LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
python run_llm_demo.py                 # DEMO_N=6 to shrink for a CPU node
```

Both `Qwen2.5-0.5B` and `Qwen2.5-7B` are already cached in the shared
`HF_HOME` (`/project/dtyu/ms3235/hf_cache`), so no download is needed. On a GPU
compute node, use the 7B model (`device_map="auto"` picks up the GPU
automatically); or serve it with vLLM and set
`LLM_BACKEND=openai-compatible` for fast batched inference.

**Validated (Qwen2.5-0.5B on the CPU login node):** the LLM path works
end-to-end — the model returns parseable JSON that becomes proper
`StockMessage`s with natural-language theses (not the fallback). It also
exposed the expected limitation: the 0.5B model gives **near-identical scores**
to every stock, so it cannot rank cross-sectionally. That is exactly why the
real experiments need a 7B+ model on GPU — and why the "coordination protocol >
model scale" hypothesis is worth testing rather than assuming.

## Harness upgrade — costs + significance (added)

`evaluation.py` and `run_analysis.py` gained the two things that make backtest
numbers trustworthy (and that most papers in this field skip):

- **Transaction costs.** `turnover()` measures how much of the book changes each
  week; `apply_cost()` charges a round-trip cost (default 20 bps) on it. The
  report now shows gross **and** net cumulative return.
- **Monte-Carlo null test** (MarketSenseAI-style). `monte_carlo_null()` builds
  5,000 random long/short books over the same weeks; `percentile_and_p()` says
  where a strategy sits in that null and gives a one-sided p-value. A strategy
  that can't beat random picking is exposed immediately.

Latest run (still rule-based agents — this validates the *harness*):

```
configuration             meanIC grossCum%  netCum%  Sharpe  MCpct      p
random (floor)            -0.040     -33.3    -41.0   -1.92     4%  0.958
momentum-only              0.006      -3.9     -7.5   -0.06    46%  0.538
B2 no-comm (round 0)       0.012     -12.6    -19.9   -0.44    31%  0.694
comm: full                 0.012     -12.8    -20.1   -0.45    30%  0.697
comm: sparse               0.004     -20.0    -26.6   -0.63    18%  0.817
comm: sector               0.018      -6.0    -13.8   -0.25    42%  0.575
```

The null test is doing its job: the random floor sits at the 4th percentile
(p ≈ 0.96 — random books mostly beat it), and no rule-based config is
significant (p ≫ 0.05), as expected for placeholder agents. Costs shave several
points off every book. When Phase-2 LLM agents run through this same harness on
GPU, these columns become the real verdict.

## Batched LLM walk-forward via vLLM (added)

The single-call LLM path is fine for a demo but far too slow for the full
walk-forward (`#dates × (1+rounds) × 30` calls). This adds a **batched** path
built around a local **vLLM** server, plus the script that actually tests the
hypothesis end-to-end.

| File | Role |
|------|------|
| `batch_llm.py` | `HttpBatchClient` fires a whole round's prompts at an OpenAI-compatible endpoint (vLLM/Ollama) **concurrently** (thread pool) — vLLM batches them on the GPU. `SequentialBatchClient` wraps the in-process model for CPU smoke tests. Same `generate_json_batch(prompts)` interface. |
| `batch_orchestration.py` | `BatchScheduler` runs the same rounds as `RoundScheduler` but one **batched** LLM call per round, and returns **both** the round-0 (independent) and final (post-communication) messages from one pass. |
| `run_llm_analysis.py` | Walk-forward that grades **no-comm (round 0)** vs **comm (topology)** from that single batched pass, through the full harness (IC, costs, MC null). This is the direct test of H1. Saves results to `results/`. |
| `serve_vllm_gpu.sh` | SLURM job: install vLLM if missing, serve Qwen2.5-7B as a local OpenAI endpoint, run `run_llm_analysis.py` against it, shut down. |

To enable prompt-building for batching, `stock_agent.py` was refactored so
`initial_prompt` / `revise_prompt` (build the text) are separate from the model
call and `message_from_data` (parse). The single-call path uses the exact same
pieces — no behaviour change (the rule-based demo output is byte-identical).

### Full per-agent transcript (`transcript.py`)

`run_llm_analysis.py` records **every message from every agent, every round**,
to `results/transcript_<topologies>_<N>x<D>.jsonl` — one JSON object per line:

```json
{"date":"2024-12-31","topology":"round0-shared","round":0,"ticker":"AAPL",
 "score":85.0,"direction":"LONG","confidence":0.9,"thesis":"...","peers_seen":[]}
```

Round-0 rows (`topology":"round0-shared"`) are each agent's independent view;
revision rows carry `peers_seen` — the exact neighbour messages (ticker, score,
direction) the agent read before revising — so information propagation through
the topology is fully reconstructable. The same records are echoed to the run
log as readable one-liners (`tail -f` shows the live agent-by-agent play-by-play,
e.g. `[2024-12-31|sparse|r1] AAPL score=35.3 LONG ... <- saw MSFT:40,NVDA:3`).

Analyse it with pandas: `pd.read_json(path, lines=True)` → group by
`ticker`/`round` to see how each agent's score evolved through the conversation.

Run the real thing on a GPU node:

```bash
sbatch serve_vllm_gpu.sh          # serves Qwen2.5-7B + runs the analysis
```

Or smoke-test the batched wiring on CPU (slow, tiny):

```bash
export LLM_BACKEND=transformers LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
DEMO_N=10 MAX_DATES=1 python run_llm_analysis.py
```

**How to read the output:** two rows, `no-comm (round 0)` and `comm: <topology>`.
Communication helps if the `comm` row beats `no-comm` on net cumulative return
**and** clears the Monte-Carlo null (low p). Because both come from the *same*
LLM pass, it's a clean matched comparison.

## What is still *not* done

- The full 74-week 7B run must be launched on a GPU node (`sbatch
  serve_vllm_gpu.sh`); the CPU login node can only smoke-test the wiring.
- `value` is a price-based reversal proxy, not real fundamentals.
- Index membership is the current list, not historical point-in-time (Phase 3).
- No borrow-cost/short-availability model for the short leg yet (Phase 3).

See `../agent-orchestration-gap-analysis.md` §6 for the full roadmap.
