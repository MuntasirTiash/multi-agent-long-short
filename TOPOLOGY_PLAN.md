# Communication topology roadmap

Design plan for extending `topology.py` beyond `full | sparse | sector`, covering the three
settings requested (percentage-sparse, correlation-gated, hierarchical industry-leader) plus a
brainstorm of further topologies and the evaluation design needed to make claims from them.

Nothing here is implemented yet. Numbers quoted as "measured" were run against the current
498-firm universe on 2026-07-29.

---

## 1. Where we are today

`topology.build_topology(name, tickers, sectors, degree, seed)` returns `{ticker: [neighbours]}` —
the list of peers each agent is *allowed to read*. The `MessageBus` enforces it; `RoundScheduler`
(rule) and `BatchScheduler` (LLM) both consume it unchanged. **Any new topology is just a new
builder returning that same dict**, which is the property that makes this cheap to extend.

| Topology | Peers per agent (498 firms) | Worst revise prompt |
|---|---|---|
| `full` | 497 | ~3,110 tokens — **over vLLM's 4096 window once system+schema+output are added** |
| `sector` | 20–78 (mean 54) | ~527 tokens |
| `sparse(3)` | 3 | ~68 tokens |

Three properties of the current code that the plan has to respect:

- **`sparse` is directed, `sector`/`full` are symmetric.** `sparse_topology` samples each agent's
  peers independently, so A reading B does not imply B reads A. Out-degree is exactly `k`; in-degree
  is ~Poisson(k), so some agents are read by nobody and others by many. This is a live confound in
  any existing `sparse` vs `sector` comparison and should be stated in results, or controlled with a
  symmetrised variant.
- **Prompt size scales with degree**, so degree is simultaneously an information variable and a cost
  variable. See §3.
- **Within a round, every agent reads the same pre-revision snapshot** (both schedulers). New
  topologies must not break this, or agent order becomes a hidden variable.

---

## 2. The three requested settings

### T4 — Sparse by percentage of the cross-section

Replace the absolute `SPARSE_DEGREE = 3` with a fraction of the universe, so degree scales with
universe size instead of silently becoming sparser as the universe grows (3 of 30 = 10%; 3 of 498 =
0.6%, which is a different experiment wearing the same name).

```
degree = max(MIN_DEGREE, round(SPARSE_PCT * (N - 1)))
```

Measured revision-prompt cost at each setting (N=498, real prompts):

| `SPARSE_PCT` | degree | revise prompt |
|---|---|---|
| 1% | 5 | 84 tokens |
| 5% | 25 | 211 tokens |
| **10%** | **50** | **366 tokens** |
| 20% | 99 | 679 tokens |
| 50% | 248 | 1,624 tokens |
| 100% (= `full`) | 497 | 3,206 tokens — over budget once system+schema+output are added |

Keep `SPARSE_DEGREE` working as an override so old runs reproduce. Config: `SPARSE_PCT = 0.10`,
`SPARSE_MODE = "pct" | "absolute"`.

**Note for interpretation:** at 10% each agent reads 50 peers out of 498. Random peers are mostly
*uninformative* for a specific stock, so this mainly tests how much the model anchors on an
unselected crowd — which is exactly why it is the right control for the correlation and sector
topologies (§3).

### T5 — Correlation-gated communication

Connect agents whose stocks co-move. The economic argument: co-moving stocks share risk factors, so
a peer's view carries information about your own name.

Two forms, and the choice matters more than it looks:

| Form | Rule | Degree |
|---|---|---|
| **T5a threshold** | edge if `corr(i,j) >= CORR_THRESHOLD` | **uncontrolled** — varies per agent and drifts over time |
| **T5b top-k by correlation** | each agent reads its `k` most correlated peers | fixed at `k` |

**Recommendation: implement both, but make T5b the primary experiment.** A pure threshold produces
wildly uneven degree (a utility in a tightly co-moving sector gets dozens of neighbours; an
idiosyncratic name gets none), which confounds structure with degree *and* makes prompt length —
hence cost and truncation risk — vary per agent. T5a is still worth having because "who has peers at
all" is itself a finding, but it needs a degree cap (`CORR_MAX_DEGREE`) to stay runnable.

Design decisions to settle:

- **Sign.** Use `corr >= τ` (co-movers only) or `|corr| >= τ` (include strongly *negatively*
  correlated names)? For a long/short book an anti-correlated peer is arguably *more* informative:
  "my hedge is rallying" is a signal. Propose running both; default to signed (`corr >= τ`) and treat
  absolute as a variant.
- **Window and refresh.** 60 trading days of daily returns, recomputed every `CORR_REFRESH` dates.
  Measured cost of the full 123,753-pair matrix: **~1.2 s per date** in pure Python → ~8 min for a
  373-date daily run if rewired daily, ~25 s if rewired monthly (`CORR_REFRESH = 21`). Monthly
  refresh is the sane default; daily is affordable if wanted.
- **Point-in-time.** Correlations use returns dated `<= asof` only. `features.PriceHistory` already
  holds per-ticker daily returns and does bisect-based windows, so this reuses existing machinery and
  needs no new data.
- **Threshold calibration is universe-dependent** — the same trap as the manager's score gate. At
  τ=0.5 in a bull tape most large caps may connect to everything; at τ=0.8 the graph may be nearly
  empty. **Measure the degree distribution across several τ before picking a default**, and report
  mean degree alongside τ in every result.

Config: `CORR_WINDOW = 60`, `CORR_THRESHOLD = 0.5`, `CORR_TOP_K = 10`, `CORR_REFRESH = 21`,
`CORR_ABS = False`, `CORR_MAX_DEGREE = 50`.

### T6 — Hierarchical industry-leader (new setting, new manager)

A three-tier structure, and the first setting that changes the *pipeline*, not just the graph.

```
tier 1   498 StockAgents            independent opinion on own ticker (unchanged)
              |  star: members -> leader
tier 2   11 IndustryLeaders         read every opinion in their industry, emit an
              |                     IndustryReport: ranked picks + per-firm score + thesis
              |                     + an industry outlook
tier 3   IndustryManagerAgent       reads the 11 reports -> dollar-neutral book
```

**Grouping: use GICS Sector (11 groups, 21–79 firms, no singletons).** Sub-industry is unusable for
this — measured 127 groups, mean 3.9 firms, and **26 singletons that would be their own leader with
no followers**. GICS *Industry Group* (25) would be the ideal middle layer but is not in the
Wikipedia table; it would need a hand-written sub-industry→group map or another source. Sector first,
and treat a finer grouping as a later variant.

**Leader selection = largest market cap in the industry.** Market cap is *not* currently available
and Yahoo's `quote`/`quoteSummary` endpoints now return **401** without a crumb. Two viable paths:

1. **SEC XBRL (recommended, verified working).** `data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/
   EntityCommonStockSharesOutstanding.json` returns 200 with a plain User-Agent and gives shares
   outstanding stamped with both `end` and `filed` dates — measured 69 datapoints for AAPL. Market
   cap at `t` = (shares from the latest filing *filed* `<= t`) × close(`t`). This is genuinely
   point-in-time, so the leader can change over time as caps change, and there is no look-ahead. The
   CIK is already in `data/sp500/sp500_constituents.csv`. Cost: one API call per firm, cached to
   `data/shares_outstanding/`.
2. **Dollar-volume proxy (zero new data).** Rank by trailing 20-day dollar volume, already computed
   as `features.dollar_vol`. Correlates with size, fully point-in-time, available immediately. Good
   enough to build and test the plumbing while the SEC fetch is written.

Plan: build T6 against the dollar-volume proxy first so the hierarchy is testable in a day, then
swap in real market cap. **Do not pick leaders using end-of-sample caps** — that is look-ahead on the
most important structural variable in the setting.

**New message type.** `StockMessage` stays the wire format between stock agents. The leader emits
something structurally different, so add to `messages.py`:

```python
@dataclass
class IndustryReport:
    industry: str
    leader: str                     # ticker of the leader agent
    longs:  List[Tuple[str, float, str]]   # (ticker, score, thesis) best first
    shorts: List[Tuple[str, float, str]]   # worst first
    outlook: str                    # <=50 word industry-level view
    round_num: int = 0
```

**New manager.** `IndustryManagerAgent` — separate from `ManagerAgent` because its input is 11
reports, not 498 rows. It must keep two contracts so nothing downstream changes: return a
`ManagedPortfolio` with dollar-neutral signed weights, and keep the dual-brain fallback (rule version
= pool all leader picks, rank by score, apply conviction weights). Extra capability worth having
here: **industry-neutrality** — cap exposure per industry, or force one long and one short per
industry, which is natural in this structure and impossible in the flat one.

**Prompt budget (fits 4096 comfortably):** leader reads up to 79 member opinions ≈ 2,000 tokens;
manager reads 11 reports × ~6 picks ≈ 1,500 tokens.

**Matched baseline is essential.** The hierarchy changes both the graph *and* the aggregation, so
"hierarchy beat flat" would be uninterpretable without holding one fixed. Compare against:
(a) flat `sector` topology + existing `ManagerAgent` — isolates the aggregation change;
(b) hierarchy + a rule-based `IndustryManagerAgent` — isolates the LLM manager's contribution;
(c) hierarchy where the leader is chosen *at random* within the industry — isolates whether
"largest firm" carries information or whether any designated aggregator would do. **(c) is the
cheapest and most revealing control** and I would run it first.

---

## 3. The methodological point that governs all of this

**Degree and structure are currently confounded.** `full`(497) vs `sector`(54) vs `sparse`(3) differ
in *how many* peers each agent reads as well as *which*. If `sector` beats `sparse`, that could be
because sector peers are informative — or just because 54 opinions average better than 3.

Fix: for every structural comparison, run a **degree-matched random control**. With T4's
percentage-sparse this becomes easy — pick `SPARSE_PCT` so random degree equals the structural
topology's mean degree:

| Structural topology | Mean degree | Degree-matched control |
|---|---|---|
| `sector` | 54 | `sparse_pct` ≈ 10.9% |
| `corr_topk(10)` | 10 | `sparse(10)` |
| `corr_threshold(τ)` | measure it | `sparse(that mean)` |

Report both, and the claim becomes "sector structure adds X over the same number of random peers"
rather than "sector beats sparse".

**Second control worth building: a placebo.** Same graph, but peer messages are *shuffled* between
agents (a valid message from the wrong stock). If performance is unchanged, the gain came from
anchoring/averaging rather than from information — a cheap and very persuasive test of the core
hypothesis.

**Metrics beyond return.** The hypothesis is about information propagation, so log per round:
score dispersion (herding), mean absolute score revision (adoption), rank-IC of round 0 vs final
(already logged), and how often an agent's revision moves *toward* its peers' mean. `transcript.jsonl`
already records `peers_seen`, so this is post-hoc analysis on existing output, not new plumbing.

---

## 4. Further topology ideas

Ordered by what each one tests that the others do not.

| # | Topology | Rule | What it tests |
|---|---|---|---|
| **T7** | **Anti-correlation / diversity** | connect to the `k` *least* correlated peers | The direct rival hypothesis to T5: is complementary information better than redundant information? Correlated peers tell you what you already know. **Strongest single addition to the research design** — it turns T5 from "does communication help" into "what kind of information helps". |
| **T8** | **Small-world (Watts–Strogatz)** | ring lattice, rewire each edge with prob. β | Interpolates regular↔random at constant degree, so β sweeps structure with degree held fixed *by construction*. The canonical multi-agent-communication result lives here. |
| **T9** | **Scale-free (Barabási–Albert)** | preferential attachment; a few hubs | Do influential hubs help, or cause error cascades / herding? Directly relevant to the "one agent's mistake spreads to everyone" concern already noted in `topology.py`. |
| **T10** | **Size / liquidity lead-lag** | large caps → small caps (directed) | Finance-native: the documented lead-lag effect where large-cap information predicts small-cap returns. Uses the same market-cap data as T6. |
| **T11** | **Factor-neighbour** | connect agents with similar factor exposures (beta, vol, momentum) using `features.py` | "Peers" in factor space rather than GICS space — tests whether the useful notion of similarity is statistical or industrial. Zero new data. |
| **T12** | **Correlation clusters** | cluster the correlation matrix; connect within cluster | Data-driven sectors. If T12 beats `sector`, GICS labels are a worse peer definition than the market's own behaviour. |
| **T13** | **Adaptive / learned** | rewire toward peers whose past advice improved accuracy | Uses `memory.py`, which is currently write-only. The most novel but also the most work — and it introduces a fitting loop that needs its own out-of-sample discipline. |
| **T14** | **Ring / lattice** | each agent reads its 2 neighbours in a fixed order | Degenerate long-path-length control: how much does *any* structure beat a maximally slow one? |

Two cheap non-topology controls that belong in the same experiment grid:

- **`none` (round-0 only)** — already the no-comm baseline in `run_llm_analysis.py`.
- **`self-loop`** — the agent re-reads its *own* round-0 message as if it were a peer. Isolates "a
  second pass at the same information" from "peer information", which no current baseline separates.

---

## 5. Data prerequisites

| Need | Status | Action |
|---|---|---|
| Daily returns for correlation | **available** | `features.PriceHistory` already holds them |
| Market cap (point-in-time) | **not available**; Yahoo 401 | SEC XBRL fetch → `data/shares_outstanding/{TICKER}.json`, verified working |
| Market cap (proxy, interim) | **available** | `features.dollar_vol` |
| Industry grouping (11) | **available** | `Sector` in `sp500_ticker.csv` |
| Industry grouping (25 groups) | not available | needs a sub-industry→GICS-industry-group map |
| Sub-industry (127) | available but too granular | keep for a 3-tier variant only |

---

## 6. Config surface (proposed)

```python
TOPOLOGY = "sparse"          # + "corr_threshold" | "corr_topk" | "corr_anti" | "small_world"
                             #   | "scale_free" | "factor" | "corr_cluster" | "lead_lag" | "ring"
SPARSE_MODE = "pct"          # "pct" (of cross-section) | "absolute" (legacy SPARSE_DEGREE)
SPARSE_PCT = 0.10
SPARSE_SYMMETRIC = False     # symmetrise the graph, removing the directedness confound

CORR_WINDOW, CORR_THRESHOLD, CORR_TOP_K = 60, 0.5, 10
CORR_REFRESH, CORR_ABS, CORR_MAX_DEGREE = 21, False, 50

HIERARCHY_GROUPING = "sector"      # "sector" | "sub_industry"
LEADER_METRIC = "dollar_volume"    # "market_cap" once the SEC fetch lands | "random" (control)
LEADER_PICKS_PER_SIDE = 3          # how many longs/shorts a leader nominates
INDUSTRY_NEUTRAL = False           # cap per-industry net exposure in the hierarchical manager

SMALL_WORLD_BETA, SCALE_FREE_M = 0.1, 3
PLACEBO_SHUFFLE_PEERS = False      # the placebo control of §3
```

---

## 7. Implementation phases

**Phase 1 — cheap, high value (no new data)**
1. T4 percentage-sparse + `SPARSE_SYMMETRIC` + degree-matched control helper.
2. A `degree_report(topology)` utility printing degree distribution and estimated prompt tokens, so
   every topology is characterised before a GPU run is spent on it.
3. T5b `corr_topk` and T5a `corr_threshold` with monthly refresh; T7 `corr_anti` (same matrix, other
   end of the sort — nearly free once T5 exists).

**Phase 2 — hierarchy**
4. `IndustryReport` in `messages.py`; `IndustryLeader` + `IndustryManagerAgent` (new `hierarchy.py`),
   both dual-brain, using the dollar-volume leader proxy.
5. A `run_hierarchy_backtest.py` entry point (or a `--mode hierarchy` switch on the daily backtest),
   plus the random-leader control (c).
6. SEC shares-outstanding fetch → real point-in-time market cap; swap `LEADER_METRIC`.

**Phase 3 — structural sweep and analysis**
7. T8/T9/T11/T12 builders (each ~20-40 lines given the shared interface).
8. Placebo shuffle; propagation metrics from `transcript.jsonl`.
9. Extend `evaluation.monte_carlo_null` to accept per-date book sizes, so a flexible-size manager book
   can be significance-tested (currently it cannot — see `CLAUDE.md`).

---

## 8. Open questions

1. **Correlation sign** — signed or absolute? (Affects whether hedges count as informative peers.)
2. **Rewiring frequency** — is a monthly graph acceptable, or is a daily-rewired graph part of the
   research claim?
3. **Leader definition** — strictly largest market cap, or largest *within* a liquidity screen? And
   should a leader also trade its own book, or only aggregate?
4. **Leader scope** — do leaders talk to *each other* (a leader council / second-tier graph), or only
   upward to the manager? A leader council is a natural T6 variant and cheap to add.
5. **Does the hierarchical setting keep the 498-agent tier 1**, or do leaders read raw features
   directly (making it 11 agents, far cheaper but a different experiment)?
6. **How many topologies per GPU run?** At 498 firms a weekly walk-forward across 3 topologies is
   ~258k LLM calls; each added topology is ~74k more. The experiment grid needs to be chosen against a
   GPU-hour budget, not enumerated exhaustively.
