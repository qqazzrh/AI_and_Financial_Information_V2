# GPSS Pipeline — Goal Post Shifting Score Detection

Detects pre-earnings-miss linguistic drift in SaaS company earnings calls using an LLM-as-Judge approach.

## Hypothesis

**H\*:** Firms that experience a negative earnings outcome in quarter *t* exhibit higher Goal Post Shifting Scores (GPSS) — as assessed by a structured LLM rubric — in their earnings call transcripts from quarter *t−1*, relative to firm-quarters not followed by a negative outcome.

## How It Works

1. **Ingest** earnings call transcripts, EPS data, and daily stock prices via Alpha Vantage API
2. **Label** each firm-quarter as MISS or NO_MISS (EPS miss or abnormal return ≤ −5%)
3. **Score** each transcript on 6 goal-post-shifting dimensions (D1–D6) using a structured LLM rubric
4. **Align** scores from quarter *t* with outcomes from quarter *t+1*
5. **Test** whether GPSS is significantly higher before misses (Welch's t-test, Mann-Whitney U, logistic regression)

## GPSS Rubric Dimensions

| Dim | Name | Detects |
|-----|------|---------|
| D1 | Metric Substitution | Shift from hard KPIs to soft/adjusted metrics |
| D2 | Timeline Deferral | Near-term → "longer-term" framing |
| D3 | Guidance Vagueness | Precise ranges → qualitative language |
| D4 | Hedging Escalation | Increased conditional/uncertain language |
| D5 | External Attribution | Blame shifted to macro/FX/supply chain |
| D6 | Q&A Deflection | Analyst questions redirected or non-answered |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your real keys

# Run the notebook
jupyter notebook gpss_pipeline.ipynb
```

## API Keys Required

- **Alpha Vantage** (free tier: 25 req/day): [Get key](https://www.alphavantage.co/support/#api-key)
- **Anthropic** (for Claude LLM scoring): [Get key](https://console.anthropic.com/)

## Rate Limit Strategy

All API responses are cached to disk. If you hit the daily limit, re-run the cell the next day — it picks up where it left off. On the free Alpha Vantage tier, full ingestion takes ~10 days; on paid tier, ~30 minutes.
