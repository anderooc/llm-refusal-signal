# Original course submission

This project began as the final project for **CS 312** at Emory University. The submitted
artefact was a single Jupyter notebook plus a plain-text README. Both are preserved here
so the provenance of the results is auditable.

- Notebook as submitted: [`notebooks/original_submission.ipynb`](../notebooks/original_submission.ipynb)
- Paper as submitted: [`paper/refusal_signal_extraction.pdf`](../paper/refusal_signal_extraction.pdf)
- Results as submitted: everything in [`results/`](../results), unmodified

## Authorship attestation

The following statement accompanied the original submission and applies to the code as
submitted — the notebook now archived at `notebooks/original_submission.ipynb`:

> ```
> /* THIS CODE IS MY OWN WORK . IT WAS WRITTEN WITHOUT
>  CONSULTING CODE WRITTEN BY OTHER STUDENTS OR RELYING
>  SOLELY ON LARGE LANGUAGE MODELS SUCH AS CHATGPT .
> Andrew Chang */
> ```

The repository was subsequently restructured into the `refusal_signal` package for
publication. That restructuring is packaging work carried out after the course concluded
and is not covered by the attestation above; the experimental design, prompts, metrics,
and results it operates on are unchanged from the submitted version.

## What changed in the restructuring

Nothing that affects a reported number. `tests/test_regression.py` enforces this by
recomputing `results/aggregate_metrics.csv` from `results/raw_responses.csv` on every test
run, and the refactored prompt builders were diffed against the notebook's originals until
they matched character for character.

| Aspect | Original submission | Now |
|---|---|---|
| Hidden rules and descriptions | 3 rules in the notebook | identical, in `rules.py` |
| System prompt text | inline f-string | identical, in `rules.py` |
| Probe wording | inline lists | identical, in `probes.py` |
| Sweep order | single-turn phase, then why-chains | identical, in `experiment.py` |
| Keyword metric | substring match, case-insensitive | identical, in `metrics.py` |
| Semantic metric | `all-MiniLM-L6-v2` cosine | identical, in `metrics.py` |
| Reconstruction prompt | inline f-string | identical, in `reconstruction.py` |
| Published metrics | `results/aggregate_metrics.csv` | byte-identical, regression-tested |
| Figures | inline matplotlib defaults | restyled for print, same underlying values |

The one deliberate deviation: figure styling. The published figures were regenerated from
the same `aggregate_metrics.csv` with a print-oriented style, so their appearance differs
while the plotted values do not.

## Original installation instructions

Reproduced from the submitted `README.txt`, superseded by the
[quickstart in the README](../README.md#quickstart):

```
1. Clone or unzip the project directory
   Ensure the following files are present:
   - refusal_signal_extraction.ipynb
   - requirements.txt
   - README.txt

2. Create and activate a virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate

3. Run all cells of jupyter notebook
   3.1 Launch jupyter notebook if needed:
       jupyter notebook
```

Expected outputs, as documented in the original README:

1. **Console output** — the system prompt containing the hidden rules, progress logs for
   the single-turn and multi-turn experiments, summary statistics for keyword and semantic
   leakage, and the reconstructed policy text.
2. **Figures** — bar plots of keyword leakage rate and average semantic leakage score by
   probing category.
3. **Files** — `results/raw_responses.csv`, `results/aggregate_metrics.csv`, and
   `results/reconstructed_policies.txt`.

All three are still produced, now via `refusal-signal generate | score | figures | reconstruct`.
