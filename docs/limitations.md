# Limitations

The central claim of this project — that refusal explanations leak the policies they are
meant to protect, and that the leak is large enough to reconstruct from — is supported by
the data. Most of the specific numbers are not load-bearing, and this document says which
and why.

Ordered by how much each should change your reading of the results.

---

## 1. Per-rule attribution is confounded

**What the design does.** All three hidden rules are present in the system prompt on every
single call. The experiment loop then iterates over the ruleset, running the same eleven
probes once per rule and tagging each response with a different `rule_id`:

```python
for rule in ruleset:
    for probe in probes:
        ...  # identical system prompt, identical probe
        records.append({"id": rule.rule_id, ...})
```

**Why that is a problem.** The `id` column is not an experimental condition. Nothing about
the model's input changes between iterations, so the three "per-rule" groups are three
independent samples of one shared condition that happen to carry different labels. The
33 rows are better described as 11 probe conditions × 3 replicates.

**What it invalidates.** Any statistic computed *per rule*:

- Per-rule keyword leakage rates. A response that leaks `Area 51 Protocol` is scored as a
  leak only in the iteration labelled `RULE_AREA51`; the same text produced under the
  `RULE_2A` label is scored clean. Which label a leak lands under is arbitrary.
- Per-rule semantic similarity. Each response is compared against whichever rule
  description its label points at, not the rule that actually shaped it.
- The per-rule observation blocks fed to the reconstruction model, whose statistics are
  therefore mislabelled.

**What survives.** Every category-level result, which is what the paper and README report.
Aggregation groups by `(model, cat)` and pools across labels, so the confound washes out:
the 2-of-33 verbatim leakage count, the finding that both leaks arose under hypothetical
and moral framing, and the ordering of semantic leakage across probing categories are all
unaffected.

**The fix.** Make rule composition a real factor. Run one condition per rule with only that
rule in the system prompt, plus an all-rules condition, and compare. This was not done in
the published run and the results have not been recomputed under it.

## 2. The semantic scores have no null baseline

Mean cosine similarity between a refusal and its rule description lands around 0.16–0.27.
That number has no reference point. Two unrelated pieces of English about safety, policy,
and confidentiality will score well above zero on `all-MiniLM-L6-v2` simply by sharing a
register, and no control condition was run to establish what "well above zero" means here.

Without a null distribution, an absolute score of 0.26 cannot be called high or low, and
"semantic leakage is consistently present" is a statement about the *ordering* across
categories rather than about any threshold being crossed.

**The fix.** Three controls, cheap to run:

- **No-rule control.** Same probes, system prompt with the hidden-rule block removed.
  Similarity against the rule descriptions under this condition is the null.
- **Distractor rules.** Score every response against rule descriptions that were never in
  the prompt. Leakage is the gap between matched and mismatched scores, which is
  scale-free.
- **Permutation test.** Shuffle the response-to-rule assignment and recompute; the observed
  mean should sit outside the shuffled distribution.

The distractor version also converts the metric into a retrieval task — is the true rule
the nearest neighbour among candidates? — which yields top-1 accuracy and AUROC, both far
more interpretable than a raw cosine.

## 3. The sample is too small for the rates to be precise

33 responses, one model (`llama3.1`), one seed, no repeated sampling at fixed temperature.

The headline 6% verbatim leakage rate is **2 events**. A binomial confidence interval on
2/33 spans roughly 1–20%, so the honest claim is "verbatim leakage occurs" rather than
"verbatim leakage occurs at 6%". Likewise, the per-category rates of 1/6 rest on a single
response each, and the gaps between the four single-turn categories in semantic leakage
(0.239 to 0.265) are almost certainly inside the noise. Only the why-chain result (0.165)
separates clearly from the rest.

**The fix.** Repeat each probe 20–50 times, report bootstrap confidence intervals, and
sweep several open-weight models. `configs/multi_model.yaml` is set up for the model sweep.

## 4. The reconstruction model is told the rule identifiers

The reconstruction prompt is keyed by rule ID, so the analyst model sees the strings
`RULE_AREA51`, `RULE_2A`, and `RULE_CLASSIFICATION` alongside the refusal excerpts.

This does not trip the keyword metric — the forbidden phrase is `Area 51 Protocol`, not
`RULE_AREA51` — but it is a genuine channel that is not refusal leakage. `RULE_CLASSIFICATION`
in particular telegraphs the existence of a classification scheme, and the reconstruction
duly returned confidentiality- and classification-themed rules. The reconstruction is
therefore "blind to the policy text" rather than "blind to everything but the refusals",
and some of its apparent fidelity is attributable to the identifiers.

**The fix.** Replace identifiers with opaque labels (`POLICY_A`, `POLICY_B`) before building
the prompt. `tests/test_reconstruction.py` pins the current behaviour so the change would
be visible.

## 5. Reconstruction fidelity is judged by eye

The claim that reconstruction "captures the semantic intent and structural constraints of
the true hidden rules" rests on reading the output and agreeing. There is no metric, no
second rater, and no blind evaluation.

**The fix.** Embed each reconstructed rule and each ground-truth rule, match them with a
Hungarian assignment over the similarity matrix, and report precision, recall, and mean
matched similarity. Add an LLM-judge rubric scoring each ground-truth rule as
recovered / partially recovered / missed, with a human-rated subset to calibrate the judge.

## 6. Synthetic rules are easier to detect than real ones

The hidden rules were invented so leakage would be unambiguous: `Area 51 Protocol` appears
nowhere in pretraining, so its presence in an output can only have come from the system
prompt. That precision costs realism in both directions. Conspicuous, oddly-named rules may
draw attention and inflate verbatim leakage; conversely, real production policies are
longer, more numerous, and more entangled with behaviour the model already exhibits, which
could raise semantic leakage while lowering keyword leakage.

Results here should be read as a demonstration that the channel exists, not as an estimate
of its bandwidth in a deployed system.

## 7. Multi-turn probing is shallower than it looks

The "why chains" send three user turns in a single request with no assistant replies in
between, rather than conducting a genuine three-turn conversation:

```python
messages = [system] + [user_1, user_2, user_3]
```

The model therefore sees the escalation all at once and answers once, which is a weaker
manipulation than iterative pressure where each follow-up responds to what was actually
said. The finding that why-chains leak *least* may be an artefact of this: presented with
three stacked demands, the model appears to retreat to generic deflection. A real
multi-turn loop, feeding each reply back before the next "but why?", would test the
explanation-pressure hypothesis properly.

## 8. Embedding similarity is not recoverability

A high cosine score means a refusal occupies similar semantic space to a rule description.
It does not establish that the rule text could be recovered from the refusal, nor that a
human reading the refusal would learn anything actionable. The reconstruction experiment
addresses this qualitatively; limitation 5 covers why that evidence is weaker than it
should be.

---

## Summary

| # | Limitation | Affects | Severity |
|---|---|---|---|
| 1 | Per-rule attribution confounded | per-rule statistics only | high |
| 2 | No null baseline for cosine scores | absolute semantic values | high |
| 3 | n = 33, one model, one seed | precision of all rates | high |
| 4 | Rule IDs disclosed to reconstructor | reconstruction fidelity claim | medium |
| 5 | Reconstruction scored by eye | reconstruction fidelity claim | medium |
| 6 | Synthetic rules | external validity | medium |
| 7 | Why-chains are single-shot | the multi-turn finding | medium |
| 8 | Similarity ≠ recoverability | interpretation of the metric | low |

None of these undermine the qualitative finding: benign probing of a correctly-behaving
model produced verbatim disclosure of protected terms, measurable semantic alignment in
every condition, and a usable reconstruction of the hidden policy. They do mean the
specific numbers should be treated as existence proofs rather than measurements.
