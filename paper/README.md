# Paper

**Refusal Signal Extraction: Semantic Leakage through LLM Refusals**
Andrew Chang, Emory University

- [`refusal_signal_extraction.pdf`](refusal_signal_extraction.pdf) — compiled paper
- [`refusal_signal_extraction.tex`](refusal_signal_extraction.tex) — LaTeX source
- [`refs.bib`](refs.bib) — bibliography

## Building

```bash
make paper          # from the repository root
# or
latexmk -pdf refusal_signal_extraction.tex
```

The source is self-contained: it uses only packages shipped with a standard TeX Live
installation, so no conference style file needs to be downloaded. The layout approximates
the ACL two-column format.

Figures are pulled directly from `../results/figures/` via `\graphicspath`, so there is one
source of truth for every plotted value. Regenerate them with `make figures` before
rebuilding the paper if the metrics change.

## Relationship to the committed PDF

The PDF is the version submitted for CS 312. The LaTeX source is a revision of that
manuscript covering the same experiments and reporting the same numbers, with three
editorial changes:

1. **An explicit threat model section.** The submitted version described the adversary in
   prose scattered through the introduction; it is now stated as a capability table.
2. **An expanded limitations section.** The submitted version noted three limitations. The
   revision documents six, including the per-rule confound and the absence of a null
   baseline for the semantic scores, and names the specific experiment that would resolve
   each. See [`../docs/limitations.md`](../docs/limitations.md) for the long form.
3. **A concrete mitigations discussion.** The submitted version called for future work on
   mitigation; the revision proposes three specific directions that follow from the
   results.

No experimental result differs between the two. Regenerate the PDF from source to pick up
the revisions.
