# PCR²: Psychometric Content Routing with Principal Component Regression

**Paper**: *PCR²: Psychometric Content Routing with Principal Component Regression for User-Conditioned LLM Processing*
**Author**: Jung Min Kang (Independent Researcher, Seoul)
**Contact**: gangjeongmin23@gmail.com | ORCID: 0009-0007-9599-2792

## Key results (58 real Wikipedia summaries × 100 synthetic users)
Only the text is real: user profiles, covariates (generated as linear functions of true θ plus noise), responses, and the relevance oracle are synthetic.

- θ estimation: Pearson r = 0.963 (Spearman ρ = 0.961) from cold-start covariates (no interaction history)
- Routing quality: 96.9% of oracle (d = 1.84 vs difficulty-only, p < 10⁻¹⁵)
- **Honest negative results**: LLTM features fail on real text (ρ = 0.068); simulated relevance oracle achieves 99.4%

## Reproducing

```bash
pip install numpy scipy scikit-learn matplotlib
python src/run_experiment.py      # Main results + bootstrap CIs + scaling + ablation
python src/generate_figures.py    # All 7 paper figures (no Type 3 fonts; TrueType/Type 42)
cd latex && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Structure
```
pcr-squared/
├── latex/                  # LaTeX source (compile from here)
│   ├── main.tex, main.pdf, references.bib, neurips_2024.sty
├── figures/                # All 7 figures (PDF, no Type 3 fonts; TrueType/Type 42)
├── src/
│   ├── run_experiment.py, generate_figures.py, extract_lltm_features.py
├── data/wiki_corpus.json   # 58 Wikipedia article summaries (CC BY-SA 4.0, see DATA_LICENSE.md)
├── results/                # JSON outputs (main, theta, scaling, ablation)
├── README.md, LICENSE, DATA_LICENSE.md, requirements.txt
```

## Citation
```bibtex
@article{kang2026pcr2,
  title={PCR$^2$: Psychometric Content Routing with Principal Component Regression
         for User-Conditioned {LLM} Processing},
  author={Kang, Jung Min},
  year={2026},
  note={Preprint}
}
```

## License
- Code: MIT (see [LICENSE](LICENSE)).
- Corpus: `data/wiki_corpus.json` contains Wikipedia text and is licensed CC BY-SA 4.0, not MIT. See [DATA_LICENSE.md](DATA_LICENSE.md) for attribution.
