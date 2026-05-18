# PCR²: Psychometric Content Routing with Principal Component Regression

**Paper**: *PCR²: Psychometric Content Routing with Principal Component Regression for User-Conditioned LLM Processing*
**Author**: Jung Min Kang (Independent Researcher, Seoul)
**Contact**: gangjeongmin23@gmail.com | ORCID: 0009-0007-9599-2792

## Key results (58 Wikipedia summaries × 100 users)
- θ estimation: ρ = 0.963 from cold-start covariates (no interaction history)
- Routing quality: 96.9% of oracle (d = 1.84 vs difficulty-only, p < 10⁻¹⁵)
- **Honest negative results**: LLTM features fail on real text (ρ = 0.068); simulated relevance oracle achieves 99.4%

## Reproducing

```bash
pip install numpy scipy scikit-learn matplotlib
python src/run_experiment.py      # Main results + bootstrap CIs + scaling + ablation
python src/generate_figures.py    # All 7 paper figures (Type 1 fonts)
cd latex && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Structure
```
pcr-squared/
├── latex/                  # LaTeX source (compile from here)
│   ├── main.tex, main.pdf, references.bib, neurips_2024.sty
├── figures/                # All 7 figures (PDF, Type 1 fonts)
├── src/
│   ├── run_experiment.py, generate_figures.py, extract_lltm_features.py
├── data/wiki_corpus.json   # 58 Wikipedia article summaries
├── results/                # JSON outputs (main, theta, scaling, ablation)
├── README.md, LICENSE, requirements.txt
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
MIT
