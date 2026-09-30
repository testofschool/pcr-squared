# Data License

## `data/wiki_corpus.json` — Wikipedia text, CC BY-SA 4.0

The corpus in `data/wiki_corpus.json` consists of the summary text of 58 English
Wikipedia articles, retrieved via the Wikipedia REST API `/page/summary/` endpoint.
Wikipedia text is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
Because that license is share-alike, **this file is distributed under CC BY-SA 4.0, not
under the MIT License** that covers the code in this repository.

- **Authors:** Wikipedia contributors. The full authorship of each article is available from
  the article's page history on Wikipedia.
- **Source:** the article pages listed below (English Wikipedia).
- **License:** CC BY-SA 4.0 — https://creativecommons.org/licenses/by-sa/4.0/
- **Changes:** each record stores the article `title` and summary `text` together with two
  fields added for this project: `level` (the difficulty label assigned for the PCR² experiments)
  and `length` (character count of `text`). The retrieval date is not recorded in this repository.

## Attribution — source articles (58)

| # | Article | Source |
|---|---------|--------|
| 1 | Water | https://en.wikipedia.org/wiki/Water |
| 2 | Sun | https://en.wikipedia.org/wiki/Sun |
| 3 | Plant | https://en.wikipedia.org/wiki/Plant |
| 4 | Animal | https://en.wikipedia.org/wiki/Animal |
| 5 | Rock (geology) | https://en.wikipedia.org/wiki/Rock_(geology) |
| 6 | Rain | https://en.wikipedia.org/wiki/Rain |
| 7 | Cloud | https://en.wikipedia.org/wiki/Cloud |
| 8 | Moon | https://en.wikipedia.org/wiki/Moon |
| 9 | Fire | https://en.wikipedia.org/wiki/Fire |
| 10 | Soil | https://en.wikipedia.org/wiki/Soil |
| 11 | Seed | https://en.wikipedia.org/wiki/Seed |
| 12 | Leaf | https://en.wikipedia.org/wiki/Leaf |
| 13 | Wind | https://en.wikipedia.org/wiki/Wind |
| 14 | Ice | https://en.wikipedia.org/wiki/Ice |
| 15 | Sand | https://en.wikipedia.org/wiki/Sand |
| 16 | Photosynthesis | https://en.wikipedia.org/wiki/Photosynthesis |
| 17 | DNA | https://en.wikipedia.org/wiki/DNA |
| 18 | Atom | https://en.wikipedia.org/wiki/Atom |
| 19 | Electricity | https://en.wikipedia.org/wiki/Electricity |
| 20 | Evolution | https://en.wikipedia.org/wiki/Evolution |
| 21 | Magnetism | https://en.wikipedia.org/wiki/Magnetism |
| 22 | Cell (biology) | https://en.wikipedia.org/wiki/Cell_(biology) |
| 23 | Chemical reaction | https://en.wikipedia.org/wiki/Chemical_reaction |
| 24 | Earthquake | https://en.wikipedia.org/wiki/Earthquake |
| 25 | Periodic table | https://en.wikipedia.org/wiki/Periodic_table |
| 26 | Wave | https://en.wikipedia.org/wiki/Wave |
| 27 | Energy | https://en.wikipedia.org/wiki/Energy |
| 28 | Quantum mechanics | https://en.wikipedia.org/wiki/Quantum_mechanics |
| 29 | Organic chemistry | https://en.wikipedia.org/wiki/Organic_chemistry |
| 30 | Topology | https://en.wikipedia.org/wiki/Topology |
| 31 | Biochemistry | https://en.wikipedia.org/wiki/Biochemistry |
| 32 | Genetics | https://en.wikipedia.org/wiki/Genetics |
| 33 | Quantum field theory | https://en.wikipedia.org/wiki/Quantum_field_theory |
| 34 | Homological algebra | https://en.wikipedia.org/wiki/Homological_algebra |
| 35 | Quantum chromodynamics | https://en.wikipedia.org/wiki/Quantum_chromodynamics |
| 36 | Category theory | https://en.wikipedia.org/wiki/Category_theory |
| 37 | Protein | https://en.wikipedia.org/wiki/Protein |
| 38 | Superconductivity | https://en.wikipedia.org/wiki/Superconductivity |
| 39 | Neutrino | https://en.wikipedia.org/wiki/Neutrino |
| 40 | Black hole | https://en.wikipedia.org/wiki/Black_hole |
| 41 | Antimatter | https://en.wikipedia.org/wiki/Antimatter |
| 42 | Epigenetics | https://en.wikipedia.org/wiki/Epigenetics |
| 43 | Entropy | https://en.wikipedia.org/wiki/Entropy |
| 44 | Diffraction | https://en.wikipedia.org/wiki/Diffraction |
| 45 | Redox | https://en.wikipedia.org/wiki/Redox |
| 46 | Titration | https://en.wikipedia.org/wiki/Titration |
| 47 | Mitosis | https://en.wikipedia.org/wiki/Mitosis |
| 48 | Meiosis | https://en.wikipedia.org/wiki/Meiosis |
| 49 | Fermentation | https://en.wikipedia.org/wiki/Fermentation |
| 50 | Osmosis | https://en.wikipedia.org/wiki/Osmosis |
| 51 | Topological insulator | https://en.wikipedia.org/wiki/Topological_insulator |
| 52 | Quantum decoherence | https://en.wikipedia.org/wiki/Quantum_decoherence |
| 53 | Geometric phase | https://en.wikipedia.org/wiki/Geometric_phase |
| 54 | Soliton | https://en.wikipedia.org/wiki/Soliton |
| 55 | Renormalization group | https://en.wikipedia.org/wiki/Renormalization_group |
| 56 | Conformal field theory | https://en.wikipedia.org/wiki/Conformal_field_theory |
| 57 | Tensor network | https://en.wikipedia.org/wiki/Tensor_network |
| 58 | Spintronics | https://en.wikipedia.org/wiki/Spintronics |

## Summary

| Artifact | License |
|----------|---------|
| `data/wiki_corpus.json` (Wikipedia text) | CC BY-SA 4.0 |
| Code (`src/*.py`) | MIT (see `LICENSE`) |
