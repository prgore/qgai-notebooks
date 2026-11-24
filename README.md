# QGAI Open Notebooks

Reproducible research notebooks from **[QGAI](https://www.qgai.in)**, the Quantum Generative AI Lab founded by Prakash Gore in Pune, India.

Every notebook follows the same rules:

- **A strong classical baseline.** Quantum methods are always compared with the best classical method we can reasonably build.
- **Results reported as they come out**, including when quantum does *not* win.
- **Reproducible:** fixed random seeds, pinned package versions, and outputs saved in the notebooks.
- **Laptop-sized:** each notebook runs on a CPU in seconds to a few minutes, using simulators.

## The notebooks

| # | Notebook | Track | What it does | Result in brief |
|---|---|---|---|---|
| 00 | [Qubits, superposition and the Born rule as a generative model](notebooks/00_born_rule_generative_model.ipynb) | Foundations | Shows that a measured quantum circuit is a probability distribution you can sample, and that entanglement creates correlations no independent-bits model can represent. | An independent model misses the Bell state by exactly log 2 nats. No advantage claimed. |
| 01 | [Quantum Circuit Born Machine vs classical baselines](notebooks/01_qcbm_vs_classical.ipynb) | Quantum Generative AI | Trains a 9-qubit QCBM on 3×3 bars-and-stripes and compares it with independent-pixel, pairwise Boltzmann and memorisation baselines. | **Negative result:** the 45-parameter classical Boltzmann machine fits perfectly (KL ≈ 0); the 99-parameter QCBM reaches KL 0.27. |
| 02 | [Barren plateaus: when quantum models stop learning](notebooks/02_barren_plateaus.ipynb) | Quantum Generative AI | Measures how gradient variance shrinks with qubit count for global vs local cost functions and shallow vs deep circuits. | Gradients shrink ~2.4× per qubit with a global cost, ~1.9× for deep circuits, and ~1.5× for shallow local costs (the trainable regime). |
| 03 | [Quantum dependence model for synthetic tabular data](notebooks/03_quantum_synthetic_data.ipynb) | Quantum Synthetic Data | Generates synthetic payments data with a rare three-way "fraud ring" pattern and compares six generators against the known true distribution. | Pairwise models (incl. the Gaussian copula) miss the rare pattern by ~70%; the QCBM (4% ± 3% over 5 seeds) and a classical 3-way model (11%) both capture it. **No quantum advantage claimed**: the gap is within noise. |
| 04 | [RAG context selection as a QUBO: QAOA vs greedy vs annealing](notebooks/04_qubo_rag_context_selection.ipynb) | Quantum RAG Systems | Formulates choosing RAG passages (relevant but not redundant) as a QUBO and solves it with top-k, MMR, simulated annealing, QAOA (p = 1–3) and exhaustive search. | The formulation is the win: top-3 covers 1 of 3 aspects, the QUBO optimum 3 of 3. QAOA reaches the optimum in 100 shots at p ≥ 2 but isn't competitive with classical solvers at this size. |
| 05 | [Complex-valued ("quantum-inspired") knowledge-graph embeddings](notebooks/05_complex_kg_embeddings.ipynb) | Quantum Knowledge Graphs | Compares real-valued DistMult with complex-valued ComplEx on a synthetic org chart, over 5 train/test splits. Fully classical, quantum-inspired. | ComplEx learns the direction of "manages" 92% of the time vs DistMult's forced 50%; both are modest at predicting hidden managers (MRR 0.23 vs 0.14). |
| 06 | [The simulation wall: what it costs to simulate qubits classically](notebooks/06_simulation_wall.ipynb) | Quantum + HPC | Measures statevector simulation time for 4–22 qubits on two simulators and computes memory needs and this machine's qubit limit. | Time roughly doubles per qubit on both simulators; the optimised one is faster by a constant factor, not a different growth rate. |
| 07 | [A learned decoder for the repetition code](notebooks/07_learned_decoder.ipynb) | Quantum-AI Convergence | Verifies a syndrome-measurement circuit on a quantum simulator, then compares minimum-weight, neural-network and maximum-likelihood decoders under independent and correlated noise. | Under correlated noise the learned decoder closes most of the gap to optimal; under independent noise it struggles at low error rates, where rare syndromes are under-sampled. |

All results above come from the saved notebook outputs; open a notebook to see the full numbers and charts.

## Run them

```bash
git clone https://github.com/prgore/qgai-notebooks.git
cd qgai-notebooks
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/      # or open in VS Code / JupyterLab (pip install jupyterlab)
```

Each notebook can also be opened in Google Colab: open `https://colab.research.google.com/github/prgore/qgai-notebooks/blob/main/notebooks/<file>.ipynb`, then run `!pip install -r https://raw.githubusercontent.com/prgore/qgai-notebooks/main/requirements.txt` in the first cell.

## How the notebooks are built

Each notebook is written as a plain Python file in [`src/`](src/) with `# %%` cell markers, which keeps diffs readable. [`build.py`](build.py) converts them to `.ipynb` and executes them so the published outputs are real:

```bash
python build.py        # rebuild and run all notebooks (about 7 minutes on a 4-core CPU)
python build.py 04     # only notebook 04
```

Timings in notebook 06 depend on hardware; every other result is deterministic for the pinned versions.

## Contributing

Issues and pull requests are welcome, especially stronger classical baselines, bugs in our comparisons, or results that contradict ours. Researchers, engineers and students interested in collaborating can reach out via [qgai.in](https://www.qgai.in/#contact).

## Licence

Code and notebooks: [MIT](LICENSE) © 2026 Prakash Gore / QGAI.
